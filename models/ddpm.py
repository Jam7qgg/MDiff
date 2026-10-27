"""DDPM model.

This code is the pytorch equivalent of:
https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py
"""
import torch
import torch.nn as nn
import functools
from torch.nn.functional import scaled_dot_product_attention

from . import model_utils, layers, normalization
from .layers import CBAMBlock  # 导入新注意力模块
RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init
dws_conv3x3=layers.ddpm_dws_conv3x3
CbamBlock=layers.CBAMBlock
CBAMAttn = functools.partial(CBAMBlock, reduction=16)  # 使用新的注意力机制


# 封装 DDPM Bottleneck 模块
class DDPMBottleneck(nn.Module):
    def __init__(self, in_ch, temb_dim=None, dropout=0.1, act=None):
        super(DDPMBottleneck, self).__init__()
        if act is None:
            act = nn.SiLU()

        self.resblock1 = ResnetBlockDDPM(
            act=act, in_ch=in_ch, out_ch=in_ch,
            temb_dim=temb_dim, dropout=dropout
        )

        self.cbam = CBAMAttn(channels=in_ch)

        self.resblock2 = ResnetBlockDDPM(
            act=act, in_ch=in_ch, out_ch=in_ch,
            temb_dim=temb_dim, dropout=dropout
        )

    def forward(self, x, temb=None):
        x = self.resblock1(x, temb)
        x = self.cbam(x)
        x = self.resblock2(x, temb)
        return x

# 通用卷积模块 (用于 CSP 结构)
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=p if p is not None else k//2)
        self.norm = nn.GroupNorm(32, c2)
        self.act = nn.SiLU()

        self.conv.weight.data = default_initializer()(self.conv.weight.shape)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# CSP 瓶颈层 - 集成 DDPMBottleneck
class C3DDPMBottleneck(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, temb_dim=None, dropout=0.1, act=None):
        super(C3DDPMBottleneck, self).__init__()
        if act is None:
            act = nn.SiLU()

        c_ = int(c2 * e)  # 隐藏通道数

        # CSP 分支
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)

        # 主路径堆叠 Bottleneck
        self.bottlenecks = nn.ModuleList()
        for _ in range(n):
            self.bottlenecks.append(
                DDPMBottleneck(
                    in_ch=c_,
                    temb_dim=temb_dim,
                    dropout=dropout,
                    act=act
                )
            )

    def forward(self, x, temb=None):
        # 路径 1: 经过 Bottlenecks
        route1 = self.cv1(x)
        for m in self.bottlenecks:
            route1 = m(route1, temb)

        # 路径 2: 短路连接
        route2 = self.cv2(x)

        # 融合
        out = self.cv3(torch.cat([route1, route2], dim=1))
        return out

@model_utils.register_model(name="ddpm")
class DDPM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)
        self.register_buffer("sigmas", torch.tensor(model_utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [
            config.data.image_size // (2**i) for i in range(num_resolutions)
        ]
        self.num_scales = config.model.num_scales # Required for caching logic

        # --- Caching attributes for accelerated sampling 1---
        self.caching_enabled = False
        self.cached_160x160_h = None  # 缓存160x160特征图（80x80->160x160上采样后的特征图）
        self.reuse_N = 0 # 缓存复用间隔
        self.reuse_counter = 0 # 当前缓存复用步数计数器
        # ----------------------------------------------------

        AttnBlock = functools.partial(layers.AttnBlock)
        self.conditional = conditional = config.model.conditional
        ResnetBlock = functools.partial(
            ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout
        )
        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_initializer()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_initializer()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels

        # [NEW] Gated Early Fusion
        modules.append(layers.GatedEarlyFusion(channels, channels))

        # Downsampling block1
        modules.append(conv3x3(channels, nf))

        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                if i_level > 0 and i_level < num_resolutions - 1:
                    custom_block = functools.partial(
                        ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout
                    )
                    modules.append(custom_block(in_ch=in_ch, out_ch=out_ch, custom_conv=dws_conv3x3))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(CBAMAttn(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                # 下采样 (部分层级使用深度可分离卷积)
                if i_level > 0 and i_level < num_resolutions - 2:
                    modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv, custom_conv=dws_conv3x3))
                else:
                    modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        # CSP 结构的瓶颈层
        modules.append(C3DDPMBottleneck(
            c1=in_ch,
            c2=in_ch,
            n=2,
            e=0.5,
            temb_dim=4*nf,
            dropout=dropout,
            act=act
        ))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                if i_level > 0 and i_level < num_resolutions - 1:
                    custom_block = functools.partial(
                        ResnetBlockDDPM, act=act, temb_dim=4 * nf, dropout=dropout
                    )
                    modules.append(custom_block(in_ch=in_ch + hs_c.pop(), out_ch=out_ch, custom_conv=dws_conv3x3))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(CBAMAttn(channels=in_ch))
            if i_level != 0:
                # 上采样 (部分层级使用深度可分离卷积)
                if i_level > 1 and i_level < num_resolutions:
                    custom_up = functools.partial(Upsample, with_conv=resamp_with_conv)
                    modules.append(custom_up(channels=in_ch, custom_conv=dws_conv3x3))
                else:
                    modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=32, eps=1e-6))
        modules.append(conv3x3(in_ch, channels, init_scale=0.0))  # 上采样最后一层也使用 conv 3x3
        self.all_modules = nn.ModuleList(modules)

        self.scale_by_sigma = config.model.scale_by_sigma

    def enable_caching(self, reuse_N):
        """Enable the N-interval caching mechanism for accelerated sampling."""
        self.caching_enabled = True
        self.reuse_N = reuse_N
        self.reuse_counter = 0
        print(f"INFO: DDPM N-interval Caching enabled, reuse_N = {reuse_N}")

    def clear_cache(self):
        """Clear the cache, must be called after a sample is generated."""
        self.cached_160x160_h = None
        self.caching_enabled = False
        self.reuse_counter = 0
        print("INFO: DDPM Cache cleared (80x80->160x160特征图).")

    def forward(self, x, labels, use_cache_this_step=False):
        modules = self.all_modules
        m_idx = 0
        hs = []
        
        # 1. 时间步/条件嵌入 (Timestep/Conditional Embedding)
        if self.conditional:
            timesteps = labels
            temb = layers.get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        # 2. 数据预处理
        if self.centered:
            h = x
        else:
            h = 2 * x - 1.0

        # 3. 推理逻辑：完整路径 vs 缓存复用路径
        if (not self.caching_enabled) or (not use_cache_this_step):
            # === 完整推理路径 (Full Pass) ===
            # 计算所有层级，并在适当时机更新缓存
            
            # (0) Early Fusion
            h = modules[m_idx](h)
            m_idx += 1

            # (1) Encoder: 初始卷积
            hs = [modules[m_idx](h)]
            m_idx += 1
            
            # (2) Encoder: 逐层特征提取
            # Level 0 (320x320)
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            
            # Downsample -> 160x160
            hs.append(modules[m_idx](hs[-1]))
            m_idx += 1
            
            # Level 1+ (160x160 -> 40x40)
            for i_level in range(1, self.num_resolutions):
                for i_block in range(self.num_res_blocks):
                    h = modules[m_idx](hs[-1], temb)
                    m_idx += 1
                    if h.shape[-1] in self.attn_resolutions:
                        h = modules[m_idx](h)
                        m_idx += 1
                    hs.append(h)
                if i_level != self.num_resolutions - 1:
                    hs.append(modules[m_idx](hs[-1]))
                    m_idx += 1
            
            # (3) Bottleneck: 中间层
            h = hs[-1]
            h = modules[m_idx](h, temb)
            m_idx += 1

            # (4) Decoder: 上采样与特征融合
            for i_level in reversed(range(self.num_resolutions)):
                for i_block in range(self.num_res_blocks + 1):
                    # 跳跃连接系数调整
                    if i_block == 0: k = 0.72
                    elif i_block == 1: k = 0.82
                    else: k = 0.90
                    
                    h = modules[m_idx](torch.cat([h, k * hs.pop()], dim=1), temb)
                    m_idx += 1
                
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                
                if i_level != 0:
                    # Upsample
                    h = modules[m_idx](h)
                    m_idx += 1
                    
                    # [Cache Update] 在 Level 2 结束 (即将进入 Level 1, 160x160) 时缓存特征图
                    if i_level == 2 and self.caching_enabled:
                        self.cached_160x160_h = h.clone()

            if self.caching_enabled:
                self.reuse_counter = 1
                
        else:
            # === 特征图复用路径 (Cache Reuse Pass) ===
            # 核心策略：
            # 1. 仅运行浅层 Encoder (Level 0, 1) 以生成浅层 Decoder 所需的 Skip Connections。
            # 2. 跳过深层 Encoder、Bottleneck 和深层 Decoder。
            # 3. 直接注入缓存的深层特征图，仅运行浅层 Decoder (Level 1, 0)。
            
            if self.cached_160x160_h is None:
                print("WARNING: Cache is None, falling back to full computation")
                return self.forward(x, labels, use_cache_this_step=False)

            # (0) Early Fusion (Same as Full Pass)
            h = modules[m_idx](h)
            m_idx += 1

            # (1) 浅层 Encoder (Partial Encoder)
            # 初始卷积
            hs = [modules[m_idx](h)]
            m_idx += 1

            # 运行 Level 0 和 Level 1
            for i_level in range(self.num_resolutions):
                if i_level < 2:
                    for i_block in range(self.num_res_blocks):
                        h = modules[m_idx](hs[-1], temb)
                        m_idx += 1
                        if h.shape[-1] in self.attn_resolutions:
                            h = modules[m_idx](h)
                            m_idx += 1
                        hs.append(h)
                    
                    if i_level != self.num_resolutions - 1:
                        # Downsample 处理
                        if i_level == 1:
                            # [关键逻辑] Level 1 结束时不执行下采样
                            # 因为我们跳过了深层网络，不需要生成 80x80 的特征图放入 hs 栈中
                            # 否则会导致 Decoder Level 1 取出错误的 Skip Connection
                            m_idx += 1
                        else:
                            hs.append(modules[m_idx](hs[-1]))
                            m_idx += 1
                else:
                    # (2) 跳过深层 Encoder (Level >= 2)
                    for i_block in range(self.num_res_blocks):
                        m_idx += 1 # ResBlock
                        if self.all_resolutions[i_level] in self.attn_resolutions:
                            m_idx += 1 # Attn
                    # Skip Downsample
                    if i_level != self.num_resolutions - 1:
                        m_idx += 1

            # (3) 跳过 Bottleneck
            m_idx += 1 
            
            # (4) 浅层 Decoder (Partial Decoder)
            for i_level in reversed(range(self.num_resolutions)):
                if i_level >= 2:
                    # 跳过深层 Decoder (Level 3, 2)
                    for i_block in range(self.num_res_blocks + 1):
                        m_idx += 1 # ResBlock
                    
                    if self.all_resolutions[i_level] in self.attn_resolutions:
                        m_idx += 1 # Attn
                    
                    if i_level != 0:
                        m_idx += 1 # Upsample
                    
                    # [Cache Injection] 注入缓存
                    if i_level == 2:
                        h = self.cached_160x160_h
                else:
                    # 运行浅层 Decoder (Level 1, 0)
                    for i_block in range(self.num_res_blocks + 1):
                        # 系数调整
                        if i_block == 0: k = 0.72
                        elif i_block == 1: k = 0.82
                        else: k = 0.90
                        
                        # 使用计算好的浅层 hs 和注入的特征图进行融合
                        h = modules[m_idx](torch.cat([h, k * hs.pop()], dim=1), temb)
                        m_idx += 1
                    
                    if h.shape[-1] in self.attn_resolutions:
                        h = modules[m_idx](h)
                        m_idx += 1
                    
                    if i_level != 0:
                        h = modules[m_idx](h)
                        m_idx += 1

            self.reuse_counter += 1

        # 4. 输出层 (Output Projection)
        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)

        if self.scale_by_sigma:
            used_sigmas = self.sigmas[labels.long(), None, None, None]
            h = h / used_sigmas
            
        return h
