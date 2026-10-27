# Multimodal MRI Reconstruction

基于 PyTorch 的多模态 MRI 重建核心实现，使用 HFS-SDE 扩散模型，以 PD 为辅助模态重建 PDFS。默认输入是目标与辅助模态的复数图像，分别以实部、虚部表示，共 4 个通道，空间尺寸为 320 × 320。

本仓库提供核心模型、训练、采样和评估流程。数据集、灵敏度图、采样掩码、训练权重及实验结果需要自行准备；消融实验与专用批处理脚本不包含在此核心发布版中。

## 代码结构

```text
main.py                         训练与采样入口
run_lib.py                      数据预处理、训练循环、采样及指标汇总
losses.py                       扩散训练损失与梯度累积
sampling.py                     自适应 Heun 采样
sde_lib.py                      SDE 定义与反向过程
save_data_slice.py              检查配对数据并生成切片索引
models/
  ddpm.py                       U-Net、CSP 瓶颈与特征缓存
  layers.py                     卷积、残差块、CBAM 与早期门控融合
  model_utils.py                模型注册与 score 包装
  normalization.py              归一化模块
  ema.py                        指数滑动平均
configs/
  default_fastMRI_configs.py    优化器、随机种子与公共配置
  hfssde/ddpm_continuous.py     fastMRI 多模态训练与采样配置
utils/
  datasets.py                   配对数据加载
  utils.py                      复数运算、FFT、掩码与 checkpoint 工具
  calc.py                       PSNR、SSIM、NMSE、HFEN
requirements.txt
```

## 环境

目标运行环境为 Linux、Python 3.10 和 NVIDIA GPU。核心模型计算使用 PyTorch；TensorFlow CPU 包用于现有文件接口。以下组合是整理后的依赖基线，未在本次源码发布时重新完成全量 GPU 训练验证。

```bash
conda create -n HFS python=3.10 -y
conda activate HFS
python -m pip install --upgrade pip
# CUDA 12.1 示例；按服务器驱动选择官方提供的 CUDA wheel。
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch 与 torchvision 的配对及其他 CUDA 安装命令见 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/#v251)。已安装对应 CUDA wheel 后，requirements 中的版本约束会复用它。

## 数据与掩码

在仓库根目录准备以下结构。外部数据目录也可以通过 Linux 符号链接放到 `data/`。

```text
data/
  train/
    pdfs/kspace/*.h5
    pdfs/map/*.h5
    pd/kspace/*.h5
    knee-pair.csv
    data_slice.pkl
  test/
    pdfs/kspace/*.h5
    pdfs/map/*.h5
    pd/kspace/*.h5
    test-pair.csv
    data_slice.pkl
mask/
  low_frequency_acs16.mat
  uniform_acc4_acs24.mat
```

- `kspace` HDF5 数据集形状为 `[切片数, 线圈数, H, W]`，保存复数 k-space。
- `s_maps` HDF5 数据集保存 PDFS 的复数灵敏度图，文件名与对应 PDFS 文件一致。当前实现也用这些图合成配对 PD，输入需满足对应的空间与线圈约定。
- 配对 CSV 格式为 `PD,PDFS,score`，前两列是不带 `.h5` 的文件名。必须确保配对切片空间对应。
- 当前 fastMRI 加载器在图像域中心裁剪至 320 × 320，并在训练与测试中跳过每个扫描的前 8 张切片。
- MAT 掩码必须包含二维数组 `mask`，形状为 `[320,320]`。训练默认使用 ACS=16 的低频掩码，测试默认使用 4 倍等距欠采样、ACS=24 的掩码。二者用途不同。

生成并检查训练索引：

```bash
python save_data_slice.py
```

生成测试索引：

```bash
python save_data_slice.py \
  --directory data/test/pdfs/kspace \
  --maps-directory data/test/pdfs/map \
  --pd-directory data/test/pd/kspace \
  --pairs data/test/test-pair.csv \
  --output data/test/data_slice.pkl
```

## 训练

在仓库根目录运行单 GPU 训练：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=train --workdir=results
```

双 GPU 训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=train --workdir=results
```

默认配置为 190 epoch、每进程 batch size 8、梯度累积拆分数 2、随机种子 1000。日志中的 `Step` 是 batch 迭代编号；训练会随机采样扩散时间，并非每次训练都执行完整反向重建。损失为带频率投影的 score matching 损失，具体定义见 `losses.py`。

权重保存至 `results/<时间戳与配置>/checkpoints/`。每 10 epoch 保存一次，150 epoch 后每 5 epoch 保存一次。恢复训练时同时指定权重和已完成的 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=train --workdir=results \
  --config.training.resume_checkpoint=/absolute/path/to/checkpoint_70.pth \
  --config.training.resume_epoch=70
```

## 采样与评估

使用自己的已训练 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=sample --workdir=results \
  --config.sampling.checkpoint_path=/absolute/path/to/checkpoint_190.pth \
  --config.sampling.ckpt=190 \
  --config.sampling.folder=fastMRI
```

也可以通过 `MRI_CHECKPOINT` 环境变量指定权重。默认使用 raw 权重、batch size 1、关闭鲁棒性扰动及特征复用。需要 EMA 时设置 `--config.sampling.weight_source=ema`。

默认保留历史评估方式 `metric_protocol=legacy`；需要标准评估方式时设置 `--config.sampling.metric_protocol=standard`。比较不同实验前应统一权重来源、评估方式、掩码和数据划分，具体实现见 `utils/calc.py`。

每张切片输出 PSNR、SSIM、NMSE、HFEN、推理时间、GPU 峰值显存及模型参数量；结束后汇总质量指标均值与标准差。日志位于 `data_analy/robust/fastMRI/`，运行结果目录位于 `results/fastMRI/`。默认关闭 MAT 输出，可用 `--config.sampling.save_mat_outputs=True` 开启。

采样采用自适应步长，实际迭代数因输入与模型而异；当前循环上限为 200，`model.num_scales=1000` 不代表固定执行 1000 次采样。GPU 峰值显存来自 PyTorch allocated memory。FLOPs 来自 profiler 支持的算子，单切片值包含基于网络调用次数的估计，不能视为包含所有 FFT 等操作的精确端到端 FLOPs。

## 来源说明

源码保留了上游文件中的版权和许可声明。相关实现包括 [Score SDE PyTorch](https://github.com/yang-song/score_sde_pytorch)、[DDPM](https://github.com/hojonathanho/diffusion) 及 [pytorch_ema](https://github.com/fadel/pytorch_ema)；复用相应文件时请遵循文件及上游项目的许可。
