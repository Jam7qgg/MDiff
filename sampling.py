"""Various sampling methods."""
print("--- DIAGNOSTIC: Loading MODIFIED sampling.py ---")
import functools
import time

import torch

import abc
import sde_lib
import numpy as np
from datetime import datetime
from models.model_utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from models import model_utils as mutils
from utils.utils import *
from pathlib import Path
import cv2
from scipy import integrate

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps, atb_mask, train_mask, device=None):
    """Create a sampling function.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
    """
    if device is None:
        device = config.device

    sampler_name = config.sampling.method
    if sampler_name.lower() == "pc":
       predictor = get_predictor(config.sampling.predictor.lower())
       sampling_fn = get_pc_sampler(sde=sde,
                                 config=config,
                                 shape=shape,
                                 predictor=predictor,
                                 denoise=True,
                                 eps=eps,
                                 device=device,
                                 atb_mask=atb_mask,
                                 train_mask=train_mask,
                                 abstol=config.sampling.abstol,
                                 reltol=config.sampling.reltol,
                                 safety=config.sampling.safety,
                                 exp=config.sampling.exp,
                                 adaptive=config.sampling.method == "adaptive",
                                 h_init=config.sampling.h_init,
                                 inverse_scaler=inverse_scaler)
    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, atb_mask,train_mask,probability_flow=False,shape=None,eps=1e-3,
                 abstol=1e-2, reltol=1e-2, safety=.9, exp=0.9):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn,probability_flow)
        self.score_fn = score_fn
        self.atb_mask = atb_mask
        self.train_mask = train_mask

    @abc.abstractmethod
    def update_fn(self, x, t, h, x_prev,atb,csm):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

@register_predictor(name='heun')
class AdaptivePredictor(Predictor):
    def __init__(self, sde, score_fn, atb_mask, train_mask, shape,probability_flow=False, eps=1e-3,
                 abstol=0.05, reltol=0.1, safety=0.9, exp=0.9):
        super().__init__(sde, score_fn, atb_mask, train_mask, probability_flow)
        self.h_min = 1e-4  # min step-size
        self.t = sde.T  # starting t
        self.eps = eps  # end t
        self.abstol = abstol
        self.reltol = reltol
        self.error_use_prev = True
        self.safety = safety
        self.n = shape[1] * shape[2] * shape[3]  # size of each sample
        self.exp = exp


        # "L2_scaled":
        def norm_fn(x):
            return torch.sqrt(torch.sum((x) ** 2, dim=(1, 2, 3), keepdim=True) / self.n)

        self.norm_fn = norm_fn

    def update_fn(self, x, t, h, x_prev, atb, csm, use_cache_this_step=False):
        # Note: both h and t are vectors with batch_size elems (this is because we want adaptive step-sizes for each sample separately)
        device = x.device
        my_rsde = self.rsde.sde

        h_ = h[:, None, None, None].to(device).float()  # expand for multiplications
        t_ = t[:, None, None, None].to(device).float()  # expand for multiplications
        with torch.amp.autocast(device_type='cuda',dtype=torch.float16):  # 使用混合精度推理，结果没区别，计算速度快1/3左右

            z = torch.randn_like(x).to(device)
            if x.shape[1] == 4:
                z_pdfs = z[:, :2, :, :]
                z_pd = z[:, 2:, :, :]

                z_pdfs = (
                    c2r(ifft2c_2d((1 - self.train_mask) * fft2c_2d(r2c(z_pdfs))))
                    .type(torch.FloatTensor)
                    .to(x.device)
                )
                # z_pd is now 2 channels, treat as complex
                z_pd_complex = r2c(z_pd)

                z_pd_masked_complex = ifft2c_2d((1 - self.train_mask) * fft2c_2d(z_pd_complex))

                # Convert back to 2 channels
                z_pd = c2r(z_pd_masked_complex).type(torch.FloatTensor).to(x.device)

                z = torch.cat([z_pdfs, z_pd], dim=1)
            else:
                z = (
                    c2r(ifft2c_2d((1 - self.train_mask) * fft2c_2d(r2c(z))))
                    .type(torch.FloatTensor)
                    .to(x.device)
                )

            drift, diffusion = my_rsde(x, t, atb, csm, self.atb_mask, use_cache_this_step=use_cache_this_step)
            drift = drift.float()
            # Heun's method for SDE (while Lamba method only focuses on the non-stochastic part, this also includes the stochastic part)
            K1_mean = -h_ * drift
            K1 = K1_mean + diffusion[:, None, None, None] * torch.sqrt(h_) * z
            drift_Heun, diffusion_Heun = my_rsde(x + K1, t - h, atb, csm, self.atb_mask, use_cache_this_step=use_cache_this_step)
            drift_Heun = drift_Heun.float()
            K2_mean = -h_ * drift_Heun
            K2 = K2_mean + diffusion_Heun[:, None, None, None] * torch.sqrt(h_) * z

            E = 1 / 2 * (K2 - K1)  # local-error between EM and Heun (SDEs) (right one)


            # Extrapolate using the Heun's method result
            x_new = x + (1 / 2) * (K1 + K2)
            x_check = x + K1


            # Calculating the error-control
            if self.error_use_prev:
                reltol_ctl = torch.maximum(torch.abs(x_prev), torch.abs(x_check)) * self.reltol
            else:
                reltol_ctl = torch.abs(x_check) * self.reltol
            err_ctl = torch.clamp(reltol_ctl, min=self.abstol)

            # Normalizing for each sample separately
            E_scaled_norm = self.norm_fn(E / err_ctl)

            accept = E_scaled_norm <=torch.ones_like(E_scaled_norm)

            x = torch.where(accept, x_new, x)
            x_prev = torch.where(accept, x_check, x_prev)
            t_ = torch.where(accept, t_ -3*h_, t_)

            # PD Replacement: Enforce consistency with img_pd
            if x.shape[1] == 4 and isinstance(atb, (list, tuple)):
                _, img_pd = atb
                if img_pd is not None:
                    # Optimized: Calculate std directly to avoid expensive FFT in marginal_prob
                    # Formula from HFS_SDE.marginal_prob
                    beta_0 = self.rsde.config.model.beta_min
                    beta_1 = self.rsde.config.model.beta_max
                    log_mean_coeff = (
                        -0.25 * t_ ** 2 * (beta_1 - beta_0) - 0.5 * t_ * beta_0
                    )
                    std_t = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))
                    # std_t is already (B, 1, 1, 1) because t_ was expanded

                    z_pd_new = torch.randn_like(x[:, 2:, :, :])
                    x_pd_enforced = img_pd + std_t * z_pd_new

                    # Replace PD channel
                    x = torch.cat([x[:, :2, :, :], x_pd_enforced], dim=1)

            # Change the step-size
            h_max = torch.clamp(t_ - 1e-3,min=0)  # max step-size must be the distance to the end (we use maximum between that and zero in case of a tiny but negative value: -1e-10)  #self.eps修改为1e-3


            E_scaled_norm = torch.clamp(E_scaled_norm, min=1e-8)
            E_pow = torch.where(h_ == 0, h_, torch.pow(E_scaled_norm,
                                                       -self.exp))  # Only applies power when not zero, otherwise, we get nans
            h_new = torch.minimum(h_max, self.safety * h_ * E_pow)
            h_new = torch.maximum(h_new, self.h_min*torch.ones_like(h_new))  #新加入使用 h_min

            return x, x_prev, t_.reshape((-1)), h_new.reshape((-1)),accept


def shared_predictor_update_fn(x, t, h, atb, csm, atb_mask, train_mask, sde, model, predictor,
                               x_prev=None, shape=None, eps=1e-3, abstol=1e-2, reltol=0.05,
                               safety=0.9, exp=0.9, use_cache_this_step=False):
    """A wrapper that configures and returns the update function of predictors."""

    predictor_obj = predictor(sde, model, atb_mask, train_mask, shape=shape, eps=eps,
                              abstol=abstol, reltol=reltol, safety=safety, exp=exp)

    result = predictor_obj.update_fn(x, t, h, x_prev, atb, csm, use_cache_this_step=use_cache_this_step)

    return result


def get_pc_sampler(config, sde, shape, predictor, train_mask, atb_mask,
                   denoise=True, device='cuda',
                   eps=1e-3, abstol=1e-2, reltol=0.05, safety=0.9, exp=0.9, adaptive=False, h_init=0.1,
                   inverse_scaler=None):
    """Create a SDE sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            atb_mask=atb_mask,
                                            train_mask=train_mask,
                                            shape=shape,
                                            predictor=predictor,
                                            eps=eps, abstol=abstol, reltol=reltol,
                                            safety=safety,
                                            exp=exp)

    def pc_sampler_adaptive(model, atb, atb_to_image, csm):
        """ The PC sampler function with N-interval cache reuse."""
        # Ensure GPU is synced before starting timer for accurate measurement 11
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.time()

        caching_activated = False
        reuse_N = config.model.reuse_N
        reuse_counter = 0
        with torch.no_grad():
            # Define t (initial time) early as it is needed for PD initialization
            t = (torch.ones(shape[0]).to(device) * sde.T).type(torch.FloatTensor).to(device)

            # Initial sample
            if isinstance(sde, sde_lib.HFS_SDE):
                z = sde.prior_sampling(shape).to(config.device)

                if isinstance(atb, (list, tuple)):
                    atb_pdfs, img_pd = atb
                else:
                    atb_pdfs = atb
                    img_pd = None

                low_fre_img_pdfs = (
                    c2r(Emat_xyt_complex(atb_pdfs * train_mask, True, r2c(csm), 1.0))
                    .type(torch.FloatTensor)
                    .to(config.device)
                )

                # Case 2: 4 channels (2 PDFS + 2 PD)
                if img_pd is not None and shape[1] == 4:
                     print(f">>> DEBUG Sampler: Initializing with PD auxiliary modality. Shifted? {img_pd.shape}")
                     img_pd = img_pd.to(config.device, dtype=torch.float32)
                     z_pdfs = z[:, :2, :, :]
                     z_pd = z[:, 2:, :, :]

                     x_pdfs = low_fre_img_pdfs + c2r(
                        ifft2c_2d((1 - train_mask) * fft2c_2d(r2c(z_pdfs)))
                     ).type(torch.FloatTensor).to(config.device)


                     _, std_T = sde.marginal_prob(img_pd, t)



                     z_pd_init = torch.randn_like(img_pd)
                     x_pd = img_pd + std_T[:, None, None, None] * z_pd_init

                     x = torch.cat([x_pdfs, x_pd], dim=1)

                else:
                    # Default case (e.g. only PDFS or unexpected shape)
                    # If z has 4 channels but we fall here, r2c(z) will fail if not handled.
                    # We must ensure we only process the relevant channels or handle the full tensor.

                    if z.shape[1] == 4:
                         # Fallback for 4-channel noise without specific PD handling above
                         # Split processing
                         z_pdfs = z[:, :2, :, :]
                         z_pd = z[:, 2:, :, :]

                         x_pdfs = low_fre_img_pdfs + c2r(
                            ifft2c_2d((1 - train_mask) * fft2c_2d(r2c(z_pdfs)))
                         ).type(torch.FloatTensor).to(config.device)

                         # For PD, if no img_pd provided, just pure noise?
                         # Or if we have img_pd but missed the block above:
                         x_pd = c2r(ifft2c_2d(fft2c_2d(r2c(z_pd)))).type(torch.FloatTensor).to(config.device)

                         x = torch.cat([x_pdfs, x_pd], dim=1)
                    else:
                        x = low_fre_img_pdfs + c2r(
                            ifft2c_2d((1 - train_mask) * fft2c_2d(r2c(z)))
                        ).type(torch.FloatTensor).to(config.device)

            else:
                x = sde.prior_sampling(shape).to(device)

            h = (torch.ones(shape[0]).to(device) * h_init).type(torch.FloatTensor).to(device)# initial step_size

            # t is already defined above

            x_prev = x
            N = 0
            max_N=200
            total_iterations = 0

            while ((t > 1e-3) & (torch.abs(t - 1e-3) > 0.01)).any() and total_iterations < max_N:
                # Check if caching should be enabled based on config (reuse_N > 0)
                if not caching_activated and reuse_N > 0:
                    # 兼容DataParallel封装
                    if hasattr(model, 'module') and hasattr(model.module, 'enable_caching'):
                        model.module.enable_caching(reuse_N=reuse_N)
                    elif hasattr(model, 'enable_caching'):
                        model.enable_caching(reuse_N=reuse_N)
                    caching_activated = True
                    print(f"[CACHE] Enabled N-interval caching with reuse_N={reuse_N}")

                # 计算当前步是否复用缓存
                use_cache_this_step = False
                cycle_idx = 0
                if caching_activated:
                    cycle_len = reuse_N + 1
                    cycle_idx = reuse_counter % cycle_len
                    use_cache_this_step = (cycle_idx != 0)  # 0表示完整推理，其他为复用
                mode_str = "CACHE" if use_cache_this_step else "FULL"

                x, x_prev, t, h, accept = predictor_update_fn(
                    x, t, h, atb, csm, x_prev=x_prev, model=model, use_cache_this_step=use_cache_this_step)

                # 统计本步accepted数目（batch内样本数）
                try:
                    accept_count = int(accept.sum().item())
                except Exception:
                    accept_count = int(bool(accept))
                N += accept_count


                # if total_iterations % 10 == 0 or total_iterations < 5:
                #     print(f"Iter {total_iterations:4d} | mode={mode_str:5s} | cycle_idx={cycle_idx}/{reuse_N} | reuse_counter={reuse_counter} | h={h} | t={t} | accepted+= {accept_count} | total_accepted={N}")

                reuse_counter += 1
                total_iterations += 1

            if denoise:
                eps_t = torch.ones(shape[0]).to(device) * 1e-3
                u, std = sde.marginal_prob(x, eps_t)

                # Check for NaNs before denoise step
                if torch.isnan(x).any():
                     print("Warning: x contains NaN before denoise step")

                with torch.amp.autocast(device_type='cuda',dtype=torch.float16):
                    model_output = model(x, eps_t)
                    if torch.isnan(model_output).any():
                        print("Warning: model_output contains NaN")
                        model_output = torch.where(torch.isnan(model_output), torch.zeros_like(model_output), model_output)

                    x = x + (std[:, None, None, None] ** 2) * model_output

                # Check for NaNs after denoise step
                if torch.isnan(x).any():
                     print("Warning: x contains NaN after denoise step")

            # Ensure GPU is synced before stopping timer
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            elapsed_time = end_time - start_time
            batch_size = shape[0]
            time_per_slice = elapsed_time / batch_size if batch_size > 0 else elapsed_time
            avg_time_per_step = (elapsed_time / total_iterations) * 1000 if total_iterations > 0 else 0
            peak_gpu_memory_mb = 0.0
            if torch.cuda.is_available():
                peak_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            # --- Disable and Clear Caching for the model ---
            if caching_activated:
                if hasattr(model, 'module') and hasattr(model.module, 'clear_cache'):
                    model.module.clear_cache()
                elif hasattr(model, 'clear_cache'):
                    model.clear_cache()
            # ---------------------------------------------

            print("-" * 60)
            print(f"Sampling Statistics:")
            print(f"  Total Time       : {elapsed_time:.4f} seconds")
            print(f"  Time/Slice       : {time_per_slice:.4f} seconds")
            print(f"  Total Steps      : {total_iterations}")
            print(f"  Avg Time/Step    : {avg_time_per_step:.2f} ms/step")
            print(f"  Effective Steps  : {N}")
            print(f"  Skipped Steps    : {total_iterations - N}")
            print(f"  Peak GPU Memory  : {peak_gpu_memory_mb:.2f} MB")
            print("-" * 60)

            if inverse_scaler is not None:
                x = inverse_scaler(x)

            stats = {
                "total_time_sec": elapsed_time,
                "time_per_slice_sec": time_per_slice,
                "total_steps": total_iterations,
                "avg_time_per_step_ms": avg_time_per_step,
                "effective_steps": N,
                "skipped_steps": total_iterations - N,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "batch_size": batch_size,
            }

            return x, N + 1, stats

    return pc_sampler_adaptive


def add_title(path, title):
    img1 = cv2.imread(path)

    black = [0, 0, 0]
    constant = cv2.copyMakeBorder(
        img1, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=black
    )
    height = 20
    violet = np.zeros((height, constant.shape[1], 3), np.uint8)
    violet[:] = (255, 0, 180)

    vcat = cv2.vconcat((violet, constant))

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(
        vcat, str(title), (violet.shape[1] // 2, height - 2), font, 0.5, (0, 0, 0), 1, 0
    )
    cv2.imwrite(path, vcat)











