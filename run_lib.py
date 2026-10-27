"""Training and evaluation for score-based generative models. """
import os
import time
import logging
import re
import math
import csv
import numpy as np
from PIL import Image
from models import ddpm
import losses
import sampling
from models import model_utils as mutils
from models.ema import ExponentialMovingAverage
import sde_lib
from absl import flags
import torch
import torch.nn.functional as F
from torch.utils import tensorboard
from utils.utils import *
import utils.datasets as datasets
import tensorflow as tf
from models.model_utils import from_flattened_numpy, to_flattened_numpy, get_score_fn

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

FLAGS = flags.FLAGS


def _infer_resume_epoch(path):
    match = re.search(r"checkpoint_(\d+)\.pth$", str(path))
    return int(match.group(1)) if match else 0


def _format_large_number(value):
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.3f} T"
    if value >= 1e9:
        return f"{value / 1e9:.3f} G"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K"
    return f"{value:.0f}"


def _count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def _as_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _first_text(value):
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def _magnitude_image(tensor):
    array = torch.abs(tensor.detach()).float().cpu().numpy()
    return np.squeeze(array)


def _save_ixi_visualization(recon, output_path):
    """Save the reconstructed image as one grayscale PNG."""
    recon_image = _magnitude_image(recon)
    image_min = float(np.min(recon_image))
    image_max = float(np.max(recon_image))
    scale = max(image_max - image_min, 1e-8)
    png_image = np.uint8(np.clip((recon_image - image_min) / scale, 0.0, 1.0) * 255.0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(png_image, mode="L").save(output_path)


def _gaussian_kernel2d(sigma, device, dtype):
    if sigma <= 0:
        raise ValueError("elastic_sigma must be greater than 0")
    radius = max(int(math.ceil(3.0 * sigma)), 1)
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-(coordinates ** 2) / (2.0 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d[None, None]


def _elastic_deform(image, max_displacement, sigma, seed):
    """Warp every channel with one smooth field capped at max_displacement pixels."""
    if max_displacement <= 0:
        return image, 0.0
    if image.ndim != 4:
        raise ValueError(f"Elastic deformation expects BCHW, got {tuple(image.shape)}")

    batch, _, height, width = image.shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(
        batch,
        2,
        height,
        width,
        generator=generator,
        dtype=torch.float32,
    ).to(image.device)

    kernel = _gaussian_kernel2d(float(sigma), image.device, torch.float32)
    padding = kernel.shape[-1] // 2
    displacement = F.conv2d(
        noise,
        kernel.expand(2, 1, -1, -1),
        padding=padding,
        groups=2,
    )
    magnitude = torch.sqrt(torch.sum(displacement ** 2, dim=1, keepdim=True))
    field_max = torch.amax(magnitude, dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    displacement = displacement * (float(max_displacement) / field_max)

    y = torch.linspace(-1.0, 1.0, height, device=image.device, dtype=torch.float32)
    x = torch.linspace(-1.0, 1.0, width, device=image.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    base_grid = torch.stack((grid_x, grid_y), dim=-1)[None].expand(batch, -1, -1, -1)
    offset_x = 2.0 * displacement[:, 0] / max(width - 1, 1)
    offset_y = 2.0 * displacement[:, 1] / max(height - 1, 1)
    sampling_grid = base_grid + torch.stack((offset_x, offset_y), dim=-1)

    warped = F.grid_sample(
        image.float(),
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).to(image.dtype)
    actual_max = float(
        torch.sqrt(torch.sum(displacement ** 2, dim=1)).amax().detach().cpu().item()
    )
    return warped, actual_max


def _robustness_tag(config):
    if not getattr(config.sampling, "robustness_perturbation", False):
        return "none"
    perturbation_type = str(
        getattr(config.sampling, "robustness_type", "shift")
    ).lower()
    if perturbation_type == "elastic":
        displacement = getattr(config.sampling, "elastic_max_displacement", 0)
        return f"elastic_max{float(displacement):g}px"
    if perturbation_type == "shift":
        direction = getattr(config.sampling, "robustness_shift_direction", "horizontal")
        pixels = getattr(config.sampling, "robustness_shift_pixels", 0)
        return f"shift_{direction}_{pixels}px"
    raise ValueError(
        f"Unsupported robustness_type={perturbation_type!r}; "
        "expected 'shift' or 'elastic'"
    )


def _profile_callable_flops(callable_fn):
    """Return profiler-supported FLOPs for one callable invocation."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.synchronize()
    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            callable_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)


def _profile_inference_flops(model, config, sample_batch_size=1):
    """Profile full, cache-partial, and gate forward paths."""
    profile = {
        "full_forward_flops": None,
        "partial_forward_flops": None,
        "gate_flops": None,
        "data_consistency_flops": None,
    }
    model_core = model.module if hasattr(model, "module") else model
    was_training = model.training
    try:
        model.eval()
        dummy_x = torch.randn(
            sample_batch_size,
            config.data.num_channels,
            config.data.image_size,
            config.data.image_size,
            device=config.device,
        )
        dummy_t = torch.ones(sample_batch_size, device=config.device) * 1e-3
        if (
            hasattr(model_core, "enable_caching")
            and hasattr(model_core, "reset_ablation_stats")
        ):
            for cache_name in (
                "cached_40x40_h",
                "cached_80x80_h",
                "cached_160x160_h",
            ):
                if hasattr(model_core, cache_name):
                    setattr(model_core, cache_name, None)
            model_core.caching_enabled = True
            model_core.reuse_N = 1
            model_core.reuse_counter = 0
            profile["full_forward_flops"] = _profile_callable_flops(
                lambda: model(dummy_x, dummy_t, use_cache_this_step=False)
            )
            profile["partial_forward_flops"] = _profile_callable_flops(
                lambda: model(dummy_x, dummy_t, use_cache_this_step=True)
            )
            model_core.caching_enabled = False
            model_core.reuse_counter = 0
            for cache_name in (
                "cached_40x40_h",
                "cached_80x80_h",
                "cached_160x160_h",
            ):
                if hasattr(model_core, cache_name):
                    setattr(model_core, cache_name, None)
        else:
            profile["full_forward_flops"] = _profile_callable_flops(
                lambda: model(dummy_x, dummy_t)
            )

        gate_module = next(
            (
                module
                for module in model_core.modules()
                if module.__class__.__name__ == "GatedEarlyFusion"
            ),
            None,
        )
        if gate_module is not None:
            profile["gate_flops"] = _profile_callable_flops(
                lambda: gate_module(dummy_x)
            )

    except Exception as exc:
        print(f"[Profile] FLOPs profiling failed: {exc}")
    finally:
        if hasattr(model_core, "caching_enabled"):
            model_core.caching_enabled = False
        if hasattr(model_core, "reuse_counter"):
            model_core.reuse_counter = 0
        for cache_name in (
            "cached_40x40_h",
            "cached_80x80_h",
            "cached_160x160_h",
        ):
            if hasattr(model_core, cache_name):
                setattr(model_core, cache_name, None)
        if hasattr(model_core, "reset_ablation_stats"):
            model_core.reset_ablation_stats()
        model_core._ablation_flops_profile = profile
        if was_training:
            model.train()
    return profile


def train(config, workdir):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    # Initialize DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank != -1

    if is_distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        config.device = torch.device(f"cuda:{local_rank}")
        # Adjust batch size per GPU
        # config.training.batch_size = config.training.batch_size // dist.get_world_size()
        # (Usually we keep batch_size as per-gpu in config or adjust it.
        #  If config.training.batch_size is global, we divide.
        #  But default_fastMRI_configs.py says batch_size=8. Assuming it is per-GPU or total?
        #  Usually configs specify per-GPU batch size. Let's assume it is per-GPU for now or keep as is.)

        # Seed adjustment for DDP
        seed = config.seed + local_rank
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Only rank 0 does logging and checkpointing
    if is_distributed and local_rank != 0:
        def logging_info(*args, **kwargs): pass
        logging.info = logging_info

    # The directory for saving test results during training
    sample_dir = os.path.join(workdir, "samples_in_train")
    if not is_distributed or local_rank == 0:
        tf.io.gfile.makedirs(sample_dir)

        tb_dir = os.path.join(workdir, "tensorboard")
        tf.io.gfile.makedirs(tb_dir)
        writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    score_model = mutils.create_model(config)
    # Remove DataParallel wrapper if present, as we use DDP or manage device manually
    if isinstance(score_model, torch.nn.DataParallel):
        score_model = score_model.module

    # Move model to device
    score_model = score_model.to(config.device)

    if is_distributed:
        score_model = DDP(score_model, device_ids=[local_rank], output_device=local_rank)

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    optimizer = losses.get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    if not is_distributed or local_rank == 0:
        tf.io.gfile.makedirs(checkpoint_dir)

    resume_checkpoint = getattr(config.training, "resume_checkpoint", "")
    start_epoch = 0
    if resume_checkpoint:
        state = restore_checkpoint(resume_checkpoint, state, device=config.device)
        start_epoch = getattr(
            config.training,
            "resume_epoch",
            _infer_resume_epoch(resume_checkpoint),
        )
        if not is_distributed or local_rank == 0:
            print(f"Resumed training from: {resume_checkpoint}")
            print(f"Continue from epoch: {start_epoch + 1}")

    initial_step = int(state["step"])

    # Build pytorch dataloader for training
    train_dl = datasets.get_dataset(config, "training")

    # Create data scaler and its inverse
    scaler = get_data_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
    elif config.training.sde.lower() == "hfssde":
        sde = sde_lib.HFS_SDE(config)
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(
        config,
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
    )

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for epoch in range(start_epoch, config.training.epochs):
        loss_sum = 0
        for step, batch in enumerate(train_dl):
            t0 = time.time()
            k0, csm, k0_pd = batch
            k0 = k0.to(config.device)
            csm = csm.to(config.device)
            k0_pd = k0_pd.to(config.device)

            # TODO: mask condition
            # PDFS Reconstruction
            label = Emat_xyt_complex(k0, True, csm, 1)  # 1x1x320x320
            label = c2r(label).float()
            label = scaler(label)

            # PD Reconstruction (Complex 2-channel) and Concatenation
            # k0_pd already on device
            # Use Emat_xyt_complex to combine coils using CSM (assuming same sensitivity as PDFS)
            img_pd_complex = Emat_xyt_complex(k0_pd, True, csm, 1.0)
            # img_pd_rss = sos(c2r(img_pd_complex))
            img_pd_2ch = c2r(img_pd_complex)
            img_pd_2ch = img_pd_2ch.float()
            img_pd_2ch = scaler(img_pd_2ch)

            # Concatenate (PDFS: 2 channels, PD: 2 channels -> 4 channels)
            label = torch.cat([label, img_pd_2ch], dim=1)

            loss = train_step_fn(state, label)
            loss_sum += loss

            param_num = sum(param.numel() for param in state["model"].parameters())
            # if step % 10 == 0:
            if not is_distributed or local_rank == 0:
                print(
                    "Epoch",
                    epoch + 1,
                    "/",
                    config.training.epochs,
                    "Step",
                    step,
                    "loss = ",
                    loss.cpu().data.numpy(),
                    "loss mean =",
                    loss_sum.cpu().data.numpy() / (step + 1),
                    "time",
                    time.time()-t0,
                    "param_num",
                    param_num,
                )

        # Save checkpoints at the configured epoch intervals.
        if (epoch + 1) % 10 ==0 or (epoch+1>150 and (epoch+1)%5== 0):
            if not is_distributed or local_rank == 0:
                save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_{epoch + 1}.pth"), state
                )

def sample(config, workdir):
    """Generate samples.

    Args:
      config: Configuration to use.
      workdir: Working directory.
    """
    # Create the log file path.
    save_dir = f"./data_analy/robust"
    save_dir=os.path.join(save_dir,config.sampling.folder)
    os.makedirs(save_dir, exist_ok=True)  # Create the output directory if needed.
    robustness_tag = _robustness_tag(config)
    folder_label = os.path.basename(os.path.normpath(str(config.sampling.folder)))
    file_name = (
        f"{folder_label}_{config.sampling.mask_type}_"
        f"{config.sampling.acc}x_{robustness_tag}.txt"
    )
    file_path = os.path.join(save_dir, file_name)

    # Initialize model
    score_model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate
    )
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    checkpoint_dir = os.path.join(workdir, "checkpoints")
    configured_ckpt_path = getattr(config.sampling, "checkpoint_path", "")
    ckpt_path = (
        configured_ckpt_path
        if configured_ckpt_path
        else os.path.join(
            checkpoint_dir,
            f"checkpoint_{config.sampling.ckpt}.pth",
        )
    )
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Set MRI_CHECKPOINT or "
            "--config.sampling.checkpoint_path to a trained checkpoint."
        )
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    print("load weights:", ckpt_path)

    weight_source = str(
        getattr(config.sampling, "weight_source", "raw")
    ).lower()
    if weight_source == "ema":
        state["ema"].copy_to(score_model.parameters())
    elif weight_source != "raw":
        raise ValueError(
            f"Unsupported sampling.weight_source={weight_source!r}; "
            "expected 'raw' or 'ema'"
        )
    score_model.eval()
    print(f"sampling weights: {weight_source}; model mode: eval")

    param_num = _count_parameters(score_model)
    flops_profile = _profile_inference_flops(score_model, config, sample_batch_size=1)
    forward_flops = flops_profile["full_forward_flops"]
    partial_forward_flops = flops_profile["partial_forward_flops"]
    gate_flops = flops_profile["gate_flops"]
    print("-" * 60)
    print("Inference Model Profile:")
    print(f"  Parameters      : {param_num} ({_format_large_number(param_num)})")
    print(f"  Full Forward    : {_format_large_number(forward_flops)}FLOPs")
    print(f"  Partial Forward : {_format_large_number(partial_forward_flops)}FLOPs")
    print(f"  Gate (included) : {_format_large_number(gate_flops)}FLOPs")
    print("  FLOPs scope     : profiler-supported ops; FFT kernels may be undercounted")
    print("-" * 60)


    if FLAGS.config.sampling.datashift == "brain":
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                FLAGS.config.sampling.datashift,
                FLAGS.config.sampling.fft,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(FLAGS.config.data.centered),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "datashift"
        )  # Load the requested evaluation split.
    elif FLAGS.config.sampling.datashift == "photom":
        SAMPLING_FOLDER_ID = "_".join(
            [
                FLAGS.config.sampling.acc,
                FLAGS.config.sampling.acs,
                FLAGS.config.sampling.mask_type,
                "ckpt",
                str(config.sampling.ckpt),
                FLAGS.config.sampling.predictor,
                FLAGS.config.training.mean_equal,
                FLAGS.config.sampling.datashift,
                FLAGS.config.sampling.fft,
                str(config.sampling.snr),
                "predictor_mse",
                str(FLAGS.config.sampling.mse),
                "corrector_mse",
                str(FLAGS.config.sampling.corrector_mse),
                str(FLAGS.config.data.centered),
                str(
                    FLAGS.config.sampling.N
                    if FLAGS.config.sampling.accelerated_sampling
                    else ""
                ),
                "photom",
                "seed",
                str(FLAGS.config.seed),
            ]
        )
        test_dl = datasets.get_dataset(
            config, "photom"
        )  # Load the requested evaluation split.
    else:
        if FLAGS.config.data.dataset_name == "IXI":
            mask_name = str(FLAGS.config.sampling.mask_type)
            acc_name = str(FLAGS.config.sampling.acc)
            SAMPLING_FOLDER_ID = f"IXI_{mask_name}{acc_name}x"
        else:
            SAMPLING_FOLDER_ID = "_".join(
                [
                    FLAGS.config.sampling.acc,
                    FLAGS.config.sampling.acs,
                    FLAGS.config.sampling.mask_type,
                    "ckpt",
                    str(config.sampling.ckpt),
                    FLAGS.config.sampling.predictor,
                    FLAGS.config.training.mean_equal,
                    str(config.sampling.snr),
                    "predictor_mse",
                    str(FLAGS.config.sampling.mse),
                    "corrector_mse",
                    str(FLAGS.config.sampling.corrector_mse),
                    str(
                        FLAGS.config.data.centered,
                    ),
                    str(
                        FLAGS.config.sampling.N
                        if FLAGS.config.sampling.accelerated_sampling
                        else ""
                    ),
                    "--",
                    "seed",
                    str(FLAGS.config.seed),
                ]
            )
        test_dl = datasets.get_dataset(
            config, "test"
        )  # Load the requested evaluation split.

    if robustness_tag != "none":
        SAMPLING_FOLDER_ID = f"{SAMPLING_FOLDER_ID}_{robustness_tag}"
    FLAGS.config.sampling.folder = os.path.join(FLAGS.workdir, SAMPLING_FOLDER_ID)
    tf.io.gfile.makedirs(FLAGS.config.sampling.folder)
    # Create data scaler and its inverse
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Setup SDEs
    if config.training.sde.lower() == "vpsde":
        sde = sde_lib.VPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "subvpsde":
        sde = sde_lib.subVPSDE(config)
        sampling_eps = 1e-3
    elif config.training.sde.lower() == "vesde":
        sde = sde_lib.VESDE(config)
        sampling_eps = 1e-5
    elif config.training.sde.lower() == "hfssde":
        sde = sde_lib.HFS_SDE(config)
        sampling_eps = 1e-3  # TODO
    else:
        raise NotImplementedError(f"SDE {config.training.sde} unknown.")

    atb_mask = get_mask(config, "sample")
    train_mask = get_mask(config, "sde")

    # Build the sampling function when sampling is enabled

    sampling_shape = (
        config.sampling.batch_size,
        config.data.num_channels,
        config.data.image_size,
        config.data.image_size,
    )
    sampling_fn = sampling.get_sampling_fn(
        config, sde, sampling_shape, inverse_scaler, sampling_eps, atb_mask, train_mask
    )

    slice_stats = []

    for index, point in enumerate(test_dl):


        if len(point) == 5:
            k0, csm, k0_pd, subject_ids, slice_names = point
            subject_id = _first_text(subject_ids)
            slice_name = _first_text(slice_names)
        else:
            k0, csm, k0_pd = point
            subject_id = ""
            slice_name = f"{index:04d}.png"
        k0 = k0.to(config.device)
        csm = csm.to(config.device)
        k0_pd = k0_pd.to(config.device)

        # PD Reconstruction (Complex 2-channel)
        # Use Emat_xyt_complex to combine coils using CSM (assuming same sensitivity as PDFS)
        # This ensures we get 1 complex channel (-> 2 real channels) instead of multiple coil channels
        img_pd_complex = Emat_xyt_complex(k0_pd, True, csm, 1.0)
        # img_pd_rss = sos(c2r(img_pd_complex))
        img_pd_2ch = c2r(img_pd_complex)
        img_pd_2ch = img_pd_2ch.type(torch.FloatTensor).to(config.device)
        img_pd_2ch = scaler(img_pd_2ch)
        img_pd_reference = img_pd_2ch.detach().clone()

        # Robustness Analysis: Apply manual perturbation to PD modality
        if getattr(config.sampling, "robustness_perturbation", False):
            perturbation_type = str(
                getattr(config.sampling, "robustness_type", "shift")
            ).lower()
            if perturbation_type == "shift":
                shift_pixels = getattr(config.sampling, "robustness_shift_pixels", 0)
                direction = getattr(
                    config.sampling,
                    "robustness_shift_direction",
                    "horizontal",
                )
                if shift_pixels != 0:
                    if direction == "horizontal":
                        img_pd_2ch = torch.roll(
                            img_pd_2ch, shifts=shift_pixels, dims=-1
                        )
                    elif direction == "vertical":
                        img_pd_2ch = torch.roll(
                            img_pd_2ch, shifts=shift_pixels, dims=-2
                        )
                    else:
                        raise ValueError(
                            f"Unsupported robustness_shift_direction={direction!r}"
                        )
                print(
                    ">>> DEBUG: Robustness rigid shift applied: "
                    f"direction={direction}, pixels={shift_pixels}"
                )
            elif perturbation_type == "elastic":
                max_displacement = float(
                    getattr(config.sampling, "elastic_max_displacement", 0.0)
                )
                elastic_sigma = float(
                    getattr(config.sampling, "elastic_sigma", 12.0)
                )
                elastic_seed = int(
                    getattr(config.sampling, "elastic_seed", config.seed)
                ) + index
                img_pd_2ch, actual_max = _elastic_deform(
                    img_pd_2ch,
                    max_displacement=max_displacement,
                    sigma=elastic_sigma,
                    seed=elastic_seed,
                )
                print(
                    ">>> DEBUG: Robustness elastic deformation applied: "
                    f"requested_max={max_displacement:.2f}px, "
                    f"actual_max={actual_max:.2f}px, sigma={elastic_sigma:.2f}, "
                    f"seed={elastic_seed}"
                )
            else:
                raise ValueError(
                    f"Unsupported robustness_type={perturbation_type!r}; "
                    "expected 'shift' or 'elastic'"
                )

            center_y = img_pd_2ch.shape[-2] // 2
            center_x = img_pd_2ch.shape[-1] // 2
            center_patch = img_pd_2ch[
                :, :, center_y - 5:center_y + 5, center_x - 5:center_x + 5
            ]
            perturbation_delta = torch.abs(img_pd_2ch - img_pd_reference)
            print(
                ">>> DEBUG: PD patch mean (center 10x10): "
                f"{center_patch.mean().item():.6f}"
            )
            print(
                ">>> DEBUG: PD perturbation difference: "
                f"MAE={perturbation_delta.mean().item():.10f}, "
                f"MAX={perturbation_delta.max().item():.10f}, "
                f"deformed_mean={img_pd_2ch.mean().item():.10f}, "
                f"deformed_std={img_pd_2ch.std().item():.10f}"
            )
            if (
                perturbation_type == "elastic"
                and float(getattr(config.sampling, "elastic_max_displacement", 0.0)) > 0
                and perturbation_delta.max().item() == 0
            ):
                raise RuntimeError("Elastic deformation was enabled but PD did not change")

        label = Emat_xyt_complex(k0, True, csm, 1.0).to(config.device)

        label_dir = os.path.join("results", FLAGS.config.sampling.datashift)
        if not tf.io.gfile.exists(label_dir):
            tf.io.gfile.makedirs(label_dir)
        save_mat(label_dir, label.to(label), "label", index, normalize=False)

        atb = k0 * atb_mask
        atb_to_image_pdfs = (
            c2r(Emat_xyt_complex(atb, True, csm, 1))
            .type(torch.FloatTensor)
            .to(config.device)
        )  # 1x2x320x320

        # Concatenate PDFS (2ch) and PD (2ch) -> 4ch
        atb_to_image = torch.cat([atb_to_image_pdfs, img_pd_2ch], dim=1)

        if (
            index == 0
            and getattr(config.sampling, "robustness_perturbation", False)
            and str(getattr(config.sampling, "robustness_type", "shift")).lower()
            == "elastic"
        ):
            probe_labels = torch.full(
                (img_pd_2ch.shape[0],),
                0.5,
                device=config.device,
                dtype=torch.float32,
            )
            reference_input = torch.cat(
                [atb_to_image_pdfs, img_pd_reference], dim=1
            )
            with torch.no_grad():
                reference_output = score_model(reference_input, probe_labels)
                deformed_output = score_model(atb_to_image, probe_labels)
            target_response = torch.abs(
                deformed_output[:, :2] - reference_output[:, :2]
            )
            print(
                ">>> DEBUG: Model target response to elastic PD: "
                f"MAE={target_response.mean().item():.10f}, "
                f"MAX={target_response.max().item():.10f}"
            )

        csm = c2r(csm).type(torch.FloatTensor).to(config.device)

        # Pass both PDFS k-space (atb) and PD image (img_pd_2ch) to sampling function
        # We wrap them in a tuple/list to be handled by sde_lib.py
        atb_combined = (atb, img_pd_2ch)

        recon, n, infer_stats = sampling_fn(score_model, atb_combined, atb_to_image, csm) #Run the configured reconstruction sampler.


        # Extract PDFS part for evaluation (channels 0,1)
        recon_pdfs = recon[:, :2, :, :]
        recon_pd = recon[:, 2:, :, :]
        recon = r2c(recon_pdfs)
        recon_pd_complex = r2c(recon_pd)

        should_visualize = (
            config.data.dataset_name == "IXI"
            and config.sampling.mask_type == "uniform"
            and str(config.sampling.acc) == "4"
            and str(config.sampling.acs) == "24"
            and subject_id == str(getattr(config.sampling, "visualize_subject", "146"))
        )
        if should_visualize:
            visualization_dir = os.path.join(
                FLAGS.config.sampling.folder,
                "visualizations",
                f"subject_{subject_id}",
            )
            visualization_name = os.path.splitext(slice_name)[0] + ".png"
            _save_ixi_visualization(
                recon[0],
                os.path.join(visualization_dir, visualization_name),
            )

        if getattr(config.sampling, "save_mat_outputs", True):
            save_mat(
                FLAGS.config.sampling.folder,
                recon.to(recon),
                "recon",
                index,
                normalize=False,
            )

            # Save PD reconstruction to verify shift
            save_mat(
                FLAGS.config.sampling.folder,
                recon_pd_complex.to(recon_pd_complex),
                "recon_pd",
                index,
                normalize=False,
            )

        from utils.calc import Evaluation_metrics, HFEN_metric

        metric_protocol = str(
            getattr(config.sampling, "metric_protocol", "legacy")
        ).lower()
        ssim, psnr, nmse = Evaluation_metrics(
            label,
            recon,
            True if FLAGS.config.sampling.datashift == "photom" else False,
            protocol=metric_protocol,
        )
        ssim_value = _as_float(ssim)
        psnr_value = _as_float(psnr)
        nmse_value = _as_float(nmse)
        hfen_value = HFEN_metric(label, recon)
        # print("---------------------------------------------")
        # print("---------------- point:", index, "------------------")
        # print("---------------------------------------------")
        # print(f"Sampling steps: {n}")

        print(
            f"mse_{config.sampling.mse}_snr_{config.sampling.snr}_cmse_{config.sampling.corrector_mse}:"
        )


        estimated_slice_flops = infer_stats.get("slice_reconstruction_flops")
        is_cache_ablation = "full_unet_forward_calls" in infer_stats
        if (
            estimated_slice_flops is None
            and forward_flops is not None
            and not is_cache_ablation
        ):
            estimated_slice_flops = forward_flops * (
                2 * infer_stats["total_steps"] + 1
            )
        slice_partial_flops = infer_stats.get(
            "partial_forward_flops", partial_forward_flops
        )
        slice_dc_flops = infer_stats.get("data_consistency_flops")
        slice_gate_flops = infer_stats.get("gate_flops", gate_flops)
        baseline_slice_flops = infer_stats.get("baseline_slice_flops")
        flops_reduction_vs_n1 = infer_stats.get("flops_reduction_vs_n1")
        theoretical_flops_speedup = infer_stats.get("theoretical_flops_speedup")

        slice_stats.append(
            {
                "num_slices": int(recon_pdfs.shape[0]),
                "sampling_steps": n,
                "full_unet_steps": infer_stats.get("full_unet_steps"),
                "cache_reuse_steps": infer_stats.get("cache_reuse_steps"),
                "full_unet_forward_calls": infer_stats.get(
                    "full_unet_forward_calls"
                ),
                "cache_reuse_forward_calls": infer_stats.get(
                    "cache_reuse_forward_calls"
                ),
                "cache_fallback_calls": infer_stats.get("cache_fallback_calls"),
                "estimated_slice_flops": estimated_slice_flops,
                "partial_forward_flops": slice_partial_flops,
                "data_consistency_flops": slice_dc_flops,
                "gate_flops": slice_gate_flops,
                "baseline_slice_flops": baseline_slice_flops,
                "flops_reduction_vs_n1": flops_reduction_vs_n1,
                "theoretical_flops_speedup": theoretical_flops_speedup,
                "time_per_slice_sec": infer_stats["time_per_slice_sec"],
                "total_time_sec": infer_stats["total_time_sec"],
                "peak_gpu_memory_mb": infer_stats["peak_gpu_memory_mb"],
                "nmse": nmse_value,
                "ssim": ssim_value,
                "psnr": psnr_value,
                "hfen": hfen_value,
            }
        )

        reduction_text = (
            "N/A"
            if flops_reduction_vs_n1 is None
            else f"{100.0 * flops_reduction_vs_n1:.2f}%"
        )
        speedup_text = (
            "N/A"
            if theoretical_flops_speedup is None
            else f"{theoretical_flops_speedup:.4f}x"
        )
        profile_msg = (
            f"Weight source: {weight_source}\n"
            f"Metric protocol: {metric_protocol}\n"
            f"Parameters: {param_num} ({_format_large_number(param_num)})\n"
            f"Single full U-Net forward FLOPs: {_format_large_number(forward_flops)}FLOPs\n"
            f"Single partial U-Net forward FLOPs: {_format_large_number(slice_partial_flops)}FLOPs\n"
            f"FLOPs per data consistency call: {_format_large_number(slice_dc_flops)}FLOPs\n"
            f"Gate FLOPs (included in forward FLOPs): {_format_large_number(slice_gate_flops)}FLOPs\n"
            f"Full-slice reconstruction FLOPs: {_format_large_number(estimated_slice_flops)}FLOPs\n"
            f"Full-computation baseline FLOPs (N=1): {_format_large_number(baseline_slice_flops)}FLOPs\n"
            f"FLOPs reduction relative to N=1: {reduction_text}\n"
            f"Theoretical FLOPs speedup: {speedup_text}\n"
            f"Full U-Net iterations: {infer_stats.get('full_unet_steps', 'N/A')}\n"
            f"Cache reuse iterations: {infer_stats.get('cache_reuse_steps', 'N/A')}\n"
            f"Full U-Net forward calls: {infer_stats.get('full_unet_forward_calls', 'N/A')}\n"
            f"Partial U-Net forward calls: {infer_stats.get('cache_reuse_forward_calls', 'N/A')}\n"
            f"Inference time per slice: {infer_stats['time_per_slice_sec']:.4f} s\n"
            f"Inference time per batch: {infer_stats['total_time_sec']:.4f} s\n"
            f"Peak GPU memory: {infer_stats['peak_gpu_memory_mb']:.2f} MB"
        )
        print(profile_msg)

        log_msg = f"---------------------------------------------\n ----------------point: {index}----------------\n  ---------------------------------------------\n Sampling steps: {n}\n {profile_msg}\n NMSE: ({nmse_value:.4f})\n SSIM: ({ssim_value:.4f})\n PSNR: ({psnr_value:.4f})\n HFEN: ({hfen_value:.4f})"
        print(log_msg)
        with open(file_path, 'a') as f:
            f.write(log_msg + "\n")

    if slice_stats:
        def mean_stat(key):
            weighted_values = [
                (item[key], item["num_slices"])
                for item in slice_stats
                if item[key] is not None
            ]
            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight == 0:
                return None
            return sum(value * weight for value, weight in weighted_values) / total_weight

        def std_stat(key):
            weighted_values = [
                (item[key], item["num_slices"])
                for item in slice_stats
                if item[key] is not None
            ]
            total_weight = sum(weight for _, weight in weighted_values)
            if total_weight == 0:
                return None
            mean_value = sum(
                value * weight for value, weight in weighted_values
            ) / total_weight
            variance = sum(
                weight * (value - mean_value) ** 2
                for value, weight in weighted_values
            ) / total_weight
            return math.sqrt(variance)

        def fixed_stat(value, digits=2):
            if value is None:
                return "N/A"
            return f"{value:.{digits}f}"

        def percent_stat(value):
            if value is None:
                return "N/A"
            return f"{100.0 * value:.2f}%"

        def speedup_stat(value):
            if value is None:
                return "N/A"
            return f"{value:.4f}x"

        total_slices = sum(item["num_slices"] for item in slice_stats)
        summary_msg = (
            "=============================================\n"
            "Aggregate results across all slices\n"
            "=============================================\n"
            f"Number of slices: {total_slices}\n"
            f"Weight source: {weight_source}\n"
            f"Metric protocol: {metric_protocol}\n"
            f"Parameters: {param_num} ({_format_large_number(param_num)})\n"
            f"Single full U-Net forward FLOPs: {_format_large_number(forward_flops)}FLOPs\n"
            f"Single partial U-Net forward FLOPs: {_format_large_number(mean_stat('partial_forward_flops'))}FLOPs\n"
            f"FLOPs per data consistency call: {_format_large_number(mean_stat('data_consistency_flops'))}FLOPs\n"
            f"Gate FLOPs (included in forward FLOPs): {_format_large_number(mean_stat('gate_flops'))}FLOPs\n"
            f"Mean full-slice reconstruction FLOPs: {_format_large_number(mean_stat('estimated_slice_flops'))}FLOPs\n"
            f"Full-computation baseline FLOPs (N=1): {_format_large_number(mean_stat('baseline_slice_flops'))}FLOPs\n"
            f"Mean FLOPs reduction relative to N=1: {percent_stat(mean_stat('flops_reduction_vs_n1'))}\n"
            f"Mean theoretical FLOPs speedup: {speedup_stat(mean_stat('theoretical_flops_speedup'))}\n"
            f"Mean inference time per slice: {mean_stat('time_per_slice_sec'):.4f} s\n"
            f"Mean inference time per batch: {mean_stat('total_time_sec'):.4f} s\n"
            f"Mean peak GPU memory: {mean_stat('peak_gpu_memory_mb'):.2f} MB\n"
            f"Mean sampling steps: {mean_stat('sampling_steps'):.2f}\n"
            f"Mean full U-Net iterations: {fixed_stat(mean_stat('full_unet_steps'))}\n"
            f"Mean cache reuse iterations: {fixed_stat(mean_stat('cache_reuse_steps'))}\n"
            f"Mean full U-Net forward calls: {fixed_stat(mean_stat('full_unet_forward_calls'))}\n"
            f"Mean partial U-Net forward calls: {fixed_stat(mean_stat('cache_reuse_forward_calls'))}\n"
            f"NMSE mean: {mean_stat('nmse'):.6f}\n"
            f"NMSE standard deviation: {std_stat('nmse'):.6f}\n"
            f"SSIM mean: {mean_stat('ssim'):.6f}\n"
            f"SSIM standard deviation: {std_stat('ssim'):.6f}\n"
            f"PSNR mean: {mean_stat('psnr'):.6f}\n"
            f"PSNR standard deviation: {std_stat('psnr'):.6f}\n"
            f"HFEN mean: {mean_stat('hfen'):.6f}\n"
            f"HFEN standard deviation: {std_stat('hfen'):.6f}"
        )
        print(summary_msg)
        with open(file_path, 'a') as f:
            f.write(summary_msg + "\n")

        combined_summary_path = os.environ.get("IXI_MASK_SUMMARY_FILE")
        if combined_summary_path:
            mask_subdir = getattr(config.sampling, "mask_subdir", "")
            mask_name = (
                f"{config.sampling.mask_type}_acc{config.sampling.acc}"
                f"_acs{config.sampling.acs}_{config.data.image_size}.mat"
            )
            if mask_subdir:
                mask_name = os.path.join(mask_subdir, mask_name)
            row = [
                mask_name,
                config.sampling.ckpt,
                total_slices,
                param_num,
                forward_flops,
                mean_stat("partial_forward_flops"),
                mean_stat("gate_flops"),
                mean_stat("data_consistency_flops"),
                mean_stat("estimated_slice_flops"),
                std_stat("estimated_slice_flops"),
                mean_stat("baseline_slice_flops"),
                mean_stat("flops_reduction_vs_n1"),
                mean_stat("theoretical_flops_speedup"),
                mean_stat("full_unet_steps"),
                mean_stat("cache_reuse_steps"),
                mean_stat("full_unet_forward_calls"),
                mean_stat("cache_reuse_forward_calls"),
                mean_stat("cache_fallback_calls"),
                mean_stat("psnr"),
                std_stat("psnr"),
                mean_stat("ssim"),
                std_stat("ssim"),
                mean_stat("nmse"),
                std_stat("nmse"),
                mean_stat("hfen"),
                std_stat("hfen"),
                mean_stat("time_per_slice_sec"),
                std_stat("time_per_slice_sec"),
                mean_stat("peak_gpu_memory_mb"),
                std_stat("peak_gpu_memory_mb"),
            ]
            experiment = os.environ.get("CACHE_ABLATION_EXPERIMENT")
            if experiment:
                row = [
                    experiment,
                    os.environ.get("CACHE_ABLATION_INTERVAL", ""),
                    os.environ.get("CACHE_ABLATION_BOUNDARY", ""),
                    getattr(config.sampling, "solver_schedule", "legacy"),
                    weight_source,
                    metric_protocol,
                ] + row
            with open(combined_summary_path, "a", newline="") as summary_file:
                writer = csv.writer(summary_file)
                writer.writerow(row)
