# MDiff

Code for **MDiff: A Lightweight Multi Modal Diffusion Framework for Accelerated MRI Reconstruction via Feature Reuse**.

**Journal:** Biomedical Signal Processing and Control

**Publication status:** Accepted for publication

**Code repository:** [github.com/Jam7qgg/MDiff](https://github.com/Jam7qgg/MDiff)

MDiff is a lightweight multimodal MRI reconstruction framework based on diffusion modeling and feature reuse. This repository contains the core PyTorch model, training, sampling, and evaluation implementation. The default HFS-SDE configuration reconstructs PDFS images using PD as the auxiliary modality. Complex-valued images are represented by real and imaginary channels, giving four input channels at a spatial resolution of 320 x 320.

## Repository Structure

```text
main.py                         Training and sampling entry point
run_lib.py                      Preprocessing, training, reconstruction, and metrics
losses.py                       Diffusion training losses and gradient accumulation
sampling.py                     Adaptive Heun sampling and feature reuse scheduling
sde_lib.py                      Forward and reverse SDE definitions
save_data_slice.py              Paired-data validation and slice-index generation
models/
  ddpm.py                       U-Net, CSP bottleneck, and cached features
  layers.py                     Convolutions, residual blocks, CBAM, and gated fusion
  model_utils.py                Model registration and score-function wrappers
  normalization.py              Normalization layers
  ema.py                        Exponential moving average
configs/
  default_fastMRI_configs.py    Optimizer, random seed, and shared settings
  hfssde/ddpm_continuous.py      Multimodal fastMRI training and sampling settings
utils/
  datasets.py                   Paired-data loading
  utils.py                      Complex operations, FFT, masks, and checkpoints
  calc.py                       PSNR, SSIM, NMSE, and HFEN
requirements.txt
```

The internal model identifier remains `ddpm` to preserve compatibility with the existing configuration and checkpoint format.

## Installation

The target runtime is Linux with Python 3.10 and an NVIDIA GPU. Model computation uses PyTorch; the TensorFlow CPU package supports the existing file interfaces. The dependency versions below form the source-release baseline. A full GPU training run was not repeated as part of repository packaging.

```bash
git clone https://github.com/Jam7qgg/MDiff.git
cd MDiff

conda create -n HFS python=3.10 -y
conda activate HFS
python -m pip install --upgrade pip
# CUDA 12.1 example: select an official wheel compatible with your GPU driver.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip check
```

See the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/#v251) for matching torchvision versions and alternative CUDA wheels. Installing the matching CUDA wheels first allows the requirements file to reuse them.

## Data Preparation

Prepare the following layout under the repository root. External data directories can also be linked to `data/` using Linux symbolic links.

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

- Each k-space HDF5 file contains a complex-valued dataset named `kspace`, with shape `[slices, coils, H, W]`.
- Sensitivity-map files contain a complex-valued dataset named `s_maps` and have the same filenames as the corresponding PDFS files. The current implementation also uses these maps to combine the paired PD data, so the inputs must satisfy the corresponding spatial and coil conventions.
- Pairing CSV files use the format `PD,PDFS,score`. The first two columns contain filenames without the `.h5` extension. Paired slices must be spatially aligned.
- The fastMRI loader center-crops in the image domain to 320 x 320 and skips the first eight slices of each scan for both training and testing.
- Each MAT mask file must contain a two-dimensional array named `mask` with shape `[320, 320]`. Training uses a low-frequency mask with ACS=16 by default. Testing uses uniform 4x undersampling with ACS=24.

Validate the training data and generate its slice index:

```bash
python save_data_slice.py
```

Generate the test index:

```bash
python save_data_slice.py \
  --directory data/test/pdfs/kspace \
  --maps-directory data/test/pdfs/map \
  --pd-directory data/test/pd/kspace \
  --pairs data/test/test-pair.csv \
  --output data/test/data_slice.pkl
```

## Training

Run single-GPU training from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=train --workdir=results
```

For two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=train --workdir=results
```

Default settings are 190 epochs, a per-process batch size of 8, two micro-batches for gradient accumulation, and random seed 1000. The logged `Step` is a training batch iteration. Training samples diffusion times randomly and does not perform a complete reverse reconstruction for every batch. See `losses.py` for the frequency-projected score-matching objective.

Checkpoints are stored in `results/<timestamp_and_configuration>/checkpoints/`. They are saved every 10 epochs, and every 5 epochs after epoch 150. To resume training, specify both the checkpoint and the number of completed epochs:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=train --workdir=results \
  --config.training.resume_checkpoint=/absolute/path/to/checkpoint_70.pth \
  --config.training.resume_epoch=70
```

## Reconstruction and Evaluation

Evaluate a trained checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config=configs/hfssde/ddpm_continuous.py \
  --mode=sample --workdir=results \
  --config.sampling.checkpoint_path=/absolute/path/to/checkpoint_190.pth \
  --config.sampling.ckpt=190 \
  --config.sampling.folder=fastMRI
```

The checkpoint path can alternatively be supplied through the `MRI_CHECKPOINT` environment variable. Defaults use raw model weights, batch size 1, no robustness perturbation, and no feature reuse. Select EMA weights with `--config.sampling.weight_source=ema`.

The default evaluation protocol is `metric_protocol=legacy`. Select the standard protocol with `--config.sampling.metric_protocol=standard`. Use the same weight source, metric protocol, masks, and data split when comparing experiments. Metric definitions are implemented in `utils/calc.py`.

Per-slice output includes PSNR, SSIM, NMSE, HFEN, reconstruction time, peak GPU memory, and model parameter count. Evaluation also reports quality-metric means and standard deviations across the processed slices. Logs are written under `data_analy/robust/fastMRI/`, with run directories under `results/fastMRI/`. MAT output is disabled by default; enable it with `--config.sampling.save_mat_outputs=True`.

### Feature Reuse

The core sampler exposes `config.model.reuse_N`:

- `reuse_N=0`: full computation at every sampling iteration.
- `reuse_N=k > 0`: one full-computation iteration followed by `k` cache-reuse iterations.

For example, append `--config.model.reuse_N=1` to the reconstruction command to alternate full and reuse iterations. In the 320 x 320 configuration, the core model caches a 160 x 160 decoder feature after the 80 x 80 to 160 x 160 upsampling operation. Reuse iterations recompute the shallow encoder and decoder while bypassing the deeper encoder, bottleneck, and decoder.

The sampler uses adaptive step sizes; the actual number of iterations depends on the input and model. The current iteration cap is 200. Setting `model.num_scales=1000` does not force 1000 sampling iterations.

Peak GPU memory is measured using PyTorch allocated memory. FLOPs cover profiler-supported operations; full-slice estimates use model-call counts and should not be interpreted as exact end-to-end counts including all FFT operations.

## Code and Data Availability

The core model, training, and sampling code accompanying the paper is publicly available at [https://github.com/Jam7qgg/MDiff](https://github.com/Jam7qgg/MDiff).

This repository distributes source code. MRI datasets, sensitivity maps, sampling masks, pretrained checkpoints, and experiment outputs are not included. Prepare these resources separately in accordance with their original access conditions. Ablation runners and experiment-specific batch scripts are outside the scope of this core release.

## Acknowledgments

The source retains the copyright and license notices of the upstream files. Related implementations include [Score SDE PyTorch](https://github.com/yang-song/score_sde_pytorch), [DDPM](https://github.com/hojonathanho/diffusion), and [pytorch_ema](https://github.com/fadel/pytorch_ema). Follow the license terms in the respective files and upstream projects when reusing their code.
