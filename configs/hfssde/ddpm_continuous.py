# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Training DDPM with VP SDE."""

import os

from configs.default_fastMRI_configs import get_default_configs


def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = "hfssde"
    training.continuous = True
    training.reduce_mean = True
    training.mask_type = "low_frequency"  # low_frequency, u
    training.acc = "None"  # center low None
    training.acs = "16"
    training.mean_equal = "noequal"  # equal or noequal
    training.resume_checkpoint = ""
    training.resume_epoch = 0

    # sampling
    sampling = config.sampling
    sampling.batch_size = 1
    sampling.method = "pc"     #Sampling method
    sampling.predictor = "heun"  # reverse_diffusion or euler_maruyama or adaptive or none
    sampling.corrector = "none"  #   none
    sampling.folder = "fastMRI"

    sampling.ckpt =190
    sampling.checkpoint_path = os.environ.get("MRI_CHECKPOINT", "")
    sampling.weight_source = "raw"
    sampling.metric_protocol = "legacy"
    sampling.save_mat_outputs = False
    sampling.mask_type = "uniform"  # uniform, cartesian, random_uniform or center
    sampling.acc = "4"
    sampling.acs = "24"
    sampling.fft = "nofft"  # fft or nofft

    sampling.accelerated_sampling =False
    if training.sde == "hfssde":
        sampling.snr = 0.16
        if sampling.accelerated_sampling:
            sampling.snr = 0.32
    elif training.sde == "vpsde":
        sampling.snr = 0.27  # 0.27
        if sampling.accelerated_sampling:
            sampling.snr = 0.43
    if sampling.accelerated_sampling:
        config.sampling.N =600

    sampling.mse = 0.92
    sampling.corrector_mse = 0.1  ###
    sampling.datashift = "knee"  ### brain  knee  or  photom

    # Robustness Analysis
    sampling.robustness_perturbation = False  # Enable explicitly for robustness tests.
    sampling.robustness_type = "shift"  # "shift" or "elastic"
    sampling.robustness_shift_direction = "vertical"  # "horizontal" or "vertical"
    sampling.robustness_shift_pixels = 20 # Shift magnitude in pixels
    sampling.elastic_max_displacement = 3.0  # Maximum local displacement in pixels
    sampling.elastic_sigma = 12.0  # Gaussian smoothing scale for the displacement field, in pixels
    sampling.elastic_seed = 2026  # Use seed + slice index for each slice.

    # data

    data = config.data
    data.centered = False  # True: Input is in [-1, 1]
    data.dataset_name = "fastMRI_knee"
    data.image_size = 320
    data.normalize_type = "std"  # minmax or std or img_std
    data.normalize_coeff = 1.5  # normalize coefficient1
    data.num_channels = 4 # 2 channels for PDFS (Re/Im) + 2 channels for PD (Re/Im)

    # model
    model = config.model
    model.name = "ddpm"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 2
    model.attn_resolutions = (40,)
    model.resamp_with_conv = True
    model.conditional = True
    model.reuse_N = 0 # Reuse interval: one full pass every N+1 iterations, followed by N reuse iterations.
                                                                                                                                                                                                                                                                                                                                                                 
    return config
