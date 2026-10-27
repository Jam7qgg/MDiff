import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 8 #  Batch size per process
    training.epochs = 190
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = False
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow =False
    sampling.snr = 0.16
    sampling.mse = 2.5     #Conditional diffusion hyperparameter
    sampling.corrector_mse = 5.0
    #Adaptive sampling settings
    sampling.adaptive =True
    sampling.h_init = 0.1    #Initial step size
    sampling.abstol = 0.05
    sampling.reltol = 0.1
    sampling.error_use_prev = True
    sampling.norm = "L2_scaled"
    sampling.safety = 0.9
    sampling.extrapolation = True
    sampling.sde_improved_euler = True
    sampling.exp = 0.9

    # data
    config.data = data = ml_collections.ConfigDict()
    data.image_size = 320
    data.random_flip = True
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 2   #U-Net input channels

    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_min = 0.01
    model.sigma_max = 348
    model.num_scales = 1000 #Number of SDE discretization scales
    model.beta_min = 0.1
    model.beta_max = 20.0
    model.dropout = 0.1
    model.reuse_N = 0  # Reuse interval: one full pass every N+1 iterations, followed by N reuse iterations.
    model.matrix = True
    model.embedding_type = "fourier"
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = "Adam"
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.0

    config.seed = 1000
    config.device =torch.device("cuda:0")
    optim.gradient_accumulation_steps = 2


    return config
