<p align="center">
  <img src="docs/assets/header.png" width="300" />
</p>


# Alchemy Lab

Alchemy Lab is a modular research infrastructure for building, training, and deploying diffusion models.

It is designed to:
- Enable fast, configuration-driven experimentation
- Facilitate evaluation, monitoring, and analysis
- Bridge research protoypes and scalable systems

It is not intended to compete with high-level libraries such as those offered by Hugging Face, or vLLM/sglang. Instead, it is closer to a personal research platform.

## Features
- Modular diffusion core
- Composable UNet-style architectures
- Support for latent diffusion
- Configuration-driven experiment management
- Structured training harness with clear separation from model primitives
- Support for distributed training (DDP)
- Explicit pathway from research experimentation to deployment runtime

## Repository Structure
Alchemy Lab is organised as a monorepo with three pillars:
- **core** - mathematical primitives and model components
- **lab** - experiment configuration and training infrastructure
- **runtime** - inference and deployment capabilities 

```
src/alchemy/
|-- core/        # diffusion primitives, model components
|-- lab/         # training infrastructure
|-- runtime/     # inference (planned)
```

## Installation & Usage
Alchemy Lab may be installed using `uv` after cloning to your local machine:

```bash
git clone https://github.com/j9smith/alchemy-lab
cd alchemy-lab
uv sync
```

Once installed, experiments can be parameterised by amending the configuration files found in `lab/configs`, and then executed via the entrypoint `lab/cli/train.py`:
```bash
cd alchemy-lab/src/alchemy/lab/cli
uv run python train.py
```

Example config file:
```yaml
train:
  resume: "checkpoint.pt"
  precision: fp32
  max_steps: 50000
  ema_decay: 0.9999

logging:
  tensorboard:
    _target_: alchemy.lab.loggers.tensorboard.TensorBoardLogger
    cfg:
      _target_: alchemy.lab.loggers.tensorboard.TensorBoardLoggerConfig
      log_dir: "~/logging"
      experiment_name: "default"
      enabled: true
  terminal:
    _target_: alchemy.lab.loggers.terminal.TerminalLogger
    cfg:
      _target_: alchemy.lab.loggers.terminal.TerminalLoggerConfig
      total_steps: 10000
      print_every_n_steps: 10
      enabled: true 

checkpoints:
  _target_: alchemy.lab.training.checkpoints.CheckpointManager
  cfg:
    _target_: alchemy.lab.training.checkpoints.CheckpointManagerConfig
    save_every_n_steps: 1000
    path: "weights.pt"
    prefix: "unet"

dist:
  backend: nccl

model:
  _target_: alchemy.core.models.unet.unet2d.UNet2D
  cfg:
    _target_: alchemy.core.models.unet.unet2d.UNet2DConfig
    in_channels: 4
    out_channels: 4
    base_channels: 64
    channel_multipliers: [1,2,4,8]
    attn_levels: [1, 2]
    use_mid_attn: true
    attn_num_heads: 8
    num_res_blocks: 2
    time_embed_dim: 256
    norm_groups: 32
    dropout: 0.1
    conv_bias: false

vae:
  _target_: alchemy.core.models.vae.pretrained.PretrainedVAE
  cfg:
    _target_: alchemy.core.models.vae.pretrained.PretrainedVAEConfig
    pretrained_model: "stabilityai/sd-vae-ft-mse"
    scaling_factor: 0.18215

optim:
  _target_: alchemy.lab.training.optim.build_optimiser
  cfg:
    _target_: alchemy.lab.training.optim.OptimiserConfig
    name: adamw
    lr: 0.0002

loss:
  _target_: alchemy.lab.training.losses.DiffusionLossFn
  loss_cfg:
    _target_: alchemy.lab.training.losses.DiffusionLossWrapperConfig
    objective: eps
    beta_schedule_cfg:
      _target_: alchemy.core.diffusion.schedules.BetaScheduleConfig
      type: linear
      T: 1000
      beta_start: 0.0001
      beta_end: 0.02

data:
  image:
    channels: 3
    resolution: 256
  dataset:
    _target_: alchemy.lab.data.dataset.build_dataset
    cfg:
      _target_: alchemy.lab.data.dataset.DatasetConfig
      resolution: 256
  loader:
    _target_: alchemy.lab.data.loader.build_dataloader
    cfg:
      _target_: alchemy.lab.data.loader.LoaderConfig
      batch_size: 32
      num_workers: 8
      shuffle: true
      drop_last: true
      pin_memory: true
      persistent_workers: true

```

Images can be sampled by loading saved checkpoints via the `cli/sample.py` script:
```bash
uv run python sample.py --ckpt ./weights/unet_stepXXX.pt --device cuda --use_ema --n 24
```
Sampled images are stored in `output/samples.png`.

## Roadmap
Alchemy Lab is very much a work in progress. Planned extensions include:
- DiT architecture
- Mixed precision training
- Distributed training (FSDP)
- Performance enhancements
- ONNX export
- Generic deployment infrastructure
