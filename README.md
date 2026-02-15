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

## Roadmap
Alchemy Lab is very much a work in progress. Planned extensions include:
- Latent diffusion
- DiT architecture
- Mixed precision training
- Distributed training (FSDP)
- Performance enhancements
- ONNX export
- Generic deployment infrastructure
