<p align="center">
  <img src="docs/assets/header.png" width="300" />
</p>


# Alchemy Lab

This project aims to provide a research-grade generative modelling and inference framework, particularly focussed on diffusion models. It is designed as a long-lived codebase for learning and experimentation. It prioritises explicit implementations, clear abstractions, and separation of concerns. 

## Project Goals

Alchemy Lab is built to:
- Encourage deep understanding of theory and implementation;
- support reproducible, configurable experimentation; 
- explore the systems that facilitate the training and deployment of machine learning models;
- and provide a clear path between research code, training infrastructure, and inference/runtime.

It is not intended to compete with high-level libraries such as those offered by Hugging Face, or vLLM/sglang. Instead, it is closer to a personal research platform.

## Repository Structure
Alchemy Lab is organised as a monorepo with three pillars: core, which contains implementations of modules that compose generative models; lab, which provides the training infrastructure; and runtime, which provides inference infrastructure. 