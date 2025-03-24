# Scalable Diffusion Backmapping for Polymer Melts

![Diffusion Backmapping Scheme](docs/conditional_gen_diff_.png)

## Overview

This repository contains the implementation of **Scalable Diffusion Backmapping**, a method for reconstructing atomistic polymer configurations from coarse-grained (CG) models. The approach utilizes **Equivariant Graph Neural Networks (EGNNs)** and **Denoising Diffusion Probabilistic Models (DDPMs)** to efficiently and accurately restore fine-grained details of polymer melts from CG representations.

The method was developed as part of research on **machine-learning-enhanced molecular simulations**, aiming to overcome the challenges posed by computationally expensive all-atom molecular dynamics (MD) simulations.

## Key Features

- **Equivariant Graph Neural Networks (GNNs)** for encoding CG and Atomistic molecular structures.
- **Diffusion-based generative modeling** to reconstruct atomistic configurations from CG ones.
- **Energy minimization and constrained optimization** for realistic polymer reconstructions.
- **Periodic boundary conditions (PBCs)** to ensure physical consistency.
- **Scalability to large polymer systems** with varying degrees of coarse-graining.

## Installation

To set up the required dependencies, install the necessary Python packages using the pyproject.toml file:

```bash
poetry install
```
## Usage

### 1. Data Preparation

The input consists of coarse-grained molecular structures. Prepare datasets in the following format: 
-  representations: Center of Mass (COM) coordinates of monomers.
- Reference atomistic structures: Used for training and evaluation.

### 2. Training the Backmapping Model

To train the Diffusion Backmapping model, run:
```bash
poetry run python src/train.py train=diffusion
```
### 3. Generating Atomistic Configurations

Once the model is trained, generate atomistic polymer structures using:
```bash
poetry run python src/sample.py
```

## Results
  - Higher fidelity backmapping compared to naïve sampling.
  - Improved RDFs in comparison to standard CG models.
  - Scalable framework suitable for various polymer systems.

## Future Work
 -	Refinement of angle distributions using additional loss terms.
 -	Improved handling of larger polymer melts with advanced architectures.
 -	Hybrid physics-informed and ML-based approaches for enhanced accuracy.

