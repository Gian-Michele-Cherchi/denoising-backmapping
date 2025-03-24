# Scalable Diffusion Backmapping for Polymer Melts

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

