# Hyperspectral Image Super-Resolution GAN (HS-SRGAN)

## Overview
This repository contains the implementation of a Generative Adversarial Network (GAN) designed for hyperspectral (HS) image super-resolution (SR). The network consists of a generator and a discriminator, both built using PyTorch.

## Table of Contents
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
- [Model Architecture](#model-architecture)
  - [Generator](#generator)
  - [Discriminator](#discriminator)
- [Dataset](#dataset)
- [Logging and Results](#logging-and-results)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/javiersn99/HS-SRGAN.git
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training
To train the GAN, run the following command:
```bash
python HS-SRGAN(v9).py
```
### README.md

Some parameters that might be changed are:

- `--train-batch-size`:
  - **Type:** int
  - **Default:** 12
  - **Metavar:** N
  - **Help:** input batch size for training (default: 12)
  
- `--test-batch-size`:
  - **Type:** int
  - **Default:** 1
  - **Metavar:** N
  - **Help:** input batch size for testing (default: 1)
  
- `--eval-batch-size`:
  - **Type:** int
  - **Default:** 12
  - **Metavar:** N
  - **Help:** input batch size for evaluation (default: 12)
  
- `--epochs`:
  - **Type:** int
  - **Default:** 100
  - **Metavar:** N
  - **Help:** number of epochs to train (default: 100)
  
- `--lr`:
  - **Type:** float
  - **Default:** 0.0001
  - **Metavar:** LR
  - **Help:** learning rate (default: 0.0001)
  
- `--log-interval`:
  - **Type:** int
  - **Default:** 10
  - **Metavar:** N
  - **Help:** how many batches to wait before logging training status
  
- `--log-dir`:
  - **Type:** str
  - **Default:** logs
  - **Metavar:** N
  - **Help:** directory to save logs
  
- `--patience`:
  - **Type:** int
  - **Default:** 25
  - **Metavar:** N
  - **Help:** number of epochs to wait before early stopping
  
- `--path_model`:
  - **Type:** str
  - **Default:** ../model/last
  - **Help:** path to model

### Model Architecture
Generator
The generator uses a series of convolutional layers, residual blocks, and pixel shuffle layers to upscale the input low-resolution images to high-resolution images.

### Discriminator
The discriminator uses convolutional layers with batch normalization and LeakyReLU activations to distinguish between real high-resolution images and generated images.

### Dataset
The dataset should be organized into three folders: train, eval, and test. Each folder should contain two subfolders: HR for high-resolution images and LR for low-resolution images. Images should be stored in .mat format where the first dimention is the spectral one.

### Logging and Results
Training logs and results are saved in the results/logs/ directory. Each training run creates a unique subdirectory named with the current date and time.
