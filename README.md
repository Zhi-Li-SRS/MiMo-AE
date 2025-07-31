# MiMo-AE: A Multimodal Autoencoder for Physiological Signals

This project implements a Multimodal Autoencoder (MiMo-AE) in PyTorch, designed to learn robust and compressed representations from various physiological signals simultaneously. 

## Project Structure

```
.
├── cleaned_data/       # Raw CSV data
├── processed_data/     # Preprocessed NPZ data
├── outputs/
│   ├── checkpoints/    # Saved model weights (.pth)
├── config.yaml         # Main configuration file
├── data_prepare.py     # Script for data preprocessing
├── train.py            # Main script for training and evaluation
├── model.py            # Contains the MiMoAE model architecture
├── loss.py             # Defines the custom loss functions
├── utils.py            # Utility functions (e.g., logging, metrics)
└── README.md           # This file
```

## Key Features

- **Multimodal Architecture**: Utilizes separate encoders for each signal modality to learn specialized features before fusing them into a shared representation.
- **Deep Residual Network**: Employs 1D ResNet blocks in the encoders and decoders for effective gradient flow and learning of complex patterns.
- **Configurable Fusion**: Supports multiple fusion strategies (`gated`, `attention`, `concat`) to combine features from different modalities.
- **Hybrid Loss Function**: Combines time-domain reconstruction loss (MSE) with a frequency-domain loss (FFT-based) to ensure both temporal and spectral characteristics are preserved.
- **Configuration-Files**: All aspects of the model, data, and training are controlled via a single `config.yaml` file for easy experimentation.
- **Experiment Tracking**: Integrated with Weights & Biases (`wandb`) for comprehensive logging of metrics, loss curves, and reconstruction visualizations.
- **End-to-End Pipeline**: Includes scripts for data preprocessing, training, and evaluation.

## Architecture Overview

The model follows a symmetric Encoder-Decoder structure:

1.  **Encoders**: Each modality is passed through its own dedicated 1D ResNet-based encoder to extract a high-level feature representation.
2.  **Fusion**: The features from all encoders are combined into a single vector using a specified fusion module (e.g., Gated Fusion).
3.  **Decoders**: The latent vector is projected back and fed into a set of parallel 1D ResNet-based decoders, which attempt to reconstruct the original signals for each modality.

## Getting Started

### 1. Prerequisites

- Python 3.8+

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd MiMo-AE
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install torch numpy pandas scipy pyyaml matplotlib tqdm wandb
    ```

### 3. Data Preparation
1.  Place your raw signal data in `.csv` format and filter into the `cleaned_data/` directory to only include the columns of interest. Each file should correspond to one subject.
2.  Update the `subjects` section in `config.yaml` to define your training, validation, and test splits.
3.  Run the data preprocessing script. This will filter, downsample, window, and normalize the data, saving the output as `.npz` files in the `processed_data/` directory.
    ```bash
    python data_prepare.py
    ```

## Usage

### 1. Configure Your Experiment
Modify `config.yaml` to set up the desired parameters for your experiment. Key sections include:
- `preprocessing`: Signal parameters, windowing, and sampling rates.
- `model`: Architecture details like latent dimension, fusion type, and dropout.
- `training`: Hyperparameters like learning rate, batch size, and number of epochs.
- `loss`: Weights for different loss components.

### 2. Train the Model

Start the training process by running:
```bash
python train.py --config config.yaml
```
Training progress, metrics, and visualizations will be logged to your Weights & Biases account. Model checkpoints will be saved in the `outputs/checkpoints/` directory.


