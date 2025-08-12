# MiMo-AE: A Multimodal Autoencoder for Physiological Signals

This project implements a Multimodal Autoencoder (MiMo-AE), designed to learn robust and compressed representations from various physiological signals simultaneously. The model supports multiple fusion strategies and includes comprehensive evaluation tools for comparing different approaches within current datasets size.

## Project Structure
```
.
├── cleaned_data/         # filtered data with only the columns of interest
├── data/                 # raw data
├── processed_data/       # Preprocessed NPZ data files
├── outputs/
│   ├── checkpoints/      # Saved model weights (.pth)
│   └── comparison_plots/ # Model comparison visualizations
├── wandb/               # Weights & Biases experiment logs
├── config.yaml          # Main configuration file
├── data_prepare.py      # Script for data preprocessing
├── train.py             # Main script for training and evaluation
├── trainer.sh           # Training pipeline startup script
├── model.py             # Contains the MiMoAE model architecture
├── loss.py              # Defines the custom loss functions
├── utils.py             # Utility functions (e.g., logging, metrics)
├── module_compare.py    # Comprehensive fusion strategy comparison tool
├── raw_visual_test.py   # Inital data visualization when getting the raw data
├── requirements.txt     # Dependencies
└── README.md            # This file
```


## Key Features
- **Multimodal Architecture**: Utilizes encoders for each signal modality (BP, respiratory, PPG) to learn specialized features before fusing them into a shared representation.
- **Deep Residual Network**: Employs 1D ResNet blocks in the encoders and decoders for effective gradient flow and learning of complex patterns.
- **Multiple Fusion Strategies**: Supports three fusion approaches:
  - **Concatenation**: Simple feature concatenation
  - **Gated Fusion**: Learnable gating mechanism for feature selection
  - **Attention Fusion**: Multi-head attention for dynamic feature weighting
- **Comprehensive Evaluation**: Includes `module_compare.py` for comparison of fusion strategies using multiple metrics (MSE, SSIM, correlation, spectral analysis)
- **Hybrid Loss Function**: Combines time-domain reconstruction loss (MSE) with a frequency-domain loss (FFT-based) to ensure both temporal and spectral characteristics are preserved.
- **Configuration**: All aspects of the model, data, and training are within `config.yaml` file for easy experimentation.
- **Experiment Tracking**: Integrated with Weights & Biases (`wandb`) for comprehensive logging of metrics, loss curves, and reconstruction visualizations.

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
    git clone https://github.com/Zhi-Li-SRS/MiMo-AE.git
    cd MiMo-AE
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    conda create -n mimoae python=3.10
    conda activate mimoae
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
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

Start the training process using the automated script:
```bash
bash trainer.sh
```

Or train directly:
```bash
python train.py --config config.yaml
```

Training progress, metrics, and visualizations will be logged to your Weights & Biases account. Model checkpoints will be saved in the `outputs/checkpoints/` directory.

### 3. Compare Fusion Strategies

After training models with different fusion strategies, compare their performance:
```bash
python module_compare.py --config config.yaml
```
This will generate:
- Comprehensive metrics comparison (MSE, SSIM, correlation, SNR)
- Box plots showing performance distributions
- Radar charts for key metrics visualization
- CSV results file with detailed statistics

## Supported Signal Types

The current implementation processes three physiological signal modalities:
- **BP (Blood Pressure)**: Continuous blood pressure measurements
- **Respiratory (breath_upper)**: Upper respiratory signal
- **PPG (ppg_fing)**: Photoplethysmography from finger sensor

## Model Configurations

### Fusion Strategies
1. **Concatenation (`concat`)**: Simple concatenation of encoder outputs
2. **Gated Fusion (`gated`)**: Learnable gates control information flow
3. **Attention Fusion (`attention`)**: Multi-head attention mechanism for dynamic weighting



