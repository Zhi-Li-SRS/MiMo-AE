"""
Training utilities for MiMo-AE (Multimodal Autoencoder).

This module provides utility functions for training, including:
- Model checkpointing and loading
- Metrics computation and logging
- Visualization functions for reconstruction comparison
- Early stopping implementation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import json
import wandb
from typing import Dict, List, Tuple, Optional
import yaml
import os


class MetricsTracker:
    """Track and compute various metrics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.losses = []
        self.recon_losses = {}
        self.freq_losses = []
        
    def update(self, loss_dict: Dict[str, torch.Tensor]):
        """Update metrics with new loss values."""
        self.losses.append(loss_dict['total'].item())
        
        modalities = ['bp', 'breath_upper', 'ppg_fing']
        for modality in modalities:
            key = f'recon_{modality}'
            if key in loss_dict:
                if key not in self.recon_losses:
                    self.recon_losses[key] = []
                self.recon_losses[key].append(loss_dict[key].item())
        
        if 'frequency' in loss_dict:
            self.freq_losses.append(loss_dict['frequency'].item())
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all tracked metrics."""
        averages = {}
        
        if self.losses:
            averages['total_loss'] = np.mean(self.losses)
        
        for key, values in self.recon_losses.items():
            if values:
                averages[key] = np.mean(values)
        
        if self.freq_losses:
            averages['freq_loss'] = np.mean(self.freq_losses)
            
        return averages


class EarlyStopping:
    """Early stopping implementation."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better) or 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            metric_value: Current metric value to monitor
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if self.best_score is None:
            self.best_score = metric_value
        elif self._is_better(metric_value, self.best_score):
            self.best_score = metric_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current value is better than best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta


class CheckpointManager:
    """Manage model checkpoints."""
    
    def __init__(self, save_dir: str, monitor: str = 'val_total_loss', mode: str = 'min'):
        """
        Args:
            save_dir: Directory to save checkpoints
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for the monitored metric
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        
    def save_checkpoint(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
        ) -> str:
        
        """Save model checkpoint."""
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        if scheduler is not None:
            ckpt['scheduler_state_dict'] = scheduler.state_dict()
        
        ckpt_path = self.save_dir / f'ckpt_epoch_{epoch}.pth'
        
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(ckpt, best_path)
            print(f"New best model saved at epoch {epoch}")
        
        return ckpt_path
    
    def load_ckpt(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        ckpt_path: str
        ) -> Tuple[int, Dict[str, float]]:
        
        """Load model checkpoint."""
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        return ckpt['epoch'], ckpt['metrics']
    
    def is_best_model(self, current_metric: float) -> bool:
        """Check if current metric is the best so far."""
        if self.best_score is None:
            self.best_score = current_metric
            return True
        
        if self.mode == 'min' and current_metric < self.best_score:
            self.best_score = current_metric
            return True
        
        elif self.mode == 'max' and current_metric > self.best_score:
            self.best_score = current_metric
            return True
        
        return False


class ReconVisualizer:
    """Visualize reconstruction results."""
    
    def __init__(self, modality_names: List[str], sampling_rate: float = 30.0):
        """
        Args:
            modality_names: Names of the modalities
            sampling_rate: Sampling rate of the signals
        """
        self.modality_names = modality_names
        self.sampling_rate = sampling_rate
        
    def plot_recon_comparison(
        self, 
        original: torch.Tensor, 
        recon: torch.Tensor,
        num_samples: int = 3,
        save_path: Optional[str] = None) -> plt.Figure:
        
        """
        Plot comparison between original and reconstructed signals.
        
        Args:
            original: Original signals (batch_size, num_modalities, seq_len)
            recon: Reconstructed signals (batch_size, num_modalities, seq_len)
            num_samples: Number of samples to plot
            save_path: Path to save the plot
            
        Returns:
            figure
        """
        batch_size, num_modalities, seq_len = original.shape
        num_samples = min(num_samples, batch_size)
        
        # Create time axis
        time_axis = np.arange(seq_len) / self.sampling_rate
        
        # Set up the plot
        fig, axes = plt.subplots(num_samples, num_modalities, 
                                figsize=(15, 4 * num_samples))
        
        plt.style.use('seaborn-v0_8-pastel')
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each sample and modality
        for sample_idx in range(num_samples):
            for mod_idx, modality in enumerate(self.modality_names):
                ax = axes[sample_idx, mod_idx]
                
                # Get signals for this sample and modality
                orig_signal = original[sample_idx, mod_idx, :].detach().cpu().numpy()
                recon_signal = recon[sample_idx, mod_idx, :].detach().cpu().numpy()
                
                # Plot signals
                ax.plot(time_axis, orig_signal, label='Original', 
                       color='blue', linewidth=1.5, alpha=0.8)
                ax.plot(time_axis, recon_signal, label='Reconstructed', 
                       color='red', linewidth=1.5, alpha=0.8, linestyle='--')
                
                # Compute and display MSE
                mse = np.mean((orig_signal - recon_signal) ** 2)
                ax.text(0.02, 0.98, f'MSE: {mse:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Formatting
                ax.set_title(f'Sample {sample_idx + 1} - {modality}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Intensity')
                ax.legend()
                ax.grid(True, alpha=0.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return fig
    
    def plot_loss_curves(
        self, 
        train_losses: Dict[str, List[float]], 
        val_losses: Dict[str, List[float]],
        save_path: Optional[str] = None
        ) -> plt.Figure:
        
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses: Dictionary of training losses over epochs
            val_losses: Dictionary of validation losses over epochs
            save_path: Path to save the plot
            
        Returns:
            figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        plt.style.use('seaborn-v0_8-pastel')
        axes = axes.flatten()
        
        # Plot total loss
        ax = axes[0]
        if 'total_loss' in train_losses:
            ax.plot(train_losses['total_loss'], label='Train', color='blue')
        if 'total_loss' in val_losses:
            ax.plot(val_losses['total_loss'], label='Validation', color='red')
        ax.set_title('Total Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.1)
        
        # Plot reconstruction losses
        ax = axes[1]
        recon_keys = [k for k in train_losses.keys() if k.startswith('recon_')]
        for key in recon_keys:
            if key in train_losses:
                ax.plot(train_losses[key], label=f'Train {key}', alpha=0.7)
            if key in val_losses:
                ax.plot(val_losses[key], label=f'Val {key}', alpha=0.7, linestyle='--')
        ax.set_title('Reconstruction Losses')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.1)
        
        # Plot frequency loss
        ax = axes[2]
        if 'freq_loss' in train_losses:
            ax.plot(train_losses['freq_loss'], label='Train', color='green')
        if 'freq_loss' in val_losses:
            ax.plot(val_losses['freq_loss'], label='Validation', color='orange')
        ax.set_title('Frequency Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.1)
        
        # Hide unused subplot
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
        return fig


# Load config file
def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config: Dict, model: nn.Module) -> None:
    """Initialize Weights & Biases logging."""
    wandb_config = config['logging']['wandb']
    
    # Initialize wandb
    wandb.init(
        project=wandb_config['project'],
        entity=wandb_config.get('entity'),
        tags=wandb_config.get('tags', []),
        notes=wandb_config.get('notes', ''),
        config=config
    )
    
    # Watch model
    wandb.watch(model, log_freq=100)


def log_metrics_to_wandb(metrics: Dict[str, float], step: int, prefix: str = '') -> None:
    """Log metrics to Weights & Biases."""
    logged_metrics = {}
    for key, value in metrics.items():
        logged_key = f"{prefix}/{key}" if prefix else key
        logged_metrics[logged_key] = value
    
    wandb.log(logged_metrics, step=step)


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, 
    scheduler_config: Dict
    ):
    
    """Create learning rate scheduler based on configuration."""
    scheduler_type = scheduler_config.get('scheduler_type', 'cosine')
    params = scheduler_config.get('scheduler_params', {})
    
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=params.get('T_max', 100)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get('step_size', 30),
            gamma=params.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")