"""
training script for MiMo-AE (Multimodal Autoencoder).

This script provides a complete training pipeline with:
- Configuration-based setup
- Model checkpointing and early stopping
- Reconstruction visualization
- Comprehensive logging with wandb
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
import wandb
from typing import Dict, Optional
import os

from model import MiMoAE
from loss import create_loss_function
from data_prepare import MultimodalDataset
from utils import *


class Trainer:
    """Main trainer class for MiMo-AE."""
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup directories
        self._setup_directories()
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary directories."""
        paths = self.config['paths']
        self.data_dir = Path(paths['data_dir'])
        self.output_dir = Path(paths['output_dir'])
        
        # Create output directories
        self.ckpt_dir = Path(self.output_dir) / self.config['logging']['checkpoint']['save_dir']
        self.plot_dir = Path(self.output_dir) / self.config['logging']['visualization']['plot_dir']
        
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
    def _setup_data(self):
        """Setup data loaders."""
        print("Setting up data loaders...")
        
        train_config = self.config['training']
        
        # Load datasets
        train_data_path = os.path.join(self.data_dir, 'train_windows.npz')
        eval_data_path = os.path.join(self.data_dir, 'eval_windows.npz')
        test_data_path = os.path.join(self.data_dir, 'test_windows.npz')
        
        train_dataset = MultimodalDataset(str(train_data_path))
        eval_dataset = MultimodalDataset(str(eval_data_path))
        test_dataset = MultimodalDataset(str(test_data_path))
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=train_config['batch_size'],
            shuffle=True,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory'],
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            eval_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory']
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=train_config['num_workers'],
            pin_memory=train_config['pin_memory']
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(eval_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
    def _setup_model(self):
        """Setup model and move to device."""
        print("Setting up model...")
        
        model_config = self.config['model']
        self.model = MiMoAE(model_config).to(self.device)
        
        
    def _setup_training(self):
        """Setup training components (optimizer, scheduler, loss, etc.)."""
        print("Setting up training components...")
        
        train_config = self.config['training']
        loss_config = self.config['loss']
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = create_lr_scheduler(self.optimizer, train_config)
        
        # Setup loss function
        self.criterion = create_loss_function(
            loss_type=loss_config['loss_type'],
            modality_weights=loss_config['modality_weights'],
            lambda_frequency=loss_config['lambda_frequency']
        )
        
        # Setup early stopping
        early_stop_config = train_config['early_stopping']
        self.early_stopping = EarlyStopping(
            patience=early_stop_config['patience'],
            min_delta=early_stop_config['min_delta'],
            mode='min'
        )
        
        # Setup checkpoint manager
        ckpt_config = self.config['logging']['checkpoint']
        self.checkpoint_manager = CheckpointManager(
            save_dir=str(self.ckpt_dir),
            monitor=ckpt_config['monitor'],
            mode=ckpt_config['mode']
        )
        
        # Setup metrics trackers
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Track loss history for plotting
        self.train_loss_history = {'total_loss': [], 'recon_bp': [], 'recon_breath_upper': [], 
                                  'recon_ppg_fing': [], 'freq_loss': []}
        self.val_loss_history = {'total_loss': [], 'recon_bp': [], 'recon_breath_upper': [], 
                                'recon_ppg_fing': [], 'freq_loss': []}
        
    def _setup_logging(self):
        """Setup logging components."""
        
        # Setup wandb
        setup_wandb(self.config, self.model)
        
        # Setup visualizer
        modality_names = self.config['preprocessing']['signal_cols']
        self.visualizer = ReconVisualizer(
            modality_names=modality_names,
            sampling_rate=self.config['preprocessing']['target_rate']
        )
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for _, data in enumerate(pbar):
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            reconstructed, _ = self.model(data)
            
            # Compute loss
            total_loss, loss_dict = self.criterion(reconstructed, data)
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            grad_clip = self.config['training']['grad_clip']
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            self.optimizer.step()
            
            # Update metrics
            self.train_metrics.update(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        return self.train_metrics.get_averages()
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
    
        self.model.eval()
        self.val_metrics.reset()
        
        # For visualization
        sample_original = None
        sample_reconstructed = None
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, data in enumerate(pbar):
                data = data.to(self.device)
                
                # Forward pass
                reconstructed, latent_vec = self.model(data)
                
                # Compute loss
                total_loss, loss_dict = self.criterion(reconstructed, data)
                
                # Update metrics
                self.val_metrics.update(loss_dict)
                
                # Save first batch for visualization
                if batch_idx == 0:
                    sample_original = data[:3].cpu()  # Take first 3 samples
                    sample_reconstructed = reconstructed[:3].cpu()
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        # Visualization
        vis_config = self.config['logging']['visualization']
        if (epoch + 1) % vis_config['plot_every_n_epochs'] == 0 and sample_original is not None:
            self._visualize_recon(epoch, sample_original, sample_reconstructed)
        
        return self.val_metrics.get_averages()
    
    def _visualize_recon(self, epoch: int, original: torch.Tensor, reconstructed: torch.Tensor):
        """Create and save reconstruction visualization."""
        vis_config = self.config['logging']['visualization']
        
        # Create plot
        save_path = self.plot_dir / f'recon_epoch_{epoch+1}.png'
        fig = self.visualizer.plot_recon_comparison(
            original, reconstructed,
            num_samples=vis_config['num_samples_to_plot'],
            save_path=str(save_path) if vis_config['save_plots'] else None
        )
        
        # Log to wandb
        wandb.log({'recon_comparison': wandb.Image(fig)}, step=epoch)
        plt.close(fig)
    
    def _update_loss_history(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Update loss history for plotting."""
        for key in self.train_loss_history.keys():
            self.train_loss_history[key].append(train_metrics.get(key, 0))
            self.val_loss_history[key].append(val_metrics.get(key, 0))
    
    def _plot_loss_curves(self, epoch: int):
        """Plot and save loss curves."""
        save_path = self.plot_dir / f'loss_curves_epoch_{epoch+1}.png'
        fig = self.visualizer.plot_loss_curves(
            self.train_loss_history, 
            self.val_loss_history,
            save_path=str(save_path)
        )
        
        # Log to wandb
        wandb.log({'loss_curves': wandb.Image(fig)}, step=epoch)
        plt.close(fig)
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        train_config = self.config['training']
        checkpoint_config = self.config['logging']['checkpoint']
        vis_config = self.config['logging']['visualization']
        
        num_epochs = train_config['num_epochs']
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate_epoch(epoch)
            
            # Update loss history
            self._update_loss_history(train_metrics, val_metrics)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):      
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Logging
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # Log to wandb
            log_metrics_to_wandb(train_metrics, epoch, 'train')
            log_metrics_to_wandb(val_metrics, epoch, 'val')
            wandb.log({'epoch': epoch, 'learning_rate': self.optimizer.param_groups[0]['lr']}, step=epoch)
            
            # Check for best model
            current_val_loss = val_metrics['total_loss']
            is_best = self.checkpoint_manager.is_best_model(current_val_loss)
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_config['save_every_n_epochs'] == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, self.scheduler,
                    epoch, val_metrics, is_best
                )
            
            # Plot loss curves
            if (epoch + 1) % vis_config['plot_every_n_epochs'] == 0:
                self._plot_loss_curves(epoch)
            
            # Early stopping
            if self.early_stopping(current_val_loss):
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break
        
        print("Training completed!")
        
        # Save final model
        final_checkpoint_path = self.checkpoint_manager.save_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch, val_metrics, is_best=False
        )
        print(f"Final model saved to {final_checkpoint_path}")
        
        # Final loss curves
        self._plot_loss_curves(epoch)
    
    def test(self, ckpt_path: Optional[str] = None):
        """Evaluate on test set."""
        print("Running test evaluation...")
        
        # Load best model if checkpoint provided
        if ckpt_path:
            print(f"Loading checkpoint from {ckpt_path}")
            epoch, metrics = self.checkpoint_manager.load_ckpt(
                self.model, self.optimizer, self.scheduler, ckpt_path
            )
            print(f"Loaded model from epoch {epoch}")
        
        self.model.eval()
        test_metrics = MetricsTracker()
        
        # Store all results for comprehensive analysis
        all_original = []
        all_reconstructed = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Testing')
            
            for data in pbar:
                data = data.to(self.device)
                
                reconstructed, latent_vec = self.model(data)
                
                total_loss, loss_dict = self.criterion(reconstructed, data)
                
                test_metrics.update(loss_dict)
                
                # Store for analysis
                all_original.append(data.cpu())
                all_reconstructed.append(reconstructed.cpu())
                
                pbar.set_postfix({'loss': f'{total_loss.item():.4f}'})
        
        # Get final metrics
        final_metrics = test_metrics.get_averages()
        
        print("Test Results:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Log to wandb
        log_metrics_to_wandb(final_metrics, 0, 'test')
        
        # Create comprehensive visualization
        all_original = torch.cat(all_original, dim=0)
        all_reconstructed = torch.cat(all_reconstructed, dim=0)
        
        # Plot a few test samples
        vis_config = self.config['logging']['visualization']
        test_save_path = self.plot_dir / 'test_recon_samples.png'
        fig = self.visualizer.plot_recon_comparison(
            all_original[:vis_config['num_samples_to_plot']], 
            all_reconstructed[:vis_config['num_samples_to_plot']],
            num_samples=vis_config['num_samples_to_plot'],
            save_path=str(test_save_path) if vis_config['save_plots'] else None
        )
        
        # Log to wandb
        wandb.log({'test_reconstruction': wandb.Image(fig)})
        plt.close(fig)
        
        return final_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train MiMo-AE model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--test-only', default=False,
                       help='Only run testing (requires checkpoint)')
    parser.add_argument('--ckpt', type=str, default=None,
                       help='Path to checkpoint for testing')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    if args.test_only:
        if args.ckpt is None:
            ckpt_path = os.path.join(trainer.ckpt_dir, 'best_model.pth')
        else:
            ckpt_path = args.ckpt
        
        if not Path(ckpt_path).exists():
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        
        trainer.test(str(ckpt_path))
    else:
        trainer.train()
        
        # Test with best model
        best_model_path = os.path.join(trainer.ckpt_dir, 'best_model.pth')
        if best_model_path.exists():
            trainer.test(str(best_model_path))


if __name__ == '__main__':
    main()