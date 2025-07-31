"""
Loss functions for Multimodal Autoencoder (MiMo-AE).

This module implements various loss functions specifically designed for
multimodal signal reconstruction, including reconstruction losses,
and frequency domain loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class MiMoLoss(nn.Module):
    """
    Comprehensive loss function for multimodal autoencoder.
    
    Combines multiple loss components:
    - Reconstruction losses
    - frequency domain loss
    """
    
    def __init__(self, 
                 modality_weights: Optional[Dict[str, float]] = None,
                 lambda_frequency: float = 0.2,
        ):
        """
        Initialize the multimodal loss function.
        
        Args:
            modality_weights: Weights for each modality ['bp', 'breath_upper', 'ppg_fing']
            lambda_frequency: Weight for frequency domain loss
        """
        super().__init__()
        
        if modality_weights is None:
            self.modality_weights = {
                'bp': 1.0,           # Blood pressure
                'breath_upper': 1.0, # Breathing signal (slightly less weight)
                'ppg_fing': 1.0      # PPG signal
            }
        else:
            self.modality_weights = modality_weights
            
        self.lambda_frequency = lambda_frequency
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        
        self.modality_names = ['bp', 'breath_upper', 'ppg_fing']
        
    def forward(self, 
                reconstructed: torch.Tensor, 
                original: torch.Tensor, 
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the total loss and component losses.
        
        Args:
            reconstructed: Reconstructed signals (batch_size, 3, seq_len)
            original: Original signals (batch_size, 3, seq_len)
            
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # 1. Reconstruction Losses
        recon_losses = self._compute_reconstruction_losses(reconstructed, original)
        loss_dict.update(recon_losses)
        
        # 2. Frequency Domain Loss
        frequency_loss = self._compute_frequency_loss(reconstructed, original)
        loss_dict['frequency'] = frequency_loss
        
        
        # Compute total weighted loss
        total_loss = (
            (1 - self.lambda_frequency) * loss_dict['total_reconstruction'] +
            self.lambda_frequency * loss_dict['frequency'] 
        )
        
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def _compute_reconstruction_losses(self, reconstructed: torch.Tensor, original: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reconstruction losses for each modality and total."""
        losses = {}
        
        # Per-modality reconstruction losses
        modality_losses = []
        for i, modality in enumerate(self.modality_names):
            # Extract modality signals
            recon_modal = reconstructed[:, i, :]  # (batch_size, seq_len)
            orig_modal = original[:, i, :]        # (batch_size, seq_len)
            
            # Compute MSE loss for this modality
            modal_loss = self.mse_loss(recon_modal, orig_modal)
            losses[f'recon_{modality}'] = modal_loss
            
            # Weight the loss
            weighted_loss = self.modality_weights[modality] * modal_loss
            modality_losses.append(weighted_loss)
        
        total_recon_loss = sum(modality_losses)
        losses['total_reconstruction'] = total_recon_loss
        
        return losses
    
    def _compute_frequency_loss(self, reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Compute frequency domain loss using FFT.
        Preserves important frequency characteristics of physiological signals.
        """
        freq_losses = []
        
        for i in range(original.shape[1]):  # For each modality
            # Extract signals
            orig_modal = original[:, i, :]
            recon_modal = reconstructed[:, i, :]
            
            # Compute FFT magnitude spectra
            orig_fft = torch.fft.fft(orig_modal, dim=-1)
            recon_fft = torch.fft.fft(recon_modal, dim=-1)
            
            orig_magnitude = torch.abs(orig_fft)
            recon_magnitude = torch.abs(recon_fft)
            
            # L2 loss in frequency domain
            freq_loss = self.mse_loss(recon_magnitude, orig_magnitude)
            freq_losses.append(freq_loss)
        
        return torch.mean(torch.stack(freq_losses))
    

class AdaptiveMiMoLoss(MiMoLoss):
    """
    Adaptive version of multimodal loss that adjusts weights during training.
    """
    
    def __init__(self, *args, **kwargs):
        kwargs.pop('lambda_correlation', None)
        super().__init__(*args, **kwargs)
        
        # Initialize adaptive weights
        self.register_buffer('loss_history', torch.zeros(2))  # Track last 2 losses
        self.register_buffer('step_counter', torch.tensor(0))
        
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with adaptive weighting."""
        total_loss, loss_dict = super().forward(reconstructed, original)
        
        # Update adaptive weights every 100 steps
        if self.step_counter % 100 == 0:
            self._update_adaptive_weights(loss_dict)
        
        self.step_counter += 1
        
        return total_loss, loss_dict
    
    def _update_adaptive_weights(self, loss_dict: Dict[str, torch.Tensor]):
        """Update adaptive weights based on loss trends."""
        # Simple adaptive strategy: increase weight if loss is not decreasing
        current_losses = torch.stack([
            loss_dict['total_reconstruction'],
            loss_dict['frequency'],
        ])
        
        self.loss_history = current_losses.detach()
        


def create_loss_function(loss_type: str = 'standard', **kwargs) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: Type of loss ('standard' or 'adaptive')
        **kwargs: Additional arguments for loss function
        
    Returns:
        Configured loss function
    """
    if loss_type == 'standard':
        return MiMoLoss(**kwargs)
    elif loss_type == 'adaptive':
        return AdaptiveMiMoLoss(**kwargs)