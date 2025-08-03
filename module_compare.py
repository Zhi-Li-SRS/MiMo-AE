"""
This script compares the performance of three fusion strategies:
- Concatenation (concat)
- Gated fusion (gated) 
- Attention fusion (attention)

Evaluates on multiple metrics including reconstruction quality,
frequency domain preservation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import signal
from scipy.stats import pearsonr
from scipy.spatial.distance import correlation
import argparse
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for bold fonts globally
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'

# Local imports
from model import MiMoAE
from loss import create_loss_function
from data_prepare import MultimodalDataset
from utils import load_config, set_random_seed
from torch.utils.data import DataLoader


def ssim_1d(x, y, window_size=11, sigma=1.5):
    """
    Compute 1D SSIM for signals.
    """
    def gaussian_window(size, sigma):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1)
    
    if len(x.shape) == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    if len(y.shape) == 1:
        y = y.unsqueeze(0).unsqueeze(0)
    
    window = gaussian_window(window_size, sigma).to(x.device)
    
    mu1 = torch.nn.functional.conv1d(x, window, padding=window_size//2)
    mu2 = torch.nn.functional.conv1d(y, window, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv1d(x * x, window, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.conv1d(y * y, window, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.conv1d(x * y, window, padding=window_size//2) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_psd_correlation(x, y, fs=30):
    """Compute power spectral density correlation."""
    freqs1, psd1 = signal.welch(x, fs=fs, nperseg=min(64, len(x)//4))
    freqs2, psd2 = signal.welch(y, fs=fs, nperseg=min(64, len(y)//4))
    
    if len(psd1) != len(psd2):
        min_len = min(len(psd1), len(psd2))
        psd1 = psd1[:min_len]
        psd2 = psd2[:min_len]
    
    corr, _ = pearsonr(psd1, psd2)
    return corr if not np.isnan(corr) else 0



class MetricsCalculator:
    """Calculate comprehensive metrics for signal reconstruction evaluation."""
    
    def __init__(self, modality_names=['bp', 'breath_upper', 'ppg_fing'], fs=30):
        self.modality_names = modality_names
        self.fs = fs
        
    def calculate_all_metrics(self, original, reconstructed):
        """Calculate all metrics for given signals."""
        batch_size, n_modalities, seq_len = original.shape
        
        metrics = {
            'mse': [], 'mae': [], 'ssim': [], 'pearson_corr': [],
            'psd_corr': [], 'snr': []
        }
        
        # Add per-modality storage - use static list to avoid iteration issues
        base_metric_keys = list(metrics.keys())
        for mod in self.modality_names:
            for metric in base_metric_keys:
                metrics[f'{metric}_{mod}'] = []
        
        for batch_idx in range(batch_size):
            metric_keys = list(metrics.keys())
            batch_metrics = {'total': {metric: [] for metric in metric_keys}}
            
            for mod_idx, mod_name in enumerate(self.modality_names):
                mod_metrics = {}
                
                orig_sig = original[batch_idx, mod_idx, :].cpu().numpy()
                recon_sig = reconstructed[batch_idx, mod_idx, :].cpu().numpy()
                
                # Basic reconstruction metrics
                mod_metrics['mse'] = mean_squared_error(orig_sig, recon_sig)
                mod_metrics['mae'] = mean_absolute_error(orig_sig, recon_sig)
                
                # SSIM
                orig_tensor = torch.tensor(orig_sig)
                recon_tensor = torch.tensor(recon_sig)
                mod_metrics['ssim'] = ssim_1d(orig_tensor, recon_tensor)
                
                # Correlation metrics
                try:
                    corr, _ = pearsonr(orig_sig, recon_sig)
                    mod_metrics['pearson_corr'] = corr if not np.isnan(corr) else 0
                except:
                    mod_metrics['pearson_corr'] = 0
                
                
                # Frequency domain metrics
                mod_metrics['psd_corr'] = compute_psd_correlation(orig_sig, recon_sig, self.fs)
                
                # SNR (Signal-to-Noise Ratio)
                signal_power = np.mean(orig_sig ** 2)
                noise_power = np.mean((orig_sig - recon_sig) ** 2)
                mod_metrics['snr'] = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
                
                # Store per-modality metrics
                for metric_name, value in mod_metrics.items():
                    metrics[f'{metric_name}_{mod_name}'].append(value)
                    batch_metrics['total'][metric_name].append(value)
            
            # Store overall metrics (averaged across modalities)
            for metric_name in metric_keys:
                if not metric_name.endswith(tuple(self.modality_names)):
                    if metric_name in batch_metrics['total'] and batch_metrics['total'][metric_name]:
                        metrics[metric_name].append(np.mean(batch_metrics['total'][metric_name]))
        
        return metrics


class ModelComparator:
    """Compare different fusion strategy models."""
    
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed for reproducibility
        if 'random_seed' in self.config['training']:
            set_random_seed(self.config['training']['random_seed'])
        
        # Model configurations for different fusion types - check which ones exist
        available_models = []
        self.model_paths = {}
        
        potential_models = {
            'concat': 'outputs/checkpoints/best_model_concat.pth',
            'gated': 'outputs/checkpoints/best_model_gated.pth', 
            'attention': 'outputs/checkpoints/best_model_attention.pth'
        }
        
        for fusion_type, path in potential_models.items():
            if Path(path).exists():
                # Check if this is actually the correct fusion type
                try:
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    fusion_keys = [key for key in checkpoint['model_state_dict'].keys() if 'fusion_module' in key]
                    
                    if not fusion_keys:  # No fusion keys = concat
                        if fusion_type == 'concat':
                            available_models.append(fusion_type)
                            self.model_paths[fusion_type] = path
                    elif 'gate' in str(fusion_keys):  # Gate keys = gated
                        if fusion_type == 'gated':
                            available_models.append(fusion_type)
                            self.model_paths[fusion_type] = path
                    elif 'attention' in str(fusion_keys):  # Attention keys = attention
                        if fusion_type == 'attention':
                            available_models.append(fusion_type)
                            self.model_paths[fusion_type] = path
                except Exception as e:
                    print(f"Warning: Could not validate {fusion_type} model: {e}")
        
        self.fusion_types = available_models
        print(f"Available models: {self.fusion_types}")
        
        # Initialize metrics calculator
        self.metrics_calc = MetricsCalculator(
            modality_names=self.config['preprocessing']['signal_cols'],
            fs=self.config['preprocessing']['target_rate']
        )
        
        self._setup_data()
        
    def _setup_data(self):
        """Setup data loaders."""
        print("Setting up data loaders...")
        
        # Load eval and test datasets
        data_dir = Path(self.config['paths']['data_dir'])
        eval_data_path = data_dir / 'eval_windows.npz'
        test_data_path = data_dir / 'test_windows.npz'
        
        eval_dataset = MultimodalDataset(str(eval_data_path))
        test_dataset = MultimodalDataset(str(test_data_path))
        
        batch_size = self.config['training']['batch_size']
        
        self.eval_loader = DataLoader(
            eval_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=2, pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        print(f"Eval samples: {len(eval_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def load_model(self, fusion_type):
        """Load model with specific fusion type."""
        # Create model config with specific fusion type
        model_config = self.config['model'].copy()
        model_config['fusion_type'] = fusion_type
        
        # Create model
        model = MiMoAE(model_config).to(self.device)
        
        # Load checkpoint
        checkpoint_path = self.model_paths[fusion_type]
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded {fusion_type} model from {checkpoint_path}")
        return model
    
    def evaluate_model(self, model, data_loader, dataset_name):
        """Evaluate a single model on given dataset."""
        print(f"Evaluating model on {dataset_name} dataset...")
        
        all_metrics = {
            'mse': [], 'mae': [], 'ssim': [], 'pearson_corr': [],
            'psd_corr': [], 'snr': []
        }
        
        base_metrics = list(all_metrics.keys())
        for mod in self.metrics_calc.modality_names:
            for metric in base_metrics:
                all_metrics[f'{metric}_{mod}'] = []
        
        with torch.no_grad():
            for batch_idx, data in enumerate(data_loader):
                data = data.to(self.device)
                reconstructed, _ = model(data)
                
                # Calculate metrics for this batch
                batch_metrics = self.metrics_calc.calculate_all_metrics(data, reconstructed)
                
                # Accumulate metrics using a copy of keys to avoid iteration issues
                for metric_name in list(batch_metrics.keys()):
                    if metric_name in all_metrics:
                        all_metrics[metric_name].extend(batch_metrics[metric_name])
        
        # Convert to numpy arrays and compute statistics
        results = {}
        for metric_name, values in all_metrics.items():
            if values:  # Only if there are values
                values_array = np.array(values)
                results[metric_name] = {
                    'mean': np.mean(values_array),
                    'std': np.std(values_array),
                    'median': np.median(values_array),
                    'values': values_array
                }
        
        return results
    
    def compare_all_models(self):
        """Compare all fusion strategies on both eval and test sets."""
        print("Starting comprehensive model comparison...")
        
        results = {}
        
        for fusion_type in self.fusion_types:
            print(f"\n{'='*50}")
            print(f"Evaluating {fusion_type.upper()} fusion model")
            print(f"{'='*50}")
            
            try:
                # Load model
                model = self.load_model(fusion_type)
                
                # Evaluate on both datasets
                results[fusion_type] = {}
                results[fusion_type]['eval'] = self.evaluate_model(model, self.eval_loader, 'eval')
                results[fusion_type]['test'] = self.evaluate_model(model, self.test_loader, 'test')
                
                # Print summary
                self._print_model_summary(fusion_type, results[fusion_type])
                
            except Exception as e:
                print(f"Error evaluating {fusion_type} model: {e}")
                continue
        
        return results
    
    def _print_model_summary(self, fusion_type, model_results):
        """Print summary statistics for a model."""
        print(f"\n{fusion_type.upper()} Model Summary:")
        print("-" * 40)
        
        for dataset_name in ['eval', 'test']:
            if dataset_name in model_results:
                print(f"\n{dataset_name.upper()} Dataset:")
                data = model_results[dataset_name]
                
                key_metrics = ['mse', 'ssim', 'pearson_corr', 'psd_corr']
                for metric in key_metrics:
                    if metric in data:
                        mean_val = data[metric]['mean']
                        std_val = data[metric]['std']
                        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
    
    def create_comparison_plots(self, results, save_dir='outputs/comparison_plots'):
        """Create comprehensive comparison visualizations."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Overall metrics comparison (boxplots)
        self._plot_overall_metrics_comparison(results, save_dir)
        
        # Radar plot for key metrics
        self._plot_radar_chart(results, save_dir)
        
        print(f"Comparison plots saved to {save_dir}")
    
    def _plot_overall_metrics_comparison(self, results, save_dir):
        """Create boxplots comparing overall metrics across fusion types."""
        # Define metrics with directional indicators
        metric_directions = {
            'mse': '↓', 'mae': '↓', 'ssim': '↑', 
            'pearson_corr': '↑', 'psd_corr': '↑', 'snr': '↑'
        }
        key_metrics = list(metric_directions.keys())
        datasets = ['eval', 'test']
        
        plt.style.use('seaborn-v0_8-pastel')
        fig, axes = plt.subplots(len(datasets), len(key_metrics), 
                                figsize=(20, 8), squeeze=False)
        
        for dataset_idx, dataset in enumerate(datasets):
            for metric_idx, metric in enumerate(key_metrics):
                ax = axes[dataset_idx, metric_idx]
                
                data_for_plot = []
                labels = []
                
                for fusion_type in self.fusion_types:
                    if (fusion_type in results and 
                        dataset in results[fusion_type] and 
                        metric in results[fusion_type][dataset]):
                        
                        values = results[fusion_type][dataset][metric]['values']
                        data_for_plot.append(values)
                        labels.append(fusion_type.capitalize())
                
                if data_for_plot:
                    bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
                    
                    # Color the boxes with alpha
                    colors = ['lightblue', 'lightgreen', 'lightcoral']
                    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    # Make mean lines black and more visible
                    for median in bp['medians']:
                        median.set_color('black')
                        median.set_linewidth(2)
                
                # Add directional arrow to title
                direction = metric_directions[metric]
                ax.set_title(f'{metric.upper()} {direction} - {dataset.upper()}', 
                           fontweight='bold', fontsize=12)
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(True, alpha=0.1)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'overall_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    
    def _plot_radar_chart(self, results, save_dir):
        """Create radar chart comparing key metrics."""
        # Add directional indicators to metric labels
        metric_labels = {
            'ssim': 'SSIM ↑', 
            'pearson_corr': 'Pearson Corr ↑', 
            'psd_corr': 'PSD Corr ↑', 
            'snr': 'SNR ↑'
        }
        key_metrics = list(metric_labels.keys())
        
        fusion_data = {}
        for fusion_type in self.fusion_types:
            if fusion_type in results and 'test' in results[fusion_type]:
                fusion_data[fusion_type] = []
                for metric in key_metrics:
                    if metric in results[fusion_type]['test']:
                        value = results[fusion_type]['test'][metric]['mean']
                        if metric == 'snr':
                            value = min(value / 50, 1)  # Normalize SNR
                        fusion_data[fusion_type].append(max(0, min(value, 1)))
                    else:
                        fusion_data[fusion_type].append(0)
        
        if fusion_data:
            plt.style.use('seaborn-v0_8-pastel')
            # Create radar chart with reduced white space
            angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            colors = ['lightblue', 'lightgreen', 'lightcoral']  
            linestyles = ['-', '-', '-']
            
            for i, (fusion_type, values) in enumerate(fusion_data.items()):
                values = np.concatenate((values, [values[0]]))  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=3, 
                       label=fusion_type.capitalize(), 
                       color=colors[i % len(colors)],
                       linestyle=linestyles[i % len(linestyles)],
                       markersize=8)
                ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([metric_labels[m] for m in key_metrics], 
                             fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], 
                             fontsize=10, fontweight='bold')
            
            ax.set_title('Key Metrics Comparison (Higher is Better)', 
                        size=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), 
                     fontsize=11, prop={'weight': 'bold'})
            ax.grid(True, alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'radar_chart.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_results_to_csv(self, results, save_path='outputs/comparison_results.csv'):
        """Save comparison results to CSV."""
        rows = []
        
        for fusion_type in self.fusion_types:
            if fusion_type not in results:
                continue
                
            for dataset in ['eval', 'test']:
                if dataset not in results[fusion_type]:
                    continue
                    
                for metric_name, metric_data in results[fusion_type][dataset].items():
                    row = {
                        'fusion_type': fusion_type,
                        'dataset': dataset,
                        'metric': metric_name,
                        'mean': metric_data['mean'],
                        'std': metric_data['std'],
                        'median': metric_data['median']
                    }
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare MiMo-AE fusion strategies')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    comparator = ModelComparator(args.config)
    
    print("Starting model comparison...")
    results = comparator.compare_all_models()
    
    if results:
        # Create visualizations
        plot_dir = Path(args.output_dir) / 'comparison_plots'
        comparator.create_comparison_plots(results, plot_dir)
        
        # Save results
        csv_path = Path(args.output_dir) / 'comparison_plots' / 'comparison_results.csv'
        comparator.save_results_to_csv(results, csv_path)
        
        print("\nComparison completed successfully!")
        print(f"Results saved to {args.output_dir}")
        
        # Print final summary
        print("\n" + "="*60)
        print("FINAL COMPARISON SUMMARY")
        print("="*60)
        
        key_metrics = ['mse', 'mae', 'ssim', 'pearson_corr', 'psd_corr', 'snr']
        for metric in key_metrics:
            print(f"\n{metric.upper()} (Test Dataset):")
            for fusion_type in comparator.fusion_types:
                if (fusion_type in results and 'test' in results[fusion_type] 
                    and metric in results[fusion_type]['test']):
                    mean_val = results[fusion_type]['test'][metric]['mean']
                    print(f"  {fusion_type.capitalize()}: {mean_val:.4f}")
    else:
        print("No results to display. Check model checkpoints.")


if __name__ == '__main__':
    main()