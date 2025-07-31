"""
Data preprocessing pipeline for multimodal auto-encoder training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt
import json
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import argparse
import yaml


class MultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal physiological signals."""
    
    def __init__(self, data_path: str):
        """
        Initialize dataset from .npz file.
        windows: (N, 3, 240)
        Args:
            data_path: Path to .npz file containing 'windows' array of shape (N, 3, 240)
        """
        data = np.load(data_path)
        self.windows = torch.FloatTensor(data['windows'])
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return self.windows[idx]


class DataPreprocessor:
    """Preprocessing physiological signal data."""
    
    def __init__(self, raw_dir: str, 
                 out_dir: str, 
                 config_path: str = 'config.yaml'):
        """
        Initialize preprocessor.
        
        Args:
            raw_dir: Directory containing cleaned CSV files
            out_dir: Output directory for processed data
            config_path: Path to configuration YAML file
            target_rate: Target sampling rate in Hz (default: 30) - can be overridden by config
            window_len: Window length in seconds (default: 8.0) - can be overridden by config
            stride_ratio: Stride as fraction of window_len (1.0 = non-overlapping, 0.5 = 50% overlap) - can be overridden by config
        """
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.config_path = Path(config_path)
        
        # Load configuration
        self._load_config()
        
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self):
        """Load configuration from YAML file."""
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load subject splits
        subjects_config = config.get('subjects', {})
        self.train_subjects = subjects_config.get('train', [])
        self.eval_subjects = subjects_config.get('eval', [])
        self.test_subjects = subjects_config.get('test', [])
        
        # Load preprocessing parameters (with fallback to defaults)
        preprocessing_config = config.get('preprocessing', {})
        self.target_rate = preprocessing_config.get('target_rate')
        self.window_len = preprocessing_config.get('window_len')
        self.stride_ratio = preprocessing_config.get('stride_ratio')
        self.signal_cols = preprocessing_config.get('signal_cols', ['bp', 'breath_upper', 'ppg_fing'])
        
        print(f"Loaded configuration from {self.config_path}")
        print(f"  Train subjects: {len(self.train_subjects)}")
        print(f"  Eval subjects: {len(self.eval_subjects)}")
        print(f"  Test subjects: {len(self.test_subjects)}")
            
        
    
    def load_subject_data(self, subject: str) -> pd.DataFrame:
        """Load CSV data for a subject."""
        
        csv_path = self.raw_dir / f"{subject}_biopac_v1.0.0.csv"
        return pd.read_csv(csv_path)
    
    def exploratory_check(self) -> Dict:
        """
        Perform check on all subjects.
        Returns sampling rate info and creates visualization plots.
        """
        print("Step 1: Exploratory check...")
        
        sampling_info = {}
        fig, axes = plt.subplots(2, 5, figsize=(20, 8)) # 2 rows, 5 columns to visualize 10 subjects
        plt.style.use('seaborn-v0_8-pastel')
        axes = axes.flatten()
        
        all_subjects = self.train_subjects + self.eval_subjects + self.test_subjects
        
        for i, subject in enumerate(all_subjects):
            df = self.load_subject_data(subject)
            
            time_diff = df['time'].diff().dropna()
            dt = time_diff.median()
            sampling_rate = 1.0 / dt
            sampling_info[subject] = {
                'sampling_rate': sampling_rate,
                'duration': df['time'].iloc[-1] - df['time'].iloc[0],
                'n_samples': len(df)
            }
            
            # Plot first 8 seconds of data
            end_idx = min(int(8 * sampling_rate), len(df))
            plot_data = df.iloc[:end_idx]
            
            ax = axes[i]
            for j, col in enumerate(self.signal_cols):
                ax.plot(plot_data['time'], plot_data[col], label=col, alpha=0.7)
            ax.set_title(f'{subject} (fs≈{sampling_rate:.0f}Hz)')
            ax.set_xlabel('Time (s)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.out_dir / 'check_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save sampling info
        with open(self.out_dir / 'sampling_info.json', 'w') as f:
            json.dump(sampling_info, f, indent=2)
            
        print(f"Data check complete. Found sampling rates around {np.mean([info['sampling_rate'] for info in sampling_info.values()]):.0f} Hz")
        return sampling_info
    
    def apply_bandpass_filter(self, data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """
        Apply bandpass filtering to physiological signals with different frequency ranges.
        
        Args:
            data: Array of shape (N_samples, 3) with signals [bp, breath_upper, ppg_fing]
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Filtered array of same shape
        """
        filtered_data = data.copy()
        
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            print(f"  Warning: Input data contains NaN or Inf values!")
            # Replace NaN/Inf with interpolated values
            for channel_idx in range(data.shape[1]):
                signal = data[:, channel_idx]
                if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                    # Simple forward-fill for NaN/Inf values
                    mask = np.isfinite(signal)
                    if np.any(mask):
                        signal = np.interp(np.arange(len(signal)), 
                                         np.where(mask)[0], 
                                         signal[mask])
                        filtered_data[:, channel_idx] = signal
        
        filter_configs = {
            0: {'low': 0.5, 'high': 8.0, 'name': 'bp'},           # bp: 0.5-8 Hz
            1: {'low': 0.1, 'high': 2.0, 'name': 'breath_upper'}, # breath: 0.1-2 Hz  
            2: {'low': 0.5, 'high': 8.0, 'name': 'ppg_fing'}      # ppg: 0.5-8 Hz
        }
        
        for channel_idx, config in filter_configs.items():
            # Calculate normalized frequencies (0-1, where 1 is Nyquist frequency)
            nyquist = sampling_rate / 2
            low_norm = config['low'] / nyquist
            high_norm = config['high'] / nyquist
            
            # Ensure frequencies are within valid range
            low_norm = max(0.001, min(low_norm, 0.99))  # Avoid edge cases
            high_norm = max(low_norm + 0.001, min(high_norm, 0.99))
            
            min_samples_needed = 6  # 3x the filter order (2)
            if len(filtered_data) < min_samples_needed:
                print(f"  Warning: Signal too short ({len(filtered_data)} samples) for filtering {config['name']}, skipping")
                continue
            

            b, a = butter(2, [low_norm, high_norm], btype='band')
            
            # Apply zero-phase filtering to avoid phase distortion
            filtered_signal = filtfilt(b, a, filtered_data[:, channel_idx])
            
            # Check if filtering introduced NaN values
            if np.any(np.isnan(filtered_signal)) or np.any(np.isinf(filtered_signal)):
                print(f"  Warning: Filtering introduced NaN/Inf for {config['name']}, skipping filter")
                # Keep original signal if filtering fails
                continue
            else:
                filtered_data[:, channel_idx] = filtered_signal
                
            print(f"  Applied {config['low']}-{config['high']}Hz bandpass filter to {config['name']}")
            
        
        return filtered_data

    def downsample_signals(self, data: np.ndarray, original_rate: float) -> np.ndarray:
        """
        Downsample signals to target rate with anti-aliasing.
        
        Args:
            data: Array of shape (N_samples, 3) with signals
            original_rate: Original sampling rate
            
        Returns:
            Downsampled array of shape (N_samples_ds, 3)
        """
        if original_rate <= self.target_rate:
            return data
            
        # Calculate decimation factor
        decimation_factor = int(original_rate / self.target_rate)
        
        # Apply decimation with anti-aliasing filter to each channel
        downsampled_channels = []
        for i in range(3):
            downsampled_channels.append(signal.decimate(data[:, i], decimation_factor, ftype='iir'))

        # Truncate to the minimum length to handle potential off-by-one errors
        min_len = min(len(ch) for ch in downsampled_channels)
        downsampled = np.zeros((min_len, 3))
        for i in range(3):
            downsampled[:, i] = downsampled_channels[i][:min_len]

        return downsampled
    
    def segment_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Segment data into windows.
        
        Args:
            data: Downsampled array of shape (N_samples, 3)
            
        Returns:
            Windowed array of shape (n_windows, 3, window_samples)
        """
        window_samples = int(self.window_len * self.target_rate)  # 240 samples for 8s at 30Hz
        stride_samples = int(window_samples * self.stride_ratio)
        
        n_windows = (len(data) - window_samples) // stride_samples + 1
        
        if n_windows <= 0:
            return np.array([]).reshape(0, 3, window_samples)
            
        windows = np.zeros((n_windows, 3, window_samples))
        
        for i in range(n_windows):
            start_idx = i * stride_samples
            end_idx = start_idx + window_samples
            windows[i] = data[start_idx:end_idx].T  # Transpose to get (3, 240)
            
        return windows
    
    def process_subject(self, subject: str, sampling_rate: float) -> np.ndarray:
        """Process a single subject through filtering, downsampling and windowing."""
        df = self.load_subject_data(subject)
        
        signal_data = df[self.signal_cols].values
        print(f"  Applying bandpass filters to {subject}...")
        filtered_data = self.apply_bandpass_filter(signal_data, sampling_rate)
        
        downsampled = self.downsample_signals(filtered_data, sampling_rate)
        
        windows = self.segment_windows(downsampled)
        
        return windows
    
    def normalize_windows_per_sample(self, windows: np.ndarray) -> np.ndarray:
        """
        Apply per-window normalization to focus on signal shape rather than absolute values.
        Each window is normalized independently using its own statistics.
        
        Args:
            windows: Array of shape (n_windows, 3, 240)
            
        Returns:
            Normalized array of same shape
        """
        normalized = windows.copy()
        
        # Normalize each window independently
        for i in range(len(windows)):
            for channel in range(3):
                signal = windows[i, channel, :]  # Shape: (240,)
                
                # Check for NaN or inf values in the signal
                if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                    print(f"  Warning: NaN/Inf detected in window {i}, channel {channel}, setting to zero")
                    normalized[i, channel, :] = np.zeros_like(signal)
                    continue
                
                # Calculate per-window statistics
                mean = signal.mean()
                std = signal.std()
                
                # Check if mean or std are NaN/inf
                if np.isnan(mean) or np.isinf(mean) or np.isnan(std) or np.isinf(std):
                    print(f"  Warning: NaN/Inf in statistics for window {i}, channel {channel}, setting to zero")
                    normalized[i, channel, :] = np.zeros_like(signal)
                    continue
                
                # Avoid division by zero for constant signals
                if std > 1e-8:
                    normalized_signal = (signal - mean) / std
                    # Final check for NaN/inf after normalization
                    if np.any(np.isnan(normalized_signal)) or np.any(np.isinf(normalized_signal)):
                        print(f"  Warning: Normalization produced NaN/Inf for window {i}, channel {channel}, setting to zero")
                        normalized[i, channel, :] = np.zeros_like(signal)
                    else:
                        normalized[i, channel, :] = normalized_signal
                else:
                    # For constant signals, center around zero
                    normalized[i, channel, :] = signal - mean
        
        print(f"Applied per-window normalization to {len(windows)} windows")
        return normalized
    
    def process_all_subjects(self):
        print("Processing all subjects (filter → downsample → window → normalize)...")
        
        # Load sampling info
        with open(self.out_dir / 'sampling_info.json', 'r') as f:
            sampling_info = json.load(f)
        
        # Process each split
        splits = {
            'train': self.train_subjects,
            'eval': self.eval_subjects,
            'test': self.test_subjects
        }
        
        all_windows = {}
        
        for split_name, subjects in splits.items():
            print(f"Processing {split_name} split ({len(subjects)} subjects)...")
            
            split_windows = []
            for subject in subjects:
                sr = sampling_info[subject]['sampling_rate']
                windows = self.process_subject(subject, sr)
                split_windows.append(windows)
                print(f"  {subject}: {len(windows)} windows")
            
            # Concatenate all windows for this split
            if split_windows:
                all_windows[split_name] = np.concatenate(split_windows, axis=0)
            else:
                all_windows[split_name] = np.array([]).reshape(0, 3, int(self.window_len * self.target_rate))
        
        # Apply per-window normalization to all splits
        for split_name in splits.keys():
            if len(all_windows[split_name]) > 0:
                print(f"Normalizing {split_name} split...")
                all_windows[split_name] = self.normalize_windows_per_sample(all_windows[split_name])
        
        # Save processed data
        for split_name, windows in all_windows.items():
            save_path = self.out_dir / f'{split_name}_windows.npz'
            np.savez_compressed(save_path, windows=windows)
            print(f"Saved {split_name}: {windows.shape} to {save_path}")
        
        return all_windows
    
    def verify_and_visualize(self):
        """
        Verification and visualization of processed data.
        """
        print("Verification and visualization...")
        
        splits = ['train', 'eval', 'test'] 
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        plt.style.use('seaborn-v0_8-pastel')
        
        for i, split in enumerate(splits):
            data_path = self.out_dir / f'{split}_windows.npz'
            if not data_path.exists():
                continue
                
            data = np.load(data_path)
            windows = data['windows']
            
            if len(windows) > 0:
                # --- Plot the first window ---
                sample_window = windows[0]
                time_axis = np.arange(sample_window.shape[1]) / self.target_rate
                signal_names = ['bp', 'breath_upper', 'ppg_fing']
                
                for ch_idx, name in enumerate(signal_names):
                    axes[0, i].plot(time_axis, sample_window[ch_idx], label=f'{name}', alpha=0.7)
                
                axes[0, i].set_title(f'{split} Sample Window')
                axes[0, i].set_xlabel('Time (s)')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.2)
                
                # --- Plot value distribution ---
                flat_data = windows.flatten()
                finite_data = flat_data[np.isfinite(flat_data)]
                
                axes[1, i].hist(finite_data, bins=50, alpha=0.7, density=True)
                axes[1, i].set_title(f'{split} Value Distribution')
                axes[1, i].set_xlabel('Normalized value')
                axes[1, i].grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(self.out_dir / 'verification_plots.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Verification complete.")
    
    def create_dataloaders(self, batch_size: int = 128, num_workers: int = 4) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders for each split.
        
        Args:
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            
        Returns:
            Dictionary with DataLoaders for each split
        """
        dataloaders = {}
        
        for split in ['train', 'eval', 'test']:
            data_path = self.out_dir / f'{split}_windows.npz'
            if data_path.exists():
                dataset = MultimodalDataset(str(data_path))
                shuffle = (split == 'train')
                
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=True
                )
                dataloaders[split] = dataloader
                print(f"Created {split} DataLoader: {len(dataset)} samples")
        
        return dataloaders
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline."""
        print("Starting full preprocessing pipeline...")
        
        self.exploratory_check()
       
        self.process_all_subjects()

        self.verify_and_visualize()
        
        print(f"Data Pipeline complete! Processed data saved to {self.out_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Preprocess multimodal physiological data')
    parser.add_argument('--raw_dir', type=str, default='cleaned_data', 
                       help='Directory containing raw CSV files')
    parser.add_argument('--out_dir', type=str, default='processed_data',
                       help='Output directory for processed data')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration YAML file')
    
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        config_path=args.config,
    )
    
    preprocessor.run_full_pipeline()


if __name__ == '__main__':
    main()