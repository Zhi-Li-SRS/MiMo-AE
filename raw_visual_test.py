import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def plot_normalized_signals(csv_file, window_duration=8):
    """Plot normalized physiological signals from a cleaned data file for specified duration."""
    df = pd.read_csv(csv_file)
    
    # Filter data to only include the first 8 seconds (or specified window)
    df_window = df[df['time'] <= window_duration].copy()
    
    signals = ['bp', 'breath_upper', 'ppg_fing']
    
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-pastel')
    colors = ['red', 'blue', 'green']
    
    for i, signal in enumerate(signals):
        scaler = MinMaxScaler()
        normalized_signal = scaler.fit_transform(df_window[signal].values.reshape(-1, 1)).flatten()
        
        plt.plot(df_window['time'], normalized_signal, 
                color=colors[i], label=f'{signal}', 
                alpha=0.8, linewidth=1.5)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Normalized Signal Value (within 8s window)')
    plt.title(f'Normalized Physiological Signals - First {window_duration} seconds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, window_duration)  # Set x-axis limit to 8 seconds
    plt.tight_layout()
    plt.savefig(f'raw_visual_test_8s.png', dpi=300)
    
    print(f"\nData summary for {csv_file} (first {window_duration} seconds):")
    print(f"Window duration: {df_window['time'].max():.2f} seconds")
    print(f"Samples in window: {len(df_window)}")
    print(f"Sampling rate: ~{len(df_window) / df_window['time'].max():.1f} Hz")
    print(f"Original total duration: {df['time'].max():.2f} seconds")

def main():
    csv_file = 'cleaned_data/SF_0034_biopac_v1.0.0.csv'
    plot_normalized_signals(csv_file)

if __name__ == "__main__":
    main()