#!/bin/bash

# MiMo-AE Training Startup 

echo "🚀 MiMo-AE Training Pipeline"
echo "=============================="

# Check if processed data exists
if [ ! -d "processed_data" ]; then
    echo "❌ Processed data directory not found!"
    echo "Please run data preprocessing first:"
    echo "python data_prepare.py"
    exit 1
fi

if [ ! -f "processed_data/train_windows.npz" ]; then
    echo "❌ Training data not found!"
    echo "Please run data preprocessing first:"
    echo "python data_prepare.py"
    exit 1
fi

echo "✓ Data found"

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "❌ config.yaml not found!"
    exit 1
fi

echo "✓ Configuration found"


# Check wandb configuration
echo ""
echo "🔍 Checking WANDB configuration..."


# Start training
echo ""
echo "🎯 Starting training..."
echo "=============================="

python train.py

echo ""
echo "🏁 Training completed!"
echo "Check outputs/ directory for results"