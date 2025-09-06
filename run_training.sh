#!/bin/bash

# Quick start training script for Transformer

echo "Starting Transformer Training..."
echo "==============================="

# Check if data files exist
if [ ! -f "EUbookshop.fi" ] || [ ! -f "EUbookshop.en" ]; then
    echo "Error: Data files (EUbookshop.fi, EUbookshop.en) not found!"
    echo "Please make sure the dataset files are in the current directory."
    exit 1
fi

# Create necessary directories
mkdir -p checkpoints logs vocab

echo "Data files found. Starting training..."

# Train with RoPE positional encoding
echo "Training with RoPE positional encoding..."
python3 train.py --config config.yaml

echo "Training completed!"

# Test the model
echo "Testing the trained model..."
python3 test.py --config config.yaml

echo "Testing completed!"
echo "Check the 'logs' directory for training curves and results."