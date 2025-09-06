import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from encoder import TransformerEncoder
from decoder import Transformer
from utils import (
    load_config, load_data, Vocabulary, DataLoader, 
    create_padding_mask, create_subsequent_mask,
    LabelSmoothingLoss, WarmupScheduler, save_checkpoint,
    plot_training_curves
)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(progress_bar):
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        
        # Create masks
        src_mask = create_padding_mask(src, 0)  # Assuming PAD token is 0
        tgt_mask = create_subsequent_mask(tgt_input.size(1)).to(device)
        tgt_padding_mask = create_padding_mask(tgt_input, 0)
        tgt_mask = tgt_mask & tgt_padding_mask
        
        # Forward pass
        optimizer.zero_grad()
        
        output, attention_weights = model(src, tgt_input, src_mask, tgt_mask)
        
        # Reshape for loss calculation
        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        
        # Calculate loss
        loss = criterion(F.log_softmax(output, dim=-1), tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        
        optimizer.step()
        
        # Update learning rate
        lr = scheduler.step()
        
        # Update progress
        total_loss += loss.item()
        n_batches += 1
        avg_loss = total_loss / n_batches
        
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'LR': f'{lr:.2e}'
        })
    
    return total_loss / n_batches

def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in dataloader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            # Create masks
            src_mask = create_padding_mask(src, 0)
            tgt_mask = create_subsequent_mask(tgt_input.size(1)).to(device)
            tgt_padding_mask = create_padding_mask(tgt_input, 0)
            tgt_mask = tgt_mask & tgt_padding_mask
            
            # Forward pass
            output, _ = model(src, tgt_input, src_mask, tgt_mask)
            
            # Reshape for loss calculation
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(F.log_softmax(output, dim=-1), tgt_output)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches

def main():
    parser = argparse.ArgumentParser(description='Train Transformer for Machine Translation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    print(f"Using device: {args.device}")
    print(f"Configuration: {config}")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and split data
    print("Loading data...")
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = load_data(
        config['data']['train_src'], 
        config['data']['train_tgt'],
        config['data']['train_split'],
        config['data']['val_split']
    )
    
    print(f"Training samples: {len(train_src)}")
    print(f"Validation samples: {len(val_src)}")
    print(f"Test samples: {len(test_src)}")
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = Vocabulary()
    tgt_vocab = Vocabulary()
    
    src_vocab.build_vocab(train_src, config['model']['vocab_size_src'])
    tgt_vocab.build_vocab(train_tgt, config['model']['vocab_size_tgt'])
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Save vocabularies
    os.makedirs(config['paths']['vocab_path'], exist_ok=True)
    src_vocab.save(os.path.join(config['paths']['vocab_path'], 'src_vocab.pkl'))
    tgt_vocab.save(os.path.join(config['paths']['vocab_path'], 'tgt_vocab.pkl'))
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_src, train_tgt, src_vocab, tgt_vocab, 
        config['training']['batch_size'], config['data']['max_len']
    )
    
    val_loader = DataLoader(
        val_src, val_tgt, src_vocab, tgt_vocab, 
        config['training']['batch_size'], config['data']['max_len']
    )
    
    # Initialize model
    print("Initializing model...")
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        pos_encoding_type=config['positional_encoding']['type']
    )
    
    model = model.to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize loss function
    criterion = LabelSmoothingLoss(
        size=len(tgt_vocab),
        padding_idx=tgt_vocab.word2idx[tgt_vocab.PAD_TOKEN],
        smoothing=config['training']['label_smoothing']
    )
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['training']['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Initialize scheduler
    scheduler = WarmupScheduler(
        optimizer, 
        config['model']['d_model'], 
        config['training']['warmup_steps']
    )
    
    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    os.makedirs(config['paths']['model_save_path'], exist_ok=True)
    os.makedirs(config['paths']['log_path'], exist_ok=True)
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, args.device, config)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, args.device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['paths']['model_save_path'], 
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                config['paths']['model_save_path'], 
                'best_model.pt'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_save_path'], 'final_model.pt')
    save_checkpoint(model, optimizer, config['training']['num_epochs'], train_losses[-1], final_model_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config['paths']['log_path'], 'training_curves.png'))
    plt.close()
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Final validation loss: {val_losses[-1]:.4f}")

if __name__ == '__main__':
    main()