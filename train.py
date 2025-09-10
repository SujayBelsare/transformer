import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from vocabulary import Vocabulary
from encoder import TransformerEncoder
from decoder import Transformer
from preprocessor import Preprocessor
from utils import (
    load_config, load_data_splits, DataLoader,
    create_padding_mask, create_subsequent_mask,
    WarmupScheduler, save_checkpoint, load_checkpoint,
    plot_training_curves, setup_misc_config
)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, config):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    # Get gradient accumulation steps from config
    gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    # Initialize gradients
    optimizer.zero_grad()
    
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
        output, attention_weights = model(src, tgt_input, src_mask, tgt_mask)
        
        # Reshape for loss calculation
        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        
        # Calculate loss and normalize by accumulation steps
        loss = criterion(output, tgt_output) / gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every gradient_accumulation_steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping
            if config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Update learning rate
            lr = scheduler.step()
        
        # Update progress (multiply loss back to get actual loss for logging)
        total_loss += loss.item() * gradient_accumulation_steps
        n_batches += 1
        avg_loss = total_loss / n_batches
        
        # Get current learning rate for display
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        
        progress_bar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'LR': f'{current_lr:.2e}',
            'Step': f'{batch_idx + 1}/{len(dataloader)}'
        })
    
    # Handle any remaining gradients at the end of epoch
    if len(dataloader) % gradient_accumulation_steps != 0:
        if config['training']['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clip'])
        optimizer.step()
        optimizer.zero_grad()
    
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
            loss = criterion(output, tgt_output)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / n_batches

def main():
    parser = argparse.ArgumentParser(description='Train Transformer for Machine Translation')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup and validate misc configuration
    misc_config = setup_misc_config(config)
    
    # Handle device configuration from misc settings
    device = misc_config.get('device', 'auto')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility from misc settings
    seed = misc_config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
    
    # Check if data splits exist, if not create them
    train_json_path = os.path.join(args.data_dir, 'train.json')
    if not os.path.exists(train_json_path):
        print("Data splits not found. Creating splits from raw data...")
        
        # Check for raw data files
        fi_path = os.path.join(args.data_dir, f"{config['data']['fi']}")
        en_path = os.path.join(args.data_dir, f"{config['data']['en']}")
        print("Using Raw files:")
        print(f"  Finnish: {fi_path}")
        print(f"  English: {en_path}")

        if not os.path.exists(fi_path) or not os.path.exists(en_path):
            raise FileNotFoundError(f"Raw data files not found. Please ensure {fi_path} and {en_path} exist.")
        
        # Create preprocessor and generate splits
        preprocessor = Preprocessor(
            fi_path=fi_path,
            en_path=en_path,
            train_ratio=0.8,
            val_ratio=0.1,
            seed=seed,
            out_dir=args.data_dir
        )
        
        preprocessor.create_splits_and_save()
        print("Data splits created successfully!")
    
    # Load preprocessed data splits
    (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt) = load_data_splits(args.data_dir)
    print(f"Training samples: {len(train_src)}")
    print(f"Validation samples: {len(val_src)}")
    print(f"Test samples: {len(test_src)}")

    # Loading or training vocabulary    
    vocab_model_path = os.path.join('vocab', 'sentencepiece_model.model')
    if os.path.exists(vocab_model_path):
        print("Loading existing vocabulary model...")
        vocab = Vocabulary()
        vocab.sp.Load(vocab_model_path)
        vocab.model_path = vocab_model_path
    else:
        print("Vocabulary model not found. Training new vocabulary...")
        
        # Collect all sentences for vocabulary training
        all_sentences = []
        for src_sent, tgt_sent in zip(train_src + val_src + test_src, train_tgt + val_tgt + test_tgt):
            all_sentences.append(src_sent)
            all_sentences.append(tgt_sent)
        
        print(f"Training vocabulary on {len(all_sentences)} sentences...")
        
        # Create vocabulary directory
        os.makedirs('vocab', exist_ok=True)
        
        # Train vocabulary
        vocab = Vocabulary()
        vocab.train(
            sentences=all_sentences,
            vocab_size=config['model'].get('vocab_size', 8000),
            model_prefix=os.path.join('vocab', 'sentencepiece_model'),
            character_coverage=0.9995
        )
        
        print("Vocabulary training completed!")
    
    print(f"Vocabulary size: {vocab.sp.GetPieceSize()}")

    # Ensure vocabulary directory exists (vocabulary is already saved during training)
    os.makedirs(config['paths']['vocab_path'], exist_ok=True)

    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_src, train_tgt, vocab, 
        config['training']['batch_size'], config['data']['max_len']
    )
    
    val_loader = DataLoader(
        val_src, val_tgt, vocab, 
        config['training']['batch_size'], config['data']['max_len']
    )
    
    # Initialize model
    print("Initializing model...")
    vocab_size = len(vocab)
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        pos_encoding_type=config['positional_encoding']['type']
    )
    
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize loss function - using Cross Entropy Loss
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.PAD_ID)
    
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
    
    # Initialize training variables
    start_epoch = 0
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Resume from checkpoint if specified in misc settings
    resume_training = misc_config.get('resume_training', False)
    if resume_training:
        resume_model_name = misc_config.get('resume_model_name', 'best_model.pt')
        resume_path = os.path.join(config['paths']['model_save_path'], resume_model_name)
        
        if os.path.exists(resume_path):
            print(f"Resuming training from checkpoint: {resume_path}")
            start_epoch, last_loss, train_losses, val_losses, best_val_loss = load_checkpoint(
                model, optimizer, scheduler, resume_path
            )
            start_epoch += 1  # Start from next epoch
            print(f"Resumed from epoch {start_epoch}, last loss: {last_loss:.4f}, best val loss: {best_val_loss:.4f}")
        else:
            print(f"Warning: Resume requested but checkpoint file {resume_path} not found. Starting from scratch.")
    
    # Training loop
    print("Starting training...")
    
    os.makedirs(config['paths']['model_save_path'], exist_ok=True)
    os.makedirs(config['paths']['log_path'], exist_ok=True)
    
    for epoch in range(start_epoch, config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, config)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(
                config['paths']['model_save_path'], 
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, checkpoint_path, 
                          config, train_losses, val_losses, best_val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                config['paths']['model_save_path'], 
                'best_model.pt'
            )
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_model_path, 
                          config, train_losses, val_losses, best_val_loss)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_save_path'], 'final_model.pt')
    save_checkpoint(model, optimizer, scheduler, config['training']['num_epochs'] - 1, 
                   train_losses[-1], final_model_path, config, train_losses, val_losses, best_val_loss)
    
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