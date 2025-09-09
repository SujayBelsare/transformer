import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
from collections import Counter
import pickle
import os
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import sacrebleu

import sentencepiece as sp
import pickle

class DataLoader:
    """Custom DataLoader for translation data with JSON format support"""
    def __init__(self, src_data, tgt_data, vocab, batch_size, max_len=256):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.vocab = vocab  # Using shared vocabulary
        self.batch_size = batch_size
        self.max_len = max_len
        
        # Encode and filter data
        self.encoded_data = []
        for src_sent, tgt_sent in zip(src_data, tgt_data):
            src_encoded = vocab.encode(src_sent)
            tgt_encoded = vocab.encode(tgt_sent)
            
            # Filter by length
            if len(src_encoded) <= max_len and len(tgt_encoded) <= max_len:
                self.encoded_data.append((src_encoded, tgt_encoded))
        
        self.n_batches = len(self.encoded_data) // batch_size
    
    def __iter__(self):
        # Shuffle data
        np.random.shuffle(self.encoded_data)
        
        for i in range(self.n_batches):
            batch_data = self.encoded_data[i * self.batch_size:(i + 1) * self.batch_size]
            
            # Separate source and target
            src_batch = [item[0] for item in batch_data]
            tgt_batch = [item[1] for item in batch_data]
            
            # Pad sequences
            src_batch = self.pad_batch(src_batch, self.vocab.PAD_ID)
            tgt_batch = self.pad_batch(tgt_batch, self.vocab.PAD_ID)
            
            # Create input and output for decoder
            # tgt_input: <s> + tokens (without </s>)
            # tgt_output: tokens + </s> (without <s>)
            tgt_input = []
            tgt_output = []
            
            for seq in tgt_batch:
                # Remove existing SOS/EOS and add properly
                if seq[0] == self.vocab.SOS_ID:
                    seq = seq[1:]  # Remove SOS
                if seq[-1] == self.vocab.EOS_ID:
                    seq = seq[:-1]  # Remove EOS
                
                # Create input (SOS + tokens) and output (tokens + EOS)
                input_seq = [self.vocab.SOS_ID] + seq
                output_seq = seq + [self.vocab.EOS_ID]
                
                tgt_input.append(input_seq)
                tgt_output.append(output_seq)
            
            # Pad the modified sequences
            tgt_input = self.pad_batch(tgt_input, self.vocab.PAD_ID)
            tgt_output = self.pad_batch(tgt_output, self.vocab.PAD_ID)
            
            yield (torch.LongTensor(src_batch), 
                   torch.LongTensor(tgt_input), 
                   torch.LongTensor(tgt_output))
    
    def pad_batch(self, batch, pad_token):
        """Pad batch to same length"""
        max_len = max(len(seq) for seq in batch)
        padded_batch = []
        for seq in batch:
            padded_seq = seq + [pad_token] * (max_len - len(seq))
            padded_batch.append(padded_seq)
        return padded_batch
    
    def __len__(self):
        return self.n_batches

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_json_data(json_file):
    """Load data from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    src_sentences = []
    tgt_sentences = []
    
    for item in data:
        src_sentences.append(item['fi'])
        tgt_sentences.append(item['en'])
    
    return src_sentences, tgt_sentences

def load_data_splits(data_dir):
    """Load preprocessed train/val/test splits from JSON files"""
    train_src, train_tgt = load_json_data(os.path.join(data_dir, 'train.json'))
    val_src, val_tgt = load_json_data(os.path.join(data_dir, 'val.json'))
    test_src, test_tgt = load_json_data(os.path.join(data_dir, 'test.json'))
    
    return (train_src, train_tgt), (val_src, val_tgt), (test_src, test_tgt)

def create_padding_mask(seq, pad_token=0):
    """Create padding mask for attention"""
    return (seq != pad_token).unsqueeze(-2)

def create_subsequent_mask(size):
    """Create mask for decoder self-attention (causal mask)"""
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.uint8)
    return mask == 0

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss"""
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

class WarmupScheduler:
    """Learning rate scheduler with warmup"""
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        self.last_lr = 0
        
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), 
                                          self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.last_lr = lr
        return lr
    
    def get_last_lr(self):
        """Get the last learning rate"""
        return [self.last_lr]

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)

def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def plot_training_curves(losses, path):
    """Plot training loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(path)
    plt.close()

def compute_bleu(predictions, references):
    """Compute BLEU score using sacrebleu"""
    # Ensure predictions and references are lists of strings
    if isinstance(predictions, list) and isinstance(references, list):
        # sacrebleu expects references as a list of lists (multiple references per prediction)
        # For single reference per prediction, we wrap each reference in a list
        refs = [[ref] for ref in references]
        
        # Calculate BLEU score using sacrebleu
        bleu = sacrebleu.corpus_bleu(predictions, refs)
        return bleu.score
    else:
        raise ValueError("Predictions and references must be lists of strings")