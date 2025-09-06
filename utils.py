import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import yaml
from collections import Counter
import pickle
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

class Vocabulary:
    """Vocabulary class for handling text tokenization"""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        self.special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        
        # Initialize special tokens
        for token in self.special_tokens:
            self.add_word(token)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
        self.word_count[word] += 1
    
    def build_vocab(self, sentences, max_vocab_size=32000):
        """Build vocabulary from sentences"""
        for sentence in sentences:
            for word in sentence.split():
                self.add_word(word)
        
        # Keep only most frequent words
        if len(self.word2idx) > max_vocab_size:
            # Sort by frequency, keep special tokens
            word_freq = [(word, count) for word, count in self.word_count.items() 
                        if word not in self.special_tokens]
            word_freq.sort(key=lambda x: x[1], reverse=True)
            
            # Reset vocabulary
            new_word2idx = {}
            new_idx2word = {}
            
            # Add special tokens first
            for i, token in enumerate(self.special_tokens):
                new_word2idx[token] = i
                new_idx2word[i] = token
            
            # Add most frequent words
            for word, _ in word_freq[:max_vocab_size - len(self.special_tokens)]:
                new_word2idx[word] = len(new_word2idx)
                new_idx2word[len(new_idx2word)] = word
            
            self.word2idx = new_word2idx
            self.idx2word = new_idx2word
    
    def encode(self, sentence):
        """Convert sentence to list of indices"""
        return [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) 
                for word in sentence.split()]
    
    def decode(self, indices):
        """Convert list of indices to sentence"""
        return ' '.join([self.idx2word.get(idx, self.UNK_TOKEN) for idx in indices])
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        """Save vocabulary to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)

class DataLoader:
    """Custom DataLoader for translation data"""
    def __init__(self, src_data, tgt_data, src_vocab, tgt_vocab, batch_size, max_len=256):
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.batch_size = batch_size
        self.max_len = max_len
        
        # Encode and filter data
        self.encoded_data = []
        for src_sent, tgt_sent in zip(src_data, tgt_data):
            src_encoded = src_vocab.encode(src_sent)
            tgt_encoded = tgt_vocab.encode(tgt_sent)
            
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
            src_batch = self.pad_batch(src_batch, self.src_vocab.word2idx[self.src_vocab.PAD_TOKEN])
            tgt_batch = self.pad_batch(tgt_batch, self.tgt_vocab.word2idx[self.tgt_vocab.PAD_TOKEN])
            
            # Add SOS and EOS tokens to target
            tgt_input = self.add_sos_eos(tgt_batch, self.tgt_vocab, add_eos=False)
            tgt_output = self.add_sos_eos(tgt_batch, self.tgt_vocab, add_sos=False)
            
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
    
    def add_sos_eos(self, batch, vocab, add_sos=True, add_eos=True):
        """Add SOS/EOS tokens to batch"""
        result = []
        for seq in batch:
            new_seq = seq.copy()
            if add_sos:
                new_seq = [vocab.word2idx[vocab.SOS_TOKEN]] + new_seq
            if add_eos:
                new_seq = new_seq + [vocab.word2idx[vocab.EOS_TOKEN]]
            result.append(new_seq)
        return result
    
    def __len__(self):
        return self.n_batches

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(src_file, tgt_file, train_split=0.8, val_split=0.1):
    """Load and split parallel data"""
    with open(src_file, 'r', encoding='utf-8') as f:
        src_lines = [line.strip().lower() for line in f.readlines()]
    
    with open(tgt_file, 'r', encoding='utf-8') as f:
        tgt_lines = [line.strip().lower() for line in f.readlines()]
    
    # Ensure same length
    min_len = min(len(src_lines), len(tgt_lines))
    src_lines = src_lines[:min_len]
    tgt_lines = tgt_lines[:min_len]
    
    # Shuffle data
    combined = list(zip(src_lines, tgt_lines))
    np.random.shuffle(combined)
    src_lines, tgt_lines = zip(*combined)
    
    # Split data
    n_train = int(len(src_lines) * train_split)
    n_val = int(len(src_lines) * val_split)
    
    train_src = src_lines[:n_train]
    train_tgt = tgt_lines[:n_train]
    
    val_src = src_lines[n_train:n_train + n_val]
    val_tgt = tgt_lines[n_train:n_train + n_val]
    
    test_src = src_lines[n_train + n_val:]
    test_tgt = tgt_lines[n_train + n_val:]
    
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
        
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(self.step_num ** (-0.5), 
                                          self.step_num * self.warmup_steps ** (-1.5))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

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
    """Simple BLEU score computation"""
    # This is a simplified version - you might want to use sacrebleu for better results
    from collections import Counter
    import math
    
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    
    def bleu_score(pred_tokens, ref_tokens, max_n=4):
        scores = []
        for n in range(1, max_n + 1):
            pred_ngrams = Counter(get_ngrams(pred_tokens, n))
            ref_ngrams = Counter(get_ngrams(ref_tokens, n))
            
            overlap = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            if total == 0:
                scores.append(0)
            else:
                scores.append(overlap / total)
        
        # Brevity penalty
        bp = min(1, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if len(pred_tokens) > 0 else 0
        
        # Geometric mean
        if all(score > 0 for score in scores):
            bleu = bp * math.exp(sum(math.log(score) for score in scores) / len(scores))
        else:
            bleu = 0
        
        return bleu
    
    total_bleu = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        total_bleu += bleu_score(pred_tokens, ref_tokens)
    
    return total_bleu / len(predictions)