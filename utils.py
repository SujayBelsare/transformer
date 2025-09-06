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

class Vocabulary:
    """Vocabulary class using SentencePiece tokenizer"""
    def __init__(self, model_path=None):
        # Import here to avoid requiring sentencepiece if not using it
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            if model_path and os.path.exists(model_path):
                self.sp.load(model_path)
                self.model_loaded = True
            else:
                self.model_loaded = False
        except ImportError:
            print("Warning: SentencePiece not available. Using simple vocabulary.")
            self.sp = None
            self.model_loaded = False
            
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SOS_TOKEN = '<s>'
        self.EOS_TOKEN = '</s>'
        
        # Special token IDs for SentencePiece
        self.PAD_ID = 0
        self.UNK_ID = 1  
        self.SOS_ID = 2
        self.EOS_ID = 3
        
        if not self.model_loaded:
            # Fallback vocabulary for compatibility
            self.word2idx = {
                self.PAD_TOKEN: 0,
                self.UNK_TOKEN: 1,
                self.SOS_TOKEN: 2,
                self.EOS_TOKEN: 3
            }
            self.idx2word = {v: k for k, v in self.word2idx.items()}
    
    def encode(self, text):
        """Encode text to token IDs"""
        if self.model_loaded and self.sp:
            # Use SentencePiece encoding
            return self.sp.encode(text, add_bos=True, add_eos=True)
        else:
            # Fallback: simple word-based encoding
            tokens = text.lower().split()
            encoded = [self.word2idx.get(self.SOS_TOKEN, 2)]
            for token in tokens:
                encoded.append(self.word2idx.get(token, self.word2idx.get(self.UNK_TOKEN, 1)))
            encoded.append(self.word2idx.get(self.EOS_TOKEN, 3))
            return encoded
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        if self.model_loaded and self.sp:
            # Use SentencePiece decoding
            return self.sp.decode(token_ids)
        else:
            # Fallback: simple word-based decoding
            tokens = []
            for idx in token_ids:
                if idx in [self.PAD_ID, self.SOS_ID, self.EOS_ID]:
                    continue
                tokens.append(self.idx2word.get(idx, self.UNK_TOKEN))
            return ' '.join(tokens)
    
    def __len__(self):
        if self.model_loaded and self.sp:
            return self.sp.get_piece_size()
        else:
            return len(self.word2idx)
    
    @classmethod
    def load(cls, path):
        """Load vocabulary from file"""
        if path.endswith('.model'):
            # SentencePiece model
            return cls(path)
        else:
            # Pickle file (for compatibility)
            with open(path, 'rb') as f:
                vocab_data = pickle.load(f)
            vocab = cls()
            vocab.word2idx = vocab_data.get('word2idx', vocab.word2idx)
            vocab.idx2word = vocab_data.get('idx2word', vocab.idx2word)
            return vocab
    
    def save(self, path):
        """Save vocabulary to file"""
        if not self.model_loaded:
            # Save as pickle for simple vocabulary
            vocab_data = {
                'word2idx': self.word2idx,
                'idx2word': self.idx2word
            }
            with open(path, 'wb') as f:
                pickle.dump(vocab_data, f)
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