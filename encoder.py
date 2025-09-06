import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class RoPEPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)"""
    def __init__(self, d_model, max_seq_len=5000):
        super(RoPEPositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Create rotation matrix
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos_emb = emb.cos()
        sin_emb = emb.sin()
        
        return cos_emb, sin_emb

def rotate_half(x):
    """Rotate half the hidden dims of the input"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding"""
    # q, k shape: [batch_size, n_heads, seq_len, d_k]
    # cos, sin shape: [seq_len, d_k]
    # Need to add batch and head dimensions to cos, sin
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, d_k]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RelativePositionBias(nn.Module):
    """Relative Position Bias for attention"""
    def __init__(self, n_heads, max_distance=128):
        super(RelativePositionBias, self).__init__()
        self.n_heads = n_heads
        self.max_distance = max_distance
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_distance - 1), n_heads)
        )
        nn.init.normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, seq_len):
        # Create relative position indices
        coords = torch.arange(seq_len)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += self.max_distance - 1
        relative_coords = torch.clamp(relative_coords, 0, 2 * self.max_distance - 2)
        
        # Get relative position bias
        relative_position_bias = self.relative_position_bias_table[relative_coords]
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # (n_heads, seq_len, seq_len)
        
        return relative_position_bias.unsqueeze(0)  # (1, n_heads, seq_len, seq_len)

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""
    def __init__(self, d_model, n_heads, dropout=0.1, pos_encoding_type="rope"):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.pos_encoding_type = pos_encoding_type
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        if pos_encoding_type == "rope":
            self.rope = RoPEPositionalEmbedding(self.d_k)  # Use d_k not d_model
        elif pos_encoding_type == "relative_bias":
            self.relative_bias = RelativePositionBias(n_heads)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.w_q.weight)
        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_v.weight)
        nn.init.xavier_uniform_(self.w_o.weight)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len_q, d_model = query.size()
        seq_len_k = key.size(1)
        seq_len_v = value.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        
        # Apply positional encoding
        if self.pos_encoding_type == "rope":
            # For self-attention, all sequences have same length
            # For cross-attention, query and key/value may have different lengths
            cos_q, sin_q = self.rope(query)
            if seq_len_k == seq_len_q:
                # Same sequence length (self-attention case)
                cos_k, sin_k = cos_q, sin_q
            else:
                # Different sequence lengths (cross-attention case)
                cos_k, sin_k = self.rope(key)
            
            # Apply RoPE to Q with query positions, K with key positions
            cos_q = cos_q.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, d_k]
            sin_q = sin_q.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, d_k]
            cos_k = cos_k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_k, d_k]
            sin_k = sin_k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_k, d_k]
            
            Q = (Q * cos_q) + (rotate_half(Q) * sin_q)
            K = (K * cos_k) + (rotate_half(K) * sin_k)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias if using relative position encoding
        if self.pos_encoding_type == "relative_bias":
            relative_bias = self.relative_bias(seq_len_q).to(attention_scores.device)
            attention_scores += relative_bias
        
        # Apply mask
        if mask is not None:
            # mask shape: [batch_size, 1, seq_len] 
            # attention_scores shape: [batch_size, n_heads, seq_len, seq_len]
            # Need to expand mask to [batch_size, 1, 1, seq_len] for proper broadcasting
            if mask.dim() == 3:  # [batch_size, 1, seq_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, 1, seq_len]
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, d_model
        )
        
        # Final linear projection
        output = self.w_o(context)
        
        return output, attention_weights

class PositionwiseFeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    """Single Encoder Layer"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pos_encoding_type="rope"):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout, pos_encoding_type)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (fallback)"""
    def __init__(self, d_model, max_seq_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TransformerEncoder(nn.Module):
    """Complete Transformer Encoder"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_seq_len=5000, dropout=0.1, pos_encoding_type="rope"):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (only for non-RoPE methods)
        if pos_encoding_type not in ["rope", "relative_bias"]:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, pos_encoding_type)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
        
    def forward(self, src, src_mask=None):
        # Embedding and scaling
        x = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding (only for methods that need it)
        if self.pos_encoding_type not in ["rope", "relative_bias"]:
            x += self.pos_encoding(x)
            
        x = self.dropout(x)
        
        # Pass through encoder layers
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            attention_weights.append(attn_weights)
        
        x = self.norm(x)
        
        return x, attention_weights