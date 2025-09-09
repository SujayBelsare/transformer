import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder import MultiHeadAttention, PositionwiseFeedForward, SinusoidalPositionalEncoding, TransformerEncoder

class DecoderLayer(nn.Module):
    """Single Decoder Layer"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, pos_encoding_type="rope"):
        super(DecoderLayer, self).__init__()
        
        # Self-attention (masked)
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout, pos_encoding_type)
        
        # Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout, pos_encoding_type)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Self-attention with causal mask
        self_attn_output, self_attn_weights = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

class TransformerDecoder(nn.Module):
    """Complete Transformer Decoder"""
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, 
                 max_seq_len=5000, dropout=0.1, pos_encoding_type="rope"):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.pos_encoding_type = pos_encoding_type
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (only for non-RoPE methods)
        if pos_encoding_type not in ["rope", "relative_bias"]:
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, pos_encoding_type)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
        nn.init.normal_(self.output_projection.weight, mean=0, std=self.d_model**-0.5)
        
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embedding and scaling
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        
        # Add positional encoding (only for methods that need it)
        if self.pos_encoding_type not in ["rope", "relative_bias"]:
            x += self.pos_encoding(x)
            
        x = self.dropout(x)
        
        # Pass through decoder layers
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, src_mask, tgt_mask
            )
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)
        
        x = self.norm(x)
        
        # Output projection to vocabulary
        output = self.output_projection(x)
        
        return output, self_attention_weights, cross_attention_weights

class Transformer(nn.Module):
    """Complete Transformer Model for Machine Translation"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_seq_len=5000, dropout=0.1, 
                 pos_encoding_type="rope"):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, n_layers, d_ff, 
            max_seq_len, dropout, pos_encoding_type
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_heads, n_layers, d_ff, 
            max_seq_len, dropout, pos_encoding_type
        )
        
        # Share embedding weights between decoder embedding and output projection
        self.decoder.output_projection.weight = self.decoder.embedding.weight
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        encoder_output, encoder_attention = self.encoder(src, src_mask)
        
        # Decode target sequence
        decoder_output, decoder_self_attention, decoder_cross_attention = self.decoder(
            tgt, encoder_output, src_mask, tgt_mask
        )
        
        return decoder_output, {
            'encoder_attention': encoder_attention,
            'decoder_self_attention': decoder_self_attention,
            'decoder_cross_attention': decoder_cross_attention
        }
    
    def encode(self, src, src_mask=None):
        """Encode source sequence only"""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """Decode target sequence only"""
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

# Decoding Strategies
class GreedyDecoder:
    """Greedy decoding strategy"""
    def __init__(self, model, src_vocab, tgt_vocab, max_length=256, device='cpu'):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_length = max_length
        self.device = device
        
    def decode(self, src_sentence):
        """Decode a single source sentence"""
        self.model.eval()
        with torch.no_grad():
            # Encode source
            src_tokens = self.src_vocab.encode(src_sentence)
            src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(self.device)
            
            # Create source mask
            from utils import create_padding_mask
            src_mask = create_padding_mask(src_tensor, self.src_vocab.PAD_ID)
            
            # Encode
            encoder_output, _ = self.model.encode(src_tensor, src_mask)
            
            # Initialize target with SOS token
            tgt_tokens = [self.tgt_vocab.SOS_ID]
            
            for _ in range(self.max_length):
                tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(self.device)
                
                # Create target mask
                from utils import create_subsequent_mask
                tgt_mask = create_subsequent_mask(len(tgt_tokens)).to(self.device)
                
                # Decode
                output, _, _ = self.model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)
                
                # Get next token
                next_token = output[0, -1, :].argmax().item()
                tgt_tokens.append(next_token)
                
                # Check for EOS token
                if next_token == self.tgt_vocab.EOS_ID:
                    break
            
            # Convert to sentence
            result_tokens = tgt_tokens[1:]  # Remove SOS token
            if result_tokens[-1] == self.tgt_vocab.EOS_ID:
                result_tokens = result_tokens[:-1]  # Remove EOS token
            
            return self.tgt_vocab.decode(result_tokens)

class BeamSearchDecoder:
    """Beam search decoding strategy"""
    def __init__(self, model, src_vocab, tgt_vocab, beam_size=5, max_length=256, device='cpu'):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.beam_size = beam_size
        self.max_length = max_length
        self.device = device
        
    def decode(self, src_sentence):
        """Decode using beam search"""
        self.model.eval()
        with torch.no_grad():
            # Encode source
            src_tokens = self.src_vocab.encode(src_sentence)
            src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(self.device)
            
            # Create source mask
            from utils import create_padding_mask
            src_mask = create_padding_mask(src_tensor, self.src_vocab.PAD_ID)
            
            # Encode
            encoder_output, _ = self.model.encode(src_tensor, src_mask)
            
            # Initialize beams
            beams = [(0.0, [self.tgt_vocab.SOS_ID])]
            completed_beams = []
            
            for step in range(self.max_length):
                candidates = []
                
                for score, tokens in beams:
                    if tokens[-1] == self.tgt_vocab.EOS_ID:
                        completed_beams.append((score, tokens))
                        continue
                    
                    # Create target tensor
                    tgt_tensor = torch.LongTensor(tokens).unsqueeze(0).to(self.device)
                    
                    # Create target mask
                    from utils import create_subsequent_mask
                    tgt_mask = create_subsequent_mask(len(tokens)).to(self.device)
                    
                    # Decode
                    output, _, _ = self.model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)
                    
                    # Get top-k tokens
                    probs = F.softmax(output[0, -1, :], dim=-1)
                    top_k_probs, top_k_indices = torch.topk(probs, self.beam_size)
                    
                    for prob, idx in zip(top_k_probs, top_k_indices):
                        new_score = score - torch.log(prob).item()  # Negative log probability
                        new_tokens = tokens + [idx.item()]
                        candidates.append((new_score, new_tokens))
                
                # Select best candidates
                candidates.sort(key=lambda x: x[0])
                beams = candidates[:self.beam_size]
                
                # Check if all beams are completed
                if len(completed_beams) >= self.beam_size:
                    break
            
            # Add remaining beams to completed
            completed_beams.extend(beams)
            
            # Select best beam
            if completed_beams:
                best_beam = min(completed_beams, key=lambda x: x[0])
                result_tokens = best_beam[1][1:]  # Remove SOS token
                if result_tokens[-1] == self.tgt_vocab.EOS_ID:
                    result_tokens = result_tokens[:-1]  # Remove EOS token
                return self.tgt_vocab.decode(result_tokens)
            else:
                return ""

class TopKSamplingDecoder:
    """Top-K sampling decoding strategy"""
    def __init__(self, model, src_vocab, tgt_vocab, k=50, temperature=1.0, max_length=256, device='cpu'):
        self.model = model
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.k = k
        self.temperature = temperature
        self.max_length = max_length
        self.device = device
        
    def decode(self, src_sentence):
        """Decode using top-k sampling"""
        self.model.eval()
        with torch.no_grad():
            # Encode source
            src_tokens = self.src_vocab.encode(src_sentence)
            src_tensor = torch.LongTensor(src_tokens).unsqueeze(0).to(self.device)
            
            # Create source mask
            from utils import create_padding_mask
            src_mask = create_padding_mask(src_tensor, self.src_vocab.PAD_ID)
            
            # Encode
            encoder_output, _ = self.model.encode(src_tensor, src_mask)
            
            # Initialize target with SOS token
            tgt_tokens = [self.tgt_vocab.SOS_ID]
            
            for _ in range(self.max_length):
                tgt_tensor = torch.LongTensor(tgt_tokens).unsqueeze(0).to(self.device)
                
                # Create target mask
                from utils import create_subsequent_mask
                tgt_mask = create_subsequent_mask(len(tgt_tokens)).to(self.device)
                
                # Decode
                output, _, _ = self.model.decode(tgt_tensor, encoder_output, src_mask, tgt_mask)
                
                # Apply temperature scaling
                logits = output[0, -1, :] / self.temperature
                
                # Top-k filtering
                top_k_logits, top_k_indices = torch.topk(logits, min(self.k, logits.size(-1)))
                
                # Sample from top-k
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()
                next_token = top_k_indices[int(next_token_idx)].item()
                
                tgt_tokens.append(next_token)
                
                # Check for EOS token
                if next_token == self.tgt_vocab.EOS_ID:
                    break
            
            # Convert to sentence
            result_tokens = tgt_tokens[1:]  # Remove SOS token
            if result_tokens[-1] == self.tgt_vocab.EOS_ID:
                result_tokens = result_tokens[:-1]  # Remove EOS token
            
            return self.tgt_vocab.decode(result_tokens)