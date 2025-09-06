# Transformer from Scratch - Machine Translation

This project implements a complete Transformer architecture from scratch for Finnish to English machine translation, as part of the Advanced NLP course assignment.

## Features

### Core Implementation
- **Complete Transformer Architecture**: Encoder-decoder model implemented from scratch
- **Two Positional Encoding Strategies**:
  - Rotary Positional Embedding (RoPE)
  - Relative Position Bias (additive bias to attention scores)
- **Three Decoding Strategies**:
  - Greedy Decoding
  - Beam Search
  - Top-k Sampling
- **Training Pipeline**: Full training loop with teacher forcing, checkpointing, and logging

### Key Components
- Multi-head attention mechanism
- Position-wise feed-forward networks
- Layer normalization and residual connections
- Label smoothing loss
- Warmup learning rate scheduler

## Project Structure

```
transformer/
├── config.yaml          # Configuration file
├── encoder.py           # Transformer encoder implementation
├── decoder.py           # Transformer decoder and decoding strategies
├── train.py            # Training script
├── test.py             # Testing and evaluation script
├── utils.py            # Helper functions and utilities
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── EUbookshop.fi       # Finnish training data
├── EUbookshop.en       # English training data
├── checkpoints/        # Model checkpoints (created during training)
├── logs/              # Training logs and plots (created during training)
└── vocab/             # Vocabulary files (created during training)
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify data files**:
   - Ensure `EUbookshop.fi` and `EUbookshop.en` are in the project directory
   - These should contain parallel Finnish-English sentences

## Usage

### 1. Training

#### Basic Training
```bash
python train.py
```

#### Training with Custom Configuration
```bash
python train.py --config custom_config.yaml --device cuda
```

#### Key Configuration Options

Edit `config.yaml` to customize:

**Model Architecture**:
```yaml
model:
  d_model: 512          # Model dimension
  n_heads: 8            # Number of attention heads
  n_layers: 6           # Number of encoder/decoder layers
  d_ff: 2048           # Feed-forward dimension
```

**Positional Encoding**:
```yaml
positional_encoding:
  type: "rope"  # Options: "rope", "relative_bias"
```

**Training Parameters**:
```yaml
training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 50
  warmup_steps: 4000
```

### 2. Testing

#### Test All Decoding Strategies
```bash
python test.py
```

#### Test Specific Strategy
```bash
python test.py --strategy greedy
python test.py --strategy beam
python test.py --strategy top_k
```

#### Test with Custom Model
```bash
python test.py --model_path checkpoints/best_model.pt --strategy beam
```

#### Compare Positional Encodings
```bash
python test.py --compare_pos_encodings
```

### 3. Interactive Translation

You can add an interactive mode to test translations:

```python
from test import interactive_translation
interactive_translation('config.yaml', 'checkpoints/best_model.pt', 'cuda')
```

## Training Process

### 1. Data Preprocessing
- Loads parallel Finnish-English sentences
- Builds vocabularies for source and target languages
- Splits data into train/validation/test sets (80/10/10)
- Creates data loaders with proper padding and masking

### 2. Model Training
- Uses teacher forcing for training
- Implements label smoothing for better generalization
- Applies warmup learning rate scheduling
- Saves checkpoints every 5 epochs
- Tracks training and validation losses

### 3. Model Checkpoints
- `best_model.pt`: Model with lowest validation loss
- `final_model.pt`: Model after final epoch
- `checkpoint_epoch_X.pt`: Regular checkpoints

## Evaluation

### BLEU Score Calculation
The model is evaluated using BLEU scores on the test set for all decoding strategies.

### Expected Results
Different decoding strategies typically show:
- **Greedy**: Fastest but may miss optimal translations
- **Beam Search**: Better quality translations with controlled search
- **Top-k Sampling**: More diverse but potentially less consistent translations

## Implementation Details

### Positional Encodings

#### RoPE (Rotary Positional Embedding)
- Applied directly to query and key vectors in attention
- Preserves relative positional information
- More effective for longer sequences

#### Relative Position Bias
- Adds learnable bias terms to attention scores
- Based on relative distances between positions
- Allows model to learn position-dependent attention patterns

### Attention Mechanisms

#### Multi-Head Attention
- Splits attention into multiple heads
- Each head learns different types of relationships
- Combines outputs through linear projection

#### Masking
- **Padding Mask**: Ignores padded tokens
- **Causal Mask**: Prevents decoder from seeing future tokens
- **Combined Masking**: Applies both masks appropriately

### Decoding Strategies

#### Greedy Decoding
```python
# Always selects highest probability token
next_token = torch.argmax(logits, dim=-1)
```

#### Beam Search
```python
# Maintains top-B candidate sequences
# Selects sequence with highest overall score
```

#### Top-k Sampling
```python
# Samples from top-k most probable tokens
# Introduces randomness while maintaining quality
```

## Performance Tips

### For Better Training
1. **Increase batch size** if you have more GPU memory
2. **Adjust warmup steps** based on dataset size
3. **Use gradient accumulation** for larger effective batch sizes
4. **Monitor attention weights** to debug attention patterns

### For Better Inference
1. **Use beam search** for better translation quality
2. **Adjust beam size** based on quality vs. speed trade-off
3. **Cache encoder outputs** when decoding multiple sentences

## Troubleshooting

### Common Issues

#### Out of Memory
- Reduce batch size in `config.yaml`
- Reduce model dimensions (`d_model`, `d_ff`)
- Use gradient checkpointing (can be added)

#### Poor Translation Quality
- Increase training epochs
- Check data quality and preprocessing
- Verify vocabulary coverage
- Adjust learning rate and warmup steps

#### Slow Training
- Use mixed precision training (can be added)
- Increase batch size if memory allows
- Use data loading optimizations

### Model Loading Issues
```python
# If you get state_dict loading errors, check:
# 1. Model architecture matches saved model
# 2. Positional encoding type is correct
# 3. Vocabulary sizes match
```

## Extensions and Improvements

### Potential Enhancements
1. **Mixed Precision Training**: For faster training with fp16
2. **Gradient Checkpointing**: To save memory during training
3. **Multi-GPU Training**: For scaling to larger models
4. **Advanced Optimizations**: Like AdaFactor optimizer
5. **Better Tokenization**: Subword tokenization (BPE/SentencePiece)

### Advanced Features
1. **Attention Visualization**: Plot attention weights
2. **Learning Rate Scheduling**: More sophisticated schedules
3. **Model Averaging**: Ensemble multiple checkpoints
4. **Knowledge Distillation**: Train smaller models from larger ones

## Pre-trained Model

A pre-trained model will be available after training completion. The model will be saved in the `checkpoints/` directory.

**Model Download**: [Link to be added after training]

## Assignment Requirements Checklist

- ✅ **Core Components**: All transformer components implemented from scratch
- ✅ **Positional Encodings**: Both RoPE and relative position bias implemented
- ✅ **Training**: Full training loop with teacher forcing
- ✅ **Decoding Strategies**: Greedy, beam search, and top-k sampling implemented
- ✅ **Evaluation**: BLEU score calculation and comparative analysis
- ✅ **Code Organization**: Proper file structure with encoder.py, decoder.py, etc.
