import torch
import argparse
import os
import numpy as np
from tqdm import tqdm

from decoder import Transformer, GreedyDecoder, BeamSearchDecoder, TopKSamplingDecoder
from utils import load_config, load_data_splits, Vocabulary, compute_bleu

def load_model_and_vocabs(config, model_path, device):
    """Load trained model and vocabularies"""
    
    # Load vocabulary
    vocab_model_path = os.path.join('data_transformations', 'finnish_english.model')
    
    if os.path.exists(vocab_model_path):
        vocab = Vocabulary(vocab_model_path)
    else:
        # Load from pickle file as fallback
        vocab_path = os.path.join(config['paths']['vocab_path'], 'vocab.pkl')
        vocab = Vocabulary.load(vocab_path)
    
    # Initialize model
    model = Transformer(
        src_vocab_size=len(vocab),
        tgt_vocab_size=len(vocab),
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model']['dropout'],
        pos_encoding_type=config['positional_encoding']['type']
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['loss']:.4f}")
    
    return model, vocab, vocab

def test_model(model, src_vocab, tgt_vocab, test_data, config, device):
    """Test model with different decoding strategies"""
    
    test_src, test_tgt = test_data
    
    results = {}
    
    # Test different decoding strategies
    decoding_strategies = ['greedy', 'beam', 'top_k']
    
    for strategy in decoding_strategies:
        print(f"\n{'='*50}")
        print(f"Testing with {strategy.upper()} decoding")
        print(f"{'='*50}")
        
        # Initialize decoder
        if strategy == 'greedy':
            decoder = GreedyDecoder(
                model, src_vocab, tgt_vocab, 
                config['decoding']['max_length'], device
            )
        elif strategy == 'beam':
            decoder = BeamSearchDecoder(
                model, src_vocab, tgt_vocab, 
                config['decoding']['beam_size'],
                config['decoding']['max_length'], device
            )
        elif strategy == 'top_k':
            decoder = TopKSamplingDecoder(
                model, src_vocab, tgt_vocab,
                config['decoding']['k'],
                config['decoding']['temperature'],
                config['decoding']['max_length'], device
            )
        
        # Decode test sentences
        predictions = []
        references = []
        
        # Test on subset for faster evaluation (you can change this)
        test_subset = min(1000, len(test_src))
        
        for i in tqdm(range(test_subset), desc=f"Decoding with {strategy}"):
            src_sentence = test_src[i]
            tgt_sentence = test_tgt[i]
            
            # Decode
            prediction = decoder.decode(src_sentence)
            
            predictions.append(prediction)
            references.append(tgt_sentence)
            
            # Show some examples
            if i < 5:
                print(f"\nExample {i + 1}:")
                print(f"Source:     {src_sentence}")
                print(f"Reference:  {tgt_sentence}")
                print(f"Prediction: {prediction}")
        
        # Calculate BLEU score
        bleu_score = compute_bleu(predictions, references)
        
        results[strategy] = {
            'bleu': bleu_score,
            'predictions': predictions[:10],  # Store first 10 for analysis
            'references': references[:10]
        }
        
        print(f"\nBLEU Score ({strategy}): {bleu_score:.4f}")
    
    return results
        
        # Compute BLEU score
        bleu_score = compute_bleu(predictions, references)
        print(f"\nBLEU Score ({strategy}): {bleu_score:.4f}")
        
        results[strategy] = {
            'bleu_score': bleu_score,
            'predictions': predictions,
            'references': references
        }
    
    return results

def compare_positional_encodings(config_path, device):
    """Compare different positional encoding methods"""
    
    config = load_config(config_path)
    
    # Load test data
    _, _, (test_src, test_tgt) = load_data(
        config['data']['train_src'], 
        config['data']['train_tgt'],
        config['data']['train_split'],
        config['data']['val_split']
    )
    
    pos_encoding_types = ['rope', 'relative_bias']
    results_comparison = {}
    
    for pos_type in pos_encoding_types:
        print(f"\n{'='*60}")
        print(f"Testing with {pos_type.upper()} positional encoding")
        print(f"{'='*60}")
        
        # Update config for current positional encoding
        config['positional_encoding']['type'] = pos_type
        
        # Look for model file with this encoding type
        model_path = os.path.join(config['paths']['model_save_path'], f'best_model_{pos_type}.pt')
        if not os.path.exists(model_path):
            model_path = os.path.join(config['paths']['model_save_path'], 'best_model.pt')
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        
        try:
            model, src_vocab, tgt_vocab = load_model_and_vocabs(config, model_path, device)
            results = test_model(model, src_vocab, tgt_vocab, (test_src, test_tgt), config, device)
            results_comparison[pos_type] = results
        except Exception as e:
            print(f"Error testing {pos_type}: {e}")
            continue
    
    return results_comparison

def analyze_results(results):
    """Analyze and compare results"""
    print(f"\n{'='*80}")
    print("RESULTS ANALYSIS")
    print(f"{'='*80}")
    
    # Create results table
    print("\nBLEU Score Comparison:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'BLEU Score':<15}")
    print("-" * 60)
    
    for strategy, result in results.items():
        bleu = result['bleu_score']
        print(f"{strategy:<15} {bleu:<15.4f}")
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x]['bleu_score'])
    print(f"\nBest decoding strategy: {best_strategy.upper()}")
    print(f"Best BLEU score: {results[best_strategy]['bleu_score']:.4f}")
    
    # Show some example comparisons
    print(f"\nExample Translation Comparisons:")
    print("-" * 80)
    
    for i in range(min(3, len(results['greedy']['references']))):
        print(f"\nExample {i + 1}:")
        print(f"Reference:  {results['greedy']['references'][i]}")
        
        for strategy in results.keys():
            print(f"{strategy.capitalize():>10}: {results[strategy]['predictions'][i]}")

def main():
    parser = argparse.ArgumentParser(description='Test Transformer Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--model_path', type=str, help='Path to trained model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--compare_pos_encodings', action='store_true', 
                       help='Compare different positional encodings')
    parser.add_argument('--strategy', type=str, choices=['greedy', 'beam', 'top_k'], 
                       help='Specific decoding strategy to test')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    print(f"Using device: {args.device}")
    
    if args.compare_pos_encodings:
        # Compare positional encodings
        results_comparison = compare_positional_encodings(args.config, args.device)
        
        # Analyze comparison results
        for pos_type, results in results_comparison.items():
            print(f"\n{'='*50}")
            print(f"Results for {pos_type.upper()}:")
            print(f"{'='*50}")
            analyze_results(results)
        
        return
    
    # Load test data
    _, _, (test_src, test_tgt) = load_data_splits(args.data_dir)
    
    print(f"Test samples: {len(test_src)}")
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(config['paths']['model_save_path'], 'best_model.pt')
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Load model and vocabularies
    model, src_vocab, tgt_vocab = load_model_and_vocabs(config, model_path, args.device)
    
    if args.strategy:
        # Test specific strategy
        print(f"Testing with {args.strategy} decoding strategy")
        
        # Update config for specific strategy
        config['decoding']['strategy'] = args.strategy
        
        # Test model
        results = test_model(model, src_vocab, src_vocab, (test_src, test_tgt), config, args.device)
        
        # Analyze results
        analyze_results(results)
    else:
        # Test all strategies
        results = test_model(model, src_vocab, src_vocab, (test_src, test_tgt), config, args.device)
        
        # Analyze results
        analyze_results(results)
    
    print("\nTesting completed!")

def interactive_translation(config_path, model_path, device):
    """Interactive translation interface"""
    
    config = load_config(config_path)
    
    # Load model and vocabularies
    model, vocab, _ = load_model_and_vocabs(config, model_path, device)
    
    # Initialize decoder (default to greedy)
    decoder = GreedyDecoder(model, vocab, vocab, device=device)
    
    print("Interactive Translation Interface")
    print("Enter 'quit' to exit, 'strategy' to change decoding strategy")
    print("-" * 50)
    
    while True:
        user_input = input("\nEnter Finnish sentence: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'strategy':
            print("Available strategies: greedy, beam, top_k")
            strategy = input("Choose strategy: ").strip().lower()
            
            if strategy == 'greedy':
                decoder = GreedyDecoder(model, vocab, vocab, device=device)
            elif strategy == 'beam':
                decoder = BeamSearchDecoder(model, vocab, vocab, device=device)
            elif strategy == 'top_k':
                decoder = TopKSamplingDecoder(model, vocab, vocab, device=device)
            else:
                print("Invalid strategy!")
                continue
            
            print(f"Switched to {strategy} decoding")
            continue
        
        if user_input:
            try:
                translation = decoder.decode(user_input.lower())
                print(f"Translation: {translation}")
            except Exception as e:
                print(f"Translation error: {e}")

if __name__ == '__main__':
    main()