import sentencepiece as spm
import os
import tempfile
from typing import List, Union, Optional
from collections import Counter


class Vocabulary:
    """Vocabulary class using SentencePiece BPE for Finnish and English tokenization"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.sp = spm.SentencePieceProcessor()
        self.model_path = model_path
        
        # Special token IDs (SentencePiece handles these automatically)
        self.PAD_ID = 0  # <pad>
        self.UNK_ID = 1  # <unk>  
        self.BOS_ID = 2  # <s> (Beginning of sentence)
        self.EOS_ID = 3  # </s> (End of sentence)
        
        # Load existing model if provided
        if model_path and os.path.exists(model_path):
            self.sp.load(model_path)
    
    def train(self, 
              sentences: List[str], 
              vocab_size: int = 32000,
              model_prefix: str = "sentencepiece_model",
              character_coverage: float = 0.9995):
        """
        Train SentencePiece BPE model on Finnish and English text
        
        Args:
            sentences: List of training sentences (Finnish + English)
            vocab_size: Target vocabulary size (16k-32k recommended)
            model_prefix: Prefix for saved model files
            character_coverage: Character coverage (0.9995 good for Finnish+English)
        """
        # Create temporary file with training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for sentence in sentences:
                f.write(sentence.strip() + '\n')
            temp_file = f.name
        
        try:
            # Train SentencePiece model
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type='bpe',  # Use BPE algorithm
                character_coverage=character_coverage,
                pad_id=self.PAD_ID,
                unk_id=self.UNK_ID,
                bos_id=self.BOS_ID,
                eos_id=self.EOS_ID,
                pad_piece='<pad>',
                unk_piece='<unk>',
                bos_piece='<s>',
                eos_piece='</s>',
                user_defined_symbols=[],  # Add custom symbols if needed
                split_by_whitespace=True,
                split_by_unicode_script=True,  # Helps with multilingual text
                split_by_number=True,
                split_digits=True,
                treat_whitespace_as_suffix=False,
                allow_whitespace_only_pieces=True,
                normalization_rule_name='nfkc',  # Unicode normalization
            )
            
            # Load the trained model
            model_file = f"{model_prefix}.model"
            self.sp.load(model_file)
            self.model_path = model_file
            
            print(f"SentencePiece model trained successfully!")
            print(f"Vocabulary size: {len(self)}")
            print(f"Model saved as: {model_file}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """
        Encode text to subword token IDs
        
        Args:
            text: Input text (Finnish or English)
            add_bos: Add beginning-of-sentence token
            add_eos: Add end-of-sentence token
            
        Returns:
            List of token IDs
        """
        if not hasattr(self.sp, 'encode') or self.sp.get_piece_size() == 0:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        # Encode text to IDs
        token_ids = self.sp.encode(text.strip())
        
        # Add special tokens if requested
        if add_bos:
            token_ids = [self.BOS_ID] + token_ids
        if add_eos:
            token_ids = token_ids + [self.EOS_ID]
            
        return token_ids
    
    def encode_as_pieces(self, text: str) -> List[str]:
        """
        Encode text to subword pieces (for debugging/visualization)
        
        Args:
            text: Input text
            
        Returns:
            List of subword pieces
        """
        if not hasattr(self.sp, 'encode') or self.sp.get_piece_size() == 0:
            raise ValueError("Model not loaded. Train or load a model first.")
            
        return self.sp.encode_as_pieces(text.strip())
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Remove special tokens from output
            
        Returns:
            Decoded text
        """
        if not hasattr(self.sp, 'decode') or self.sp.get_piece_size() == 0:
            raise ValueError("Model not loaded. Train or load a model first.")
        
        if skip_special_tokens:
            # Remove special tokens
            filtered_ids = [tid for tid in token_ids 
                          if tid not in [self.PAD_ID, self.BOS_ID, self.EOS_ID]]
            return self.sp.decode(filtered_ids)
        else:
            return self.sp.decode(token_ids)
    
    def __len__(self) -> int:
        """Return vocabulary size"""
        return self.sp.get_piece_size() if hasattr(self.sp, 'get_piece_size') else 0
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self)
    
    def id_to_piece(self, token_id: int) -> str:
        """Convert token ID to piece"""
        return self.sp.id_to_piece(token_id)
    
    def piece_to_id(self, piece: str) -> int:
        """Convert piece to token ID"""
        return self.sp.piece_to_id(piece)
    
    def save(self, model_prefix: str):
        """
        Save the trained model
        
        Args:
            model_prefix: Prefix for model files
        """
        if self.model_path and os.path.exists(self.model_path):
            # Copy current model to new location
            import shutil
            new_model_path = f"{model_prefix}.model"
            new_vocab_path = f"{model_prefix}.vocab"
            
            shutil.copy(self.model_path, new_model_path)
            if os.path.exists(self.model_path.replace('.model', '.vocab')):
                shutil.copy(self.model_path.replace('.model', '.vocab'), new_vocab_path)
            
            self.model_path = new_model_path
            print(f"💾 Model saved as: {new_model_path}")
        else:
            raise ValueError("No trained model to save. Train a model first.")
    
    @classmethod
    def load(cls, model_path: str) -> 'Vocabulary':
        """
        Load a trained SentencePiece model
        
        Args:
            model_path: Path to .model file
            
        Returns:
            Vocabulary instance with loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        vocab = cls(model_path=model_path)
        print(f"📖 Model loaded from: {model_path}")
        print(f"📊 Vocabulary size: {len(vocab)}")
        return vocab


# Example usage and testing
if __name__ == "__main__":
    # read data from files
    with open('../data/EUbookshop.fi', 'r', encoding='utf-8') as f:
        finnish_sentences = f.readlines()
    with open('../data/EUbookshop.en', 'r', encoding='utf-8') as f:
        english_sentences = f.readlines()
    combined_sentences = finnish_sentences + english_sentences

    # Create and train vocabulary
    vocab = Vocabulary()
    
    print("Training SentencePiece model...")
    vocab.train(combined_sentences, vocab_size=16000, model_prefix="finnish_english")

    # Test encoding/decoding
    test_sentence = "osallistujillamme"
    
    print(f"\nOriginal: {test_sentence}")
    
    # Encode to pieces (for visualization)
    pieces = vocab.encode_as_pieces(test_sentence)
    print(f"Pieces: {pieces}")
    
    # Encode to IDs
    token_ids = vocab.encode(test_sentence)
    print(f"Token IDs: {token_ids}")
    
    # Decode back
    decoded = vocab.decode(token_ids)
    print(f"Decoded: {decoded}")
    
    print(f"\nVocabulary size: {len(vocab)}")
