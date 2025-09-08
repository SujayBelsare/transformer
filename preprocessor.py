import os
import random
import json
import re
import unicodedata
from typing import Tuple, List, Dict

class Preprocessor:
    def __init__(self, fi_path: str, en_path: str,
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 seed: int = 2023101033,
                 out_dir: str = "./data"):
        """
        Args:
            fi_path: path to Finnish sentences file (.fi)
            en_path: path to English sentences file (.en)
            train_ratio: proportion of data for training
            val_ratio: proportion of data for validation
            seed: random seed for reproducibility
            out_dir: directory to save train/val/test splits
        """
        self.fi_path = fi_path
        self.en_path = en_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.out_dir = out_dir
        self.splits = {}

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        return unicodedata.normalize('NFC', text)
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize quotes, dashes, and spaces"""
        # Replace curly quotes with straight quotes
        text = re.sub(r'[""''`´]', '"', text)
        text = re.sub(r'[''‛]', "'", text)
        
        # Normalize dashes
        text = re.sub(r'[–—―]', '-', text)
        
        # Replace non-breaking spaces and other special spaces
        text = re.sub(r'[\u00A0\u2000-\u200B\u2028\u2029]', ' ', text)
        
        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def clean_symbols(self, text: str) -> str:
        """Remove unwanted symbols but keep important punctuation"""
        # Remove: *, », «, †, ‡, §, ¶, etc. but keep important punctuation
        # Keep: letters, numbers, spaces, and important punctuation: . ! ? , ; : - ( ) " ' /
        text = re.sub(r'[^\w\s.,!?;:()\-"\'/]', ' ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def tokenize_simple(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace"""
        return text.split()
    
    def is_valid_sentence_pair(self, fi_tokens: List[str], en_tokens: List[str]) -> bool:
        """Check if sentence pair meets quality criteria"""
        # 1. Remove empty or very short sentences (< 2 tokens)
        if len(fi_tokens) < 2 or len(en_tokens) < 2:
            return False
        
        # 2. Remove excessively long sentences (> 200 tokens)
        if len(fi_tokens) > 200 or len(en_tokens) > 200:
            return False
        
        # 3. Filter misaligned pairs (length ratio check)
        ratio = len(fi_tokens) / len(en_tokens)
        if ratio > 3.0 or ratio < 1/3.0:
            return False
        
        return True
    
    def process_sentence_pair(self, fi_sentence: str, en_sentence: str) -> Tuple[str, str, bool]:
        """
        Process a sentence pair through the full cleaning pipeline
        
        Returns:
            Tuple of (cleaned_fi, cleaned_en, is_valid)
        """
        # Step 1: Unicode normalization
        fi_clean = self.normalize_unicode(fi_sentence)
        en_clean = self.normalize_unicode(en_sentence)
        
        # Step 2: Punctuation normalization
        fi_clean = self.normalize_punctuation(fi_clean)
        en_clean = self.normalize_punctuation(en_clean)
        
        # Step 3: Symbol cleaning
        fi_clean = self.clean_symbols(fi_clean)
        en_clean = self.clean_symbols(en_clean)
        
        # Step 4: Lowercasing
        fi_clean = fi_clean.lower()
        en_clean = en_clean.lower()
        
        # Step 5: Tokenization for validation
        fi_tokens = self.tokenize_simple(fi_clean)
        en_tokens = self.tokenize_simple(en_clean)
        
        # Step 6: Validation
        is_valid = self.is_valid_sentence_pair(fi_tokens, en_tokens)
        
        return fi_clean, en_clean, is_valid

    def create_splits_and_save(self):
        """Main function to create splits and save as JSON files"""
        assert os.path.exists(self.fi_path), f"File not found: {self.fi_path}"
        assert os.path.exists(self.en_path), f"File not found: {self.en_path}"

        print("Reading data...")
        # Read data
        with open(self.fi_path, "r", encoding="utf-8") as f:
            fi_sentences = f.readlines()
        with open(self.en_path, "r", encoding="utf-8") as f:
            en_sentences = f.readlines()

        assert len(fi_sentences) == len(en_sentences), "Mismatched corpus sizes"
        print(f"Loaded {len(fi_sentences)} sentence pairs")

        # Process sentences through cleaning pipeline
        print("Processing sentences through cleaning pipeline...")
        processed_pairs = []
        seen_pairs = set()  # For duplicate removal
        
        stats = {
            'total': len(fi_sentences),
            'too_short': 0,
            'too_long': 0,
            'misaligned': 0,
            'duplicates': 0,
            'empty': 0,
            'valid': 0
        }
        
        for i, (fi_sent, en_sent) in enumerate(zip(fi_sentences, en_sentences)):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(fi_sentences)} sentences...")
            
            fi_clean, en_clean, is_valid = self.process_sentence_pair(fi_sent.strip(), en_sent.strip())
            
            # Skip empty sentences
            if not fi_clean or not en_clean:
                stats['empty'] += 1
                continue
            
            # Skip invalid pairs
            if not is_valid:
                # Determine reason for invalidity
                fi_tokens = self.tokenize_simple(fi_clean)
                en_tokens = self.tokenize_simple(en_clean)
                
                if len(fi_tokens) < 2 or len(en_tokens) < 2:
                    stats['too_short'] += 1
                elif len(fi_tokens) > 200 or len(en_tokens) > 200:
                    stats['too_long'] += 1
                else:
                    stats['misaligned'] += 1
                continue
            
            # Check for duplicates
            pair_key = (fi_clean, en_clean)
            if pair_key in seen_pairs:
                stats['duplicates'] += 1
                continue
            
            seen_pairs.add(pair_key)
            processed_pairs.append((fi_clean, en_clean))
            stats['valid'] += 1

        print(f"\nCleaning statistics:")
        print(f"Total pairs: {stats['total']}")
        print(f"Valid pairs: {stats['valid']}")
        print(f"Removed - Empty: {stats['empty']}")
        print(f"Removed - Too short: {stats['too_short']}")
        print(f"Removed - Too long: {stats['too_long']}")
        print(f"Removed - Misaligned: {stats['misaligned']}")
        print(f"Removed - Duplicates: {stats['duplicates']}")
        print(f"Retention rate: {stats['valid']/stats['total']*100:.2f}%")

        # Shuffle with fixed seed
        print("\nShuffling and creating splits...")
        random.seed(self.seed)
        random.shuffle(processed_pairs)

        # Compute splits
        n_total = len(processed_pairs)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_pairs = processed_pairs[:n_train]
        val_pairs = processed_pairs[n_train:n_train + n_val]
        test_pairs = processed_pairs[n_train + n_val:]

        # Collect splits as list of dictionaries
        self.splits = {
            "train": [{"fi": fi, "en": en} for fi, en in train_pairs],
            "val": [{"fi": fi, "en": en} for fi, en in val_pairs],
            "test": [{"fi": fi, "en": en} for fi, en in test_pairs]
        }

        # Save splits to JSON files
        print("Saving splits...")
        os.makedirs(self.out_dir, exist_ok=True)
        for split, data in self.splits.items():
            json_out = os.path.join(self.out_dir, f"{split}.json")
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {split} split: {json_out} ({len(data)} samples)")
            
        # Save cleaning statistics
        stats_out = os.path.join(self.out_dir, "cleaning_stats.json")
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved cleaning statistics: {stats_out}")

    def get_split(self, split: str) -> List[Dict[str, str]]:
        """Return sentences from a given split as list of dictionaries"""
        assert split in self.splits, "Invalid split: choose train/val/test"
        return self.splits[split]

if __name__ == "__main__":
    dataset = Preprocessor(
        fi_path="../data/EUbookshop.fi",
        en_path="../data/EUbookshop.en",
        train_ratio=0.8,
        val_ratio=0.1,
        seed=42,
        out_dir="../data"
    )
    
    # Create and save splits
    dataset.create_splits_and_save()
    
    # Print final stats
    train_data = dataset.get_split("train")
    val_data = dataset.get_split("val")
    test_data = dataset.get_split("test")
    print(f"\nFinal split sizes:")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Show examples of cleaned data
    print(f"\nExamples from training set:")
    for i in range(min(3, len(train_data))):
        print(f"\nExample {i+1}:")
        print(f"Finnish: {train_data[i]['fi']}")
        print(f"English: {train_data[i]['en']}")
        
    # Show token length statistics
    fi_lengths = [len(dataset.tokenize_simple(item['fi'])) for item in train_data[:1000]]
    en_lengths = [len(dataset.tokenize_simple(item['en'])) for item in train_data[:1000]]
    
    print(f"\nToken length statistics (first 1000 training samples):")
    print(f"Finnish - Min: {min(fi_lengths)}, Max: {max(fi_lengths)}, Avg: {sum(fi_lengths)/len(fi_lengths):.1f}")
    print(f"English - Min: {min(en_lengths)}, Max: {max(en_lengths)}, Avg: {sum(en_lengths)/len(en_lengths):.1f}")