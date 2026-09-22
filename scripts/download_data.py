"""Standalone script to download all datasets for SpamShield."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import prepare_and_save_primary_data, download_spamassassin_corpus

def main():
    print("=" * 60)
    print("SpamShield Dataset Acquisition Pipeline")
    print("=" * 60)
    
    print("\n[Step 1/2] Acquiring and cleaning primary UCI SMS Spam Collection...")
    df_primary, stats = prepare_and_save_primary_data()
    print("  Initial shape:", stats["initial_shape"])
    print("  Duplicates removed:", stats["duplicate_count"])
    print("  Final clean shape:", stats["final_shape"])
    print("  Class distribution (0=HAM, 1=SPAM):", stats["final_class_dist"])
    print(f"  Spam ratio: {stats['spam_ratio']:.2%}")

    print("\n[Step 2/2] Acquiring external generalization dataset (SpamAssassin)...")
    download_spamassassin_corpus(max_per_class=400)
    print("\nDataset acquisition complete!")

if __name__ == "__main__":
    main()
