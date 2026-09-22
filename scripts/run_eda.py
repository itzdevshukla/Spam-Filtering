"""CLI script to run full Exploratory Data Analysis on UCI SMS Spam dataset."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eda import run_full_eda

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "processed" / "sms_spam_clean.csv"

    if not data_path.exists():
        print("[Error] Processed dataset not found. Running data acquisition first...")
        from src.data_loader import prepare_and_save_primary_data
        df, _ = prepare_and_save_primary_data()
    else:
        df = pd.read_csv(data_path)

    print("=" * 60)
    print("SpamShield Exploratory Data Analysis (EDA)")
    print(f"Dataset Shape: {df.shape}")
    print("=" * 60)

    stats = run_full_eda(df)
    print("\nEDA Summary Statistics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    print("\n[Complete] Check 'outputs/eda/' for generated visualizations.")

if __name__ == "__main__":
    main()
