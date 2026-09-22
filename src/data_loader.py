"""Data acquisition and ingestion module for SpamShield.

Handles downloading, unpacking, inspecting, and standardizing the primary
UCI SMS Spam Collection dataset and the secondary SpamAssassin Mail Corpus.
"""

from __future__ import annotations
import email
from email import policy
import io
import os
from pathlib import Path
import tarfile
import urllib.request
import zipfile
import pandas as pd

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

UCI_SMS_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
SPAMASSASSIN_HAM_URL = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2"
SPAMASSASSIN_SPAM_URL = "https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2"


def download_uci_sms_dataset(force: bool = False) -> Path:
    """Download and extract the UCI SMS Spam Collection dataset."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_file = RAW_DIR / "SMSSpamCollection"
    zip_path = RAW_DIR / "sms_spam_collection.zip"

    if target_file.exists() and not force:
        print(f"[DataLoader] Raw dataset already exists at {target_file}")
        return target_file

    print(f"[DataLoader] Downloading primary dataset from {UCI_SMS_URL}...")
    headers = {"User-Agent": "SpamShield-Downloader/1.0"}
    req = urllib.request.Request(UCI_SMS_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    print("[DataLoader] Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    # In the zip, the file is named SMSSpamCollection
    if not target_file.exists():
        candidates = list(RAW_DIR.glob("*SMSSpamCollection*"))
        if candidates:
            candidates[0].rename(target_file)

    if zip_path.exists():
        zip_path.unlink()

    print(f"[DataLoader] Successfully unpacked dataset to {target_file}")
    return target_file


def load_raw_sms_dataset(raw_path: Path | None = None) -> pd.DataFrame:
    """Load the raw UCI SMS Spam Collection into a pandas DataFrame.
    
    The raw file is tab-delimited with no header:
    Column 0: label ('ham' or 'spam')
    Column 1: text message
    """
    if raw_path is None:
        raw_path = RAW_DIR / "SMSSpamCollection"

    if not raw_path.exists():
        download_uci_sms_dataset()

    # Read tab-separated format, handling potential encoding quirks
    try:
        df = pd.read_csv(
            raw_path,
            sep="\t",
            names=["raw_label", "text"],
            encoding="utf-8",
            quoting=3,  # QUOTE_NONE to avoid issues with unmatched quotes
            on_bad_lines="skip"
        )
    except Exception:
        df = pd.read_csv(
            raw_path,
            sep="\t",
            names=["raw_label", "text"],
            encoding="latin-1",
            quoting=3,
            on_bad_lines="skip"
        )

    return df


def clean_primary_dataset(df: pd.DataFrame, remove_duplicates: bool = True) -> tuple[pd.DataFrame, dict]:
    """Inspect and clean the primary SMS dataset.
    
    Operations:
    1. Check missing values and drop invalid rows.
    2. Normalize label: 'ham' -> 0, 'spam' -> 1.
    3. Trim whitespace and handle malformed rows.
    4. Deduplicate (optional, but documented for academic rigor).
    5. Return cleaned DataFrame and inspection statistics dictionary.
    """
    stats = {}
    stats["initial_shape"] = df.shape
    stats["initial_missing"] = df.isnull().sum().to_dict()
    stats["initial_class_dist"] = df["raw_label"].value_counts().to_dict()

    # Drop nulls
    cleaned = df.dropna(subset=["raw_label", "text"]).copy()
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"].str.len() > 0]

    # Map labels: ham -> 0, spam -> 1
    label_map = {"ham": 0, "spam": 1}
    cleaned = cleaned[cleaned["raw_label"].isin(label_map.keys())].copy()
    cleaned["label"] = cleaned["raw_label"].map(label_map).astype(int)

    stats["duplicate_count"] = int(cleaned.duplicated(subset=["text"]).sum())
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates(subset=["text"]).reset_index(drop=True)

    cleaned = cleaned[["label", "text", "raw_label"]]
    stats["final_shape"] = cleaned.shape
    stats["final_class_dist"] = cleaned["label"].value_counts().to_dict()
    stats["spam_ratio"] = float(cleaned["label"].mean())

    return cleaned, stats


def prepare_and_save_primary_data() -> tuple[pd.DataFrame, dict]:
    """End-to-end preparation of the primary dataset."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = download_uci_sms_dataset()
    raw_df = load_raw_sms_dataset(raw_path)
    clean_df, stats = clean_primary_dataset(raw_df, remove_duplicates=True)

    out_csv = PROCESSED_DIR / "sms_spam_clean.csv"
    clean_df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[DataLoader] Cleaned primary dataset saved to {out_csv} (Shape: {clean_df.shape})")
    return clean_df, stats


def _extract_email_body(raw_bytes: bytes) -> str:
    """Parse raw RFC-822 email bytes and return cleaned plain-text body."""
    try:
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        body_parts = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        body_parts.append(payload.decode("utf-8", errors="ignore"))
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body_parts.append(payload.decode("utf-8", errors="ignore"))
            else:
                body_parts.append(str(msg.get_payload()))
        full_text = " ".join(body_parts).strip()
        return full_text if full_text else raw_bytes.decode("latin-1", errors="ignore")[:1000]
    except Exception:
        return raw_bytes.decode("latin-1", errors="ignore")[:1000]


def download_spamassassin_corpus(max_per_class: int = 500, force: bool = False) -> Path:
    """Download a partition of SpamAssassin public corpus for external generalization testing.
    
    Extracts text body and standardizes to ['label', 'text', 'source'].
    """
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    out_file = EXTERNAL_DIR / "spamassassin_generalization.csv"
    if out_file.exists() and not force:
        print(f"[DataLoader] External SpamAssassin dataset already exists at {out_file}")
        return out_file

    records = []
    headers = {"User-Agent": "SpamShield-Downloader/1.0"}

    # 1. Download Ham
    print("[DataLoader] Downloading SpamAssassin Ham corpus...")
    req_ham = urllib.request.Request(SPAMASSASSIN_HAM_URL, headers=headers)
    with urllib.request.urlopen(req_ham, timeout=60) as resp:
        tar_bytes = io.BytesIO(resp.read())
        with tarfile.open(fileobj=tar_bytes, mode="r:bz2") as tar:
            count = 0
            for member in tar.getmembers():
                if member.isfile() and not member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f:
                        body = _extract_email_body(f.read())
                        if len(body) > 20:
                            records.append({"label": 0, "text": body, "source": "spamassassin_ham"})
                            count += 1
                            if count >= max_per_class:
                                break

    # 2. Download Spam
    print("[DataLoader] Downloading SpamAssassin Spam corpus...")
    req_spam = urllib.request.Request(SPAMASSASSIN_SPAM_URL, headers=headers)
    with urllib.request.urlopen(req_spam, timeout=60) as resp:
        tar_bytes = io.BytesIO(resp.read())
        with tarfile.open(fileobj=tar_bytes, mode="r:bz2") as tar:
            count = 0
            for member in tar.getmembers():
                if member.isfile() and not member.name.endswith(".txt"):
                    f = tar.extractfile(member)
                    if f:
                        body = _extract_email_body(f.read())
                        if len(body) > 20:
                            records.append({"label": 1, "text": body, "source": "spamassassin_spam"})
                            count += 1
                            if count >= max_per_class:
                                break

    df_ext = pd.DataFrame(records)
    df_ext = df_ext.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df_ext.to_csv(out_file, index=False, encoding="utf-8")
    print(f"[DataLoader] External evaluation dataset saved to {out_file} (Shape: {df_ext.shape})")
    return out_file


if __name__ == "__main__":
    print("Executing standalone dataset acquisition...")
    prepare_and_save_primary_data()
    download_spamassassin_corpus(max_per_class=400)
