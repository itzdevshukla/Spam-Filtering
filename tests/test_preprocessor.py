"""Unit tests for text preprocessing and signal preservation."""

import pytest
from src.preprocessor import clean_and_normalize_text, extract_meta_features


def test_clean_and_normalize_preserves_urls():
    text = "Visit http://claim-prize.com/urgent now!"
    normalized = clean_and_normalize_text(text, preserve_signals=True)
    assert "__url__" in normalized


def test_clean_and_normalize_preserves_currency():
    text = "Congratulations, you won £5000 cash or $1000 voucher!"
    normalized = clean_and_normalize_text(text, preserve_signals=True)
    assert "__currency__" in normalized


def test_clean_and_normalize_preserves_phone_numbers():
    text = "Call 08718726270 immediately to claim."
    normalized = clean_and_normalize_text(text, preserve_signals=True)
    assert "__phone__" in normalized


def test_clean_and_normalize_preserves_exclamation_bursts():
    text = "FREE PRIZE WINNER!!!!!!"
    normalized = clean_and_normalize_text(text, preserve_signals=True)
    assert "__exclburst__" in normalized


def test_extract_meta_features():
    text = "FREE! Win £1000 cash today. Call 0800123456 now!"
    feats = extract_meta_features(text)
    assert feats["char_count"] == len(text)
    assert feats["word_count"] > 0
    assert feats["currency_count"] >= 1
    assert feats["uppercase_ratio"] > 0
    assert feats["has_urgent_keyword"] == 1.0


def test_empty_string_meta_features():
    feats = extract_meta_features("")
    assert feats["char_count"] == 0.0
    assert feats["word_count"] == 0.0
