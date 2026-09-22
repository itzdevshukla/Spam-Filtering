"""Text preprocessing and spam signal extraction module for SpamShield.

Implements specialized text cleaning that rigorously preserves high-yield
spam indicators (URLs, phone numbers, currency symbols, urgent punctuation bursts,
and capitalization dynamics) rather than naively stripping them.
"""

from __future__ import annotations
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Regular expressions for key spam signals
URL_REGEX = re.compile(
    r"(?:https?://|www\.)[^\s/$.?#].[^\s]*|(?:\b[a-zA-Z0-9-]+\.(?:com|org|net|co\.uk|info|biz|tv|me|io|in|xyz)\b(?:/[^\s]*)?)",
    re.IGNORECASE,
)
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}\b|\b\d{10,12}\b"
)
CURRENCY_AMOUNT_REGEX = re.compile(
    r"(?:[\$£€¥₹]|(?:rs\.?|inr|usd|gbp|eur)\s*)\s*\d+(?:[.,]\d+)?|\b\d+(?:[.,]\d+)?\s*(?:[\$£€¥₹]|(?:rs\.?|inr|usd|gbp|eur|p|pounds))\b",
    re.IGNORECASE,
)
CURRENCY_SYMBOL_REGEX = re.compile(r"[\$£€¥₹]|(?:\brs\.?|\binr|\busd|\bgbp|\beur)\b", re.IGNORECASE)
NUMBER_REGEX = re.compile(r"\b\d+\b")
EXCLAMATION_BURST_REGEX = re.compile(r"!{2,}")
QUESTION_BURST_REGEX = re.compile(r"\?{2,}")
REPEATED_CHAR_REGEX = re.compile(r"(.)\1{3,}")


def extract_meta_features(text: str) -> dict[str, float]:
    """Extract domain-specific quantitative meta-features from raw message text.
    
    These features capture syntactic, structural, and behavioral markers
    strongly correlated with unsolicited bulk communications (Spam).
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "char_count": 0.0,
            "word_count": 0.0,
            "uppercase_count": 0.0,
            "uppercase_ratio": 0.0,
            "digit_count": 0.0,
            "digit_ratio": 0.0,
            "currency_count": 0.0,
            "url_count": 0.0,
            "phone_count": 0.0,
            "exclamation_count": 0.0,
            "question_count": 0.0,
            "exclamation_burst": 0.0,
            "has_urgent_keyword": 0.0,
        }

    char_count = float(len(text))
    words = text.split()
    word_count = float(len(words))
    uppercase_count = float(sum(1 for c in text if c.isupper()))
    digit_count = float(sum(1 for c in text if c.isdigit()))

    uppercase_ratio = uppercase_count / max(char_count, 1.0)
    digit_ratio = digit_count / max(char_count, 1.0)

    currency_count = float(len(CURRENCY_SYMBOL_REGEX.findall(text)))
    url_count = float(len(URL_REGEX.findall(text)))
    phone_count = float(len(PHONE_REGEX.findall(text)))
    exclamation_count = float(text.count("!"))
    question_count = float(text.count("?"))
    exclamation_burst = float(len(EXCLAMATION_BURST_REGEX.findall(text)))

    urgent_pattern = re.compile(
        r"\b(?:urgent|free|win|winner|cash|prize|claim|guaranteed|awarded|selected|won|credit|loan|call now|text back|congratulations)\b",
        re.IGNORECASE,
    )
    has_urgent_keyword = float(1.0 if urgent_pattern.search(text) else 0.0)

    return {
        "char_count": char_count,
        "word_count": word_count,
        "uppercase_count": uppercase_count,
        "uppercase_ratio": uppercase_ratio,
        "digit_count": digit_count,
        "digit_ratio": digit_ratio,
        "currency_count": currency_count,
        "url_count": url_count,
        "phone_count": phone_count,
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "exclamation_burst": exclamation_burst,
        "has_urgent_keyword": has_urgent_keyword,
    }


def clean_and_normalize_text(text: str, preserve_signals: bool = True) -> str:
    """Normalize raw message text into a canonical representation.
    
    If preserve_signals is True, high-information spam cues are mapped
    to distinct semantic tokens rather than being discarded:
      - URLs -> __url__
      - Phone numbers -> __phone__
      - Currency amounts -> __currency__
      - Plain numbers -> __number__
      - Exclamation bursts (!!!) -> __exclburst__
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalization (decompose characters to avoid homoglyph tricks)
    normalized = unicodedata.normalize("NFKD", text)

    # 2. Collapse extreme character repetitions (e.g., 'freeeeee' -> 'freee')
    normalized = REPEATED_CHAR_REGEX.sub(r"\1\1\1", normalized)

    if preserve_signals:
        # Preserve URLs
        normalized = URL_REGEX.sub(" __url__ ", normalized)
        # Preserve currency amounts
        normalized = CURRENCY_AMOUNT_REGEX.sub(" __currency__ ", normalized)
        # Preserve phone numbers
        normalized = PHONE_REGEX.sub(" __phone__ ", normalized)
        # Exclamation / Question bursts
        normalized = EXCLAMATION_BURST_REGEX.sub(" __exclburst__ ", normalized)
        normalized = QUESTION_BURST_REGEX.sub(" __questburst__ ", normalized)
        # Remaining numbers
        normalized = NUMBER_REGEX.sub(" __number__ ", normalized)

    # 3. Lowercase
    normalized = normalized.lower()

    # 4. Remove unwanted symbols while keeping semantic tokens and standard words
    # Keep alphanumeric, underscores (used in tokens), and basic punctuation
    normalized = re.sub(r"[^\w\s_]", " ", normalized)

    # 5. Whitespace normalization
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


class MetaFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for extracting domain meta-features."""

    def __init__(self):
        self.feature_names_ = [
            "char_count",
            "word_count",
            "uppercase_count",
            "uppercase_ratio",
            "digit_count",
            "digit_ratio",
            "currency_count",
            "url_count",
            "phone_count",
            "exclamation_count",
            "question_count",
            "exclamation_burst",
            "has_urgent_keyword",
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            rows = [extract_meta_features(str(text)) for text in X]
        elif isinstance(X, list):
            rows = [extract_meta_features(str(text)) for text in X]
        elif isinstance(X, np.ndarray):
            rows = [extract_meta_features(str(text)) for text in X.ravel()]
        else:
            rows = [extract_meta_features(str(X))]

        df_features = pd.DataFrame(rows)[self.feature_names_]
        return df_features.values

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class TextCleanerTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for text normalization."""

    def __init__(self, preserve_signals: bool = True):
        self.preserve_signals = preserve_signals

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            return X.apply(lambda t: clean_and_normalize_text(t, self.preserve_signals)).values
        elif isinstance(X, list):
            return [clean_and_normalize_text(t, self.preserve_signals) for t in X]
        elif isinstance(X, np.ndarray):
            return np.array([clean_and_normalize_text(t, self.preserve_signals) for t in X.ravel()])
        else:
            return np.array([clean_and_normalize_text(str(X), self.preserve_signals)])
