"""
NLP.py

Goal:
- Read `results_excel_schema.csv`
- Extract the "Description" field
- Build a word frequency dictionary (unigrams) and, optionally, bigrams
- Save frequencies to CSV for later keyword selection
- Produce some visualisations:
    - Bar chart of top-N words
    - Bar chart of top-N bigrams

Usage (from the same folder where results_excel_schema.csv is stored):
    python analyze_descriptions_nlp.py
"""

import re
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Basic text processing utilities
# ---------------------------------------------------------------------

def simple_tokenize(text: str) -> list:
    """
    Very simple tokenizer:
    - lowercases
    - removes non-alphabetic characters
    - splits on whitespace

    Parameters
    ----------
    text : str
        Input text (game description).

    Returns
    -------
    tokens : list of str
        List of token strings.
    """
    text = text.lower()
    # Replace anything that is not a letter with a space
    text = re.sub(r"[^a-z]+", " ", text)
    tokens = text.split()
    return tokens


def build_stopwords() -> set:
    """
    Build a stopword set for English + domain-specific noise words.

    You can adjust this list as you refine the analysis.
    """
    # Basic English stopwords (minimal list, extend as needed)
    stopwords = {
        "the", "and", "a", "an", "to", "of", "in", "on", "for", "with", "by",
        "is", "are", "was", "were", "be", "been", "being",
        "this", "that", "these", "those",
        "it", "its", "at", "from", "as", "or", "if", "you",
        "your", "we", "our", "they", "their", "them",
        "can", "will", "would", "could", "should",
        "not", "no", "yes", "do", "does", "did",
        "about", "into", "out", "up", "down", "over", "under",
        "more", "most", "less", "many", "any", "some",
        "so", "such", "also", "just", "very",
        "use", "using", "used",
    }

    # Domain-specific noise words from game descriptions
    domain_noise = {
        "game", "games", "play", "player", "players", "fun", "challenge",
        "challenging", "levels", "level", "new", "features",
        "experience", "exciting", "enjoy", "enjoyable",
        "mobile", "phone", "tablet", "device", "devices",
        "free", "version", "download",
        # Tech / Platform
        "app", "apps", "store", "ios", "iphone", "ipad", "android", "google",
        "itunes", "apple", "mac", "pc", "steam",
        # Social / Web
        "http", "https", "www", "com", "org", "net", "website", "link",
        "facebook", "twitter", "instagram", "youtube", "discord", "reddit", "tiktok",
        "social", "media", "follow", "like", "share",
        # Legal / Business
        "privacy", "policy", "terms", "service", "support", "contact", "email",
        "copyright", "rights", "reserved", "trademark", "inc", "ltd", "llc",
        "subscribe", "subscription", "purchase", "buy", "payment", "price",
        "renew", "renewal", "period", "current", "cancel", "cancellation",
        "charged", "account", "settings", "confirmation", "offered", "forfeited",
        "unused", "portion", "trial", "auto", "automatically", "bill", "billing",
        "turned", "least", "within",
        # Common filler / Time / Misc
        "before", "after", "during", "while", "when", "where", "why", "how",
        "best", "top", "great", "good", "better", "real", "world",
        "time", "day", "daily", "week", "month", "year",
        "update", "updates", "bug", "fixes", "fixed", "improvement", "improvements",
        # User-specified exclusions
        "all", "have", "get", "make", "own", "friends", "other", "create", "through",
        "online", "multiplayer", "become", "yourself", "note", "need", "start", "going",
    }

    return stopwords.union(domain_noise)


# ---------------------------------------------------------------------
# 2. Core analysis functions
# ---------------------------------------------------------------------

def load_descriptions(path: str = "results_excel_schema.csv") -> pd.Series:
    """
    Load the descriptions column from the main scraping output.

    Parameters
    ----------
    path : str
        Path to the CSV file that contains a "Description" column.

    Returns
    -------
    descriptions : pandas.Series
        Series of description strings (NaN dropped).
    """
    df = pd.read_csv(path)
    if "Description" not in df.columns:
        raise ValueError("Column 'Description' not found in the CSV file.")

    descriptions = df["Description"].dropna().astype(str)
    return descriptions


def compute_unigram_frequencies(descriptions: pd.Series,
                                min_len: int = 3) -> Counter:
    """
    Compute unigram (single word) frequencies over all descriptions.

    Parameters
    ----------
    descriptions : pandas.Series
        Series of description strings.
    min_len : int
        Minimum token length to be considered (e.g., 3 to remove 'at', 'of', etc.).

    Returns
    -------
    freq : collections.Counter
        Counter with token → count.
    """
    stopwords = build_stopwords()
    freq = Counter()

    for desc in descriptions:
        tokens = simple_tokenize(desc)
        for tok in tokens:
            if len(tok) < min_len:
                continue
            if tok in stopwords:
                continue
            freq[tok] += 1

    return freq


def compute_bigram_frequencies(descriptions: pd.Series,
                               min_len: int = 3) -> Counter:
    """
    Compute bigram frequencies (pairs of consecutive tokens).

    Parameters
    ----------
    descriptions : pandas.Series
        Series of description strings.
    min_len : int
        Minimum token length for each token in the bigram.

    Returns
    -------
    bigram_freq : collections.Counter
        Counter with ("word1 word2") → count.
    """
    stopwords = build_stopwords()
    bigram_freq = Counter()

    for desc in descriptions:
        tokens = simple_tokenize(desc)
        # Filter stopwords and short tokens before forming bigrams
        tokens = [
            t for t in tokens
            if len(t) >= min_len and t not in stopwords
        ]
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]} {tokens[i+1]}"
            bigram_freq[bigram] += 1

    return bigram_freq


# ---------------------------------------------------------------------
# 3. Visualisation helpers
# ---------------------------------------------------------------------

def plot_top_unigrams(freq: Counter, out_dir: Path, top_n: int = 30):
    """
    Plot a bar chart of the top-N unigrams (most frequent words).

    Parameters
    ----------
    freq : collections.Counter
        Unigram frequencies.
    out_dir : pathlib.Path
        Output directory for the figure.
    top_n : int
        Number of top words to display.
    """
    most_common = freq.most_common(top_n)
    if not most_common:
        return

    words, counts = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} words in game descriptions")
    plt.tight_layout()
    plt.savefig(out_dir / f"top{top_n}_unigrams.png")
    plt.close()


def plot_top_bigrams(freq: Counter, out_dir: Path, top_n: int = 20):
    """
    Plot a bar chart of the top-N bigrams.

    Parameters
    ----------
    freq : collections.Counter
        Bigram frequencies.
    out_dir : pathlib.Path
        Output directory for the figure.
    top_n : int
        Number of top bigrams to display.
    """
    most_common = freq.most_common(top_n)
    if not most_common:
        return

    bigrams, counts = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(bigrams, counts)
    plt.xticks(rotation=60, ha="right")
    plt.ylabel("Frequency")
    plt.title(f"Top {top_n} bigrams in game descriptions")
    plt.tight_layout()
    plt.savefig(out_dir / f"top{top_n}_bigrams.png")
    plt.close()


# ---------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------

def run_nlp_analysis(
    results_path: str = "results_excel_schema.csv",
    out_dir: str = "description_nlp_analysis",
    top_n_unigrams: int = 30,
    top_n_bigrams: int = 30,
):
    """
    Main pipeline:

    1. Load descriptions from `results_excel_schema.csv`.
    2. Compute unigram and bigram frequencies.
    3. Save frequencies to CSV.
    4. Generate a few plots.

    Parameters
    ----------
    results_path : str
        Path to the CSV file with game results and a `Description` column.
    out_dir : str
        Directory to store CSV outputs and visualisations.
    top_n_unigrams : int
        Number of top unigrams to plot.
    top_n_bigrams : int
        Number of top bigrams to plot.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load descriptions
    try:
        descriptions = load_descriptions(results_path)
        print(f"Loaded {len(descriptions)} descriptions.")
    except Exception as e:
        print(f"Error loading descriptions: {e}")
        return

    if descriptions.empty:
        print("No descriptions found to analyze.")
        return

    # 2. Compute frequencies
    unigram_freq = compute_unigram_frequencies(descriptions)
    bigram_freq = compute_bigram_frequencies(descriptions)

    print(f"Found {len(unigram_freq)} unique unigrams after filtering.")
    print(f"Found {len(bigram_freq)} unique bigrams after filtering.")

    # 3. Save frequencies to CSV (for later keyword selection)
    unigram_df = pd.DataFrame(
        unigram_freq.most_common(),
        columns=["word", "count"]
    )
    unigram_df.to_csv(out_path / "unigram_frequencies.csv", index=False)

    bigram_df = pd.DataFrame(
        bigram_freq.most_common(),
        columns=["bigram", "count"]
    )
    bigram_df.to_csv(out_path / "bigram_frequencies.csv", index=False)

    # 4. Plots
    plot_top_unigrams(unigram_freq, out_path, top_n=top_n_unigrams)
    plot_top_bigrams(bigram_freq, out_path, top_n=top_n_bigrams)

    print("Analysis complete.")
    print(f"- Unigram frequencies saved to: {out_path / 'unigram_frequencies.csv'}")
    print(f"- Bigram frequencies saved to: {out_path / 'bigram_frequencies.csv'}")
    print(f"- Figures saved in: {out_path.resolve()}")


if __name__ == "__main__":
    run_nlp_analysis()
