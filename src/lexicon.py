#!/usr/bin/env python
"""
Sustainability classification for App Store games.

This script extends the main App Store extraction output (`results_excel_schema.csv`)
with two additional columns:

    - Sustainability_Score
    - Sustainability_Game_Rule

The classification is based exclusively on the textual content of the "Description"
field and relies on a transparent, lexicon-based heuristic. No language model is used.

Sustainability_Score
--------------------
A deterministic score in the range [0, 1] derived from the presence of:

    1. Sustainability-related terms (e.g. environment, recycling, carbon, eco, etc.)
    2. Educational / behavioural terms (e.g. learn, teach, habits, manage, etc.)

The score is computed as:

    raw = (# sustainability hits) + (# educational hits)
    score = min(1.0, raw / 4.0)

This creates a simple stepwise saturation:

    raw = 0  -> 0.00
    raw = 1  -> 0.25
    raw = 2  -> 0.50
    raw = 3  -> 0.75
    raw >= 4 -> 1.00

The intention is not to approximate a calibrated probability, but to provide
a reproducible and interpretable indicator of “how strongly” a description
matches the sustainability+education profile.

Sustainability_Game_Rule
------------------------
A binary flag derived from the score:

    - "YES" if Sustainability_Score >= threshold
    - "NO"  otherwise

By default, threshold = 0.7 (i.e. raw >= 3). This can be overridden via CLI.

Usage
-----
    python classify_sustainability.py \
        --infile results_excel_schema.csv \
        --outfile results_with_sustainability_flag.csv \
        --threshold 0.7

Requirements
------------
    - Python 3.8+
    - pandas
"""

import argparse
import re
from typing import Set

import pandas as pd


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

#: Terms directly associated with environmental sustainability,
#: resource use, pollution, biodiversity, etc.
SUSTAINABILITY_TERMS: Set[str] = {
    "sustainable", "sustainability",
    "environment", "environmental", "eco", "ecological",
    "climate", "carbon", "footprint", "pollution",
    "recycle", "recycling", "waste", "compost", "garbage", "trash",
    "renewable", "solar", "wind", "energy", "emissions",
    "forest", "tree", "trees", "wildlife", "animals",
    "biodiversity", "nature", "natural", "species",
    "water", "ocean", "sea", "river", "clean", "green",
    "resources", "resource", "conservation", "protect", "protection",
    "planet", "earth", 
    # Added based on scraping keywords
    "ecosystem", "habitat", "marine", "global warming", "air", "circular",
    "consumption", "production", "economy", "mobility", "finance",
    "urban", "planning", "sdg", "agenda", "development",
    # Social / Equity
    "community", "equality", "equity", "inclusion", "rights", "justice",
    "poverty", "hunger", "wellbeing", "health", "fairness", "volunteer",
    "cooperation", "participation",
}

#: Terms indicative of learning, behavioural change, training,
#: and explicit educational intent.
EDUCATIONAL_TERMS: Set[str] = {
    "learn", "learning", "teach", "teaches", "teaching",
    "training", "practice", "practices",
    "habit", "habits", "behaviour", "behavior",
    "manage", "management",
    "goal", "goals",
    "improve", "improvement", "reduce", "saving", "save",
    "awareness", "educational", "education", "quiz", "quizzes",
    # Added based on scraping keywords (mechanics/cognitive)
    "decision", "making", "problem", "solving", "serious",
}


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def compute_sustainability_score(description: str) -> float:
    """
    Compute a density-based sustainability score in [0, 1] for a game description,
    normalised by the total number of tokens and rounded to 6 decimal places.

    The score reflects how much of the text is composed of:
    - sustainability-related terms
    - educational / behavioural-change terms

    Rules:
    - If the description contains zero sustainability terms OR zero educational terms,
      the score is 0.0.
    - Otherwise:
          sust_ratio = sust_hits / N
          edu_ratio  = edu_hits  / N
          score      = (sust_ratio + edu_ratio) / 2
    - The final score is rounded to 6 decimal places for Excel compatibility.
    """
    if not isinstance(description, str):
        return 0.0

    text = description.lower()
    tokens = re.findall(r"[a-z]+", text)
    if not tokens:
        return 0.0

    N = len(tokens)
    sust_hits = sum(t in SUSTAINABILITY_TERMS for t in tokens)
    edu_hits  = sum(t in EDUCATIONAL_TERMS for t in tokens)

    # The game must express BOTH dimensions to be considered relevant
    if sust_hits == 0 or edu_hits == 0:
        return 0.0

    # Additive Smoothing (Zavorra)
    # We add a constant K to the denominator to penalize very short texts.
    # This prevents short descriptions with 1-2 keywords from getting artificially high scores.
    SMOOTHING_K = 50.0

    sust_ratio = sust_hits / (N + SMOOTHING_K)
    edu_ratio  = edu_hits / (N + SMOOTHING_K)

    score = (sust_ratio + edu_ratio) / 2.0

    # Excel-friendly numeric format
    return round(float(score), 6)



def classify_from_score(score: float, threshold: float) -> str:
    """
    Map a numeric score to a binary label using a fixed threshold.

    Parameters
    ----------
    score : float
        Sustainability_Score in [0, 1].
    threshold : float
        Decision threshold in [0, 1]. Typical values: 0.7 or 0.8.

    Returns
    -------
    str
        "YES" if score >= threshold, "NO" otherwise.
    """
    return "YES" if score >= threshold else "NO"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - infile
        - outfile
        - threshold
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extend the App Store results CSV with a rule-based "
            "sustainability score and classification flag."
        )
    )
    parser.add_argument(
        "--infile",
        "-i",
        default="results_excel_schema.csv",
        help="Input CSV file (default: results_excel_schema.csv)",
    )
    parser.add_argument(
        "--outfile",
        "-o",
        default="results_with_sustainability_flag.csv",
        help="Output CSV file (default: results_with_sustainability_flag.csv)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.02,
        help="Threshold in [0,1] for classifying a game as sustainability-related "
             '(default: 0.02).',
    )

    return parser.parse_args()



def apply_lexicon_analysis(
    infile: str = "results_excel_schema.csv",
    outfile: str = "results_with_sustainability_flag.csv",
    threshold: float = 0.02
) -> None:
    """
    Entry point for the sustainability classification pipeline.
    """
    try:
        # Robust loading to handle potential semi-colon
        df = pd.read_csv(infile, sep=None, engine='python', encoding="utf-8")
    except FileNotFoundError:
        try:
             # Fallback
             df = pd.read_csv(infile, sep=";", encoding="utf-8")
        except:
             print(f"Error: Input file '{infile}' not found or unreadable.")
             return

    if "Description" not in df.columns:
        print(f"Error: Input CSV must contain a 'Description' column. Found: {df.columns}")
        return

    # Compute scores row-wise
    scores = df["Description"].fillna("").apply(compute_sustainability_score)
    df["Sustainability_Score"] = scores

    # Apply threshold to obtain binary label
    df["Sustainability_Game_Rule"] = scores.apply(
        lambda s: classify_from_score(s, threshold=threshold)
    )

    df.to_csv(
        outfile,
        index=False,
        encoding="utf-8",
        sep=";", 
        decimal=",", 
    )
    print(f"Sustainability analysis complete. Saved to {outfile}")


def main() -> None:
    # UPDATED DEFAULT FOR VALIDATION
    # We want to re-score the file that has LLM judgements to check improvements.
    infile = "results_with_llm_judge.csv"
    outfile = "results_rescored.csv"
    
    print(f"Re-scoring {infile} -> {outfile} with UPDATED LEXICON...")
    apply_lexicon_analysis(infile, outfile, threshold=0.015)


if __name__ == "__main__":
    main()

