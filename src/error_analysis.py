
import pandas as pd
import re
from collections import Counter
import os

# --- CONFIGURATION ---
INPUT_FILE = "results_with_llm_judge.csv"
OUTPUT_DIR = "validation"
THRESHOLD = 0.015  # Optimal threshold from previous analysis

# Import existing lexicon to check against
try:
    from lexicon import SUSTAINABILITY_TERMS, EDUCATIONAL_TERMS
    LEXICON = SUSTAINABILITY_TERMS.union(EDUCATIONAL_TERMS)
except ImportError:
    # Fallback if lexicon.py isn't in path or import fails
    print("Warning: Could not import lexicon.py. Using empty lexicon set.")
    LEXICON = set()

# Basic Stopwords for FN analysis (simplified)
STOPWORDS = {
    "the", "and", "to", "of", "a", "in", "is", "for", "with", "you", "your", "that", "on", "are", 
    "this", "game", "app", "play", "can", "as", "be", "it", "features", "new", "world", "from",
    "all", "have", "by", "more", "will", "or", "an", "one", "time", "level", "levels", "mode",
    "best", "fun", "free", "get", "use", "make", "up", "now", "how", "games", "players", "friends",
    "download", "enjoy", "experience", "graphics", "gameplay", "challenge", "puzzle", "action",
    "simulation", "simulator", "real", "realistic", "control", "drive", "driving", "car", "truck",
    "city", "build", "building", "create", "design", "style", "easy", "simple", "mobile", "android",
    "ios", "device", "support", "please", "rate", "us", "contact", "email", "support", "help",
    "version", "update", "updated", "bug", "fixes", "fixed", "improvements", "improved", "performance",
    "optimized", "optimization", "user", "interface", "ui", "ux", "system", "setting", "settings",
    "option", "options", "menu", "screen", "button", "buttons", "touch", "tap", "swipe", "click",
    "controller", "controls", "support", "language", "languages", "english", "french", "german",
    "italian", "spanish", "portuguese", "russian", "chinese", "japanese", "korean", "hindi", "arabic",
    "turkish", "indonesian", "thai", "vietnamese", "malay", "filipino", "polish", "dutch", "swedish",
    "danish", "norwegian", "finnish", "hungarian", "czech", "slovak", "romanian", "bulgarian", "greek"
}

def clean_and_tokenize(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    # Simple tokenization: alpha characters only
    tokens = re.findall(r"[a-z]+", text)
    return tokens

def analyze_false_positives(df):
    """
    Finds keywords from our lexicon that appear in Non-Sustainable apps with high scores.
    These are likely 'noisy' words.
    """
    # Filter: Non-Sustainable (NO-NO) BUT Score >= Threshold
    fp_df = df[
        (df["LLM_Directly"] == "NO") & 
        (df["LLM_Indirectly"] == "NO") & 
        (df["Sustainability_Score"] >= THRESHOLD)
    ]
    
    print(f"Found {len(fp_df)} False Positives.")
    if len(fp_df) == 0:
        return Counter()

    noisy_keywords = []
    
    for desc in fp_df["Description"]:
        tokens = clean_and_tokenize(desc)
        # Check which LEXICON words are present
        found = [t for t in tokens if t in LEXICON]
        noisy_keywords.extend(found)
        
    return Counter(noisy_keywords)

def analyze_false_negatives(df):
    """
    Finds frequent non-lexicon words in Strongly Sustainable apps with low scores.
    These are likely missing keywords.
    """
    # Filter: Strongly Sustainable (YES-?) BUT Score < Threshold
    # We prioritize "Directly YES" as they should definitely be caught.
    fn_df = df[
        (df["LLM_Directly"] == "YES") & 
        (df["Sustainability_Score"] < THRESHOLD)
    ]
    
    print(f"Found {len(fn_df)} False Negatives (Directly Educative but Missed).")
    if len(fn_df) == 0:
        return Counter()

    potential_keywords = []
    
    for desc in fn_df["Description"]:
        tokens = clean_and_tokenize(desc)
        # Check words NOT in lexicon and NOT in stopwords
        candidates = [
            t for t in tokens 
            if t not in LEXICON and t not in STOPWORDS and len(t) > 3
        ]
        potential_keywords.extend(candidates)
        
    return Counter(potential_keywords)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load Data
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, sep=";", encoding="utf-8")
        if "LLM_Directly" not in df.columns:
             df = pd.read_csv(INPUT_FILE, sep=",", encoding="utf-8")
             
        # Normalize Score
        if df["Sustainability_Score"].dtype == object:
            df["Sustainability_Score"] = df["Sustainability_Score"].astype(str).str.replace(",", ".").replace("nan", "0")
        df["Sustainability_Score"] = pd.to_numeric(df["Sustainability_Score"], errors='coerce').fillna(0)
             
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Run Analysis
    fp_counts = analyze_false_positives(df)
    fn_counts = analyze_false_negatives(df)
    
    # Save Report
    report_path = os.path.join(OUTPUT_DIR, "error_analysis_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"--- Error Analysis Report ---\n")
        f.write(f"Threshold Used: {THRESHOLD}\n\n")
        
        f.write(f"=== FALSE POSITIVES (Noise Analysis) ===\n")
        f.write(f"Top lexicon words found in 'Not Sustainable' apps (Count):\n")
        for word, count in fp_counts.most_common(20):
            f.write(f"- {word}: {count}\n")
        
        f.write(f"\n=== FALSE NEGATIVES (Gap Analysis) ===\n")
        f.write(f"Top frequent terms in 'Directly Sustainable' apps that we missed (Count):\n")
        for word, count in fn_counts.most_common(30):
            f.write(f"- {word}: {count}\n")
            
    print(f"Analysis complete. Report saved to {report_path}")
    
    # Print preview to console
    print("\n[Preview] Top Noisy Words (Candidates for removal/weight-reduction):")
    print(fp_counts.most_common(10))
    print("\n[Preview] Top Missing Words (Candidates for addition):")
    print(fn_counts.most_common(10))

if __name__ == "__main__":
    main()
