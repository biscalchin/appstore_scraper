import os
import re
import json
import csv
import string
from pathlib import Path
from collections import Counter
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is available
import math
import statistics

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def get_academic_stopwords():
    return {
        "et", "al", "fig", "figure", "table", "doi", "vol", "pp", "journal", 
        "university", "press", "conference", "proceedings", "abstract", 
        "introduction", "conclusion", "methodology", "results", "discussion", 
        "references", "cite", "citation", "author", "authors", "published", 
        "copyright", "license", "available", "data", "analysis", "study", 
        "research", "paper", "article", "review", "section", "chapter",
        "based", "using", "used", "approach", "case", "new", "also", "within",
        "however", "therefore", "thus", "results", "shown", "see", "can", "may",
        "result", "method", "effect", "sample", "participants", "framework", "week", "weeks",
        "month", "months", "year", "years", "day", "days", "hour", "hours", "minute", "minutes", "second", "seconds",
        "transformative", "httpsdoiorgsu", "motivation", "might", "drive", "finding", "round", "effectiveness",
        "publication", "public", "janakiraman", "httpsdoiorg", "trend", "chen", "souza",
        # EDITORIA / FORMAT
        "issue", "issues", "volume", "number", "no", "editor",
        "publisher", "elsevier", "springer", "wiley", "mdpi",
        "switzerland", "basel",  # dai metadata di Sustainability (Switzerland)
        "online", "access", "open", "accessed", "retrieved",

        # LINGUA ACCADEMICA GENERICA
        "overview", "background", "state", "art",
        "literature", "gap", "gaps", "field", "fields",
        "area", "areas", "domain", "domains",

        # NOMI PROPRI che ti sporcano le keyword (esempi trovati nel corpus)
        "janakiraman", "souza", "chen",

        # URL / DOI spezzati
        "https", "org", "com", "pdf",
        "motivational", "future", "user", "kim", "tion", "http", 
        "wang", "van", "sus", "bibliometric", "remains", "httpsdoi", 
        "prior", "gameplay", "watson", "esd", "provided", "playing",
        "tainability", "learn", "core", "crucial", "five",
                # === CONCETTI TROPPO GENERICI NEL TUO CORPUS ===
        "behavior", "behaviors", "behaviour", "behaviours",
        "behavioral",
        "cognitive",
        "dimension", "dimensions",
        "topic", "topics",
        "intervention", "interventions",
        "conceptual",
        "indicator", "indicators",
        "across",
        "exploring", "exploration",
        "retention",
        "targeted",
        "environmentally", "environ",

        "learner", "learners",
        "experience", "experiences",
        "construct", "constructs",
        "key", "keys",
        "component", "components",
        "factor", "factors",
        "affective",
        "industry", "industrial",
        "structural",
        "measurement", "measurements",
        "scenario", "scenarios",
        "transformation", "transformations",
        "comprehensive",
        "board", "boards",
        "collective",
        "enhance", "enhancing", "enhanced",
        "understanding",

        "mapping",
        "international",
        "fostering",
        "decisionmaking",
        "measure", "measures",
        "degree", "degrees",
        "tool", "tools",
        "management",
        "qualitative",
        "pedagogical",
        "competence", "competences", "competency", "competencies",
        "helped",
        "pattern", "patterns",
        "sustain", "sustained",
        "science",
        "continued",
        "requires", "required", "require",

        "relatedness",
        "least",
        "identified",
        "stakeholder", "stakeholders",
        "showed", "shown",
        "would",
        "perceived", "perception",
        "clear", "clearly",
        "test", "tests", "testing",
        "strategy", "strategies",
        "instance",
        "learned",
        "path", "paths",
        "stage", "stages",
        "focus", "focused", "focusing",

        "indicated", "indicates",
        "humancomputer",
        "emotion", "emotions",
        "uncertainty",
        "impactful",
        "good", "better", "best",
        "thing", "things",
        "people",
        "experiential",

        "suggests", "suggested",
        "psychological",
        "examine", "examines", "examined",
        "produce", "produces", "produced",
        "integration",
        "although",
        "designing",
        "greater",
        "next",

        "complexity",
        "interaction", "interactions",
        "investigated", "investigate", "investigating",
        "reveals", "revealed",
        "practical",
        "reviewed",
        "difference", "differences",
        "space", "spaces",
        "item", "items",

        # NOMI PROPRI / TITOLI DI GIOCHI NON UTILI COME SEED
        "enercities"

    }

def get_domain_generic_stopwords():
    return {
        "game", "games", "gaming", "player", "learning", "education", 
        "teaching", "teacher", "students", "course", "system", "platform", 
        "application", "app", "serious", "gamification", "digital",
        "sustainability", "sustainable", "development", "design", "model", 
        "process", "project", "impact", "social", "environmental", "economic", 
        "value", "change", "challenge", "challenges", "issue", "issues", 
        "need", "needs", "different", "important", "example", "context", 
        "level", "levels"
    }

def get_generic_exclusions():
    # Kept for backward compatibility with process_pdfs initial filtering if needed,
    # but we will rely on the new lists for the advanced step.
    # Merging the old list into the new structure effectively.
    return get_domain_generic_stopwords()

def normalize_text(text):
    # Lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and symbols
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_pdfs(papers_dir):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    academic_stops = get_academic_stopwords()
    generic_exclusions = get_generic_exclusions()
    
    # Combine all exclusions for the initial pass
    all_exclusions = stop_words.union(academic_stops).union(generic_exclusions)
    
    all_words = []
    doc_freq = Counter()
    
    papers_path = Path(papers_dir)
    if not papers_path.exists():
        print(f"Directory not found: {papers_dir}")
        return Counter(), Counter(), 0

    pdf_files = list(papers_path.glob("*.pdf"))
    n_docs = len(pdf_files)
    print(f"Found {n_docs} PDF files in {papers_dir}")

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        try:
            text_content = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + " "
            
            if not text_content:
                print(f"  Warning: No text extracted from {pdf_file.name}")
                continue

            # Normalize
            normalized_text = normalize_text(text_content)
            
            # Tokenize
            tokens = nltk.word_tokenize(normalized_text)
            
            doc_lemmas = set()
            
            # Process tokens
            for token in tokens:
                # Lemmatize first
                lemma = lemmatizer.lemmatize(token)
                
                # Filter
                if len(lemma) < 3:
                    continue
                if lemma in all_exclusions:
                    continue
                
                all_words.append(lemma)
                doc_lemmas.add(lemma)
            
            # Update Document Frequency
            doc_freq.update(doc_lemmas)
                
        except Exception as e:
            print(f"  Error processing {pdf_file.name}: {e}")

    return Counter(all_words), doc_freq, n_docs

def save_results(word_counts, output_dir):
    output_path = Path(output_dir)
    
    # 1. keywords.json
    sorted_dict = dict(word_counts.most_common())
    with open(output_path / "keywords.json", "w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, indent=4)
    
    # 2. keywords.csv
    with open(output_path / "keywords.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "count"])
        for word, count in word_counts.most_common():
            if count <= 3:
                continue
            writer.writerow([word, count])
            
    # 3. keywords_cleaned.txt
    with open(output_path / "keywords_cleaned.txt", "w", encoding="utf-8") as f:
        for word, _ in word_counts.most_common():
            f.write(f"{word}\n")

    print(f"Basic results saved to {output_path.resolve()}")

def advanced_analysis(base_dir, doc_freq, n_docs):
    print("\nStarting Advanced Analysis...")
    keywords_csv = base_dir / "keywords.csv"
    if not keywords_csv.exists():
        print("keywords.csv not found, skipping advanced analysis.")
        return

    # 1. Read keywords.csv and normalize/lemmatize again
    print("Reading and normalizing keywords.csv...")
    lemmatizer = WordNetLemmatizer()
    refined_counts = Counter()
    
    with open(keywords_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = row["keyword"]
            count = int(row["count"])
            
            # Normalize
            term = normalize_text(term)
            # Filter non-alpha
            if not term.isalpha():
                continue
            if len(term) < 3:
                continue
                
            # Lemmatize
            lemma = lemmatizer.lemmatize(term)
            refined_counts[lemma] += count

    # 2. Prepare Filters
    stop_words = set(stopwords.words('english'))
    academic_stops = get_academic_stopwords()
    domain_stops = get_domain_generic_stopwords()
    total_stopwords = stop_words.union(academic_stops).union(domain_stops)

    MIN_TF = 3
    MIN_DF_RATIO = 0.3
    
    filtered_keywords = []
    excluded_keywords = []
    
    # 3. Apply Filters
    print("Applying filters...")
    valid_tfidfs = []
    
    # First pass to collect valid items for TF-IDF threshold calculation
    temp_candidates = []
    
    for lemma, count in refined_counts.items():
        # Stopword Filter
        if lemma in total_stopwords:
            excluded_keywords.append({"keyword": lemma, "reason": "stopword"})
            continue
            
        # Min TF Filter
        if count < MIN_TF:
            excluded_keywords.append({"keyword": lemma, "reason": "low_tf"})
            continue
            
        # Min DF Ratio Filter
        df = doc_freq.get(lemma, 0)
        df_ratio = df / n_docs if n_docs > 0 else 0
        
        if df_ratio < MIN_DF_RATIO:
            excluded_keywords.append({"keyword": lemma, "reason": "low_df"})
            continue
            
        # Calculate TF-IDF
        # idf = log(N / df)
        idf = math.log(n_docs / df) if df > 0 else 0
        tfidf = count * idf
        
        temp_candidates.append({
            "keyword": lemma,
            "total_count": count,
            "doc_freq": df,
            "df_ratio": df_ratio,
            "tfidf": tfidf
        })
        valid_tfidfs.append(tfidf)

    # 4. TF-IDF Threshold Filter
    if valid_tfidfs:
        min_tfidf_threshold = 10.0
        print(f"TF-IDF Threshold (Fixed): {min_tfidf_threshold}")
        
        for cand in temp_candidates:
            if cand["tfidf"] <= min_tfidf_threshold:
                excluded_keywords.append({"keyword": cand["keyword"], "reason": "low_tfidf"})
            else:
                filtered_keywords.append(cand)
    else:
        print("No keywords survived initial filters.")

    # Sort by TF-IDF descending
    filtered_keywords.sort(key=lambda x: x["tfidf"], reverse=True)

    # 5. Write Outputs
    out_filtered = base_dir / "keywords_filtered.csv"
    with open(out_filtered, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "total_count", "doc_freq", "df_ratio", "tfidf"])
        for item in filtered_keywords:
            writer.writerow([
                item["keyword"], 
                item["total_count"], 
                item["doc_freq"], 
                f"{item['df_ratio']:.2f}", 
                f"{item['tfidf']:.4f}"
            ])
            
    out_excluded = base_dir / "keywords_excluded.csv"
    with open(out_excluded, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["keyword", "reason"])
        for item in excluded_keywords:
            writer.writerow([item["keyword"], item["reason"]])

    print(f"Advanced analysis saved to:\n- {out_filtered}\n- {out_excluded}")

def main():
    base_dir = Path(__file__).parent
    papers_dir = base_dir / "papers"
    
    print("Starting keyword extraction...")
    word_counts, doc_freq, n_docs = process_pdfs(papers_dir)
    
    if word_counts:
        save_results(word_counts, base_dir)
        advanced_analysis(base_dir, doc_freq, n_docs)
        print("Done.")
    else:
        print("No keywords extracted.")

if __name__ == "__main__":
    main()
