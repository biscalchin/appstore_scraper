# App Store Scraper: Experiential Replication Guide

This guide provides the necessary instructions and details to replicate the experiments described in the study. The pipeline extracts, classifies, and evaluates sustainability-related applications from the Apple App Store.

## Directory Structure
- `src/`: Python source code scripts for scraping, classification, and analysis.
- `config/`: Configuration files, including keyword lists and category mappings.
- `data/`: Extracted datasets and CSV results from the pipeline.
- `output/`: Generated evaluation plots, sub-evaluations, and NLP extraction results.
- `private/`: Standalone system configuration and verification scripts.

## 1. Prerequisites and Setup

### 1.1 Environment
The scripts are written in Python 3. It is recommended to use at least Python 3.9.

### 1.2 Dependencies
You will need to install the following core Python libraries to execute the entire pipeline:
```bash
pip install pandas numpy scikit-learn matplotlib python-dotenv nltk requests
```
*Note: The LLM Judge (`src/llm_judge.py`) relies on `datapizza` for interactions with the Google Gemini API. Make sure to have a valid `GOOGLE_API_KEY` in a `.env` file at the root of the directory.*

## 2. Experimental Pipeline 

The experiment is divided into four main phases:

### Phase 1: Data Acquisition (Scraping)
This step queries the Apple App Store API using a list of seed keywords and predefined categories, saving the results in a structured format.

**Command:**
```bash
python src/scraper.py --keywords config/keywords.txt --categories config/categories.json --output data/results_excel_schema.csv
```

**Inputs:**
*   `config/keywords.txt`: A file containing the seed keywords (e.g., sustainability, ecology, zero waste).
*   `config/categories.json`: A mapping of Apple App Store genre IDs to category names for filtering.

**Outputs:**
*   `data/results_excel_schema.csv`: The metadata of all retrieved apps.
*   `data/search_process_log.csv` & `data/query_stats.csv`: Logs and query statistics.

---

### Phase 2: Classification & Scoring (Lexicon Analysis)
This phase applies a rules-based classification algorithm. It scans the app descriptions against curated lexicons of environmental and educational terms to compute a "Sustainability Score" between 0 and 1.

**Command:**
```bash
# Executed automatically by scraper.py if pipeline completes. 
# Standalone usage involves reading the raw CSV and scoring:
python src/lexicon.py 
```

**Inputs:**
*   `data/results_excel_schema.csv`

**Outputs:**
*   `data/results_with_sustainability.csv` (or similar, depending on lexicon expansion name like `results_expanded_lexicon.csv`). This file includes the assigned `Sustainability_Score`.

---

### Phase 3: NLP Analysis (Optional for Replication)
This phase analyzes the language used in descriptions, removing buzzwords, extracting unigrams/bigrams, and plotting term frequencies.

**Command:**
```bash
python src/NLP.py
```
**Inputs:**
*   `data/results_with_sustainability.csv`

**Outputs:**
*   Data saved to the `output/description_nlp_analysis` directory, including frequency CSV files and visualizations (bar charts for terms).

---

### Phase 4: Validation and Ranking Analysis (LLM Judge)
To validate the effectiveness of the lexicon-based score, an LLM evaluation phase is conducted using `gemini-2.0-flash`. The LLM classifies the applications as "Directly Educational" or "Indirectly Educational" based on the app descriptions. Finally, standard Information Retrieval ranking metrics are computed.

**Step 4.1: LLM Inference**
Evaluates the descriptions and appends the ground truth labels.
```bash
python src/llm_judge.py --infile data/results_expanded_lexicon.csv --outfile data/results_with_llm_judge.csv
```
*   *Note: Ensure `.env` is properly configured with your API key.*

**Step 4.2: Ranking Evaluation**
Treats the rules-based score (`Sustainability_Score`) as the prediction mechanism and evaluates it against the LLM's classification as ground truth.
```bash
python src/ranking_analysis.py data/results_with_llm_judge.csv output/validation_v3
```

**Outputs from Validation:**
*   `output/validation_v3/ranking_at_k.csv` (Precision@K, Recall@K)
*   `output/validation_v3/ndcg_at_k.csv` (NDCG@K for graded relevance)
*   `output/validation_v3/paper_summary.csv` (Summary metrics like MAP, AUPRC)
*   `output/validation_v3/precision_at_k.png` ... (Visualizations)

## 3. Data Availability

For replication without re-running the APIs (which may yield distinct results due to store updates or API quotas), the following key datasets belong to the repository:

1.  **`data/results_excel_schema.csv`**: The raw output of Phase 1.
2.  **`data/results_expanded_lexicon.csv`**: The output of Phase 2 containing the rule-based sustainability scores.
3.  **`data/results_with_llm_judge.csv`**: The output of Phase 4 containing the LLM-derived ground truth labels (`LLM_Directly`, `LLM_Indirectly`). 

Researchers should start from `data/results_with_llm_judge.csv` and run the script `src/ranking_analysis.py` to directly replicate the validation results.
