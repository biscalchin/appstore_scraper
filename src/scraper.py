import csv
import json
import time
import urllib.parse
import requests
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Set

# Import local modules
try:
    import NLP
    import lexicon
    import stats_collection
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# === Constants ===
BASE_DIR = Path(__file__).resolve().parent
BASE_URL = "https://itunes.apple.com/search"
DEFAULT_ENTITY = "software"
DEFAULT_LIMIT = 200
REQUIRED_LANGS = ["EN"]

EXCEL_COLUMNS = [
    "Title","App ID","Description","Year","Developer/Publisher","Platform","Language (English)","Language (Others)",
    "Cost/Business Model","Release Status","Target Age","Declared / Inferred","PEGI/ESRB",
    "Età Minima Dichiarata","Educational Dimension (Knowledge/Attitude/Behavior)",
    "Environmental Dimension","Social Dimension","Economic Dimension",
    "SDG 1","SDG 2","SDG 3","SDG 4","SDG 5","SDG 6","SDG 7","SDG 8","SDG 9","SDG 10","SDG 11","SDG 12","SDG 13","SDG 14","SDG 15","SDG 16","SDG 17",
    "Genre","Narrative/Setting","Complexity (1–3)","Game Mode (Single/Multi/Co-op)","Replayability","Session Length","Reward Structure",
    "Clarity And Credibility (Engagement)","Immersion & Narrative (Engagement)","Progression & Rewards (Engagement)","Learning & Feedback (Engagement)","Systems & Social (Engagement)",
    "Credibility of Content (0–3)","Educational Intent (Declared)","Playtest Notes/DiGAP Data",
    "Query Keyword","Query Subgenre","Store Link/Reference"
]

GENRE_MAP = {
    "Simulation": "Simulation",
    "Strategy": "Strategy",
    "Puzzle": "Puzzle",
    "Role Playing": "RPG",
    "Adventure": "Adventure",
}

# === UI Helpers ===
class UI:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def print_header(text):
        print(f"\n{UI.HEADER}{UI.BOLD}=== {text} ==={UI.ENDC}")

    @staticmethod
    def print_step(step_num, total_steps, text):
        print(f"\n{UI.OKCYAN}[Step {step_num}/{total_steps}] {text}{UI.ENDC}")

    @staticmethod
    def print_success(text):
        print(f"{UI.OKGREEN}✔ {text}{UI.ENDC}")

    @staticmethod
    def print_info(text):
        print(f"{UI.OKBLUE}ℹ {text}{UI.ENDC}")

    @staticmethod
    def print_warning(text):
        print(f"{UI.WARNING}⚠ {text}{UI.ENDC}")

    @staticmethod
    def print_error(text):
        print(f"{UI.FAIL}✖ {text}{UI.ENDC}")

# === Helper Functions ===

def load_categories(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_keywords(path: Path):
    kws = []
    if not path.exists():
        return kws
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        kws.append(line)
    return kws

def build_queries(keywords, genre_ids, entity=DEFAULT_ENTITY, limit=DEFAULT_LIMIT):
    queries = []
    for kw in keywords:
        if genre_ids:
            for gid in genre_ids:
                queries.append({
                    "term": kw,
                    "entity": entity,
                    "genreId": str(gid),
                    "limit": str(limit)
                })
        else:
            queries.append({
                "term": kw,
                "entity": entity,
                "limit": str(limit)
            })
    return queries

def call_api(params, backoff=0.3, retries=3):
    q = "?" + urllib.parse.urlencode(params)
    url = BASE_URL + q
    last_exc = None
    for _ in range(retries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json().get("results", [])
            else:
                last_exc = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_exc = e
        time.sleep(backoff)
        backoff *= 2
    UI.print_warning(f"API Call Failed: {url} — {last_exc}")
    return []

def has_required_language(item, required_langs):
    langs = item.get("languageCodesISO2A") or []
    langs_up = {str(x).upper() for x in langs}
    for req in required_langs:
        if req.upper() in langs_up:
            return True
    return False

def extract_year(item):
    dt = item.get("currentVersionReleaseDate") or item.get("releaseDate")
    if not dt:
        return ""
    try:
        return str(datetime.fromisoformat(dt.replace("Z","+00:00")).year)
    except Exception:
        try:
            return dt[:4]
        except Exception:
            return ""

def map_genre(item):
    genres = item.get("genres") or []
    sub = None
    for g in genres:
        if g and g != "Games":
            sub = g
            break
    if not sub:
        return ""
    return GENRE_MAP.get(sub, "Other")

def cost_business_model(item):
    price = item.get("price")
    formatted = (item.get("formattedPrice") or "").strip().lower()
    if price is not None:
        try:
            if float(price) == 0.0:
                return "Free"
            else:
                return "Premium (One-time purchase)"
        except Exception:
            pass
    if "free" in formatted or "gratis" in formatted:
        return "Free"
    return "Premium (One-time purchase)"

def min_age_declared(item):
    rat = item.get("contentAdvisoryRating")
    return rat or ""

def subgenre_from_query(genre_id, cats):
    if not genre_id:
        return ""
    if genre_id == "6014":
        return "Games (primary)"
    name = (cats.get("game_subgenres") or {}).get(str(genre_id))
    if name:
        return name
    return (cats.get("primary_genres") or {}).get(str(genre_id), "")

def normalize_row(item, q_term, q_genre_name):
    title = item.get("trackName") or ""
    description = item.get("description") or ""
    year = extract_year(item)

    artist = item.get("artistName")
    seller = item.get("sellerName")
    if artist and seller and artist != seller:
        dev_pub = f"{artist} / {seller}"
    else:
        dev_pub = artist or seller or ""

    platform = "Apple Store"
    langs = item.get("languageCodesISO2A") or []
    has_en = "TRUE" if "EN" in [str(x).upper() for x in langs] else "FALSE"
    others = [x for x in langs if str(x).upper() != "EN"]
    lang_others = ",".join(others)

    cost = cost_business_model(item)
    min_age = min_age_declared(item)
    genre = map_genre(item)
    url = item.get("trackViewUrl") or ""

    row = {
        "Title": title,
        "App ID": item.get("trackId") or "",
        "Description": description,
        "Year": year,
        "Developer/Publisher": dev_pub,
        "Platform": platform,
        "Language (English)": has_en,
        "Language (Others)": lang_others,
        "Cost/Business Model": cost,
        "Release Status": "",
        "Target Age": "",
        "Declared / Inferred": "",
        "PEGI/ESRB": "",
        "Età Minima Dichiarata": min_age,
        "Educational Dimension (Knowledge/Attitude/Behavior)": "",
        "Environmental Dimension": "",
        "Social Dimension": "",
        "Economic Dimension": "",
        "SDG 1": "","SDG 2": "","SDG 3": "","SDG 4": "","SDG 5": "","SDG 6": "","SDG 7": "","SDG 8": "","SDG 9": "","SDG 10": "","SDG 11": "","SDG 12": "","SDG 13": "","SDG 14": "","SDG 15": "","SDG 16": "","SDG 17": "",
        "Genre": genre,
        "Narrative/Setting": "",
        "Complexity (1–3)": "",
        "Game Mode (Single/Multi/Co-op)": "",
        "Replayability": "",
        "Session Length": "",
        "Reward Structure": "",
        "Clarity And Credibility (Engagement)": "",
        "Immersion & Narrative (Engagement)": "",
        "Progression & Rewards (Engagement)": "",
        "Learning & Feedback (Engagement)": "",
        "Systems & Social (Engagement)": "",
        "Credibility of Content (0–3)": "",
        "Educational Intent (Declared)": "",
        "Playtest Notes/DiGAP Data": "",
        "Query Keyword": q_term or "",
        "Query Subgenre": q_genre_name or "",
        "Store Link/Reference": url
    }
    return row

# === Main Orchestrator ===

def run_pipeline(
    categories_path="categories.json",
    keywords_path="keywords_example.txt",
    out_csv="results_excel_schema.csv",
    stats_csv="query_stats.csv",
    process_csv="search_process_log.csv",
    genre_scope="games_only",
    required_langs=None,
    skip_scraping=False
):
    UI.print_header("APPSTORE SCRAPER PIPELINE")
    
    if required_langs is None:
        required_langs = REQUIRED_LANGS

    # --- Step 1: Setup ---
    UI.print_step(1, 4, "Setup and Configuration")
    try:
        cats = load_categories(Path(categories_path))
        UI.print_info(f"Loaded categories from {categories_path}")
        
        keywords = load_keywords(Path(keywords_path))
        UI.print_info(f"Loaded {len(keywords)} keywords from {keywords_path}")
        
        if genre_scope == "games_only":
            genre_ids = ["6014"] + list(cats.get("game_subgenres", {}).keys())
        elif genre_scope == "all":
            genre_ids = []
        else:
            genre_ids = [g.strip() for g in genre_scope.split(",") if g.strip()]
            
        queries = build_queries(keywords, genre_ids)
        UI.print_info(f"Built {len(queries)} queries to execute.")
        
    except Exception as e:
        UI.print_error(f"Setup failed: {e}")
        return

    # --- Step 2: Scraping ---
    UI.print_step(2, 4, "Scraping App Store")
    
    if skip_scraping:
        UI.print_warning("Skipping scraping as requested. Using existing data.")
    else:
        seen = set()
        rows = []
        stats_rows = []
        process_rows = []
        
        start_time = time.time()
        
        for i, q in enumerate(queries, 1):
            q_term = q.get("term")
            q_genre_id = q.get("genreId", "")
            q_genre_name = subgenre_from_query(q_genre_id, cats)
            
            print(f"\rProcessing query {i}/{len(queries)}: '{q_term}' in '{q_genre_name}'...", end="", flush=True)
            
            results = call_api(q)
            
            raw_count = len(results)
            passed_lang_count = 0
            passed_primary_games_count = 0
            uniques_added = 0
            dups_skipped = 0
            
            for rank, res in enumerate(results, start=1):
                app_id = res.get("trackId")
                title = res.get("trackName") or ""
                
                passed_lang = has_required_language(res, required_langs)
                passed_primary_games = False
                is_duplicate = False
                included = False
                
                if passed_lang:
                    passed_lang_count += 1
                    if genre_ids and "6014" in genre_ids:
                        if str(res.get("primaryGenreId")) == "6014":
                            passed_primary_games = True
                            passed_primary_games_count += 1
                    else:
                        passed_primary_games = True
                        passed_primary_games_count += 1
                
                if passed_lang and passed_primary_games:
                    key = (app_id, title)
                    if key in seen:
                        is_duplicate = True
                        dups_skipped += 1
                    else:
                        seen.add(key)
                        row = normalize_row(res, q_term, q_genre_name)
                        rows.append(row)
                        uniques_added += 1
                        included = True
                
                process_rows.append({
                    "Query Keyword": q_term or "",
                    "GenreId": q_genre_id or "",
                    "Query Subgenre": q_genre_name or "",
                    "Result Rank (1-based)": rank,
                    "App ID": app_id or "",
                    "Title": title,
                    "Passed Language Filter": "TRUE" if passed_lang else "FALSE",
                    "Passed PrimaryGenre Filter (Games)": "TRUE" if passed_primary_games else "FALSE",
                    "Is Duplicate (global)": "TRUE" if is_duplicate else "FALSE",
                    "Included In Final Dataset": "TRUE" if included else "FALSE"
                })
            
            stats_rows.append(stats_collection.compute_query_stats(
                q_term, q_genre_id, q_genre_name, raw_count, passed_lang_count, 
                passed_primary_games_count, uniques_added, dups_skipped
            ))
            
            time.sleep(0.1) # Polite delay

        print() # Newline after progress
        elapsed = time.time() - start_time
        UI.print_success(f"Scraping completed in {elapsed:.2f}s")
        UI.print_info(f"Collected {len(rows)} unique records.")
        
        # Save results
        UI.print_info(f"Saving raw results to {out_csv}...")
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=EXCEL_COLUMNS)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        
        # Save stats
        UI.print_info(f"Saving statistics to {stats_csv}...")
        stats_collection.save_query_stats(stats_rows, stats_csv)
        
        # Save logs
        UI.print_info(f"Saving process logs to {process_csv}...")
        stats_collection.save_process_log(process_rows, process_csv)

    # --- Step 3: Lexicon Analysis ---
    UI.print_step(3, 4, "Lexicon & Sustainability Analysis")
    sustainability_csv = out_csv.replace(".csv", "_with_sustainability.csv")
    try:
        lexicon.apply_lexicon_analysis(
            infile=out_csv,
            outfile=sustainability_csv,
            threshold=0.02
        )
        UI.print_success(f"Sustainability analysis completed. Output: {sustainability_csv}")
    except Exception as e:
        UI.print_error(f"Lexicon analysis failed: {e}")

    # --- Step 4: NLP Analysis ---
    UI.print_step(4, 4, "NLP Analysis")
    nlp_out_dir = "description_nlp_analysis"
    try:
        NLP.run_nlp_analysis(
            results_path=out_csv,
            out_dir=nlp_out_dir
        )
        UI.print_success(f"NLP analysis completed. Output directory: {nlp_out_dir}")
    except Exception as e:
        UI.print_error(f"NLP analysis failed: {e}")

    UI.print_header("PIPELINE COMPLETE")

def main():
    parser = argparse.ArgumentParser(description="AppStore Scraper Orchestrator")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip scraping and run analysis on existing file")
    parser.add_argument("--keywords", default=str(BASE_DIR / "keywords_example.txt"), help="Path to keywords file")
    parser.add_argument("--output", default=str(BASE_DIR / "results_excel_schema.csv"), help="Output CSV file")
    parser.add_argument("--categories", default=str(BASE_DIR / "categories.json"), help="Path to categories file")
    
    args = parser.parse_args()
    
    run_pipeline(
        categories_path=args.categories,
        keywords_path=args.keywords,
        out_csv=args.output,
        skip_scraping=args.skip_scraping
    )

if __name__ == "__main__":
    main()
