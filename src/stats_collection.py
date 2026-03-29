import csv
from pathlib import Path
from typing import List, Dict, Any

# === Statistiche per query (una riga per ogni keyword × genreId) ===
QUERY_STATS_COLUMNS = [
    "Query Keyword","GenreId","Query Subgenre","API Results (raw)",
    "Passed Language Filter","Passed PrimaryGenre Filter (Games)",
    "Unique Added","Duplicates Skipped"
]

# === Log dettagliato del processo (una riga per ogni risultato grezzo) ===
PROCESS_LOG_COLUMNS = [
    "Query Keyword","GenreId","Query Subgenre",
    "Result Rank (1-based)",
    "App ID","Title",
    "Passed Language Filter","Passed PrimaryGenre Filter (Games)",
    "Is Duplicate (global)","Included In Final Dataset"
]

def save_query_stats(stats_rows: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save query statistics to a CSV file.
    """
    file_exists = Path(output_file).exists()
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=QUERY_STATS_COLUMNS)
        writer.writeheader()
        for r in stats_rows:
            writer.writerow(r)
    print(f"Saved {len(stats_rows)} query stats to {output_file}")

def save_process_log(process_rows: List[Dict[str, Any]], output_file: str) -> None:
    """
    Save process log to a CSV file.
    """
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PROCESS_LOG_COLUMNS)
        writer.writeheader()
        for r in process_rows:
            writer.writerow(r)
    print(f"Saved {len(process_rows)} process log entries to {output_file}")

def compute_query_stats(
    q_term: str,
    q_genre_id: str,
    q_genre_name: str,
    raw_count: int,
    passed_lang_count: int,
    passed_primary_games_count: int,
    uniques_added: int,
    dups_skipped: int
) -> Dict[str, Any]:
    """
    Helper to construct a stats dictionary row.
    """
    return {
        "Query Keyword": q_term or "",
        "GenreId": q_genre_id or "",
        "Query Subgenre": q_genre_name or "",
        "API Results (raw)": raw_count,
        "Passed Language Filter": passed_lang_count,
        "Passed PrimaryGenre Filter (Games)": passed_primary_games_count,
        "Unique Added": uniques_added,
        "Duplicates Skipped": dups_skipped
    }
