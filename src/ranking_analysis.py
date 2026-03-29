import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt

def load_and_clean_data(filepath):
    """Loads CSV, cleans, and handles semantic labelling."""
    print(f"Loading data from {filepath}...")
    
    try:
        # Try reading with semicolon separator first
        df = pd.read_csv(filepath, sep=";", encoding="utf-8")
        if "LLM_Directly" not in df.columns:
            df = pd.read_csv(filepath, sep=",", encoding="utf-8")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # --- Data Sanity Checks (Initial) ---
    print(f"Total Rows Loaded: {len(df)}")

    # 1. Clean Sustainability_Score
    # Handle comma decimals
    if df["Sustainability_Score"].dtype == object:
        df["Sustainability_Score"] = df["Sustainability_Score"].astype(str).str.replace(",", ".").replace("nan", "0")
    
    df["Sustainability_Score"] = pd.to_numeric(df["Sustainability_Score"], errors='coerce')
    
    # Report Score Status
    non_null_scores = df["Sustainability_Score"].notna().sum()
    print(f"Rows with Non-Null Sustainability_Score: {non_null_scores}")
    
    # 2. Remove Duplicates by App ID
    # Keep row with max score if duplicates exist
    initial_count = len(df)
    df = df.sort_values("Sustainability_Score", ascending=False).drop_duplicates(subset=["App ID"], keep="first")
    print(f"Duplicates removed: {initial_count - len(df)} (Kept max score per App ID)")
    
    # 3. Label Engineering
    def get_binary_relevance(row):
        d = str(row.get("LLM_Directly", "")).lower()
        i = str(row.get("LLM_Indirectly", "")).lower()
        
        is_direct = any(x in d for x in ["yes", "true", "1", "direct"])
        is_indirect = any(x in i for x in ["yes", "true", "1", "indirect"])
        
        # Check for explicit negatives to be safe, though default to 0 is common
        # If both are missing/nan, we might want to return NaN? 
        # For now, following "If ambiguous/missing => set NA" rule strictly:
        if (d == "nan" or d == "") and (i == "nan" or i == ""):
            return np.nan
            
        if is_direct or is_indirect:
            return 1
        return 0

    def get_graded_relevance(row):
        d = str(row.get("LLM_Directly", "")).lower()
        i = str(row.get("LLM_Indirectly", "")).lower()
        
        is_direct = any(x in d for x in ["yes", "true", "1", "direct"])
        is_indirect = any(x in i for x in ["yes", "true", "1", "indirect"])
        
        if (d == "nan" or d == "") and (i == "nan" or i == ""):
            return np.nan

        if is_direct:
            return 2
        elif is_indirect:
            return 1
        return 0

    df["y_relevant"] = df.apply(get_binary_relevance, axis=1)
    df["y_gain"] = df.apply(get_graded_relevance, axis=1)
    
    # 4. Filter out rows with NaN labels or Scores
    # We only evaluate on rows where we have Ground Truth AND a Score
    df_eval = df.dropna(subset=["y_relevant", "y_gain", "Sustainability_Score"]).copy()
    
    print(f"Rows with Valid Labels & Scores (used for eval): {len(df_eval)}")
    print(f"Excluded (NaN labels/scores): {len(df) - len(df_eval)}")
    
    processed_counts = df_eval["y_relevant"].value_counts()
    print(f"Label Distribution (y_relevant): \n{processed_counts}")
    
    return df_eval

def compute_ranking_metrics_at_k(df, k_values_abs, k_values_perc):
    """Computes Precision@K and Recall@K."""
    results = []
    
    # Sort by Score Descending
    df_sorted = df.sort_values("Sustainability_Score", ascending=False)
    total_relevant = df_sorted["y_relevant"].sum()
    total_items = len(df_sorted)
    
    # Absolute K
    for k in k_values_abs:
        if k > total_items: k = total_items
        
        top_k = df_sorted.head(k)
        relevant_in_top_k = top_k["y_relevant"].sum()
        
        prec = relevant_in_top_k / k
        rec = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
        
        results.append({
            "K_type": "absolute",
            "K_value": k,
            "K": k,
            "precision_at_k": prec,
            "recall_at_k": rec,
            "relevant_in_top_k": relevant_in_top_k,
            "total_relevant": total_relevant,
            "total_items": total_items
        })
        
    # Percentage K
    for p in k_values_perc:
        k = max(1, int(np.ceil(p * total_items)))
        
        top_k = df_sorted.head(k)
        relevant_in_top_k = top_k["y_relevant"].sum()
        
        prec = relevant_in_top_k / k
        rec = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
        
        results.append({
            "K_type": "percentage",
            "K_value": p,
            "K": k,
            "precision_at_k": prec,
            "recall_at_k": rec,
            "relevant_in_top_k": relevant_in_top_k,
            "total_relevant": total_relevant,
            "total_items": total_items
        })
        
    return pd.DataFrame(results)

def compute_ndcg_at_k(df, k_values_abs, k_values_perc):
    """Computes NDCG@K using y_gain."""
    results = []
    
    # Sort by Score Descending (Ranking)
    df_sorted = df.sort_values("Sustainability_Score", ascending=False)
    relevance = df_sorted["y_gain"].values
    
    # Ideal Ranking (Sort by Label Descending)
    ideal_relevance = sorted(relevance, reverse=True)
    
    total_items = len(df)
    
    def dcg(rels):
        # Discounted Cumulative Gain
        # 2^rel - 1 / log2(i+1)
        res = 0
        for i, r in enumerate(rels):
            res += (2**r - 1) / np.log2(i + 2)
        return res

    all_k = []
    # Mix abs and perc to get actual K integers
    for k in k_values_abs:
        if k > total_items: k = total_items
        all_k.append(("absolute", k, k))
        
    for p in k_values_perc:
        k = max(1, int(np.ceil(p * total_items)))
        all_k.append(("percentage", p, k))
        
    for k_type, k_val, k in all_k:
        # IDCG
        idcg_k = dcg(ideal_relevance[:k])
        
        # DCG
        dcg_k = dcg(relevance[:k])
        
        ndcg = dcg_k / idcg_k if idcg_k > 0 else 0
        
        results.append({
            "K_type": k_type,
            "K_value": k_val,
            "K": k,
            "ndcg_at_k": ndcg
        })
        
    return pd.DataFrame(results)

def get_pr_curve_metrics(df):
    """Computes PR Curve points and AUPRC."""
    y_true = df["y_relevant"]
    y_scores = df["Sustainability_Score"]
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # Prevalence
    prevalence = y_true.mean()
    
    pr_df = pd.DataFrame({
        "threshold": np.append(thresholds, np.nan), # Thresholds is 1 shorter
        "precision": precision,
        "recall": recall
    })
    
    summary = pd.DataFrame({"AUPRC": [pr_auc], "Prevalence": [prevalence]})
    
    return pr_df, summary

def compute_average_precision(df):
    """Computes Average Precision (AP) globally."""
    y_true = df["y_relevant"]
    y_scores = df["Sustainability_Score"]
    
    ap = average_precision_score(y_true, y_scores)
    return pd.DataFrame({"Average_Precision": [ap]})

def analyze_query_level_metrics(df):
    """Computes AP and NDCG per query group."""
    # Group by Query Keyword + Query Subgenre
    # Handle NaNs in grouping cols
    df["Query_Group"] = df["Query Keyword"].fillna("Unknown") + " - " + df["Query Subgenre"].fillna("Unknown")
    
    groups = df.groupby("Query_Group")
    
    ap_results = []
    ndcg_results = [] # Just doing specific K=10 for summary
    
    for name, group in groups:
        n_items = len(group)
        n_relevant = group["y_relevant"].sum()
        
        # Skip small groups
        if n_items < 5:
            continue
            
        # AP
        ap = average_precision_score(group["y_relevant"], group["Sustainability_Score"])
        
        # NDCG@10
        # (Reusing logic, simplified)
        group_sorted = group.sort_values("Sustainability_Score", ascending=False)
        relevance = group_sorted["y_gain"].values
        ideal_relevance = sorted(relevance, reverse=True)
        
        def dcg_k(rels, k):
            res = 0
            for i, r in enumerate(rels[:k]):
                 res += (2**r - 1) / np.log2(i + 2)
            return res
            
        k=10
        idcg = dcg_k(ideal_relevance, k)
        dcg = dcg_k(relevance, k)
        ndcg_10 = dcg / idcg if idcg > 0 else 0
        
        ap_results.append({
            "query_group": name,
            "n_items": n_items,
            "n_relevant": n_relevant,
            "ap": ap
        })

        ndcg_results.append({
             "query_group": name,
             "ndcg_at_10": ndcg_10
        })
        
    ap_df = pd.DataFrame(ap_results)
    map_val = ap_df["ap"].mean() if not ap_df.empty else 0
    map_summary = pd.DataFrame({"MAP": [map_val], "num_queries": [len(ap_df)]})
    
    return ap_df, map_summary, pd.DataFrame(ndcg_results)

def threshold_sweep(df):
    """Sweeps thresholds to find optimal F1/Precision/Recall points."""
    y_true = df["y_relevant"].values
    y_scores = df["Sustainability_Score"].values
    
    min_score = y_scores.min()
    max_score = y_scores.max()
    thresholds = np.linspace(min_score, max_score, 200)
    
    results = []
    
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        acc = (tp + tn) / len(y_true)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        results.append({
            "threshold": t,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        })
        
    df_res = pd.DataFrame(results)
    
    # Operating Points
    # 1. Max F1
    idx_f1 = df_res["f1"].idxmax()
    row_f1 = df_res.loc[idx_f1].copy()
    row_f1["criteria"] = "Max F1"
    
    # 2. High Precision (>= 0.8) with best recall
    df_prec = df_res[df_res["precision"] >= 0.8]
    if not df_prec.empty:
        idx_prec = df_prec["recall"].idxmax()
        row_prec = df_res.loc[idx_prec].copy()
    else:
        # Fallback to max precision available
        idx_prec = df_res["precision"].idxmax()
        row_prec = df_res.loc[idx_prec].copy()
    row_prec["criteria"] = "Precision >= 0.8 (Best Recall)"
        
    # 3. High Recall (>= 0.8) with best precision
    df_rec = df_res[df_res["recall"] >= 0.8]
    if not df_rec.empty:
        idx_rec = df_rec["precision"].idxmax()
        row_rec = df_res.loc[idx_rec].copy()
    else:
        # Fallback
        idx_rec = df_res["recall"].idxmax()
        row_rec = df_res.loc[idx_rec].copy()
    row_rec["criteria"] = "Recall >= 0.8 (Best Precision)"
    
    ops = pd.DataFrame([row_f1, row_prec, row_rec])
    
    return df_res, ops

def main():
    infile = sys.argv[1] if len(sys.argv) > 1 else "../results_with_llm_judge.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        
    df = load_and_clean_data(infile)
    if df is None or df.empty:
        print("Data load failed or empty.")
        return

    # --- Ranking Metrics ---
    k_abs = [10, 20, 50, 100]
    k_perc = [0.01, 0.05, 0.10, 0.25, 0.50]
    
    # 1. Precision/Recall @ K
    df_ranking_k = compute_ranking_metrics_at_k(df, k_abs, k_perc)
    df_ranking_k.to_csv(os.path.join(outdir, "ranking_at_k.csv"), index=False)
    
    # 2. PR Curve
    df_pr, df_auprc = get_pr_curve_metrics(df)
    df_pr.to_csv(os.path.join(outdir, "pr_curve_points.csv"), index=False)
    df_auprc.to_csv(os.path.join(outdir, "auprc_summary.csv"), index=False)
    
    # 3. Average Precision
    df_ap = compute_average_precision(df)
    df_ap.to_csv(os.path.join(outdir, "average_precision.csv"), index=False)
    
    df_ap_query, df_map, df_ndcg_query = analyze_query_level_metrics(df)
    df_ap_query.to_csv(os.path.join(outdir, "ap_by_query.csv"), index=False)
    df_map.to_csv(os.path.join(outdir, "map_summary.csv"), index=False)
    df_ndcg_query.to_csv(os.path.join(outdir, "ndcg_by_query.csv"), index=False)
    
    # 4. NDCG @ K
    df_ndcg = compute_ndcg_at_k(df, k_abs, k_perc)
    df_ndcg.to_csv(os.path.join(outdir, "ndcg_at_k.csv"), index=False)
    
    # --- Threshold Metrics ---
    df_sweep, df_ops = threshold_sweep(df)
    df_sweep.to_csv(os.path.join(outdir, "threshold_sweep.csv"), index=False)
    df_ops.to_csv(os.path.join(outdir, "operating_points.csv"), index=False)
    
    # --- Summaries ---
    # Top K Examples
    top_20 = df.sort_values("Sustainability_Score", ascending=False).head(20)
    cols_to_keep = ["Title", "App ID", "Sustainability_Score", "y_relevant", "y_gain", "LLM_Reason", "Query Keyword", "Query Subgenre", "Store Link/Reference"]
    # Truncate LLM Reason
    top_20["LLM_Reason"] = top_20["LLM_Reason"].astype(str).str.slice(0, 180)
    top_20[cols_to_keep].to_csv(os.path.join(outdir, "top_k_examples.csv"), index=False)
    
    # Paper Summary
    # Extract specific values for the summary row
    ndcg_50 = df_ndcg[df_ndcg["K_type"] == "absolute"][df_ndcg["K_value"] == 50]["ndcg_at_k"].values[0] if not df_ndcg.empty else 0
    prec_50 = df_ranking_k[df_ranking_k["K_type"] == "absolute"][df_ranking_k["K_value"] == 50]["precision_at_k"].values[0] if not df_ranking_k.empty else 0
    
    summary_row = {
        "AUPRC": df_auprc["AUPRC"].values[0],
        "AP_Global": df_ap["Average_Precision"].values[0],
        "MAP_Query": df_map["MAP"].values[0],
        "NDCG@50": ndcg_50,
        "Precision@50": prec_50,
        "Prevalence": df_auprc["Prevalence"].values[0],
        "Dataset_Size": len(df)
    }
    pd.DataFrame([summary_row]).to_csv(os.path.join(outdir, "paper_summary.csv"), index=False)
    
    # --- Plots ---
    # Precision @ K
    plt.figure()
    k_plot = df_ranking_k[df_ranking_k["K_type"]=="absolute"]
    plt.plot(k_plot["K"], k_plot["precision_at_k"], marker='o')
    plt.title("Precision @ K")
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "precision_at_k.png"))
    plt.close()
    
    # NDCG @ K
    plt.figure()
    k_plot_ndcg = df_ndcg[df_ndcg["K_type"]=="absolute"]
    plt.plot(k_plot_ndcg["K"], k_plot_ndcg["ndcg_at_k"], marker='o', color='green')
    plt.title("NDCG @ K")
    plt.xlabel("K")
    plt.ylabel("NDCG")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "ndcg_at_k.png"))
    plt.close()
    
    # PR Curve
    plt.figure()
    plt.plot(df_pr["recall"], df_pr["precision"], label=f'AUPRC={summary_row["AUPRC"]:.2f}')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "pr_curve.png"))
    plt.close()
    
    print("Analysis Complete. Results saved to " + outdir)

if __name__ == "__main__":
    main()
