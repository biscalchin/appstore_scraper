import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
import sys

def load_and_clean_data(filepath):
    """Loads the CSV and cleans the relevant columns."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return None

    try:
        # Try semicolon first as it's the expected format
        df = pd.read_csv(filepath, sep=";", encoding="utf-8")
        if "LLM_Directly" not in df.columns:
             df = pd.read_csv(filepath, sep=",", encoding="utf-8")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Filter out rows where LLM didn't run
    df = df[df["LLM_Directly"].isin(["YES", "NO"]) & df["LLM_Indirectly"].isin(["YES", "NO"])]
    
    # Convert Score to float
    # Handle comma decimals if present
    if df["Sustainability_Score"].dtype == object:
        df["Sustainability_Score"] = df["Sustainability_Score"].astype(str).str.replace(",", ".").replace("nan", "0")
    
    df["Sustainability_Score"] = pd.to_numeric(df["Sustainability_Score"], errors='coerce').fillna(0)
    
    return df

def categorize_app(row):
    """Categorizes app based on User's request."""
    d = row["LLM_Directly"]
    i = row["LLM_Indirectly"]
    
    if d == "YES" and i == "YES":
        return "Strongly Sustainable (YES-YES)"
    elif (d == "YES" and i == "NO") or (d == "NO" and i == "YES"):
        return "Moderately Sustainable (Mixed)"
    else:
        return "Not Sustainable (NO-NO)"

def log_print(message, file_handle=None):
    """Helper to print to both stdout and a file."""
    print(message)
    if file_handle:
        file_handle.write(message + "\n")

def analyze_correlations(df, output_dir="."):
    """Performs statistical analysis and plotting, saving directly to output_dir."""
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open log file
    stats_file_path = os.path.join(output_dir, "validation_stats.txt")
    with open(stats_file_path, "w", encoding="utf-8") as f:
        
        log_print(f"--- Analysis Timestamp: {pd.Timestamp.now()} ---", f)
        log_print(f"Total Apps Analyzed: {len(df)}", f)

        # 1. Create Category Column
        df["Sustainability_Category"] = df.apply(categorize_app, axis=1)
        
        # Define order for plots
        order = ["Not Sustainable (NO-NO)", "Moderately Sustainable (Mixed)", "Strongly Sustainable (YES-YES)"]
        
        log_print("\n--- Descriptive Statistics by Category ---", f)
        stats = df.groupby("Sustainability_Category")["Sustainability_Score"].describe()
        log_print(stats.to_string(), f)
        
        # 2. Box Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="Sustainability_Category", y="Sustainability_Score", data=df, order=order)
        plt.title("Distribution of Sustainability Score by LLM Classification")
        plt.ylabel("Calculated Sustainability Score (0-1)")
        plt.xlabel("LLM Classification")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "score_distribution_boxplot.png"))
        plt.close()
        log_print("\n[Saved] score_distribution_boxplot.png", f)
        
        # 3. Binary Classification Analysis (Is it sustainable AT ALL?)
        # We treat YES-YES and Mixed as "Positive" and NO-NO as "Negative"
        df["Is_Sustainable_Binary"] = df["Sustainability_Category"] != "Not Sustainable (NO-NO)"
        
        y_true = df["Is_Sustainable_Binary"].astype(int)
        y_scores = df["Sustainability_Score"]
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve: Score ability to predict Any Sustainability (Direct or Indirect)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()
        log_print("\n[Saved] roc_curve.png", f)
        
        log_print(f"\n--- ROC AUC Score: {roc_auc:.3f} ---", f)
        log_print("An AUC of 0.5 is random guessing. 1.0 is perfect prediction.", f)
        
        # 4. Find Optimal Threshold (Maximizing F1 Score)
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        # Handle division by zero
        f1_scores = np.nan_to_num(f1_scores)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_pr[best_idx]
        best_f1 = f1_scores[best_idx]
        
        log_print(f"\n--- Optimal Threshold Analysis ---", f)
        log_print(f"Best Score Threshold to separate Sustainable from Non-Sustainable: {best_threshold:.4f}", f)
        log_print(f"Achieves F1-Score of: {best_f1:.3f}", f)

        # 5. Confusion Matrix at Optimal Threshold
        y_pred = (y_scores >= best_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Sustainable", "Sustainable"])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f"Confusion Matrix (Threshold = {best_threshold:.4f})")
        plt.tight_layout() # Fix truncated labels
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()
        log_print("\n[Saved] confusion_matrix.png", f)

        # Calculate Accuracy, Precision, Recall manually for log
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        log_print(f"Accuracy at Threshold: {accuracy:.3f}", f)
        log_print(f"Precision at Threshold: {tp/(tp+fp) if (tp+fp)>0 else 0:.3f}", f)
        log_print(f"Recall at Threshold: {tp/(tp+fn) if (tp+fn)>0 else 0:.3f}", f)
        
        # 6. Specific Analysis: Directly vs Indirectly Separation
        # Can the score distinguish between Direct and Indirect?
        # Filter only sustainable apps
        df_sus = df[df["Is_Sustainable_Binary"] == True].copy()
        if not df_sus.empty:
            df_sus["Is_Direct"] = df_sus["LLM_Directly"] == "YES"
            
            # Check if we have enough data for this sub-analysis
            if df_sus["Is_Direct"].nunique() > 1:
                mean_direct = df_sus[df_sus["Is_Direct"]]["Sustainability_Score"].mean()
                mean_indirect = df_sus[~df_sus["Is_Direct"]]["Sustainability_Score"].mean()
                log_print("\n--- Direct vs Indirect Distinction ---", f)
                log_print(f"Mean Score for DIRECTLY Educative: {mean_direct:.4f}", f)
                log_print(f"Mean Score for INDIRECTLY Educative: {mean_indirect:.4f}", f)
        
def main():
    infile = "results_with_llm_judge.csv" # Default
    output_dir = "validation" # Default output dir
    
    # Arg parsing: python script.py [infile] [outdir]
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]

    print(f"Loading data from {infile}...")
    print(f"Saving results to: {output_dir}")
    
    df = load_and_clean_data(infile)
    if df is None or df.empty:
        print("No valid data found to analyze.")
        return
        
    print(f"Analyzing {len(df)} classified apps.")
    analyze_correlations(df, output_dir)

if __name__ == "__main__":
    main()
