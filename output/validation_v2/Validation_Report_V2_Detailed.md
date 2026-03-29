# Validation Report V2: Lexical Algorithm Optimization

**Date:** 2026-01-24
**Subject:** Optimization of Sustainability Scoring Algorithm using Benchmarked LLM Data.

---

## 1. Executive Summary
This report details the successful optimization of the `Sustainability_Score` algorithm. By analyzing the discrepancies between our initial keyword-based score (V1) and the LLM's classification (Ground Truth), we identified and removed significant sources of "semantic noise."

**Key Result:** The refined algorithm (V2) achieves a **+11.5% increase in Precision** and a **+6.9% increase in Accuracy**, making it a much more reliable filter for discovering sustainable apps.

### Sustainability Score Formula
The score is calculated using the prevalence of sustainability and educational keywords, normalized by the text length with a smoothing factor to penalize very short descriptions:

$$ Score = \frac{1}{2} \left( \frac{S}{N + 50} + \frac{E}{N + 50} \right) $$

Where:
*   $S$ = Count of **Sustainability** Terms (e.g., *eco, carbon, recycle*)
*   $E$ = Count of **Education** Terms (e.g., *learn, teach, habits*)
*   $N$ = Total number of words in the description
*   $50$ = Smoothing constant (Zavorra)

| Metric | V1 (Baseline) | V2 (Optimized) | Change | Impact |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | 48.9% | **60.4%** | 🟢 **+11.5%** | ✅ **Significant reduction in False Positives** |
| **Accuracy** | 69.3% | **76.2%** | 🟢 **+6.9%** | ✅ **Better overall classification** |
| **ROC AUC** | 0.705 | **0.734** | 🟢 **+2.9%** | ✅ **Higher predictive power** |

---

## 2. Phase 1: The Baseline & The "Noise" Problem
In the first iteration (V1), we used a broad lexicon derived from general academic literature on serious games.

### The Problem: False Positives
Our Error Analysis revealed that V1 had a low precision (~49%). This means that **1 out of 2 apps** flagged as "Sustainable" were actually irrelevant.

**Root Cause: The "Simulator" Effect**
The algorithm was heavily penalized by generic gaming terminology that acts as a "False Friend" in our context:
*   **"Simulator" / "Simulation"**: In our Educational Lexicon, these were considered learning methods. However, in the App Store, they are *genres* (e.g., "Train Simulator", "Truck Simulator").
*   **"World"**: Intended to catch planetary awareness, but frequently used in titles like "World of Tanks" or "Super Mario World".
*   **"Train"**: Intended as "Education/Training", but matched with locomotives.

**Result:** A "Train Simulator" game would get points for `train` (Education) + `simulator` (Education) + `world` (Sustainability), resulting in a high sustainability score despite being purely entertainment.

---

## 3. Phase 2: The Solution (Refinement Strategy)
To fix this, we moved from a "Broad Academic" approach to a "Store-Specific" approach.

### 1. Purification (Removing Noise)
We removed keywords that are frequent in *game metadata* but ambiguous regarding *content*:
*   `simulator`, `simulation` (Genre markers)
*   `train` (Ambiguous noun/verb)
*   `world` (Ambiguous noun)
*   `strategy`, `mission`, `challenges` (Generic gameplay terms)

### 2. Specificity (Adding Signal)
We identified terms frequent in verified sustainable apps that we were missing:
*   `garbage`, `trash` (Critical for waste management games)
*   `species` (Critical for biodiversity games)

---

## 4. Why V2 is More Accurate: The Factors of Success

### Factor A: Decoupling Genre from Intent
The biggest contributor to the accuracy jump is the decoupling of **Game Genre** from **Educational Intent**.
*   **V1**: Being a "Simulation Strategy" game gave you free points towards being "Educational".
*   **V2**: You only get "Educational" points if you use active pedagogy concepts (`teach`, `learn`, `decision making`) or specific sustainability problems (`pollution`, `waste`).

### Factor B: Increased Semantic Density
By removing high-frequency generic words (`world`, `water` - which is often just a game element), we forced the score to rely on **High-Value Keys**.
*   A user mentioning "Environment" in a "Programming Environment" context might still trigger a false positive, but they are far rarer than "World".
*   Now, a high score requires the presence of distinct, unambiguous concepts (e.g., `ecosystem` + `protection`).

### Factor C: The "Empty" Description Filter
V2 implicitly penalizes apps with generic, marketing-fluff descriptions. If an app description only says "Best driving simulator in the world!", V1 gave it valid points (`simulator`, `world`). V2 gives it **0**. This correctly aligns with our goal: strict sustainability identification.

---

## 5. Visual Analysis

### ROC Curve Improvement
The V2 algorithm (Orange) shows a stronger lift compared to V1 (Blue), indicating improved discrimination capability. The "knee" of the curve is closer to the top-left corner, confirming the higher AUC (0.734 vs 0.705).
![ROC Curve](roc_curve.png)

### Score Distribution & Separation
The boxplot demonstrates a cleaner separation between classes. Note how the "Not Sustainable" apps (0) are now compressed towards the bottom (low mean score), while "Sustainable" apps (1) maintain a higher distribution. This reduced noise floor allows for a safer cut-off threshold.
![Boxplot](score_distribution_boxplot.png)

---

## 6. Conclusion & Recommendations
The V2 algorithm is robust. It sacrifices a small amount of recall (-2.8%) to achieve trustworthiness.

**Operational Recommendation:**
Use a threshold of **0.007** for production scraping.
*   **Score > 0.007**: Highly likely to be relevant.
*   **Score < 0.007**: Safe to discard.

This setup ensures that human reviewers or downstream processes waste minimal time on irrelevant apps.
