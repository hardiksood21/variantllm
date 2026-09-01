import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    confusion_matrix
)

# Add src to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from variantllm.inference.scorer import ZeroShotVariantScorer

def evaluate_clinvar_benchmark(data_path="data/processed/clinvar_benchmark_sample.csv"):
    print("=" * 65)
    print("VariantLLM: Zero-Shot Foundation Model ClinVar Benchmark Evaluation")
    print("=" * 65)
    
    if not os.path.exists(data_path):
        from data.curate_clinvar_benchmark import save_clinvar_benchmark
        df = save_clinvar_benchmark(data_path)
    else:
        df = pd.read_csv(data_path)
        
    print(f"Loaded {len(df)} validated human clinical variants across: {df['gene'].unique().tolist()}")
    
    scorer = ZeroShotVariantScorer(model_name="facebook/esm2_t6_8M_UR50D")
    
    y_true = []
    y_scores = []
    y_pred = []
    llr_list = []
    results = []
    
    print("\nExecuting Zero-Shot Foundation Model Scoring across all variants...")
    for idx, row in df.iterrows():
        res = scorer.score_sequence_pair(row["wt_seq"], row["mut_seq"])
        
        y_true.append(row["label"])
        y_scores.append(res["pathogenicity_prob"])
        y_pred.append(1 if res["pathogenicity_prob"] >= 0.5 else 0)
        llr_list.append(res["zero_shot_llr_score"])
        
        results.append({
            "Gene": row["gene"],
            "Protein Change": row["protein_change"],
            "ClinVar Label": row["clinical_significance"],
            "True Label": row["label"],
            "Predicted Class": "Pathogenic" if res["pathogenicity_prob"] >= 0.5 else "Benign",
            "Pathogenicity Prob": round(res["pathogenicity_prob"], 4),
            "Zero-Shot LLR (Score)": round(res["zero_shot_llr_score"], 4),
            "Site Entropy": round(res["site_entropy"], 4)
        })
        print(f"  [{row['gene']} {row['protein_change']}] True: {row['clinical_significance']} | LLR: {res['zero_shot_llr_score']:+.3f} | Prob: {res['pathogenicity_prob']:.2%}")
        
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = np.array(y_pred)
    
    # Calculate Metrics
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "=" * 65)
    print("CLINVAR ZERO-SHOT BENCHMARK METRICS SUMMARY")
    print("=" * 65)
    print(f"  * ROC-AUC Score:                {roc_auc:.4f}")
    print(f"  * PR-AUC (Avg Precision) Score: {pr_auc:.4f}")
    print(f"  * Overall Accuracy:             {acc:.2%}")
    print(f"  * F1-Score:                     {f1:.4f}")
    print(f"  * Precision (Pathogenic):       {prec:.4f}")
    print(f"  * Recall (Sensitivity):         {rec:.4f}")
    print(f"  * Matthews Correlation (MCC):   {mcc:.4f}")
    print(f"  * Confusion Matrix:\n{cm}")
    print("=" * 65)
    
    res_df = pd.DataFrame(results)
    out_csv = "data/processed/clinvar_evaluation_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved detailed evaluation metrics to {out_csv}")
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1_score": f1,
        "mcc": mcc,
        "results_df": res_df
    }

if __name__ == "__main__":
    evaluate_clinvar_benchmark()
