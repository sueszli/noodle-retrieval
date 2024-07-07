import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import functions from 3_core_metrics.py
from core_metrics import compute_exact, compute_f1, compute_precision, compute_recall


def read_csv_manual(file_path):
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return pd.DataFrame(data)


def evaluate_qa_results(result_path):
    # Load the results
    results_df = read_csv_manual(result_path)

    # Ensure the results DataFrame has the required columns
    if not all(col in results_df.columns for col in ["queryid", "documentid", "output", "text-selection"]):
        raise ValueError("Results file must contain 'queryid', 'documentid', 'output', and 'text-selection' columns")

    exact_matches = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    # Evaluate each result
    for i, row in results_df.iterrows():
        predicted_answer = row["output"]
        gold_answer = row["text-selection"]

        # Compute exact match and F1 scores
        exact_match = compute_exact(gold_answer, predicted_answer)
        f1_score = compute_f1(gold_answer, predicted_answer)
        precision = compute_precision(gold_answer, predicted_answer)
        recall = compute_recall(gold_answer, predicted_answer)

        exact_matches.append(exact_match)
        f1_scores.append(f1_score)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Calculate overall metrics
    avg_exact_match = np.mean(exact_matches)
    avg_f1_score = np.mean(f1_scores)
    avg_precision_score = np.mean(precision_scores)
    avg_recall_score = np.mean(recall_scores)

    print(f"Exact Match: {avg_exact_match:.4f}")
    print(f"F1 Score: {avg_f1_score:.4f}")
    print(f"Recall: {avg_recall_score:.4f}")
    print(f"Precision: {avg_precision_score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        result_path = Path(sys.argv[1])
        if result_path.is_dir():
            print(f"Error: {result_path} is a directory, please provide the path to the result.csv file.")
        elif not result_path.exists():
            print(f"Error: {result_path} does not exist.")
        else:
            evaluate_qa_results(result_path)
    else:
        print("Usage: python evaluate_results.py <result.csv>")
