
import os
import json
import numpy as np
from scipy.stats import kendalltau

DATA_DIR = "../patient_contributions_DataSet1"
methods = ["SHAP", "Lime", "Inherent"]

def kendalls_w_from_rank_matrix(rank_matrix):
    """Compute Kendall's W from a rank matrix (rows: methods, columns: features)."""
    rank_matrix = np.array(rank_matrix)
    k, n = rank_matrix.shape  # k = number of methods, n = number of features
    if k < 2 or n < 2:
        return None  # Not enough data
    R = np.sum(rank_matrix, axis=0)
    R_bar = np.mean(R)
    S = np.sum((R - R_bar) ** 2)
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    return W

def process_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    result = {}

    for model, methods_data in data.items():
        method_ranks = []

        # Collect scores per method
        available_methods = [m for m in methods if m in methods_data]
        if len(available_methods) < 2:
            continue

        features = set()
        for m in available_methods:
            features.update(methods_data[m].keys())
        features = sorted(features)

        for m in available_methods:
            scores = [methods_data[m].get(f, 0) for f in features]
            # Invert sign for proper "importance" ranking: higher = more important
            ranks = (-np.array(scores)).argsort().argsort() + 1  # rank 1 is highest
            method_ranks.append(ranks)

        if len(method_ranks) >= 2:
            W = kendalls_w_from_rank_matrix(method_ranks)
            result[model] = {"kendall_w": round(W, 4) if W is not None else None}

    return result

def main():
    results = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            patient_result = process_file(filepath)
            results[filename] = patient_result

    with open("kendall_per_patient.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Kendall’s W analysis completed. Results saved to 'kendall_per_patient.json'.")

if __name__ == "__main__":
    main()
