
import os
import json
import numpy as np
from scipy.stats import kendalltau

DATA_DIR = "./explanations"
methods = ["SHAP", "Lime", "Inherent"]
TOP_K = 5

def kendalls_w_from_rank_matrix(rank_matrix):
    rank_matrix = np.array(rank_matrix)
    k, n = rank_matrix.shape
    if k < 2 or n < 2:
        return None
    R = np.sum(rank_matrix, axis=0)
    R_bar = np.mean(R)
    S = np.sum((R - R_bar) ** 2)
    W = 12 * S / (k ** 2 * (n ** 3 - n))
    return W

def sign_agreement(methods_data, features):
    signs_by_method = {}
    for m in methods:
        if m in methods_data:
            signs_by_method[m] = {f: np.sign(methods_data[m].get(f, 0)) for f in features}

    agreed = 0
    total = 0
    for f in features:
        current_signs = [signs_by_method[m][f] for m in signs_by_method if f in signs_by_method[m]]
        current_signs = [s for s in current_signs if s != 0]
        if len(current_signs) >= 2:
            total += 1
            if all(s == current_signs[0] for s in current_signs[1:]):
                agreed += 1
    return round(agreed / total, 4) if total > 0 else None

def intersection_at_k(methods_data, features, k=TOP_K):
    top_k = {}
    for m in methods:
        if m in methods_data:
            sorted_feats = sorted(
                [(f, methods_data[m].get(f, 0)) for f in features],
                key=lambda x: abs(x[1]),
                reverse=True
            )
            top_k[m] = set([f for f, _ in sorted_feats[:k]])

    pairs = [("SHAP", "Lime"), ("SHAP", "Inherent"), ("Lime", "Inherent")]
    scores = []
    for m1, m2 in pairs:
        if m1 in top_k and m2 in top_k:
            inter = len(top_k[m1].intersection(top_k[m2]))
            scores.append(inter / k)
    return round(np.mean(scores), 4) if scores else None

def process_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    result = {}

    for model, methods_data in data.items():
        method_ranks = []
        available_methods = [m for m in methods if m in methods_data]
        if len(available_methods) < 2:
            continue

        all_features = set()
        for m in available_methods:
            all_features.update(methods_data[m].keys())

        # Remove features that are zero across all methods
        valid_features = set()
        for f in all_features:
            values = [methods_data[m].get(f, 0) for m in available_methods]
            if all(v != 0 for v in values):
                valid_features.add(f)
        features = sorted(valid_features)
        if not features:
            continue

        for m in available_methods:
            scores = [methods_data[m].get(f, 0) for f in features]
            ranks = (-np.array(scores)).argsort().argsort() + 1
            method_ranks.append(ranks)

        model_result = {}
        if len(method_ranks) >= 2:
            W = kendalls_w_from_rank_matrix(method_ranks)
            model_result["kendall_w"] = round(W, 4) if W is not None else None

        sign_agree = sign_agreement(methods_data, features)
        model_result["sign_agreement"] = sign_agree

        intersection_score = intersection_at_k(methods_data, features, TOP_K)
        model_result[f"intersection_at_{TOP_K}"] = intersection_score

        # Pearson correlation average (on raw values)
        pearson_scores = []
        for m1, m2 in [("SHAP", "Lime"), ("SHAP", "Inherent"), ("Lime", "Inherent")]:
            if m1 in methods_data and m2 in methods_data:
                v1 = np.array([methods_data[m1].get(f, 0) for f in features])
                v2 = np.array([methods_data[m2].get(f, 0) for f in features])
                if len(v1) > 1:
                    r = np.corrcoef(v1, v2)[0, 1]
                    pearson_scores.append(r)
        if pearson_scores:
            model_result["pearson_avg"] = round(np.mean(pearson_scores), 4)

        result[model] = model_result

    return result

def main():
    results = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            patient_result = process_file(filepath)
            results[filename] = patient_result

    with open("kendall_sign_intersection_pearson_filtered.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Analysis completed. Results saved to 'kendall_sign_intersection_pearson_filtered.json'.")

if __name__ == "__main__":
    main()
