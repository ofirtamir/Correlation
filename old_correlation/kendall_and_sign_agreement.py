
import os
import json
import numpy as np
from scipy.stats import kendalltau

DATA_DIR = "../explanations"
methods = ["SHAP", "Lime", "Inherent"]

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

def sign_agreement(methods_data):
    signs_by_method = {}
    features = set()
    for m in methods:
        if m in methods_data:
            signs_by_method[m] = {}
            for f, val in methods_data[m].items():
                features.add(f)
                signs_by_method[m][f] = np.sign(val)

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

def process_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    result = {}

    for model, methods_data in data.items():
        method_ranks = []
        available_methods = [m for m in methods if m in methods_data]
        if len(available_methods) < 2:
            continue

        features = set()
        for m in available_methods:
            features.update(methods_data[m].keys())
        features = sorted(features)

        for m in available_methods:
            scores = [methods_data[m].get(f, 0) for f in features]
            ranks = (-np.array(scores)).argsort().argsort() + 1
            method_ranks.append(ranks)

        model_result = {}
        if len(method_ranks) >= 2:
            W = kendalls_w_from_rank_matrix(method_ranks)
            model_result["kendall_w"] = round(W, 4) if W is not None else None

        sign_agree = sign_agreement(methods_data)
        model_result["sign_agreement"] = sign_agree

        result[model] = model_result

    return result

def main():
    results = {}
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            filepath = os.path.join(DATA_DIR, filename)
            patient_result = process_file(filepath)
            results[filename] = patient_result

    with open("kendall_and_sign_agreement.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Analysis completed. Results saved to 'kendall_and_sign_agreement.json'.")

if __name__ == "__main__":
    main()
