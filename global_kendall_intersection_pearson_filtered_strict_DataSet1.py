import os
import json
import numpy as np
from scipy.stats import kendalltau

# DATA_FILE = "./global_explanations_1/global_graph_data.json"
DATA_FILE = "./global_explanations_2/global_graph_DataSet2.json"
methods = ["SHAP", "Lime", "Inherent"]
TOP_K = 10

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
            top1 = top_k[m1]
            top2 = top_k[m2]

            only_in_m1 = top1 - top2  # פיצ'רים שנמצאים רק ב־m1
            only_in_m2 = top2 - top1  # פיצ'רים שנמצאים רק ב־m2

            print(f"Features in {m1} only: {only_in_m1}")
            print(f"Features in {m2} only: {only_in_m2}")

            inter = len(top1 & top2)
            scores.append(inter / k)
    return round(np.mean(scores), 4) if scores else None

def process_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    result = {}

    THRESHOLD = 1
    for model, methods_data in data.items():
        available_methods = [m for m in methods if m in methods_data]
        if len(available_methods) < 2:
            continue

        # אפס פיצ'רים שערכם קטן מהסף בכל שיטה
        all_features = set()
        for m in available_methods:
            all_features.update(methods_data[m].keys())
            for f in methods_data[m]:
                if abs(methods_data[m][f]) < THRESHOLD:
                    methods_data[m][f] = 0

        all_features_sorted = sorted(all_features)
        if not all_features_sorted:
            continue

        # מציאת פיצ'רים משותפים שאינם אפס בכל השיטות
        common_features = set(all_features_sorted)
        for m in available_methods:
            nonzero_feats = {f for f in all_features_sorted if methods_data[m].get(f, 0) != 0}
            common_features &= nonzero_feats

        common_features = sorted(common_features)
        if len(common_features) < 2:
            continue

        # חישוב דירוגים אחידים רק על הפיצ'רים המשותפים
        aligned_ranks = []
        for m in available_methods:
            scores = [methods_data[m].get(f, 0) for f in common_features]
            ranks = (-np.array(scores)).argsort().argsort() + 1
            aligned_ranks.append(ranks)

        model_result = {}
        W = kendalls_w_from_rank_matrix(np.array(aligned_ranks))
        # if W is not None:
        #     W = max(0.0, min(1.0, W))  # חסום לטווח חוקי
        model_result["kendall_w"] = round(W, 4) if W is not None else None

        intersection_score = intersection_at_k(methods_data, all_features_sorted, TOP_K)
        model_result[f"intersection_at_{TOP_K}"] = intersection_score

        # Pearson correlation average (on raw values)
        pearson_scores = []
        for m1, m2 in [("SHAP", "Lime"), ("SHAP", "Inherent"), ("Lime", "Inherent")]:
            if m1 in methods_data and m2 in methods_data:
                v1 = np.array([methods_data[m1].get(f, 0) for f in all_features_sorted])
                v2 = np.array([methods_data[m2].get(f, 0) for f in all_features_sorted])
                if len(v1) > 1:
                    r = np.corrcoef(v1, v2)[0, 1]
                    pearson_scores.append(r)
        if pearson_scores:
            model_result["pearson_avg"] = round(np.mean(pearson_scores), 4)

        result[model] = model_result

    return result

def main():
    results = process_file(DATA_FILE)

    output_filename = "global_kendall_intersection_pearson_filtered_strict_DataSet2.json"
    with open(output_filename, "w") as f:
        json.dump({os.path.basename(DATA_FILE): results}, f, indent=2)

    print(f"✅ Analysis completed. Results saved to '{output_filename}'.")

if __name__ == "__main__":
    main()
