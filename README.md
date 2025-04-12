# Explainability Correlation Analysis

This project analyzes the consistency between different explainability methods applied to machine learning models in ICU mortality prediction.

## 🔍 Goal
To evaluate how much various explanation techniques agree with one another when interpreting model predictions, across multiple patients and models.

## 🧪 Methods Included

The analysis compares three explanation methods:
- SHAP
- LIME
- Inherent (based on the model's internal structure)

Each patient has a corresponding explanation JSON file containing feature attributions from these methods per model.

### ✅ Correlation Metrics Computed

1. **Kendall’s W**  
   Measures agreement in feature rankings across all three methods.

2. **Sign Agreement**  
   Calculates the percentage of features where all three methods agree on the sign (positive/negative) of the contribution.

3. **Intersection@K (K=5)**  
   Computes the average overlap of the top 5 important features across all method pairs (e.g., SHAP–LIME).

4. **Pearson Correlation (average)**  
   Computes linear correlation between feature importance values across all method pairs, then averages the result.

## 📁 Input

Place all explanation JSON files inside the `explanations/` directory. Each file should follow this format:

```json
{
  "LogisticRegression": {
    "SHAP": { "age": 0.4, "bmi": -0.2, ... },
    "LIME": { ... },
    "Inherent": { ... }
  },
  ...
}
```

## 🚀 Run the Script

Make sure you have the dependencies installed:
```bash
pip install -r requirements.txt
```

Then run:
```bash
python kendall_sign_intersection_pearson.py
```

## 📤 Output

A single JSON file:
```
kendall_sign_intersection.json
```

Each entry includes the correlation results per patient and per model.

## 📦 Project Structure

```
explanability_project/
├── explanations/                   # Input explanation files
├── kendall_sign_intersection_pearson.py  # Main analysis script
├── requirements.txt
└── README.md
```