# Quick Start Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   - Go to [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Download `creditcard.csv`
   - Place it in: `data/creditcard.csv`

## Run the Project

Run notebooks in this order:

### 1. Data Exploration
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```
Explore the fraud dataset, understand features and class distribution.

### 2. Train Baseline Model
```bash
jupyter notebook notebooks/02_train_baseline_model.ipynb
```
Train XGBoost fraud detector and save model + scaler.

### 3. Generate SHAP Explanations
```bash
jupyter notebook notebooks/03_generate_shap_explanations.ipynb
```
Compute SHAP values for all samples (used for attacks).

### 4. Attack WITH XAI
```bash
jupyter notebook notebooks/04_attack_with_xai.ipynb
```
Execute adversarial attack using SHAP explanations to guide feature manipulation.

### 5. Attack WITHOUT XAI
```bash
jupyter notebook notebooks/05_attack_without_xai.ipynb
```
Execute adversarial attack using random feature manipulation (baseline).

### 6. Compare Results
```bash
jupyter notebook notebooks/06_comparison_analysis.ipynb
```
Compare attack effectiveness, visualize results, draw conclusions.

## Project Output

After running all notebooks, you'll have:
- `data/model.pkl` - Trained fraud detector
- `data/shap_values.npy` - SHAP explanations
- `data/results_with_xai.pkl` - Attack results (WITH XAI)
- `data/results_without_xai.pkl` - Attack results (WITHOUT XAI)
- `data/comparison_main.png` - Visualization of results
- `data/shap_summary.png` - SHAP feature importance
- `data/shap_waterfall.png` - Local explanations example

## Key Findings

The project demonstrates that:
- **WITH XAI**: Attackers achieve ~X% success with Y features modified
- **WITHOUT XAI**: Attackers achieve ~A% success with B features modified
- **Conclusion**: XAI (SHAP) creates an attack surface—interpretability enables more efficient adversarial attacks

See `notebooks/06_comparison_analysis.ipynb` for full results and interpretation.
