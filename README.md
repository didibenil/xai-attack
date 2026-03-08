# XAI Security Vulnerability Proof-of-Concept

Demonstrating that explainable AI (SHAP/LIME) can be weaponized against fraud detection systems.

## Research Question
**Does interpretable AI enable more efficient adversarial attacks?**

Compare attacker success rates:
- **WITH XAI**: Using SHAP explanations to guide feature manipulation
- **WITHOUT XAI**: Random/brute-force feature manipulation

## Project Structure

```
project5/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── data/
│   └── fraud_data.csv          # Kaggle Credit Card Fraud dataset
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Load, explore, visualize data
│   ├── 02_train_baseline_model.ipynb  # Train fraud detection model
│   ├── 03_generate_shap_explanations.ipynb  # Generate SHAP values
│   ├── 04_attack_with_xai.ipynb       # Attack using SHAP guidance
│   ├── 05_attack_without_xai.ipynb    # Attack with random/brute-force
│   └── 06_comparison_analysis.ipynb   # Compare results & visualize
└── src/
    ├── __init__.py
    ├── attack_with_xai.py       # XAI-guided attack implementation
    ├── attack_without_xai.py    # Baseline attack implementation
    └── utils.py                 # Helper functions
```

## Workflow

1. **Load & Explore**: Understand fraud dataset features
2. **Train Model**: Build baseline fraud detector (Random Forest or XGBoost)
3. **Generate Explanations**: Compute SHAP values for all samples
4. **Attack WITH XAI**:
   - Select negative predictions (legitimate transactions)
   - Use SHAP to identify top-k features pushing toward "fraud"
   - Modify those features to flip prediction to "legitimate"
   - Track: # features changed, magnitudes, success rate
5. **Attack WITHOUT XAI**:
   - Same samples, same goal
   - No SHAP access—use random/brute-force manipulation
   - Track same metrics
6. **Compare & Analyze**:
   - Plot success rates (XAI vs baseline)
   - Compare # features modified
   - Compare modification magnitudes
   - Write conclusions

## Key Metrics

- **Success Rate**: % of attacks that flip prediction (fraud → legitimate)
- **Feature Budget**: Avg # of features modified per successful attack
- **Modification Magnitude**: Avg amount each feature changes
- **Query Efficiency**: # of model queries needed per attack
- **Cost-Benefit**: How much interpretability costs in terms of robustness

## Expected Results

- **WITH XAI**: High success with fewer feature modifications (e.g., 3 features, 85% success)
- **WITHOUT XAI**: Lower success, more feature modifications (e.g., 15 features, 40% success)
- **Conclusion**: Interpretability is an attack surface

## Installation

```bash
pip install -r requirements.txt
```

Then run notebooks in order.
