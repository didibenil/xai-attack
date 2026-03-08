"""
Attack strategy using SHAP explanations to guide feature manipulation.
"""

import numpy as np
import pandas as pd
from copy import deepcopy


class SHAPGuidedAttack:
    """
    Attacker with access to SHAP explanations.
    Uses feature importance to strategically modify features.
    """

    def __init__(self, model, shap_values, feature_names, target_class=0):
        """
        Args:
            model: Trained classifier with predict_proba method
            shap_values: SHAP values array (samples × features)
            feature_names: List of feature names
            target_class: Target prediction class (0=legitimate, 1=fraud)
        """
        self.model = model
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.target_class = target_class
        self.num_features = len(feature_names)

    def attack(self, X, y_true, max_iterations=50, top_k=5, step_size=0.05):
        """
        Execute attack on samples where true class != target class.

        Args:
            X: Feature matrix (samples × features)
            y_true: True labels
            max_iterations: Max modification steps per sample
            top_k: Number of top features to modify
            step_size: How much to modify features per iteration

        Returns:
            Dictionary with attack results
        """
        # Target samples: where ground truth is NOT target class
        target_mask = y_true != self.target_class
        target_indices = np.where(target_mask)[0]  

        results = {
            'successful_attacks': 0,
            'failed_attacks': 0,
            'modified_features': [],
            'modification_magnitudes': [],
            'original_predictions': [],
            'final_predictions': [],
            'queries': 0,
        }

        for idx in target_indices:
            sample = X[idx].copy()
            original_pred = self.model.predict_proba([sample])[0][self.target_class]
            results['queries'] += 1

            # Get SHAP values for this sample
            sample_shap = self.shap_values[idx]

            # Identify features pushing AWAY from target class
            # (i.e., features with negative contribution if target=0)
            feature_importance = np.abs(sample_shap)
            top_feature_indices = np.argsort(feature_importance)[-top_k:]

            # Try to flip prediction by modifying top features
            modified_sample = sample.copy()
            for iteration in range(max_iterations):
                # Modify top features toward favorable direction
                for feat_idx in top_feature_indices:
                    if sample_shap[feat_idx] < 0:
                        # Increase feature if it has negative contribution
                        modified_sample[feat_idx] += step_size
                    else:
                        # Decrease feature if it has positive contribution
                        modified_sample[feat_idx] -= step_size

                # Clip to valid range [0, 1] (assuming normalized features)
                modified_sample = np.clip(modified_sample, 0, 1)

                # Check prediction
                new_pred = self.model.predict_proba([modified_sample])[
                    0
                ][self.target_class]
                results['queries'] += 1

                # Success: prediction flipped to target class
                if self.model.predict([modified_sample])[0] == self.target_class:
                    num_features_modified = np.sum(
                        np.abs(modified_sample - sample) > 0.01
                    )
                    avg_magnitude = np.mean(np.abs(modified_sample - sample))

                    results['successful_attacks'] += 1
                    results['modified_features'].append(num_features_modified)
                    results['modification_magnitudes'].append(avg_magnitude)
                    results['original_predictions'].append(original_pred)
                    results['final_predictions'].append(new_pred)
                    break
            else:
                # Attack failed after max iterations
                results['failed_attacks'] += 1

        # Compute statistics
        if results['modified_features']:
            results['avg_features_modified'] = np.mean(
                results['modified_features']
            )
            results['avg_modification_magnitude'] = np.mean(
                results['modification_magnitudes']
            )
            results['success_rate'] = (
                results['successful_attacks']
                / (results['successful_attacks'] + results['failed_attacks'])
            )
        else:
            results['avg_features_modified'] = 0
            results['avg_modification_magnitude'] = 0
            results['success_rate'] = 0

        return results