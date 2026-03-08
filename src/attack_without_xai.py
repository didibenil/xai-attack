"""
Baseline attack strategy without access to SHAP explanations.
Uses random feature manipulation (brute force).
"""

import numpy as np


class RandomFeatureAttack:
    """
    Attacker WITHOUT access to explanations.
    Uses random feature manipulation to find adversarial examples.
    """

    def __init__(self, model, feature_names, target_class=0, seed=42):
        """
        Args:
            model: Trained classifier with predict_proba method
            feature_names: List of feature names
            target_class: Target prediction class (0=legitimate, 1=fraud)
            seed: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.target_class = target_class
        self.num_features = len(feature_names)
        self.seed = seed
        np.random.seed(seed)

    def attack(self, X, y_true, max_iterations=50, step_size=0.05):
        """
        Execute attack using random feature modifications.

        Args:
            X: Feature matrix (samples × features)
            y_true: True labels
            max_iterations: Max modification steps per sample
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

            # Brute force: randomly modify features
            modified_sample = sample.copy()
            for iteration in range(max_iterations):
                # Randomly select a feature to modify
                feature_idx = np.random.randint(0, self.num_features)

                # Random direction (increase or decrease)
                direction = np.random.choice([-1, 1])
                modified_sample[feature_idx] += direction * step_size

                # Clip to valid range [0, 1]
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
