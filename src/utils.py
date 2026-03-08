"""Utility functions for XAI security project."""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """Evaluate model performance."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'f1': 2 * (tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0,
    }

    if y_pred_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def normalize_features(X, feature_min, feature_max):
    """Normalize features to [0, 1] range based on min/max from training data."""
    X_norm = (X - feature_min) / (feature_max - feature_min + 1e-10)
    return np.clip(X_norm, 0, 1)


def denormalize_features(X_norm, feature_min, feature_max):
    """Denormalize features back to original range."""
    X_denorm = X_norm * (feature_max - feature_min) + feature_min
    return X_denorm


def get_feature_bounds(X):
    """Get min/max bounds for each feature."""
    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    return feature_min, feature_max
