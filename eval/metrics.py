"""Metrics computation for evaluation."""

import numpy as np
from typing import List, Dict, Any


def compute_success_rate(results: List[Dict[str, Any]]) -> float:
    """Compute success rate from trial results.

    Args:
        results: List of trial result dicts

    Returns:
        success_rate: Fraction of successful trials
    """
    if not results:
        return 0.0
    successes = sum(1 for r in results if r.get("success", False))
    return successes / len(results)


def compute_pose_error_rms(pose_errors: List[float]) -> float:
    """Compute RMS pose error.

    Args:
        pose_errors: List of pose error magnitudes

    Returns:
        rms_error: Root mean square error
    """
    if not pose_errors:
        return 0.0
    return float(np.sqrt(np.mean(np.array(pose_errors) ** 2)))


def compute_force_error_rms(force_errors: List[float]) -> float:
    """Compute RMS force error.

    Args:
        force_errors: List of force error magnitudes

    Returns:
        rms_error: Root mean square error
    """
    if not force_errors:
        return 0.0
    return float(np.sqrt(np.mean(np.array(force_errors) ** 2)))


def count_safety_violations(results: List[Dict[str, Any]]) -> int:
    """Count total safety violations.

    Args:
        results: List of trial result dicts

    Returns:
        total_violations: Total count of safety violations
    """
    return sum(r.get("safety_violations", 0) for r in results)


def compute_metrics(trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute all metrics from trial results.

    Args:
        trial_results: List of trial result dicts

    Returns:
        metrics: Dict with all computed metrics
    """
    return {
        "success_rate": compute_success_rate(trial_results),
        "mean_completion_time": np.mean([r.get("completion_time", 0) for r in trial_results]),
        "pose_error_rms": compute_pose_error_rms([r.get("pose_error_rms", 0) for r in trial_results]),
        "force_error_rms": compute_force_error_rms([r.get("force_error_rms", 0) for r in trial_results]),
        "safety_violations": count_safety_violations(trial_results),
    }
