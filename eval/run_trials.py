"""Trial runner for evaluation protocol.

M0 stub: placeholder for M4 implementation.
"""

from typing import Dict, Any, Optional
import numpy as np


def run_trial(task_config: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """Run a single trial.

    Args:
        task_config: Task configuration dict
        seed: Random seed for this trial

    Returns:
        trial_result: Dict with trial results
    """
    # M0 stub - to be implemented in M4
    return {
        "seed": seed,
        "success": True,
        "completion_time": 0.0,
        "pose_error_rms": 0.0,
        "force_error_rms": 0.0,
        "safety_violations": 0,
        "failure_mode": None,
    }


def run_trials(
    task_config: Dict[str, Any],
    n_trials: int = 50,
    base_seed: int = 0,
    out_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Run multiple trials.

    Args:
        task_config: Task configuration dict
        n_trials: Number of trials to run
        base_seed: Base random seed
        out_dir: Output directory for results

    Returns:
        aggregated_results: Dict with aggregated metrics
    """
    results = []
    for i in range(n_trials):
        result = run_trial(task_config, seed=base_seed + i)
        results.append(result)

    # Aggregate results
    success_rate = np.mean([r["success"] for r in results])
    completion_times = [r["completion_time"] for r in results if r["success"]]
    mean_time = np.mean(completion_times) if completion_times else 0.0

    return {
        "n_trials": n_trials,
        "success_rate": success_rate,
        "mean_completion_time": mean_time,
        "trials": results,
    }
