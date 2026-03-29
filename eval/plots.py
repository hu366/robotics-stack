"""Plotting utilities for evaluation results.

M0 stub: placeholder for M4 implementation.
"""

from typing import Dict, Any, List
from pathlib import Path


def plot_pose_errors(pose_errors: List[float], output_path: str) -> str:
    """Plot pose error over time.

    Args:
        pose_errors: List of pose error values
        output_path: Path to save plot

    Returns:
        plot_path: Path to saved plot
    """
    # M0 stub - to be implemented in M4
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Placeholder: create empty file
    path.touch()
    return str(path)


def plot_force_errors(force_errors: List[float], output_path: str) -> str:
    """Plot force error over time.

    Args:
        force_errors: List of force error values
        output_path: Path to save plot

    Returns:
        plot_path: Path to saved plot
    """
    # M0 stub - to be implemented in M4
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return str(path)


def generate_plots(results: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Generate all plots from evaluation results.

    Args:
        results: Aggregated evaluation results
        output_dir: Directory to save plots

    Returns:
        plot_paths: Dict mapping plot names to paths
    """
    # M0 stub
    return {
        "pose_errors": plot_pose_errors([], f"{output_dir}/pose_errors.png"),
        "force_errors": plot_force_errors([], f"{output_dir}/force_errors.png"),
    }
