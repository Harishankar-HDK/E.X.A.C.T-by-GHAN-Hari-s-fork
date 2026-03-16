# evaluators/base_evaluator.py

"""
BaseEvaluator
=============
Abstract base class for all EXACT heatmap evaluators.

Every evaluator follows the same three-step interface:

    result = evaluator.evaluate(explainer_result, input_tensor, ...)
    evaluator.report(result)
    evaluator.plot(result, save_png=True)

Subclasses must implement:
    - evaluate() → dict
    - _plot_body() → called by plot() to render the evaluator-specific figure
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class BaseEvaluator(ABC):
    """
    Abstract base for all EXACT heatmap evaluators.

    Parameters
    ----------
    save_dir : str
        Directory for saved plots.
    """

    # Subclasses define their interpretation thresholds here.
    # Format: {metric_name: [(threshold, label), ...]}
    # Thresholds are checked from highest to lowest; first match wins.
    THRESHOLDS: dict[str, list[tuple[float, str]]] = {}

    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface (must implement evaluate; report/plot are shared)
    # ------------------------------------------------------------------

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> dict:
        """
        Run evaluation metrics and return a structured results dict.
        Must always include at least:
            'scores'       : dict[metric_name -> float]
            'grades'       : dict[metric_name -> str]   e.g. 'Excellent'
            'overall'      : float   weighted composite in [0, 1]
            'overall_grade': str
        """

    def report(self, results: dict) -> None:
        """Print a formatted evaluation report to the console."""
        self._print_report(results)

    def plot(
        self,
        results: dict,
        save_png: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Render and optionally save the evaluation visual report."""
        try:
            import matplotlib
            if save_png:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("pip install matplotlib")
        self._plot_body(results, save_png, filename)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _grade(self, metric: str, value: float) -> str:
        """
        Convert a numeric score to a human-readable grade using
        the subclass-defined THRESHOLDS table.
        """
        thresholds = self.THRESHOLDS.get(metric, [])
        for threshold, label in sorted(thresholds, key=lambda x: x[0], reverse=True):
            if value >= threshold:
                return label
        return "Poor"

    def _grade_color(self, grade: str) -> str:
        """Map a grade string to a hex colour for plots."""
        return {
            "Excellent": "#54C27D",
            "Good":      "#F7C948",
            "Poor":      "#F4845F",
        }.get(grade, "#8B949E")

    def _extract_heatmap(self, explainer_result: dict) -> np.ndarray:
        """
        Pull the heatmap array from any EXACT explainer result dict.
        Accepts 'heatmap' (canonical) or 'cam' (legacy).

        Note: checks key existence rather than value truthiness — numpy arrays
        raise ValueError when used with 'or', and a valid all-zero heatmap
        would be falsy even if present.
        """
        if "heatmap" in explainer_result:
            raw = explainer_result["heatmap"]
        elif "cam" in explainer_result:
            raw = explainer_result["cam"]
        else:
            raise KeyError(
                f"explainer_result has no 'heatmap' or 'cam' key. "
                f"Found keys: {list(explainer_result.keys())}"
            )
        return np.array(raw, dtype=np.float32)

    @staticmethod
    def _normalize(cam: np.ndarray) -> np.ndarray:
        mn, mx = cam.min(), cam.max()
        if mx - mn < 1e-8:
            return np.zeros_like(cam)
        return (cam - mn) / (mx - mn)

    @staticmethod
    def _weighted_composite(scores: dict[str, float], weights: dict[str, float]) -> float:
        active = {m: weights[m] for m in scores if m in weights}
        total  = sum(active.values())
        if not total:
            return float(np.mean(list(scores.values())))
        return sum(scores[m] * active[m] / total for m in active)

    def _print_report(self, results: dict) -> None:
        """Shared console report formatter."""
        name    = results.get("evaluator", self.__class__.__name__)
        scores  = results["scores"]
        grades  = results["grades"]
        overall = results["overall"]
        og      = results["overall_grade"]

        width = 58
        print("\n" + "=" * width)
        print(f"  EXACT -- {name}")
        print("=" * width)

        for metric, value in scores.items():
            grade = grades.get(metric, "")
            bar   = self._ascii_bar(value)
            print(f"  {metric:<22} {value:>6.4f}  {bar}  {grade}")

        print("-" * width)
        overall_bar = self._ascii_bar(overall)
        print(f"  {'OVERALL':<22} {overall:>6.4f}  {overall_bar}  {og}")
        print("=" * width + "\n")

    @staticmethod
    def _ascii_bar(value: float, width: int = 20) -> str:
        filled = int(round(value * width))
        return "[" + "#" * filled + "." * (width - filled) + "]"

    @abstractmethod
    def _plot_body(self, results: dict, save_png: bool, filename: Optional[str]) -> None:
        """Subclass implements the actual figure layout."""