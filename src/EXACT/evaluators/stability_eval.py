# evaluators/stability_evaluator.py

"""
StabilityEvaluator
==================
Evaluates how consistent a heatmap explanation is under small input perturbations.

A reliable XAI method should produce nearly identical heatmaps when the input
changes by an imperceptible amount (Gaussian noise). Large heatmap shifts under
tiny input changes suggest the method is sensitive to noise and may not be
trustworthy for real-world use.

Metrics
-------
Mean Deviation
    Average pixel-wise absolute difference between the reference heatmap
    and each noisy re-run. Lower = more stable.
    Reported as a stability score = (1 - mean_deviation). Higher = better.

Max Deviation
    Worst-case deviation across all noisy runs. Useful for identifying
    whether the method ever produces a drastically different heatmap.
    Reported as stability score = (1 - max_deviation). Higher = better.

Std Deviation
    Standard deviation of per-run deviations. A low std means the method
    fails consistently rather than occasionally — which may indicate a
    systematic sensitivity rather than random noise sensitivity.

Usage
-----
    from EXACT.evaluators import StabilityEvaluator

    ev = StabilityEvaluator()
    result = ev.evaluate(
        explainer_result=gradcam_result,
        explainer_obj=gradcam_exp,
        input_tensor=input_tensor,
        extra_kwargs={"method": "gradcam"},
        runs=15,
        noise_std=0.05,
    )
    ev.report(result)
    ev.plot(result, save_png=True)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch

from EXACT.evaluators import BaseEvaluator


class StabilityEvaluator(BaseEvaluator):
    """
    Evaluates heatmap stability under Gaussian input noise.

    Parameters
    ----------
    runs : int
        Number of noisy re-runs. Default 10.
    noise_std : float
        Standard deviation of Gaussian noise added to the input. Default 0.05.
    save_dir : str
        Directory for saved plots.
    """

    THRESHOLDS = {
        "mean_stability": [(0.85, "Excellent"), (0.70, "Good")],
        "max_stability":  [(0.80, "Excellent"), (0.60, "Good")],
    }

    WEIGHTS = {
        "mean_stability": 0.60,
        "max_stability":  0.40,
    }

    def __init__(
        self,
        runs: int = 10,
        noise_std: float = 0.05,
        save_dir: str = "user_saves/evaluator_saves/stability",
    ):
        super().__init__(save_dir)
        self.runs      = runs
        self.noise_std = noise_std

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        explainer_result: dict,
        explainer_obj: Any,
        input_tensor: torch.Tensor,
        extra_kwargs: Optional[dict] = None,
        runs: Optional[int] = None,
        noise_std: Optional[float] = None,
    ) -> dict:
        """
        Re-run the explainer on noisy inputs and measure heatmap deviation.

        Parameters
        ----------
        explainer_result : dict
            Output of explainer.explain() for the clean input.
            Used as the reference heatmap.
        explainer_obj : any
            The explainer instance. Must have .explain(input_tensor, **extra_kwargs).
        input_tensor : torch.Tensor
            The clean input used to generate explainer_result. Shape (1,C,H,W).
        extra_kwargs : dict, optional
            Forwarded to explainer_obj.explain() on each re-run.
            e.g. {"method": "gradcam"} for GradCAM, {} for LIME / IG.
        runs : int, optional
            Overrides instance default.
        noise_std : float, optional
            Overrides instance default.

        Returns
        -------
        dict with keys:
            'evaluator'       : str
            'scores'          : dict[metric -> float]
            'grades'          : dict[metric -> str]
            'overall'         : float
            'overall_grade'   : str
            'per_run_devs'    : list[float]  deviation each run (for plotting)
            'noise_std'       : float
            'runs_completed'  : int
        """
        runs      = runs or self.runs
        noise_std = noise_std or self.noise_std
        kwargs    = extra_kwargs or {}

        cam_ref = self._normalize(self._extract_heatmap(explainer_result))
        h, w    = cam_ref.shape

        per_run_devs = []

        for i in range(runs):
            noise = torch.randn_like(input_tensor) * noise_std
            noisy = (input_tensor + noise).clamp(0, 1)
            result = None
            try:
                result    = explainer_obj.explain(noisy, **kwargs)
                if "heatmap" in result:
                    raw = result["heatmap"]
                elif "cam" in result:
                    raw = result["cam"]
                else:
                    raise KeyError("No 'heatmap' or 'cam' key in result.")
                cam_noisy = self._normalize(self._to_hw(raw, h, w))
                dev       = float(np.mean(np.abs(cam_noisy - cam_ref)))
                per_run_devs.append(dev)
            except Exception as exc:
                warnings.warn(f"Stability run {i+1} failed: {exc}")
            finally:
                del result
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not per_run_devs:
            mean_dev = 1.0
            max_dev  = 1.0
        else:
            mean_dev = float(np.mean(per_run_devs))
            max_dev  = float(np.max(per_run_devs))

        scores = {
            "mean_stability": round(1.0 - mean_dev, 4),
            "max_stability":  round(1.0 - max_dev,  4),
        }
        grades  = {m: self._grade(m, v) for m, v in scores.items()}
        overall = round(self._weighted_composite(scores, self.WEIGHTS), 4)

        return {
            "evaluator":      "Stability Evaluator",
            "scores":         scores,
            "grades":         grades,
            "overall":        overall,
            "overall_grade":  self._grade_overall(overall),
            "per_run_devs":   per_run_devs,
            "noise_std":      noise_std,
            "runs_completed": len(per_run_devs),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_hw(cam: Any, h: int, w: int) -> np.ndarray:
        if hasattr(cam, "cpu"):
            cam = cam.cpu().numpy()
        cam = np.array(cam, dtype=np.float32)
        if cam.ndim == 3:
            cam = cam.squeeze(0) if cam.shape[0] == 1 else cam.squeeze(-1)
        if cam.shape != (h, w):
            cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam

    @staticmethod
    def _grade_overall(overall: float) -> str:
        if overall >= 0.82:
            return "Excellent"
        if overall >= 0.65:
            return "Good"
        return "Poor"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _plot_body(self, results: dict, save_png: bool, filename: Optional[str]) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        BG, CARD  = "#0D1117", "#161B22"
        TEXT, SUB = "#E6EDF3", "#8B949E"
        BLUE      = "#4E9AF1"

        scores   = results["scores"]
        grades   = results["grades"]
        overall  = results["overall"]
        og       = results["overall_grade"]
        devs     = results["per_run_devs"]
        n_runs   = results["runs_completed"]
        ns       = results["noise_std"]

        fig = plt.figure(figsize=(13, 6), facecolor=BG)
        fig.suptitle(f"EXACT -- Stability Evaluation  (noise_std={ns}, runs={n_runs})",
                     color=TEXT, fontsize=13, fontweight="bold",
                     fontfamily="monospace", y=0.97)

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4,
                               top=0.88, bottom=0.10, left=0.07, right=0.97)

        # ── Per-run deviation line ────────────────────────────────────
        ax1 = fig.add_subplot(gs[:, 0:2])
        ax1.set_facecolor(CARD)
        if devs:
            ax1.plot(range(1, len(devs)+1), devs, color=BLUE,
                     linewidth=2, marker="o", markersize=5)
            ax1.fill_between(range(1, len(devs)+1), devs, alpha=0.15, color=BLUE)
            ax1.axhline(np.mean(devs), color="#F7C948", linewidth=1.2,
                        linestyle="--", label=f"Mean = {np.mean(devs):.4f}")
            ax1.axhline(np.max(devs), color="#F4845F", linewidth=1.0,
                        linestyle=":", label=f"Max = {np.max(devs):.4f}")
            ax1.legend(facecolor=CARD, labelcolor=TEXT, fontsize=8,
                       edgecolor="#30363D")
        ax1.set_title("Per-run Heatmap Deviation  (lower = more stable)",
                      color=TEXT, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Noisy run", color=SUB, fontsize=8)
        ax1.set_ylabel("Mean absolute pixel deviation", color=SUB, fontsize=8)
        ax1.tick_params(colors=SUB, labelsize=8)
        ax1.spines[["top","right"]].set_visible(False)
        ax1.spines[["left","bottom"]].set_color("#30363D")
        ax1.set_ylim(0, max(devs) * 1.3 + 0.01 if devs else 1.0)

        # ── Score bars ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor(CARD); ax2.axis("off")
        metrics = list(scores.keys())
        vals    = [scores[m] for m in metrics]
        colors  = [self._grade_color(grades[m]) for m in metrics]
        bars    = ax2.barh(metrics[::-1], vals[::-1], color=colors[::-1],
                           height=0.5, edgecolor="#30363D")
        ax2.set_xlim(0, 1.25)
        ax2.set_facecolor(CARD)
        ax2.tick_params(axis="y", labelcolor=TEXT, labelsize=9)
        ax2.tick_params(axis="x", labelcolor=SUB, labelsize=8)
        ax2.spines[["top","right","left","bottom"]].set_color("#30363D")
        for bar, val, m in zip(bars, vals[::-1], metrics[::-1]):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}  {grades[m]}",
                     va="center", color=TEXT, fontsize=8, fontfamily="monospace")

        # ── Overall ───────────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.set_facecolor(CARD); ax3.axis("off")
        oc = self._grade_color(og)
        ax3.text(0.5, 0.65, f"{overall:.4f}", ha="center", va="center",
                 color=oc, fontsize=28, fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.5, 0.30, og, ha="center", va="center",
                 color=oc, fontsize=13, fontweight="bold", transform=ax3.transAxes)
        ax3.text(0.5, 0.10, "Overall Stability", ha="center", va="center",
                 color=SUB, fontsize=8, transform=ax3.transAxes)

        if save_png:
            fname = filename or "stability_eval.png"
            out   = self.save_dir / fname
            fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
            print(f"Saved: {out}")
        else:
            plt.show()
        plt.close(fig)