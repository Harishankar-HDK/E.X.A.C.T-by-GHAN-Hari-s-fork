# evaluators/sharpness_evaluator.py

"""
SharpnessEvaluator
==================
Evaluates how focused and well-concentrated a heatmap is.

A good XAI heatmap should highlight a compact, meaningful region rather
than spreading activation diffusely across the entire image. Diffuse
heatmaps are harder to interpret and often indicate the method is not
confidently localising the model's decision.

Metrics
-------
Sparsity
    Measures how focused the heatmap distribution is using normalised
    entropy. A perfectly focused heatmap (all energy on one pixel) gives
    sparsity = 1.0. A uniform heatmap (spread equally everywhere) gives
    sparsity ≈ 0.0. Higher = sharper.

Concentration
    Measures what fraction of the total activation energy is contained
    in the top-20% of pixels. If 80%+ of the energy is in 20% of pixels,
    the heatmap is well-concentrated. Higher = sharper.

Mass Center Stability (optional, requires multiple heatmaps)
    Not included here — this is a single-heatmap evaluator.

Usage
-----
    from EXACT.evaluators import SharpnessEvaluator

    ev     = SharpnessEvaluator()
    result = ev.evaluate(explainer_result=gradcam_result)
    ev.report(result)
    ev.plot(result, save_png=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from EXACT.evaluators import BaseEvaluator


class SharpnessEvaluator(BaseEvaluator):
    """
    Evaluates heatmap sharpness via Sparsity and Concentration.

    No model access is needed — operates purely on the heatmap array.

    Parameters
    ----------
    topk_frac : float
        Fraction of top pixels used for concentration. Default 0.20 (top 20%).
    save_dir : str
        Directory for saved plots.
    """

    THRESHOLDS = {
        "sparsity":      [(0.15, "Excellent"), (0.05, "Good")],
        "concentration": [(0.70, "Excellent"), (0.50, "Good")],
    }

    WEIGHTS = {
        "sparsity":      0.45,
        "concentration": 0.55,
    }

    def __init__(
        self,
        topk_frac: float = 0.20,
        save_dir: str = "user_saves/evaluator_saves/sharpness",
    ):
        super().__init__(save_dir)
        self.topk_frac = topk_frac

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, explainer_result: dict) -> dict:
        """
        Compute Sparsity and Concentration for the given heatmap.

        Parameters
        ----------
        explainer_result : dict
            Output of any EXACT explainer's explain() method.
            Must contain a 'heatmap' key (or legacy 'cam').

        Returns
        -------
        dict with keys:
            'evaluator'    : str
            'scores'       : dict[metric -> float]
            'grades'       : dict[metric -> str]
            'overall'      : float
            'overall_grade': str
            'heatmap'      : np.ndarray  normalised (H,W) for plotting
            'topk_mask'    : np.ndarray  binary mask of top-k pixels for plotting
        """
        cam  = self._normalize(self._extract_heatmap(explainer_result))

        sparsity      = self._sparsity(cam)
        concentration = self._concentration(cam, self.topk_frac)

        scores = {
            "sparsity":      round(sparsity, 4),
            "concentration": round(concentration, 4),
        }
        grades  = {m: self._grade(m, v) for m, v in scores.items()}
        overall = round(self._weighted_composite(scores, self.WEIGHTS), 4)

        # Build top-k mask for visualisation
        flat   = cam.flatten()
        k      = max(1, int(self.topk_frac * len(flat)))
        thresh = np.sort(flat)[::-1][k - 1]
        topk_mask = (cam >= thresh).astype(np.float32)

        return {
            "evaluator":     "Sharpness Evaluator",
            "scores":        scores,
            "grades":        grades,
            "overall":       overall,
            "overall_grade": self._grade_overall(overall),
            "heatmap":       cam,
            "topk_mask":     topk_mask,
            "topk_frac":     self.topk_frac,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _sparsity(cam: np.ndarray) -> float:
        p = cam.flatten() + 1e-9
        p = p / p.sum()
        return float(1.0 - (-np.sum(p * np.log(p))) / np.log(len(p)))

    @staticmethod
    def _concentration(cam: np.ndarray, topk_frac: float) -> float:
        flat = cam.flatten()
        k    = max(1, int(topk_frac * len(flat)))
        return float(np.sort(flat)[::-1][:k].sum() / (flat.sum() + 1e-9))

    @staticmethod
    def _grade_overall(overall: float) -> str:
        if overall >= 0.65:
            return "Excellent"
        if overall >= 0.45:
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

        scores   = results["scores"]
        grades   = results["grades"]
        overall  = results["overall"]
        og       = results["overall_grade"]
        cam      = results["heatmap"]
        topk     = results["topk_mask"]
        topk_frac = results["topk_frac"]

        fig = plt.figure(figsize=(14, 6), facecolor=BG)
        fig.suptitle("EXACT -- Sharpness Evaluation",
                     color=TEXT, fontsize=14, fontweight="bold",
                     fontfamily="monospace", y=0.97)

        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.4,
                               top=0.88, bottom=0.10, left=0.05, right=0.97)

        # ── Heatmap ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(cam, cmap="jet", vmin=0, vmax=1)
        ax1.set_title("Heatmap", color=TEXT, fontsize=10, fontweight="bold")
        ax1.axis("off")

        # ── Top-k mask ────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.imshow(topk, cmap="hot", vmin=0, vmax=1)
        ax2.set_title(f"Top {int(topk_frac*100)}% pixels", color=TEXT,
                      fontsize=10, fontweight="bold")
        ax2.axis("off")

        # ── Activation distribution ───────────────────────────────────
        ax3 = fig.add_subplot(gs[:, 2])
        ax3.set_facecolor(CARD)
        flat = cam.flatten()
        ax3.hist(flat, bins=50, color="#4E9AF1", edgecolor="#30363D", alpha=0.85)
        ax3.set_title("Activation Distribution", color=TEXT, fontsize=10, fontweight="bold")
        ax3.set_xlabel("Activation value", color=SUB, fontsize=8)
        ax3.set_ylabel("Pixel count", color=SUB, fontsize=8)
        ax3.tick_params(colors=SUB, labelsize=8)
        ax3.spines[["top","right"]].set_visible(False)
        ax3.spines[["left","bottom"]].set_color("#30363D")
        # Annotate sparsity on distribution plot
        ax3.text(0.97, 0.95,
                 f"Entropy-sparsity: {scores['sparsity']:.4f}\n[{grades['sparsity']}]",
                 ha="right", va="top", color=self._grade_color(grades["sparsity"]),
                 fontsize=8, fontweight="bold", transform=ax3.transAxes)

        # ── Score + overall ───────────────────────────────────────────
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.set_facecolor(CARD); ax4.axis("off")
        metrics = list(scores.keys())
        vals    = [scores[m] for m in metrics]
        colors  = [self._grade_color(grades[m]) for m in metrics]
        bars    = ax4.barh(metrics[::-1], vals[::-1], color=colors[::-1],
                           height=0.5, edgecolor="#30363D")
        ax4.set_xlim(0, 1.25)
        ax4.set_facecolor(CARD)
        ax4.tick_params(axis="y", labelcolor=TEXT, labelsize=9)
        ax4.tick_params(axis="x", labelcolor=SUB, labelsize=8)
        ax4.spines[["top","right","left","bottom"]].set_color("#30363D")
        for bar, val, m in zip(bars, vals[::-1], metrics[::-1]):
            ax4.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}  {grades[m]}",
                     va="center", color=TEXT, fontsize=8, fontfamily="monospace")

        ax5 = fig.add_subplot(gs[1, 3])
        ax5.set_facecolor(CARD); ax5.axis("off")
        oc = self._grade_color(og)
        ax5.text(0.5, 0.65, f"{overall:.4f}", ha="center", va="center",
                 color=oc, fontsize=28, fontweight="bold", transform=ax5.transAxes)
        ax5.text(0.5, 0.30, og, ha="center", va="center",
                 color=oc, fontsize=13, fontweight="bold", transform=ax5.transAxes)
        ax5.text(0.5, 0.10, "Overall Sharpness", ha="center", va="center",
                 color=SUB, fontsize=8, transform=ax5.transAxes)

        if save_png:
            fname = filename or "sharpness_eval.png"
            out   = self.save_dir / fname
            fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
            print(f"Saved: {out}")
        else:
            plt.show()
        plt.close(fig)