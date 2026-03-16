# evaluators/faithfulness_evaluator.py

"""
FaithfulnessEvaluator
=====================
Evaluates how faithfully a heatmap reflects the model's true decision process.

Metrics
-------
Deletion AUC (inverted)
    Progressively masks the most-activated pixels with zeros and tracks
    how fast the model's top-class confidence drops.
    A faithful heatmap highlights pixels the model truly relies on, so
    confidence should collapse quickly → low raw AUC → high inverted score.

Insertion AUC
    Starts from a blurred baseline and progressively reveals the most-
    activated pixels. Confidence should rise quickly if the heatmap
    correctly identifies what the model needs.
    Higher AUC = better.

Both metrics are model-based: they re-run the model many times, so they
are slower than sharpness or localization.

Usage
-----
    from EXACT.evaluators import FaithfulnessEvaluator

    ev = FaithfulnessEvaluator(model, device="cuda")
    result = ev.evaluate(
        explainer_result=gradcam_result,   # any explain() output with 'heatmap'
        input_tensor=input_tensor,
        steps=10,
    )
    ev.report(result)
    ev.plot(result, save_png=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from EXACT.evaluators import BaseEvaluator


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluates heatmap faithfulness via Deletion AUC and Insertion AUC.

    Parameters
    ----------
    model : torch.nn.Module
        The model that generated the predictions being explained.
    device : str
        'cpu' or 'cuda'. Default 'cpu'.
    steps : int
        Number of masking steps for AUC curves. Default 10.
        Increase to 20 for smoother curves at higher compute cost.
    save_dir : str
        Directory for saved plots. Default 'user_saves/evaluator_saves/faithfulness'.
    """

    THRESHOLDS = {
        "deletion_auc":  [(0.80, "Excellent"), (0.60, "Good")],
        "insertion_auc": [(0.75, "Excellent"), (0.50, "Good")],
    }

    WEIGHTS = {
        "deletion_auc":  0.50,
        "insertion_auc": 0.50,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        steps: int = 10,
        save_dir: str = "user_saves/evaluator_saves/faithfulness",
    ):
        super().__init__(save_dir)
        self.model  = model.eval()
        self.device = device
        self.steps  = steps

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        explainer_result: dict,
        input_tensor: torch.Tensor,
        steps: Optional[int] = None,
    ) -> dict:
        """
        Compute Deletion AUC and Insertion AUC for the given heatmap.

        Parameters
        ----------
        explainer_result : dict
            Output of any EXACT explainer's explain() method.
            Must contain a 'heatmap' key (or legacy 'cam').
        input_tensor : torch.Tensor
            The preprocessed input used to generate the heatmap. Shape (1,C,H,W).
        steps : int, optional
            Overrides the instance default number of masking steps.

        Returns
        -------
        dict with keys:
            'evaluator'    : str
            'scores'       : dict[metric -> float]
            'grades'       : dict[metric -> str]
            'overall'      : float
            'overall_grade': str
            'curves'       : dict — deletion/insertion confidence curves for plotting
        """
        steps = steps or self.steps
        cam   = self._normalize(self._extract_heatmap(explainer_result))

        # Resize CAM to tensor spatial dims for pixel masking
        _, _, th, tw = input_tensor.shape
        if cam.shape != (th, tw):
            cam = cv2.resize(cam, (tw, th), interpolation=cv2.INTER_LINEAR)

        x          = input_tensor.clone().to(self.device)
        fracs      = np.linspace(0, 1, steps + 1)
        sorted_idx = np.argsort(cam.flatten())[::-1].copy()
        n_px       = th * tw

        del_curve, ins_curve = self._run_curves(x, cam, sorted_idx, n_px, fracs)

        del_auc_raw = float(np.trapz(del_curve, fracs))
        ins_auc     = float(np.trapz(ins_curve, fracs))
        del_auc     = round(1.0 - del_auc_raw, 4)   # inverted: higher = better
        ins_auc     = round(ins_auc, 4)

        scores = {"deletion_auc": del_auc, "insertion_auc": ins_auc}
        grades = {m: self._grade(m, v) for m, v in scores.items()}
        overall = round(self._weighted_composite(scores, self.WEIGHTS), 4)

        return {
            "evaluator":     "Faithfulness Evaluator",
            "scores":        scores,
            "grades":        grades,
            "overall":       overall,
            "overall_grade": self._grade_overall(overall),
            "curves": {
                "fracs":       fracs.tolist(),
                "deletion":    del_curve,
                "insertion":   ins_curve,
                "del_auc_raw": del_auc_raw,
            },
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_curves(self, x, cam, sorted_idx, n_px, fracs):
        """Run deletion and insertion passes in a single method to share setup."""
        baseline_del = torch.zeros_like(x)
        x_np         = x[0].cpu().permute(1, 2, 0).numpy()
        blurred      = cv2.GaussianBlur(x_np, (51, 51), 0)
        baseline_ins = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(self.device)

        del_curve = []
        ins_curve = []

        with torch.no_grad():
            for frac in fracs:
                n = int(frac * n_px)

                # Deletion
                masked = x.clone().reshape(1, -1, n_px)
                if n:
                    masked[:, :, sorted_idx[:n]] = (
                        baseline_del.reshape(1, -1, n_px)[:, :, sorted_idx[:n]]
                    )
                logits = self.model(masked.reshape_as(x))
                pred   = logits.argmax(1)
                del_curve.append(F.softmax(logits, dim=1)[0, pred].item())

                # Insertion
                revealed = baseline_ins.clone().reshape(1, -1, n_px)
                if n:
                    revealed[:, :, sorted_idx[:n]] = (
                        x.reshape(1, -1, n_px)[:, :, sorted_idx[:n]]
                    )
                logits = self.model(revealed.reshape_as(x))
                pred   = logits.argmax(1)
                ins_curve.append(F.softmax(logits, dim=1)[0, pred].item())

        return del_curve, ins_curve

    @staticmethod
    def _grade_overall(overall: float) -> str:
        if overall >= 0.75:
            return "Excellent"
        if overall >= 0.55:
            return "Good"
        return "Poor"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _plot_body(self, results: dict, save_png: bool, filename: Optional[str]) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        BG, CARD     = "#0D1117", "#161B22"
        TEXT, SUB    = "#E6EDF3", "#8B949E"
        GREEN, GOLD  = "#54C27D", "#F7C948"
        RED, BLUE    = "#F4845F", "#4E9AF1"

        scores  = results["scores"]
        grades  = results["grades"]
        overall = results["overall"]
        og      = results["overall_grade"]
        curves  = results["curves"]
        fracs   = curves["fracs"]

        fig = plt.figure(figsize=(14, 7), facecolor=BG)
        fig.suptitle("EXACT -- Faithfulness Evaluation",
                     color=TEXT, fontsize=14, fontweight="bold",
                     fontfamily="monospace", y=0.97)

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.4,
                               top=0.88, bottom=0.10, left=0.07, right=0.97)

        # ── Deletion curve ────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.set_facecolor(CARD)
        ax1.plot(fracs, curves["deletion"], color=RED, linewidth=2, label="Deletion")
        ax1.fill_between(fracs, curves["deletion"], alpha=0.15, color=RED)
        ax1.set_title("Deletion Curve", color=TEXT, fontsize=10, fontweight="bold")
        ax1.set_xlabel("Fraction of pixels masked", color=SUB, fontsize=8)
        ax1.set_ylabel("Model confidence", color=SUB, fontsize=8)
        ax1.tick_params(colors=SUB, labelsize=8)
        ax1.spines[["top","right"]].set_visible(False)
        ax1.spines[["left","bottom"]].set_color("#30363D")
        ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)
        ax1.text(0.5, 0.05, f"AUC = {scores['deletion_auc']:.4f}  [{grades['deletion_auc']}]",
                 ha="center", color=self._grade_color(grades["deletion_auc"]),
                 fontsize=9, fontweight="bold", transform=ax1.transAxes)

        # ── Insertion curve ───────────────────────────────────────────
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.set_facecolor(CARD)
        ax2.plot(fracs, curves["insertion"], color=GREEN, linewidth=2, label="Insertion")
        ax2.fill_between(fracs, curves["insertion"], alpha=0.15, color=GREEN)
        ax2.set_title("Insertion Curve", color=TEXT, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Fraction of pixels revealed", color=SUB, fontsize=8)
        ax2.set_ylabel("Model confidence", color=SUB, fontsize=8)
        ax2.tick_params(colors=SUB, labelsize=8)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.spines[["left","bottom"]].set_color("#30363D")
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
        ax2.text(0.5, 0.05, f"AUC = {scores['insertion_auc']:.4f}  [{grades['insertion_auc']}]",
                 ha="center", color=self._grade_color(grades["insertion_auc"]),
                 fontsize=9, fontweight="bold", transform=ax2.transAxes)

        # ── Score summary ─────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(CARD); ax3.axis("off")
        ax3.set_title("Scores", color=TEXT, fontsize=10, fontweight="bold")
        metrics = list(scores.keys())
        vals    = [scores[m] for m in metrics]
        colors  = [self._grade_color(grades[m]) for m in metrics]
        bars    = ax3.barh(metrics[::-1], vals[::-1], color=colors[::-1],
                           height=0.5, edgecolor="#30363D")
        ax3.set_xlim(0, 1.15)
        ax3.set_facecolor(CARD)
        ax3.tick_params(axis="y", labelcolor=TEXT, labelsize=9)
        ax3.tick_params(axis="x", labelcolor=SUB, labelsize=8)
        ax3.spines[["top","right","left","bottom"]].set_color("#30363D")
        for bar, val, m in zip(bars, vals[::-1], metrics[::-1]):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}  {grades[m]}",
                     va="center", color=TEXT, fontsize=8, fontfamily="monospace")

        # ── Overall gauge ─────────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor(CARD); ax4.axis("off")
        oc   = self._grade_color(og)
        ax4.text(0.5, 0.65, f"{overall:.4f}", ha="center", va="center",
                 color=oc, fontsize=28, fontweight="bold",
                 transform=ax4.transAxes)
        ax4.text(0.5, 0.30, og, ha="center", va="center",
                 color=oc, fontsize=13, fontweight="bold",
                 transform=ax4.transAxes)
        ax4.text(0.5, 0.10, "Overall Faithfulness", ha="center", va="center",
                 color=SUB, fontsize=8, transform=ax4.transAxes)
        for spine in ax4.spines.values():
            spine.set_edgecolor("#30363D")

        if save_png:
            fname = filename or "faithfulness_eval.png"
            out   = self.save_dir / fname
            fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
            print(f"Saved: {out}")
        else:
            plt.show()
        plt.close(fig)