# evaluators/localization_evaluator.py

"""
LocalizationEvaluator
=====================
Evaluates how well a heatmap spatially localises the model's focus relative
to a ground-truth foreground mask or bounding box.

Requires a GT mask — if you don't have one, use the other evaluators instead.

Metrics
-------
IoU (Intersection over Union)
    Binarises the heatmap at a threshold and computes the overlap ratio
    between the predicted important region and the GT mask.
    Higher = better spatial alignment.

Pointing Game
    Checks whether the single pixel with the highest activation falls
    inside the GT mask. 1.0 = yes, 0.0 = no.
    A simple but intuitive check: "is the model looking at the right thing?"

Energy Inside Mask
    Fraction of the total heatmap energy (sum of activations) that falls
    inside the GT mask region. Unlike IoU, this does not require binarisation
    and is therefore threshold-free.
    Higher = more of the model's attention is on the foreground.

Usage
-----
    from EXACT.evaluators import LocalizationEvaluator

    ev = LocalizationEvaluator()
    result = ev.evaluate(
        explainer_result=gradcam_result,
        gt_mask=gt_mask,          # (H, W) binary array: 1=foreground, 0=background
        iou_threshold=0.5,
    )
    ev.report(result)
    ev.plot(result, save_png=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from EXACT.evaluators import BaseEvaluator


class LocalizationEvaluator(BaseEvaluator):
    """
    Evaluates heatmap localisation quality against a ground-truth mask.

    Parameters
    ----------
    iou_threshold : float
        Binarisation threshold for IoU. Default 0.5.
    save_dir : str
        Directory for saved plots.
    """

    THRESHOLDS = {
        "iou":           [(0.50, "Excellent"), (0.25, "Good")],
        "pointing_game": [(1.00, "Excellent"), (0.50, "Good")],
        "energy_inside": [(0.70, "Excellent"), (0.50, "Good")],
    }

    WEIGHTS = {
        "iou":           0.40,
        "pointing_game": 0.25,
        "energy_inside": 0.35,
    }

    def __init__(
        self,
        iou_threshold: float = 0.5,
        save_dir: str = "user_saves/evaluator_saves/localization",
    ):
        super().__init__(save_dir)
        self.iou_threshold = iou_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        explainer_result: dict,
        gt_mask: np.ndarray,
        iou_threshold: Optional[float] = None,
    ) -> dict:
        """
        Compute IoU, Pointing Game, and Energy Inside Mask.

        Parameters
        ----------
        explainer_result : dict
            Output of any EXACT explainer's explain() method.
            Must contain a 'heatmap' key (or legacy 'cam').
        gt_mask : np.ndarray
            Ground-truth foreground mask, shape (H, W).
            Values: 1 (or True) = foreground, 0 = background.
            Can be a segmentation mask or a bounding-box fill.
        iou_threshold : float, optional
            Overrides the instance default binarisation threshold.

        Returns
        -------
        dict with keys:
            'evaluator'    : str
            'scores'       : dict[metric -> float]
            'grades'       : dict[metric -> str]
            'overall'      : float
            'overall_grade': str
            'heatmap'      : np.ndarray  normalised (H, W)
            'gt_mask'      : np.ndarray  resized binary (H, W)
            'binary_cam'   : np.ndarray  binarised heatmap used for IoU
            'iou_threshold': float
        """
        threshold = iou_threshold or self.iou_threshold
        cam       = self._normalize(self._extract_heatmap(explainer_result))

        # Align gt_mask to heatmap spatial dims
        h, w   = cam.shape
        gt     = (gt_mask > 0).astype(np.float32)
        if gt.shape != (h, w):
            gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)

        binary_cam   = (cam >= threshold).astype(np.uint8)
        gt_bin       = (gt > 0).astype(np.uint8)

        iou           = self._iou(binary_cam, gt_bin)
        pointing      = self._pointing_game(cam, gt_bin)
        energy_inside = self._energy_inside(cam, gt_bin)

        scores = {
            "iou":           round(iou, 4),
            "pointing_game": round(pointing, 4),
            "energy_inside": round(energy_inside, 4),
        }
        grades  = {m: self._grade(m, v) for m, v in scores.items()}
        overall = round(self._weighted_composite(scores, self.WEIGHTS), 4)

        return {
            "evaluator":     "Localization Evaluator",
            "scores":        scores,
            "grades":        grades,
            "overall":       overall,
            "overall_grade": self._grade_overall(overall),
            "heatmap":       cam,
            "gt_mask":       gt,
            "binary_cam":    binary_cam,
            "iou_threshold": threshold,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
        inter = (pred & gt).sum()
        union = (pred | gt).sum()
        return float(inter / (union + 1e-9))

    @staticmethod
    def _pointing_game(cam: np.ndarray, gt: np.ndarray) -> float:
        peak = np.unravel_index(cam.argmax(), cam.shape)
        return float(gt[peak] > 0)

    @staticmethod
    def _energy_inside(cam: np.ndarray, gt: np.ndarray) -> float:
        total  = cam.sum() + 1e-9
        inside = (cam * gt).sum()
        return float(inside / total)

    @staticmethod
    def _grade_overall(overall: float) -> str:
        if overall >= 0.55:
            return "Excellent"
        if overall >= 0.35:
            return "Good"
        return "Poor"

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def _plot_body(self, results: dict, save_png: bool, filename: Optional[str]) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.colors import ListedColormap

        BG, CARD  = "#0D1117", "#161B22"
        TEXT, SUB = "#E6EDF3", "#8B949E"

        scores  = results["scores"]
        grades  = results["grades"]
        overall = results["overall"]
        og      = results["overall_grade"]
        cam     = results["heatmap"]
        gt      = results["gt_mask"]
        binary  = results["binary_cam"]
        thresh  = results["iou_threshold"]

        # Overlap visualisation: TP/FP/FN map
        tp = (binary & gt.astype(np.uint8)).astype(np.float32)     # green
        fp = (binary & (1 - gt.astype(np.uint8))).astype(np.float32)  # red
        fn = ((1 - binary) & gt.astype(np.uint8)).astype(np.float32)  # blue
        overlap_rgb = np.stack([fp, tp, fn], axis=-1)               # (H,W,3)

        fig = plt.figure(figsize=(16, 6), facecolor=BG)
        fig.suptitle(f"EXACT -- Localization Evaluation  (IoU threshold={thresh})",
                     color=TEXT, fontsize=13, fontweight="bold",
                     fontfamily="monospace", y=0.97)

        gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.5, wspace=0.35,
                               top=0.88, bottom=0.10, left=0.04, right=0.97)

        # ── Heatmap ───────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.imshow(cam, cmap="jet", vmin=0, vmax=1)
        ax1.set_title("Heatmap", color=TEXT, fontsize=10, fontweight="bold")
        ax1.axis("off")

        # ── GT mask ───────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.imshow(gt, cmap="Greens", vmin=0, vmax=1)
        ax2.set_title("GT Mask", color=TEXT, fontsize=10, fontweight="bold")
        ax2.axis("off")

        # ── Binarised CAM ─────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[:, 2])
        ax3.imshow(binary, cmap="hot", vmin=0, vmax=1)
        ax3.set_title(f"Binary CAM (thr={thresh})", color=TEXT,
                      fontsize=10, fontweight="bold")
        ax3.axis("off")

        # ── Overlap TP/FP/FN ─────────────────────────────────────────
        ax4 = fig.add_subplot(gs[:, 3])
        ax4.imshow(np.clip(overlap_rgb, 0, 1))
        ax4.set_title("TP/FP/FN  (G/R/B)", color=TEXT,
                      fontsize=10, fontweight="bold")
        ax4.axis("off")

        # Mark peak pixel for pointing game
        peak = np.unravel_index(cam.argmax(), cam.shape)
        peak_color = "#54C27D" if scores["pointing_game"] == 1.0 else "#F4845F"
        ax4.plot(peak[1], peak[0], marker="x", color=peak_color,
                 markersize=10, markeredgewidth=2,
                 label=f"Peak ({'in' if scores['pointing_game']==1.0 else 'out'})")
        ax4.legend(facecolor=CARD, labelcolor=TEXT, fontsize=7,
                   edgecolor="#30363D", loc="lower right")

        # ── Scores + overall ─────────────────────────────────────────
        ax5 = fig.add_subplot(gs[0, 4])
        ax5.set_facecolor(CARD); ax5.axis("off")
        metrics = list(scores.keys())
        vals    = [scores[m] for m in metrics]
        colors  = [self._grade_color(grades[m]) for m in metrics]
        bars    = ax5.barh(metrics[::-1], vals[::-1], color=colors[::-1],
                           height=0.5, edgecolor="#30363D")
        ax5.set_xlim(0, 1.25)
        ax5.set_facecolor(CARD)
        ax5.tick_params(axis="y", labelcolor=TEXT, labelsize=8)
        ax5.tick_params(axis="x", labelcolor=SUB, labelsize=7)
        ax5.spines[["top","right","left","bottom"]].set_color("#30363D")
        for bar, val, m in zip(bars, vals[::-1], metrics[::-1]):
            ax5.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}  {grades[m]}",
                     va="center", color=TEXT, fontsize=7, fontfamily="monospace")

        ax6 = fig.add_subplot(gs[1, 4])
        ax6.set_facecolor(CARD); ax6.axis("off")
        oc = self._grade_color(og)
        ax6.text(0.5, 0.65, f"{overall:.4f}", ha="center", va="center",
                 color=oc, fontsize=26, fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.5, 0.30, og, ha="center", va="center",
                 color=oc, fontsize=12, fontweight="bold", transform=ax6.transAxes)
        ax6.text(0.5, 0.10, "Overall Localization", ha="center", va="center",
                 color=SUB, fontsize=8, transform=ax6.transAxes)

        if save_png:
            fname = filename or "localization_eval.png"
            out   = self.save_dir / fname
            fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=BG)
            print(f"Saved: {out}")
        else:
            plt.show()
        plt.close(fig)