"""
lrp_tabular.py
========================
Layer-wise Relevance Propagation (LRP) explainer for PyTorch tabular classification models.

This file blends the robust mathematical backend of LRPImageExplainer (which seamlessly
supports MLPs) with the rich 5-chart dashboard of the tabular explainer suite to
give deep insights into which tabular features are most important for your model.
"""

import warnings
import torch
import torch.nn as nn
import numpy as np

import matplotlib
if matplotlib.get_backend() == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from typing import Dict, List, Optional
from EXACT.explainers.lrp_explainer import LRPImageExplainer

# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

def _apply_dark_theme(ax: plt.Axes) -> None:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="#ccccee", labelsize=8)
    ax.xaxis.label.set_color("#ccccee")
    ax.yaxis.label.set_color("#ccccee")
    ax.title.set_color("white")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#444466")

def _get_top_k_indices(attr: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(np.abs(attr))[::-1][:k]

# =============================================================================
# MAIN CLASS
# =============================================================================

class LRPTabularExplainer:
    """
    Layer-wise Relevance Propagation (LRP) explainer for PyTorch tabular classification MLPs.

    Usage:
        explainer = LRPTabularExplainer(model, feature_names=["age", "income"])
        results   = explainer.explain(input_tensor, training_data=X_train)
        explainer.save_dashboard(results, "explanation.png")
    """

    def __init__(
        self,
        model:         nn.Module,
        feature_names: Optional[List[str]] = None,
        device:        Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.device = device or next(model.parameters()).device
        self.feature_names = feature_names

        # Detect Style B ([batch,1] sigmoid) model
        try:
            n_in = None
            for p in model.parameters():
                if p.ndim == 2:
                    n_in = p.shape[1]
                    break
            if n_in is None:
                raise ValueError("No 2D parameter found — cannot probe model input size.")
            dummy = torch.zeros(1, n_in, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                dummy_out = model(dummy)
            self._is_style_b = (dummy_out.dim() == 2 and dummy_out.shape[1] == 1)
        except Exception:
            self._is_style_b = False

        # Use LRPImageExplainer for the math (it fully supports MLPs seamlessly)
        self._lrp_engine = LRPImageExplainer(self.model, device=self.device)
        
        if self._lrp_engine.model_type != "mlp":
            warnings.warn(
                f"Model detected as '{self._lrp_engine.model_type}'. Expected 'mlp' "
                f"for tabular explanation. It might still run if inputs are correctly formatted.",
                UserWarning,
                stacklevel=2,
            )

    def _safe_forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if output.dim() == 2 and output.shape[1] == 1:
            p = output
            output = torch.cat([1 - p, p], dim=1)
        return output

    def explain(
        self,
        input_tensor:  torch.Tensor,
        target_class:  Optional[int]         = None,
        training_data: Optional[np.ndarray]  = None,
        top_k:         Optional[int]         = None,
        eps:           float                 = 1e-6,
    ) -> Dict:
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)
            
        if input_tensor.dim() != 2 or input_tensor.shape[0] != 1:
            raise ValueError(
                f"input_tensor must be shape [F] or [1, F], "
                f"but got shape {tuple(input_tensor.shape)}."
            )

        input_tensor = input_tensor.float().to(self.device)
        F = input_tensor.shape[1]

        if self.feature_names is not None:
            names = self.feature_names
            if len(names) != F:
                raise ValueError("Feature names length does not match input length.")
        else:
            names = [f"F{i}" for i in range(F)]

        if self._is_style_b:
            warnings.warn(
                "Model output shape is [batch, 1]. Auto-converting to [batch, 2].",
                UserWarning, stacklevel=2
            )

        self.model.eval()

        with torch.no_grad():
            logits = self._safe_forward(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        f_input = float(logits[0, target_class].item())

        # Perform the actual LRP pass
        # pixel_low and pixel_high are ignored by epsilon rule for MLPs
        relevance_map, completeness_err, rule_log = self._lrp_engine._lrp_pass(
            input_tensor=input_tensor,
            target_class=target_class,
            pixel_low=0.0,
            pixel_high=1.0,
            eps=eps,
            lrp_alpha=1.0,
            lrp_beta=0.0
        )
        
        attr_np = relevance_map.squeeze(0).cpu().numpy()
        input_np = input_tensor.squeeze(0).cpu().numpy()
        
        if training_data is not None:
            base_np = np.array(training_data.mean(axis=0), dtype=np.float32)
        else:
            base_np = np.zeros_like(input_np)

        k = min(top_k if top_k is not None else F, F)

        return {
            "target_class":       target_class,
            "convergence_delta":  completeness_err,
            "feature_names":      names,
            "attributions":       attr_np,
            "input_values":       input_np,
            "baseline_values":    base_np,
            "training_data":      training_data,
            "top_k":              k,
            "chart_bar":          self._plot_bar(attr_np, names, target_class, k),
            "chart_force":        self._plot_force(attr_np, names, target_class, k, input_np, f_input),
            "chart_waterfall":    self._plot_waterfall(attr_np, names, target_class, k),
            "chart_distribution": self._plot_distribution(attr_np, names, target_class, k, input_np, base_np, training_data),
            "chart_summary":      self._plot_summary(attr_np, names, target_class, k),
        }

    def save_dashboard(
        self,
        results:    Dict,
        save_path:  str,
        class_name: Optional[str] = None,
        dpi:        int           = 150,
    ) -> None:
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        delta  = results["convergence_delta"]
        label  = class_name or f"Class {results['target_class']}"

        if delta < 0.05:
            quality = "EXCELLENT"
        elif delta < 0.15:
            quality = "OK"
        else:
            quality = "!! check completeness"

        fig = plt.figure(figsize=(20, 12), facecolor="#111122")
        fig.suptitle(
            f"LRP — Tabular Explanation\n"
            f"{label}  |  Completeness Error = {delta:.4f}  [{quality}]",
            color="white", fontsize=13, fontweight="bold", y=0.99,
        )

        outer = GridSpec(2, 1, figure=fig, hspace=0.38, top=0.93, bottom=0.03)
        top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], wspace=0.25)
        bottom = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.30)

        slots  = [top[0],    top[1],         bottom[0],         bottom[1],          bottom[2]]
        charts = ["chart_bar", "chart_force", "chart_waterfall", "chart_distribution", "chart_summary"]

        for slot, key in zip(slots, charts):
            ax = fig.add_subplot(slot)
            src_fig = results[key]
            src_fig.canvas.draw()
            w, h = src_fig.canvas.get_width_height()
            img  = np.frombuffer(src_fig.canvas.buffer_rgba(),
                                 dtype=np.uint8).reshape(h, w, 4)
            ax.imshow(img)
            ax.axis("off")

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

        for key in charts:
            plt.close(results[key])

    # =========================================================================
    # PRIVATE FORMATTING & PLOTTING METHODS
    # =========================================================================

    def _plot_bar(
        self,
        attr:         np.ndarray,
        names:        List[str],
        target_class: int,
        k:            int,
    ) -> plt.Figure:
        idx = _get_top_k_indices(attr, k)
        vals   = attr[idx][::-1]
        labels = [names[i] for i in idx][::-1]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]

        fig, ax = plt.subplots(figsize=(7, max(3, k * 0.45 + 1)), facecolor="#111122")
        _apply_dark_theme(ax)

        bars = ax.barh(labels, vals, color=colors, edgecolor="#2a2a4a", height=0.65)
        max_abs = float(np.abs(vals).max()) or 1.0

        for bar, v in zip(bars, vals):
            offset = max_abs * 0.025
            x_pos  = v + offset if v >= 0 else v - offset
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{v:+.4f}",
                va="center",
                ha="left" if v >= 0 else "right",
                color="white", fontsize=7.5, fontweight="bold",
            )

        ax.axvline(0, color="#aaaacc", lw=0.8, alpha=0.6)
        ax.set_xlabel("Relevance")
        ax.set_title(f"Feature Relevance (Top {k})  |  Class {target_class}", fontsize=10, fontweight="bold", pad=8)
        ax.legend(
            handles=[
                mpatches.Patch(color="#2ecc71", label="Supports prediction"),
                mpatches.Patch(color="#e74c3c", label="Suppresses prediction"),
            ],
            loc="lower right", facecolor="#111122", edgecolor="#444466", labelcolor="#ccccee", fontsize=7.5,
        )

        plt.tight_layout()
        return fig

    def _plot_force(
        self,
        attr:           np.ndarray,
        names:          List[str],
        target_class:   int,
        k:              int,
        in_np:          np.ndarray,
        f_input:        float,
    ) -> plt.Figure:
        idx     = _get_top_k_indices(attr, k)
        vals    = attr[idx]
        labels  = [names[i] for i in idx]
        invals  = in_np[idx]
        max_abs = float(np.abs(vals).max()) or 1.0

        total_sum      = float(attr.sum())
        baseline_score = f_input - total_sum

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#111122")
        _apply_dark_theme(ax)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-2.4, 2.4)
        ax.axis("off")
        ax.set_title(
            f"Force Plot  |  Class {target_class}  |  "
            f"Baseline = {baseline_score:.3f}   →   Prediction = {f_input:.3f}",
            fontsize=11, fontweight="bold", pad=10,
        )

        ax.annotate(
            "", xy=(0.93, 0.0), xytext=(0.07, 0.0),
            arrowprops=dict(arrowstyle="-|>", color="#555577", lw=2.5, mutation_scale=16),
        )

        ax.text(
            0.04, 0.0, f"Baseline\n{baseline_score:.3f}",
            ha="center", va="center", color="#ccccee", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a2e", ec="#555577", lw=1.5),
        )

        ax.text(
            0.96, 0.0, f"Score\n{f_input:.3f}",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="#0d2b1a", ec="#2ecc71", lw=2.0),
        )

        xs = np.linspace(0.13, 0.87, max(len(vals), 1))

        for i, (v, label, iv) in enumerate(zip(vals, labels, invals)):
            pos   = v >= 0
            color = "#2ecc71" if pos else "#5588ff"
            tc    = "#2ecc71" if pos else "#7aadff"
            sign  = 1 if pos else -1

            stem = sign * (0.15 + abs(v) / max_abs * 0.85)
            x    = xs[i]

            ax.plot([x, x], [sign * 0.10, stem], color=color, lw=1.8)
            ax.annotate(
                "", xy=(x + (0.03 if pos else -0.03), stem), xytext=(x, stem),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, mutation_scale=9),
            )

            ax.text(
                x, stem + sign * 0.22, f"{label}\n={iv:.2f}\n{v:+.3f}",
                ha="center", va="bottom" if pos else "top",
                color=tc, fontsize=6.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="#0a2a0a" if pos else "#0a0a2a", ec=color, lw=1.0),
            )

        ax.text(
            0.5, -2.2, "▲ Green (above) pushes score UP   ▼ Blue (below) pushes score DOWN",
            ha="center", color="#aaaacc", fontsize=8, style="italic",
        )

        plt.tight_layout()
        return fig

    def _plot_waterfall(
        self,
        attr:         np.ndarray,
        names:        List[str],
        target_class: int,
        k:            int,
    ) -> plt.Figure:
        idx    = _get_top_k_indices(attr, k)
        vals   = list(attr[idx])
        labels = [names[i] for i in idx]

        if k < len(attr):
            vals.append(float(attr.sum() - sum(vals)))
            labels.append("Others")

        running = np.zeros(len(vals) + 1)
        for i, v in enumerate(vals):
            running[i + 1] = running[i] + v

        bottoms = running[:-1]
        colors  = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
        xs      = np.arange(len(vals))

        fig, ax = plt.subplots(figsize=(max(5, len(vals) * 0.85 + 1), 5), facecolor="#111122")
        _apply_dark_theme(ax)

        ax.bar(xs, vals, bottom=bottoms, color=colors, edgecolor="#2a2a4a", width=0.62)

        for i in range(len(vals) - 1):
            y_top = bottoms[i] + vals[i]
            ax.plot([xs[i] + 0.31, xs[i + 1] - 0.31], [y_top, y_top], color="#7777aa", lw=0.9, ls="--", alpha=0.7)

        span = float(np.abs(running).max()) or 1.0
        for i, v in enumerate(vals):
            ax.text(
                xs[i], bottoms[i] + v + span * 0.03, f"{v:+.3f}",
                ha="center", va="bottom", color="white", fontsize=7.5, fontweight="bold",
            )

        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.axhline(0, color="#aaaacc", lw=0.9, alpha=0.6)
        ax.set_ylabel("Cumulative Relevance")
        ax.set_title(f"Waterfall — Relevance Build-Up  |  Class {target_class}", fontsize=10, fontweight="bold", pad=8)
        plt.tight_layout()
        return fig

    def _plot_distribution(
        self,
        attr:          np.ndarray,
        names:         List[str],
        target_class:  int,
        k:             int,
        in_np:         np.ndarray,
        base_np:       np.ndarray,
        training_data: Optional[np.ndarray],
    ) -> plt.Figure:
        idx   = _get_top_k_indices(attr, k)
        n     = len(idx)
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8 + 0.6),
            facecolor="#111122", squeeze=False,
        )
        fig.suptitle(
            f"Feature Value vs Training Distribution (Top {k})  |  Class {target_class}\n"
            f"Orange dashed = dataset mean   Coloured dot = this sample",
            color="white", fontsize=10, fontweight="bold", y=1.02,
        )

        for plot_i, feat_i in enumerate(idx):
            row = plot_i // ncols
            col = plot_i % ncols
            ax  = axes[row][col]
            _apply_dark_theme(ax)

            v        = attr[feat_i]
            dot_color = "#2ecc71" if v >= 0 else "#e74c3c"

            if training_data is not None:
                ax.hist(training_data[:, feat_i], bins=25, color="#445588", alpha=0.75, edgecolor="#2a2a4a", lw=0.5)
                y_max = ax.get_ylim()[1]
                ax.axvline(base_np[feat_i],  color="#f39c12", lw=1.8, ls="--")
                ax.axvline(in_np[feat_i],    color=dot_color, lw=1.4)
                ax.scatter([in_np[feat_i]], [y_max * 0.06], color=dot_color, s=80, zorder=5)
                ax.set_ylabel("Count")
            else:
                ax.bar(["Mean ref.", "Sample"], [base_np[feat_i], in_np[feat_i]], color=["#f39c12", dot_color], edgecolor="#2a2a4a", width=0.5)
                ax.set_ylabel("Value")

            ax.set_title(f"{names[feat_i]}\nrelevance = {v:+.3f}", fontsize=8, fontweight="bold", pad=4)

        for plot_i in range(n, nrows * ncols):
            axes[plot_i // ncols][plot_i % ncols].set_visible(False)

        plt.tight_layout()
        return fig

    def _plot_summary(
        self,
        attr:         np.ndarray,
        names:        List[str],
        target_class: int,
        k:            int,
    ) -> plt.Figure:
        idx     = _get_top_k_indices(attr, k)
        vals    = attr[idx]
        labels  = [names[i] for i in idx]
        abs_max = float(np.abs(vals).max()) or 1.0

        rows        = []
        cell_colors = []

        for rank, (name, v) in enumerate(zip(labels, vals), start=1):
            direction   = "▲ Supports"  if v >= 0 else "▼ Suppresses"
            block_count = max(1, int(abs(v) / abs_max * 12))
            importance  = "█" * block_count

            rows.append([str(rank), name, f"{v:+.4f}", direction, importance])
            row_bg = "#0d2b1a" if v >= 0 else "#2b0d0d"
            cell_colors.append([row_bg] * 5)

        fig, ax = plt.subplots(figsize=(8, max(2.8, k * 0.42 + 1.5)), facecolor="#111122")
        ax.set_facecolor("#111122")
        ax.axis("off")

        header      = [["Rank", "Feature", "Relevance", "Direction", "Importance"]]
        header_bg   = [["#2c2c4e"] * 5]

        table = ax.table(cellText=header + rows, cellColours=header_bg + cell_colors, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.45)

        for col in range(5):
            table[0, col].set_text_props(color="white", fontweight="bold")
            table[0, col].set_edgecolor("#555577")

        for row in range(1, len(rows) + 1):
            is_positive = rows[row - 1][2].startswith("+")
            for col in range(5):
                cell = table[row, col]
                cell.set_edgecolor("#333355")
                if col in (3, 4):
                    cell.set_text_props(color="#2ecc71" if is_positive else "#e74c3c", fontweight="bold" if col == 3 else "normal")
                else:
                    cell.set_text_props(color="white")

        ax.set_title(f"Feature Relevance Summary (Top {k})  |  Class {target_class}", color="white", fontsize=10, fontweight="bold", pad=10)
        plt.tight_layout()
        return fig
