"""
EXACT — EXplainability and Attribution for Classification Tasks
===============================================================
comparators/text_comp.py

Compares word-level importance scores from any number of text explainers.
Works with LIME, LOO, and any future explainer out of the box.

What it produces
----------------
Two separate results the user can call independently:

    plot_words(results, save_png=True)
        → One bar chart per explainer, all aligned on the same word axis,
          placed side by side in a single PNG.
          Shows which words each explainer considers important.

    plot_scores(results, save_png=True)
        → A score comparison chart showing quality metrics per explainer:
          Coverage, Concentration, Polarity Balance, Agreement, and a
          Composite score with a ranked winner.

    report(results)
        → Prints both of the above as formatted text to the terminal.

    plot(results, save_png=True)
        → Saves both panels together in one PNG.

Usage
-----
    from EXACT.comparators.text_comp import TextComparator

    cmp = TextComparator(class_names=["negative", "positive"])

    # Step 1 — run your explainers
    lime_result = lime_explainer.explain(text)
    loo_result  = loo_explainer.explain(text)

    # Step 2 — compare
    results = cmp.compare(
        explanations = {"LIME": lime_result, "LOO": loo_result},
        text = text,
    )

    # Step 3 — output
    cmp.report(results)                          # terminal
    cmp.plot_words(results,  save_png=True)      # bar plots only
    cmp.plot_scores(results, save_png=True)      # score chart only
    cmp.plot(results,        save_png=True)      # both together

Adding any future explainer
----------------------------
    shap_scores = {"horrible": 0.71, "good": -0.30}
    results = cmp.compare(
        explanations = {
            "LIME" : lime_result,
            "LOO"  : loo_result,
            "SHAP" : shap_scores,   # plain dict — just works
        },
        text = text,
    )
"""

import textwrap
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_HERE         = Path(__file__).resolve().parent   # .../src/EXACT/comparators/
_PROJECT_ROOT = _HERE.parent.parent.parent
_USER_SAVES   = _PROJECT_ROOT / "user_saves"


def _saves():
    _USER_SAVES.mkdir(parents=True, exist_ok=True)
    return _USER_SAVES


# ─────────────────────────────────────────────────────────────────────────────
# Extraction — converts any explanation to {word: score}
# ─────────────────────────────────────────────────────────────────────────────

def _extract(name, exp):
    '''
    Convert any explanation object to {word: score}.

    Auto-detects:
      LIME  →  lime.explanation.Explanation  (has .as_list + .top_labels)
      LOO   →  LOOTextExplanation            (has .tokens  + .importances)
      dict  →  {word: score}                 passed through directly

    For any other type, raises a clear error with instructions.
    '''
    if isinstance(exp, dict):
        return {str(k).lower(): float(v) for k, v in exp.items()}

    if hasattr(exp, "as_list") and hasattr(exp, "top_labels"):
        label = exp.top_labels[0]
        return {w.lower(): float(s) for w, s in exp.as_list(label=label)}

    if hasattr(exp, "tokens") and hasattr(exp, "importances"):
        out = {}
        for tok, imp in zip(exp.tokens, exp.importances):
            k = tok.lower()
            if k not in out or abs(float(imp)) > abs(out[k]):
                out[k] = float(imp)
        return out

    raise TypeError(
        f"[EXACT] Cannot read scores from '{name}' "
        f"(type: {type(exp).__name__}).\n"
        f"Extract word scores yourself and pass a plain dict:\n"
        f"    {{'word1': 0.4, 'word2': -0.2, ...}}\n"
    )


def _predicted_class(exp, class_names):
    if hasattr(exp, "class_label"):
        return exp.class_label
    if hasattr(exp, "top_labels"):
        idx = exp.top_labels[0]
        return class_names[idx] if class_names and idx < len(class_names) else str(idx)
    return "?"


def _confidence(exp):
    return float(exp.predicted_proba) if hasattr(exp, "predicted_proba") else None


def _norm(scores):
    m = max((abs(v) for v in scores.values()), default=1e-12)
    return {k: v / m for k, v in scores.items()} if m > 1e-12 else dict(scores)


def _rank(scores):
    return {w: i + 1 for i, w in
            enumerate(sorted(scores, key=lambda k: abs(scores[k]), reverse=True))}


def _sentence_words(explanations):
    for exp in explanations.values():
        if hasattr(exp, "tokens"):
            return [t.lower() for t in exp.tokens]
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ─────────────────────────────────────────────────────────────────────────────

def _coverage(scores, sentence_words):
    '''
    What fraction of the sentence words did this explainer score?

    LIME only scores its top-N features, so coverage is often < 1.0.
    LOO scores every word, so coverage is always 1.0.
    Higher is better — a fuller explanation.
    '''
    if not sentence_words:
        return 0.0
    return round(sum(1 for w in sentence_words if w in scores) / len(sentence_words), 4)


def _concentration(scores):
    '''
    What fraction of the total importance is held by the top 3 words?

    Higher = explanation is focused on a few key words (easier to read).
    Lower  = importance is spread evenly across many words.
    '''
    if not scores:
        return 0.0
    vals  = sorted(abs(v) for v in scores.values())[::-1]
    total = sum(vals)
    return round(sum(vals[:3]) / total, 4) if total > 1e-12 else 0.0


def _polarity(scores):
    '''
    How balanced are the positive and negative scores?

    1.0 = equal positive and negative mass (captures both sides of sentiment).
    0.0 = all one-sided (only positive or only negative words scored).
    '''
    if not scores:
        return 0.0
    pos = sum(v for v in scores.values() if v > 0)
    neg = sum(abs(v) for v in scores.values() if v < 0)
    tot = pos + neg
    return round(2 * min(pos, neg) / tot, 4) if tot > 1e-12 else 0.0


def _agreement(scores_a, scores_b):
    '''
    On shared words, what fraction do both explainers assign the same sign?

    1.0 = perfect agreement. 0.0 = perfect disagreement.
    Only computed on words that appear in both explanations.
    '''
    shared = set(scores_a) & set(scores_b)
    if not shared:
        return 0.0
    return round(
        sum(1 for w in shared if (scores_a[w] > 0) == (scores_b[w] > 0))
        / len(shared), 4
    )


def _weights(metrics):
    base  = {"coverage": 0.25, "concentration": 0.25,
             "polarity": 0.20, "agreement": 0.30}
    act   = {m: base[m] for m in metrics if m in base}
    total = sum(act.values())
    return {m: v / total for m, v in act.items()} if total else \
           {m: 1 / len(metrics) for m in metrics}


# ─────────────────────────────────────────────────────────────────────────────
# TextComparator
# ─────────────────────────────────────────────────────────────────────────────

class TextComparator:

    '''
    Compares word-level importance from any number of text explainers.

    Works with LIME and LOO automatically.
    Works with any future explainer when passed as a {word: score} dict.

    Parameters
    -----------
    class_names : list[str], optional
        Human-readable class labels.
    top_n : int
        Number of words to show. Default 10.
    '''

    def __init__(self, class_names=None, top_n=10):
        self.class_names = class_names
        self.top_n       = top_n


    # ─────────────────────────────────────────────────────────────────
    # compare()
    # ─────────────────────────────────────────────────────────────────

    def compare(self, explanations, text=None):
        '''
        Extract scores and compute quality metrics for every explainer.

        Parameters
        -----------
        explanations : dict
            { "LIME": lime_exp, "LOO": loo_exp, "SHAP": {word: score}, ... }
        text : str, optional
            Original input text. Auto-read from LOO if not given.

        Returns
        --------
        results dict — pass to report(), plot_words(), plot_scores(), plot().
        '''
        if not explanations:
            raise ValueError("explanations dict is empty.")

        names      = list(explanations.keys())
        raw        = {n: _extract(n, e) for n, e in explanations.items()}
        normalised = {n: _norm(s)       for n, s in raw.items()}
        ranks      = {n: _rank(s)       for n, s in raw.items()}

        # input text
        input_text = text
        if not input_text:
            for exp in explanations.values():
                if hasattr(exp, "original_text"):
                    input_text = exp.original_text
                    break
        input_text  = input_text or ""
        sent_words  = input_text.lower().split()

        # quality metrics
        metrics = ["coverage", "concentration", "polarity"]
        scores  = {n: {} for n in names}

        for n in names:
            scores[n]["coverage"]      = _coverage(raw[n], sent_words)
            scores[n]["concentration"] = _concentration(raw[n])
            scores[n]["polarity"]      = _polarity(raw[n])

        if len(names) >= 2:
            for n in names:
                pairs = [_agreement(raw[n], raw[o])
                         for o in names if o != n]
                scores[n]["agreement"] = round(float(np.mean(pairs)), 4)
            metrics.append("agreement")

        weights_dict = _weights(metrics)
        summary = {n: round(sum(weights_dict.get(m, 0) * scores[n].get(m, 0)
                               for m in metrics), 4)
                   for n in names}
        ranked  = sorted(summary.items(), key=lambda x: x[1], reverse=True)
        winner  = ranked[0][0] if ranked else None

        # words to display — union of top_n from every explainer,
        # in original sentence order when possible
        candidates = set()
        for s in raw.values():
            top = sorted(s, key=lambda k: abs(s[k]), reverse=True)
            candidates.update(top[:self.top_n])

        order = _sentence_words(explanations)
        if order:
            display = [word for word in order if word in candidates]
            for word in candidates:
                if word not in display:
                    display.append(word)
        else:
            display = sorted(candidates,
                             key=lambda word: np.mean([abs(raw[n].get(word, 0))
                                                       for n in names]),
                             reverse=True)

        # per-explainer label info
        label_info = {}
        for n in names:
            exp = explanations[n]
            label_info[n] = {
                "class": _predicted_class(exp, self.class_names),
                "conf" : _confidence(exp),
            }

        return {
            "names"       : names,
            "raw"         : raw,
            "normalised"  : normalised,
            "ranks"       : ranks,
            "scores"      : scores,
            "metrics"     : metrics,
            "weights"     : weights_dict,
            "summary"     : summary,
            "ranked"      : ranked,
            "winner"      : winner,
            "display"     : display,
            "text"        : input_text,
            "label_info"  : label_info,
        }


    # ─────────────────────────────────────────────────────────────────
    # report()  —  terminal output
    # ─────────────────────────────────────────────────────────────────

    def report(self, results):
        '''
        Print word importances and score comparison to the terminal.

        Parameters
        -----------
        results : dict   output of compare()
        '''
        names   = results["names"]
        display = results["display"]
        raw     = results["raw"]
        ranks   = results["ranks"]
        scores  = results["scores"]
        metrics = results["metrics"]
        ranked  = results["ranked"]
        winner  = results["winner"]
        text    = results["text"]
        w_map   = results["weights"]

        # ── header ───────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("  EXACT  —  Text Explanation Comparison")
        print("=" * 72)
        print(f"  Input : {textwrap.shorten(text, 65)}")
        for n in names:
            info = results["label_info"][n]
            cls  = info["class"]
            conf = f"  (conf {info['conf']:.4f})" if info["conf"] else ""
            print(f"  {n:<8} →  Predicted: {cls}{conf}")
        print("=" * 72)

        # ── PART 1: word importances ──────────────────────────────────
        print("\n  PART 1 — Word Importances")
        print("  " + "-" * 68)

        bar_w = max(14, 52 // len(names))
        w_col = 14

        # header row
        hdr = f"  {'Word':<{w_col}}"
        for n in names:
            hdr += f"  {'◀─ ' + n + ' ─▶':^{bar_w + 9}}"
        print(hdr)

        sub = f"  {'':<{w_col}}"
        for _ in names:
            sub += f"  {'Score':>9}  {'Rank':>4}  {'Bar':<{bar_w}}"
        print(sub)
        print("  " + "-" * 68)

        for word in display:
            line = f"  {word:<{w_col}}"
            for n in names:
                rv  = raw[n].get(word)
                nv  = results["normalised"][n].get(word, 0.0)
                rk  = ranks[n].get(word)

                if rv is not None:
                    fill = int(bar_w * abs(nv))
                    sign = "▶" if rv >= 0 else "◀"
                    bar  = (sign * fill).ljust(bar_w)
                    scr  = f"{rv:>+9.4f}"
                    rkk  = f"#{rk:>3}"
                else:
                    bar  = " " * bar_w
                    scr  = f"{'N/A':>9}"
                    rkk  = f"{'N/A':>4}"
                line += f"  {scr}  {rkk}  {bar}"
            print(line)

        print("  " + "-" * 68)

        # ── PART 2: score comparison ──────────────────────────────────
        print("\n  PART 2 — Score Comparison")
        print("  " + "-" * 68)

        col_w = max(12, max(len(n) for n in names) + 2)
        met_w = 20

        print(f"  {'Metric':<{met_w}}" +
              "".join(f"{n:>{col_w}}" for n in names))
        print("  " + "-" * (met_w + col_w * len(names)))

        guide = {
            "coverage"     : "fraction of sentence words scored",
            "concentration": "top-3 words share of total importance",
            "polarity"     : "balance of positive vs negative words",
            "agreement"    : "directional agreement with other explainers",
        }
        for m in metrics:
            vals = [scores[n].get(m, float("nan")) for n in names]
            best = max((v for v in vals if not np.isnan(v)), default=None)
            row  = f"  {m:<{met_w}}"
            for v in vals:
                if np.isnan(v):
                    row += f"{'N/A':>{col_w}}"
                else:
                    star = " ★" if best is not None and abs(v - best) < 1e-9 else "  "
                    row += f"{f'{v:.4f}{star}':>{col_w}}"
            print(row)

        print("  " + "-" * (met_w + col_w * len(names)))
        comp = f"  {'COMPOSITE':<{met_w}}"
        for n in names:
            comp += f"{results['summary'][n]:>{col_w}.4f}"
        print(comp)
        print("  " + "-" * 68)

        print("\n  RANKED  (higher composite = better explanation quality)")
        medals = ["[1]", "[2]", "[3]"] + ["   "] * 20
        for i, (n, s) in enumerate(ranked):
            bar = "█" * int(s * 28) + "░" * (28 - int(s * 28))
            print(f"  {medals[i]}  {n:<18}  {s:.4f}  {bar}")

        print(f"\n  Winner   : {winner}")
        print(f"  Weights  : { {k: round(v, 2) for k, v in w_map.items()} }")

        print("\n  Metric guide:")
        for m in metrics:
            if m in guide:
                print(f"    {m:<16} — {guide[m]}")
        print("=" * 72 + "\n")


    # ─────────────────────────────────────────────────────────────────
    # plot_words()  —  bar plots side by side
    # ─────────────────────────────────────────────────────────────────

    def plot_words(self, results, save_png=False, filename=None):
        '''
        Save a PNG with one bar chart per explainer, all aligned on the
        same word axis and placed side by side.

        Green bars  = word supports the prediction (positive importance).
        Red bars    = word opposes the prediction  (negative importance).
        Grey bars   = word not scored by this explainer.

        Parameters
        -----------
        results  : dict   output of compare()
        save_png : bool   save to user_saves/ if True
        filename : str    custom filename stem. Auto-generated if None.
        '''
        plt = self._get_plt()
        if plt is None:
            return

        import matplotlib.gridspec as gridspec

        names   = results["names"]
        words   = results["display"]
        n_exp   = len(names)
        n_words = len(words)
        winner  = results["winner"]
        text    = results["text"]

        PALETTE = ["#4E9AF1","#54C27D","#F4845F","#A78BFA","#F7C948","#E879A0"]
        BG, CARD      = "#0D1117", "#161B22"
        TEXT_C, SUB_C = "#E6EDF3", "#8B949E"
        GOLD          = "#F7C948"
        NEG_COL       = "#E05555"
        GREY          = "#444455"

        exp_cols = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

        fig_w = max(8, 5.5 * n_exp)
        fig_h = max(6, n_words * 0.52 + 3.0)

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
        fig.suptitle(
            f'EXACT  —  Word Importance Comparison\n'
            f'"{textwrap.shorten(text, 78)}"',
            color=TEXT_C, fontsize=11, fontweight="bold",
            fontfamily="monospace", y=0.99,
        )

        gs = gridspec.GridSpec(
            1, n_exp, figure=fig,
            wspace=0.06, top=0.88, bottom=0.07,
            left=0.10, right=0.97,
        )

        y      = np.arange(n_words)
        height = 0.64

        for col, name in enumerate(names):
            ax = fig.add_subplot(gs[col])
            ax.set_facecolor(CARD)
            ax.axvline(0, color="#666", lw=0.85, ls="--", zorder=3)
            ax.xaxis.grid(True, ls=":", alpha=0.3, color="#aaa", zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(-1.40, 1.40)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("bottom", "left"):
                ax.spines[sp].set_color("#30363D")

            norms = results["normalised"][name]
            raws  = results["raw"][name]

            bvals = []
            bcols = []
            for w in words:
                nv = norms.get(w, 0.0)
                rv = raws.get(w)
                bvals.append(nv)
                if rv is None:
                    bcols.append(GREY)
                elif nv >= 0:
                    bcols.append(exp_cols[name])
                else:
                    bcols.append(NEG_COL)

            ax.barh(y, bvals, height=height, color=bcols,
                    edgecolor="#0D1117", linewidth=0.4, zorder=2)

            for i, (nv, w) in enumerate(zip(bvals, words)):
                rv = raws.get(w)
                if rv is None:
                    continue
                ha    = "left" if nv >= 0 else "right"
                x_off = 0.04 if nv >= 0 else -0.04
                ax.text(nv + x_off, i, f"{rv:+.4f}",
                        va="center", ha=ha, fontsize=6.8,
                        color=TEXT_C, fontfamily="monospace")

            if col == 0:
                ax.set_yticks(y)
                ax.set_yticklabels(words, fontsize=9.5,
                                   color=TEXT_C, fontfamily="monospace")
            else:
                ax.set_yticks([])
            ax.invert_yaxis()
            ax.tick_params(axis="x", labelcolor=SUB_C, labelsize=7.5)
            ax.set_xlabel("Normalised Importance", color=SUB_C, fontsize=8.5)

            info  = results["label_info"][name]
            cls   = info["class"]
            conf  = f"\nconf {info['conf']:.4f}" if info["conf"] else ""
            flag  = "  ★ BEST" if name == winner else ""
            ax.set_title(
                f"{name}{flag}\nPredicted: {cls}{conf}",
                color=exp_cols[name], fontsize=10,
                fontweight="bold", pad=9,
                fontfamily="monospace",
            )

        # legend
        import matplotlib.patches as mp
        handles = []
        for i, n in enumerate(names):
            handles.append(mp.Patch(color=PALETTE[i % len(PALETTE)],
                                    label=f"{n} (supports +)"))
        handles.append(mp.Patch(color=NEG_COL, label="Opposes prediction (−)"))
        handles.append(mp.Patch(color=GREY,    label="Not scored by explainer"))
        fig.legend(handles=handles, loc="lower center",
                   ncol=min(len(handles), 4),
                   fontsize=8, framealpha=0.3,
                   facecolor=CARD, edgecolor="#333",
                   labelcolor=TEXT_C,
                   bbox_to_anchor=(0.5, 0.0))

        self._save_or_show(fig, plt, save_png,
                           filename or f"words_{'_'.join(names)}",
                           "word importance")


    # ─────────────────────────────────────────────────────────────────
    # plot_scores()  —  quality metric comparison chart
    # ─────────────────────────────────────────────────────────────────

    def plot_scores(self, results, save_png=False, filename=None):
        '''
        Save a PNG showing quality metric scores for each explainer —
        a score table and a ranked composite bar chart.

        Metrics shown:
          Coverage      — fraction of sentence words scored
          Concentration — top-3 words share of total importance
          Polarity      — balance of positive vs negative evidence
          Agreement     — directional agreement with other explainers
          Composite     — weighted average of all metrics above

        Parameters
        -----------
        results  : dict   output of compare()
        save_png : bool   save to user_saves/ if True
        filename : str    custom filename stem. Auto-generated if None.
        '''
        plt = self._get_plt()
        if plt is None:
            return

        import matplotlib.gridspec as gridspec

        names   = results["names"]
        scores  = results["scores"]
        metrics = results["metrics"]
        ranked  = results["ranked"]
        winner  = results["winner"]
        text    = results["text"]

        PALETTE = ["#4E9AF1","#54C27D","#F4845F","#A78BFA","#F7C948","#E879A0"]
        BG, CARD      = "#0D1117", "#161B22"
        TEXT_C, SUB_C = "#E6EDF3", "#8B949E"
        GOLD          = "#F7C948"

        exp_cols = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}

        fig = plt.figure(figsize=(14, 7), facecolor=BG)
        fig.suptitle(
            f'EXACT  —  Explainer Score Comparison\n'
            f'"{textwrap.shorten(text, 78)}"',
            color=TEXT_C, fontsize=11, fontweight="bold",
            fontfamily="monospace", y=1.01,
        )

        gs = gridspec.GridSpec(
            1, 2, figure=fig, wspace=0.14,
            top=0.88, bottom=0.10,
            left=0.05, right=0.97,
            width_ratios=[1.8, 1.0],
        )

        # ── left: metric score table ──────────────────────────────────
        ax_tbl = fig.add_subplot(gs[0])
        ax_tbl.set_facecolor(CARD)
        ax_tbl.axis("off")

        all_rows = metrics + ["COMPOSITE"]
        ctexts, ccolors = [], []

        for m in all_rows:
            is_comp = m == "COMPOSITE"
            base    = "#1C2030" if is_comp else CARD
            vals    = ([results["summary"][n] for n in names] if is_comp
                       else [scores[n].get(m, float("nan")) for n in names])
            valid   = [v for v in vals if not np.isnan(v)]
            best_v  = max(valid) if valid else None
            worst_v = min(valid) if valid else None

            row_t = [f"  {m}"]
            row_c = [base]
            for v in vals:
                if np.isnan(v):
                    row_t.append("N/A"); row_c.append(base)
                else:
                    row_t.append(f"{v:.4f}")
                    if best_v is not None and abs(v - best_v) < 1e-9:
                        row_c.append("#183018")   # green = best
                    elif (worst_v is not None and abs(v - worst_v) < 1e-9
                          and best_v != worst_v):
                        row_c.append("#301818")   # red = worst
                    else:
                        row_c.append(base)
            ctexts.append(row_t)
            ccolors.append(row_c)

        tbl = ax_tbl.table(
            cellText    = ctexts,
            colLabels   = ["Metric"] + names,
            cellColours = ccolors,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9.5)
        tbl.scale(1.0, 2.0)

        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#2a2a3a")
            if r == 0:
                cell.set_text_props(color=GOLD, fontweight="bold",
                                    fontfamily="monospace")
            else:
                cell.set_text_props(color=TEXT_C, fontfamily="monospace")

        ax_tbl.set_title("Score Table  (green = best  |  red = worst per row)",
                         color=SUB_C, fontsize=9,
                         fontfamily="monospace", pad=10)

        # ── right: ranked composite bar chart ─────────────────────────
        ax_bar = fig.add_subplot(gs[1])
        ax_bar.set_facecolor(CARD)
        for sp in ("top","right","left","bottom"):
            ax_bar.spines[sp].set_color("#2a2a3a")

        rnames = [r[0] for r in ranked]
        rvals  = [r[1] for r in ranked]
        rcols  = [GOLD if n == winner else exp_cols[n] for n in rnames]

        bars = ax_bar.barh(rnames[::-1], rvals[::-1],
                           color=rcols[::-1], height=0.50,
                           edgecolor="#0D1117", linewidth=0.5)

        for bar, val in zip(bars, rvals[::-1]):
            ax_bar.text(
                bar.get_width() + 0.008,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center", ha="left",
                color=TEXT_C, fontsize=9,
                fontfamily="monospace",
            )

        ax_bar.set_xlim(0, max(rvals) * 1.35 + 0.01)
        ax_bar.xaxis.grid(True, ls=":", alpha=0.3, color="#aaa")
        ax_bar.set_axisbelow(True)
        ax_bar.set_xlabel("Composite Score  (higher = better)",
                          color=SUB_C, fontsize=9)
        ax_bar.tick_params(axis="y", labelcolor=TEXT_C, labelsize=10)
        ax_bar.tick_params(axis="x", labelcolor=SUB_C, labelsize=8)
        ax_bar.set_title(
            f"Ranked  —  Winner: {winner}",
            color=GOLD, fontsize=10,
            fontweight="bold", pad=10,
            fontfamily="monospace",
        )

        plt.tight_layout()
        self._save_or_show(fig, plt, save_png,
                           filename or f"scores_{'_'.join(names)}",
                           "score comparison")


    # ─────────────────────────────────────────────────────────────────
    # plot()  —  both panels in one PNG
    # ─────────────────────────────────────────────────────────────────

    def plot(self, results, save_png=False, filename=None):
        '''
        Save both the word importance bar plots and the score comparison
        chart together in a single PNG.

        Equivalent to calling plot_words() and plot_scores() separately
        but produces one combined file.

        Parameters
        -----------
        results  : dict   output of compare()
        save_png : bool   save to user_saves/ if True
        filename : str    custom filename stem. Auto-generated if None.
        '''
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        names  = "_".join(results["names"])
        stem   = filename or f"comparison_{names}_{stamp}"

        self.plot_words( results, save_png=save_png, filename=f"{stem}_words")
        self.plot_scores(results, save_png=save_png, filename=f"{stem}_scores")


    # ─────────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────────

    def _get_plt(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            return plt
        except ImportError:
            warnings.warn("matplotlib not installed. pip install matplotlib")
            return None

    def _save_or_show(self, fig, plt, save_png, stem, label):
        if save_png:
            out = _saves() / f"{stem}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"[EXACT] {label} saved → {out}")
        else:
            plt.show()
            plt.close(fig)