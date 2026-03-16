"""
EXACT — EXplainability and Attribution for Classification Tasks
===============================================================
loo_text_explainer.py  |  Leave-One-Out explainer for PyTorch text models

Algorithm
---------
For a text of N tokens, LOO produces N importance scores:

    importance[i] = P(y=c | original text)
                  - P(y=c | text with token_i removed / masked)

A high positive score  → token strongly supports the prediction.
A negative score       → removing the token *increases* model confidence.

Supported model families (out-of-the-box)
------------------------------------------
  • HuggingFace Transformers  (BERT, RoBERTa, DistilBERT, ALBERT, XLNet …)
  • Custom LSTM / GRU / CNN text classifiers
  • Any nn.Module that accepts a tokenized tensor or an encoded dict

Quick start
-----------
    # ── HuggingFace model ──────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from EXACT.explainers.loo_text_explainer import LOOTextExplainer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    model     = AutoModelForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased-finetuned-sst-2-english")

    explainer = LOOTextExplainer(
        model         = model,
        tokenizer     = tokenizer,              # pass the HF tokenizer directly
        class_names   = ["Negative", "Positive"],
    )

    explanation = explainer.explain("The movie was absolutely fantastic!")
    explanation.show()                          # prints a coloured table
    explanation.visualize(save_png=True)        # saves PNG to user_saves/

    # ── Custom LSTM model ──────────────────────────────────────────────
    def encode(text):                           # user-supplied encode fn
        ids = [vocab.get(w, 0) for w in text.lower().split()]
        return torch.tensor(ids, dtype=torch.long)

    explainer = LOOTextExplainer(
        model      = my_lstm,
        encode_fn  = encode,
        class_names = ["neg", "pos"],
    )
    explanation = explainer.explain("the film was brilliant")
    explanation.visualize(save_png=True)
"""

from __future__ import annotations

import os
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_HERE         = Path(__file__).resolve().parent          # …/src/EXACT/explainers/
_PROJECT_ROOT = _HERE.parent.parent.parent               # src/EXACT/ -> src/ -> project root
_USER_SAVES   = _PROJECT_ROOT / "user_saves"


def _user_saves_dir() -> Path:
    """Return (and create if needed) the user_saves directory."""
    _USER_SAVES.mkdir(parents=True, exist_ok=True)
    return _USER_SAVES


# ─────────────────────────────────────────────────────────────────────────────
# Internal tokenisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_hf_tokenizer(obj) -> bool:
    """True when obj looks like a HuggingFace PreTrainedTokenizer(Fast)."""
    return hasattr(obj, "encode") and hasattr(obj, "convert_tokens_to_string")


def _whitespace_tokenize(text: str) -> List[str]:
    return text.split()


def _wordpunct_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)


def _wordpunct_join(tokens: List[str]) -> str:
    out: List[str] = []
    for tok in tokens:
        if out and re.match(r"[^\w\s]", tok):
            out[-1] += tok
        else:
            out.append(tok)
    return " ".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Explanation result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LOOTextExplanation:
    """
    The full result of a Leave-One-Out text explanation.

    Attributes
    ----------
    tokens          : list[str]   — individual tokens analysed
    importances     : np.ndarray  — shape (n_tokens,); importance per token
    predicted_class : int         — index of predicted class
    predicted_proba : float       — model confidence for predicted class
    class_names     : list[str]   — human-readable class labels (optional)
    original_text   : str         — raw input string
    label_explained : int         — the class whose probability was tracked
    all_probas      : np.ndarray  — full probability vector on original text
    """

    tokens          : List[str]
    importances     : np.ndarray
    predicted_class : int
    predicted_proba : float
    class_names     : Optional[List[str]]
    original_text   : str
    label_explained : int
    all_probas      : np.ndarray

    # ── Derived helpers ───────────────────────────────────────────────────────

    @property
    def class_label(self) -> str:
        if self.class_names and self.predicted_class < len(self.class_names):
            return self.class_names[self.predicted_class]
        return str(self.predicted_class)

    def top_tokens(self, n: int = 5) -> List[Tuple[str, float]]:
        """n most positively important tokens → [(token, score), …]"""
        idx = np.argsort(self.importances)[::-1][:n]
        return [(self.tokens[i], float(self.importances[i])) for i in idx]

    def bottom_tokens(self, n: int = 5) -> List[Tuple[str, float]]:
        """n most negatively important tokens → [(token, score), …]"""
        idx = np.argsort(self.importances)[:n]
        return [(self.tokens[i], float(self.importances[i])) for i in idx]

    def as_dict(self) -> Dict:
        return {
            "tokens"          : self.tokens,
            "importances"     : self.importances.tolist(),
            "predicted_class" : self.predicted_class,
            "predicted_proba" : self.predicted_proba,
            "class_names"     : self.class_names,
            "original_text"   : self.original_text,
            "label_explained" : self.label_explained,
            "all_probas"      : self.all_probas.tolist(),
        }

    # ── Console display ───────────────────────────────────────────────────────

    def show(self, max_tokens: Optional[int] = None) -> None:
        """Pretty-print the explanation to stdout."""
        tokens = self.tokens
        importances = self.importances

        if max_tokens is not None:
            idx = np.argsort(np.abs(importances))[::-1][:max_tokens]
            idx = sorted(idx)
            tokens = [tokens[i] for i in idx]
            importances = importances[idx]

        max_abs = max(np.abs(importances).max(), 1e-12)
        bar_w   = 30

        header = (
            f"\n{'─'*60}\n"
            f"  EXACT · LOO Text Explanation\n"
            f"{'─'*60}\n"
            f"  Input : {textwrap.shorten(self.original_text, 55)}\n"
            f"  Pred  : {self.class_label}  (confidence {self.predicted_proba:.4f})\n"
            f"{'─'*60}\n"
            f"  {'Token':<20}  {'Importance':>10}  Bar\n"
            f"{'─'*60}"
        )
        print(header)

        for tok, imp in zip(tokens, importances):
            fill  = int(bar_w * abs(imp) / max_abs)
            sign  = "▶" if imp >= 0 else "◀"
            bar   = (sign * fill).ljust(bar_w)
            print(f"  {tok:<20}  {imp:>+10.4f}  {bar}")

        print(f"{'─'*60}\n")

    # ── Matplotlib visualization ───────────────────────────────────────────────

    def visualize(
        self,
        max_tokens  : Optional[int]  = None,
        figsize     : Tuple[int,int] = (10, 6),
        title       : Optional[str]  = None,
        save_png    : bool           = False,
        filename    : Optional[str]  = None,
    ) -> Optional["plt.Figure"]:  # type: ignore[name-defined]
        """
        Render a horizontal bar chart of token importances.

        Parameters
        ----------
        max_tokens : int, optional
            Show only the N tokens with the largest |importance|.
            Default shows all tokens.
        figsize    : (width, height) in inches.  Default (10, 6).
        title      : Override the auto-generated title.
        save_png   : If True, save the figure to ``user_saves/`` as a PNG.
        filename   : Custom file stem (no extension).  Auto-generated if None.

        Returns
        -------
        matplotlib.figure.Figure or None
            The Figure object so the caller can further customise it.
            Returns None if matplotlib is not installed.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            warnings.warn(
                "matplotlib is not installed — cannot visualize.\n"
                "Install it with:  pip install matplotlib",
                stacklevel=2,
            )
            return None

        # ── select tokens ────────────────────────────────────────────────
        tokens      = list(self.tokens)
        importances = self.importances.copy()

        if max_tokens is not None:
            keep = np.argsort(np.abs(importances))[::-1][:max_tokens]
            keep = sorted(keep)
            tokens      = [tokens[i] for i in keep]
            importances = importances[keep]

        # ── colour map: green (+) / red (−) ──────────────────────────────
        max_abs = max(np.abs(importances).max(), 1e-12)
        norm    = mcolors.TwoSlopeNorm(
            vmin   = min(importances.min(), -1e-9),
            vcenter= 0.0,
            vmax   = max(importances.max(),  1e-9),
        )
        try:
            cmap   = plt.get_cmap("RdYlGn")
        except Exception:
            cmap   = plt.cm.RdYlGn  # type: ignore[attr-defined]
        colours = [cmap(norm(v)) for v in importances]

        # ── figure ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#f9f9f9")
        ax.set_facecolor("#f0f0f0")

        y_pos = np.arange(len(tokens))
        bars  = ax.barh(y_pos, importances, color=colours,
                        edgecolor="white", linewidth=0.6, height=0.65)

        # value labels on bars
        for bar, val in zip(bars, importances):
            x_off = 0.002 * max_abs
            ha    = "left" if val >= 0 else "right"
            ax.text(
                val + (x_off if val >= 0 else -x_off),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.4f}",
                va="center", ha=ha, fontsize=8, color="#333333",
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=11)
        ax.invert_yaxis()
        ax.axvline(0, color="#555555", linewidth=1.0, linestyle="--")
        ax.set_xlabel("LOO Importance  (positive → supports prediction)",
                      fontsize=10, labelpad=8)
        ax.tick_params(axis="x", labelsize=9)

        # grid
        ax.xaxis.grid(True, linestyle=":", alpha=0.5, color="#aaaaaa")
        ax.set_axisbelow(True)

        # spines
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # title
        explained_cls = (
            self.class_names[self.label_explained]
            if self.class_names and self.label_explained < len(self.class_names)
            else str(self.label_explained)
        )
        auto_title = (
            f"LOO Text Explanation  ·  Predicted: '{self.class_label}'"
            f"  (conf = {self.predicted_proba:.4f})\n"
            f"Importance w.r.t. class: '{explained_cls}'"
        )
        ax.set_title(title or auto_title, fontsize=12, fontweight="bold",
                     pad=14, color="#222222")

        plt.tight_layout()

        # ── save ─────────────────────────────────────────────────────────
        if save_png:
            save_dir = _user_saves_dir()
            if filename is None:
                stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"loo_explanation_{stamp}"
            out_path = save_dir / f"{filename}.png"
            fig.savefig(out_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[EXACT] Explanation saved → {out_path}")

        return fig

    # ── HTML (Jupyter) ────────────────────────────────────────────────────────

    def to_html(self) -> str:
        """
        Return an HTML span-highlighted string of the original text.

        Usage in Jupyter::

            from IPython.display import HTML, display
            display(HTML(explanation.to_html()))
        """
        tokens      = self.tokens
        importances = self.importances
        max_abs     = max(np.abs(importances).max(), 1e-12)

        parts = [
            "<div style='font-family:monospace;font-size:1.05em;"
            "line-height:2.4em;padding:8px'>"
        ]
        for tok, imp in zip(tokens, importances):
            intensity = int(220 * abs(imp) / max_abs)
            bg = (
                f"rgb({255-intensity},255,{255-intensity})"
                if imp >= 0
                else f"rgb(255,{255-intensity},{255-intensity})"
            )
            parts.append(
                f"<span title='importance: {imp:+.4f}' style='"
                f"background:{bg};padding:3px 5px;margin:2px 1px;"
                f"border-radius:4px;border:1px solid #ccc;"
                f"cursor:default'>{tok}</span>"
            )
        parts.append("</div>")
        return "".join(parts)

    def __repr__(self) -> str:
        top = self.top_tokens(3)
        top_str = ", ".join(f"'{t}'({s:+.3f})" for t, s in top)
        return (
            f"LOOTextExplanation("
            f"predicted='{self.class_label}' conf={self.predicted_proba:.3f}, "
            f"tokens={len(self.tokens)}, "
            f"top=[{top_str}])"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Core explainer
# ─────────────────────────────────────────────────────────────────────────────

class LOOTextExplainer:
    """
    Leave-One-Out explainer for **any PyTorch text classification model**.

    The explainer needs to know two things:
      1. How to *tokenize* the input text into units for LOO analysis.
      2. How to *encode* a string back into model-ready tensors.

    Both are handled automatically when you pass a HuggingFace tokenizer.
    For custom models, supply an ``encode_fn``.

    Parameters
    ----------
    model : nn.Module
        A trained PyTorch model.  Will be moved to ``device`` and set to
        ``eval()`` mode automatically.

    tokenizer : HF PreTrainedTokenizer | str | callable, optional
        Controls how text is split into LOO units.

        - **HuggingFace tokenizer** (recommended for transformer models):
          Pass the tokenizer object directly.  ``encode_fn`` is built
          automatically; you do NOT need to provide it separately.

        - ``"whitespace"``  (default for custom models):
          Split on whitespace.  Simple and reliable for word-level models.

        - ``"wordpunct"``:
          Split words and punctuation into separate tokens.

        - Any callable ``tok_fn(text: str) -> List[str]``:
          Custom tokenization function.

    encode_fn : callable, optional
        ``encode_fn(text: str) -> Tensor | dict``
        Converts a string into model input.  Required for non-HF models.
        Examples::

            # word-id model
            encode_fn = lambda t: torch.tensor(
                [vocab.get(w, 0) for w in t.split()])

            # HuggingFace — built automatically when tokenizer is a HF obj
            encode_fn = lambda t: hf_tok(t, return_tensors="pt",
                                          truncation=True, max_length=128)

    mask_strategy : str, optional
        How the left-out token is treated when rebuilding the text:

        - ``"remove"``      (default) — token is deleted from the sequence.
        - ``"mask_token"``  — replaced with the tokenizer's ``[MASK]`` token.
          Requires ``tokenizer`` to be a HF tokenizer or to have
          a ``.mask_token`` attribute.
        - ``"unk_token"``   — replaced with ``[UNK]``.

    output_type : str, optional
        Tells the explainer how to interpret model output:

        - ``"logits"``   (default) — softmax is applied internally.
        - ``"proba"``    — output is already a probability distribution.
        - ``"log_proba"``— output is log-probabilities; exp is applied.

    device : str | torch.device, optional
        Inference device.  Defaults to CUDA if available, else CPU.

    class_names : list[str], optional
        Human-readable class labels used in outputs and plots.

    batch_size : int, optional
        Number of perturbed texts forwarded together.  Default 32.
        Lower this if you hit GPU out-of-memory errors.

    max_length : int, optional
        Maximum sequence length passed to a HuggingFace tokenizer.
        Default 512.

    normalize : bool, optional
        If True, normalize importances so |importance|.sum() == 1.
        Default False.

    verbose : bool, optional
        Print progress during ``explain()``.  Default False.

    Examples
    --------
    **HuggingFace transformer (minimal)**::

        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from EXACT.explainers.loo_text_explainer import LOOTextExplainer

        hf_tok = AutoTokenizer.from_pretrained(
                     "distilbert-base-uncased-finetuned-sst-2-english")
        model  = AutoModelForSequenceClassification.from_pretrained(
                     "distilbert-base-uncased-finetuned-sst-2-english")

        explainer = LOOTextExplainer(
            model       = model,
            tokenizer   = hf_tok,
            class_names = ["Negative", "Positive"],
        )
        exp = explainer.explain("This film was absolutely wonderful!")
        exp.show()
        exp.visualize(save_png=True)

    **Custom LSTM**::

        explainer = LOOTextExplainer(
            model     = my_lstm,
            encode_fn = lambda t: torch.tensor([vocab[w] for w in t.split()]),
            class_names = ["neg", "pos"],
        )
        exp = explainer.explain("the acting was superb")
        exp.visualize(save_png=True)
    """

    def __init__(
        self,
        model          : nn.Module,
        tokenizer                    = "whitespace",
        encode_fn      : Optional[Callable] = None,
        mask_strategy  : str         = "remove",
        output_type    : str         = "logits",
        device                       = None,
        class_names    : Optional[List[str]] = None,
        batch_size     : int         = 32,
        max_length     : int         = 512,
        normalize      : bool        = False,
        verbose        : bool        = False,
    ):
        # ── device ──────────────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── model ────────────────────────────────────────────────────────
        self.model = model.to(self.device)
        self.model.eval()

        # ── config ───────────────────────────────────────────────────────
        self.output_type  = output_type
        self.class_names  = class_names
        self.batch_size   = batch_size
        self.max_length   = max_length
        self.normalize    = normalize
        self.verbose      = verbose
        self.mask_strategy= mask_strategy

        # ── validate mask_strategy ───────────────────────────────────────
        _valid_masks = ("remove", "mask_token", "unk_token")
        if mask_strategy not in _valid_masks:
            raise ValueError(
                f"mask_strategy must be one of {_valid_masks}, "
                f"got '{mask_strategy}'"
            )

        # ── resolve tokenizer + encode_fn ───────────────────────────────
        self._hf_tokenizer    = None
        self._hf_encode_batch = None   # only set for HF tokenizers

        if _is_hf_tokenizer(tokenizer):
            # HuggingFace tokenizer — build everything automatically
            self._hf_tokenizer = tokenizer
            # IMPORTANT: use word-level (whitespace) tokenization for LOO,
            # NOT subword tokenization. LOO removes one word at a time from
            # the original string, then re-encodes the resulting string with
            # the HF tokenizer. This way the model always receives valid,
            # naturally-written text — never broken subword fragments.
            self._tok_fn  = _whitespace_tokenize   # words the user sees
            self._join_fn = lambda toks: " ".join(toks)

            if encode_fn is not None:
                # User supplied their own; respect it
                self._encode_fn = encode_fn
            else:
                # Build encode_fn for a SINGLE string → dict of tensors
                # padding=False so single-sample encoding has no pad tokens.
                _hft = tokenizer
                _ml  = max_length
                def _hf_encode_single(t: str) -> dict:
                    enc = _hft(
                        t,
                        return_tensors = "pt",
                        truncation     = True,
                        max_length     = _ml,
                        padding        = False,
                    )
                    # Convert BatchEncoding → plain dict of tensors
                    return dict(enc)

                # Also keep a batch-encoding function used by _forward_chunk
                def _hf_encode_batch(texts) -> dict:
                    enc = _hft(
                        texts,
                        return_tensors = "pt",
                        truncation     = True,
                        max_length     = _ml,
                        padding        = True,   # pad to longest in batch
                    )
                    return dict(enc)

                self._encode_fn       = _hf_encode_single
                self._hf_encode_batch = _hf_encode_batch

        elif callable(tokenizer) and not isinstance(tokenizer, str):
            # User-supplied tokenize function
            self._tok_fn    = tokenizer
            self._join_fn   = lambda toks: " ".join(toks)
            self._encode_fn = self._require_encode_fn(encode_fn)

        elif isinstance(tokenizer, str):
            _map = {
                "whitespace": (_whitespace_tokenize, lambda t: " ".join(t)),
                "wordpunct" : (_wordpunct_tokenize,  _wordpunct_join),
            }
            if tokenizer not in _map:
                raise ValueError(
                    f"tokenizer string must be 'whitespace' or 'wordpunct', "
                    f"got '{tokenizer}'"
                )
            self._tok_fn, self._join_fn = _map[tokenizer]
            self._encode_fn = self._require_encode_fn(encode_fn)

        else:
            raise TypeError(
                "tokenizer must be a HuggingFace tokenizer, "
                "a callable, or one of 'whitespace' / 'wordpunct'."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def explain(
        self,
        text  : str,
        label : Optional[int] = None,
    ) -> LOOTextExplanation:
        """
        Explain a single text input with Leave-One-Out analysis.

        Parameters
        ----------
        text  : str
            The text to explain.
        label : int, optional
            The class index to explain.  If ``None`` (default), the model's
            top predicted class is used.

        Returns
        -------
        LOOTextExplanation
            Call ``.show()`` for a console table or ``.visualize()`` for a plot.
        """
        text = str(text).strip()
        if not text:
            raise ValueError("Input text is empty.")

        tokens = self._tok_fn(text)
        if len(tokens) == 0:
            raise ValueError(
                "Input text produced zero tokens after tokenization.  "
                "Check your tokenizer or input string."
            )

        if self.verbose:
            print(f"[EXACT·LOO] Tokenized ({len(tokens)} tokens): {tokens}")

        # ── baseline ────────────────────────────────────────────────────
        baseline_probas = self._infer_single(text)          # (n_classes,)
        predicted_class = int(np.argmax(baseline_probas))
        target_class    = label if label is not None else predicted_class
        baseline_score  = float(baseline_probas[target_class])

        if self.verbose:
            cls_str = (
                self.class_names[predicted_class]
                if self.class_names else str(predicted_class)
            )
            print(f"[EXACT·LOO] Baseline → '{cls_str}'  "
                  f"P={baseline_score:.4f}")

        # ── perturbed texts ──────────────────────────────────────────────
        perturbed = [
            self._perturb(tokens, i) for i in range(len(tokens))
        ]

        # ── batched inference ────────────────────────────────────────────
        perturbed_probas = self._infer_batch(perturbed)     # (n_tokens, n_cls)

        # ── importance = drop in class probability ───────────────────────
        importances = baseline_score - perturbed_probas[:, target_class]

        if self.normalize:
            denom = np.abs(importances).sum()
            if denom > 1e-12:
                importances = importances / denom

        return LOOTextExplanation(
            tokens          = tokens,
            importances     = importances,
            predicted_class = predicted_class,
            predicted_proba = baseline_score,
            class_names     = self.class_names,
            original_text   = text,
            label_explained = target_class,
            all_probas      = baseline_probas,
        )

    def explain_batch(
        self,
        texts  : List[str],
        labels : Optional[List[Optional[int]]] = None,
    ) -> List[LOOTextExplanation]:
        """
        Explain a list of texts.

        Parameters
        ----------
        texts  : list[str]
        labels : list[int | None], optional
            Per-text target class.  None entries use the predicted class.

        Returns
        -------
        list[LOOTextExplanation]
        """
        if labels is None:
            labels = [None] * len(texts)
        return [self.explain(t, lbl) for t, lbl in zip(texts, labels)]

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    # ── perturbation ─────────────────────────────────────────────────────────

    def _perturb(self, tokens: List[str], drop_idx: int) -> str:
        """
        Return the text string with tokens[drop_idx] removed or masked.

        tokens are always WORD-level (what the user sees).
        We operate on the word list and join back to a plain string.
        The HF tokenizer is then called on that plain string during
        inference — it never sees broken or reconstructed subword fragments.
        """
        t = tokens.copy()

        if self.mask_strategy == "remove":
            t.pop(drop_idx)

        elif self.mask_strategy == "mask_token":
            # For word-level LOO with HF models we simply remove the word.
            # Replacing a whole word with [MASK] causes the subword tokenizer
            # to produce a single [MASK] token where there may have been
            # multiple subword pieces, which still distorts context less than
            # dropping a subword fragment.  Use remove for cleanest results.
            t.pop(drop_idx)

        else:  # unk_token — remove is cleaner than inserting [UNK] as a word
            t.pop(drop_idx)

        if not t:
            return ""

        # Always join words with a space — the HF encode_fn handles the rest
        return " ".join(t)

    # ── inference ────────────────────────────────────────────────────────────

    def _infer_single(self, text: str) -> np.ndarray:
        """Run one text through the model → probability vector (n_classes,)."""
        return self._infer_batch([text])[0]

    def _infer_batch(self, texts: List[str]) -> np.ndarray:
        """
        Run a list of texts in mini-batches.
        Returns ndarray of shape (len(texts), n_classes).
        """
        chunks = [
            texts[i : i + self.batch_size]
            for i in range(0, len(texts), self.batch_size)
        ]
        parts = [self._forward_chunk(chunk) for chunk in chunks]
        return np.vstack(parts)

    def _forward_chunk(self, texts: List[str]) -> np.ndarray:
        """
        Forward a chunk of texts.

        Strategy
        --------
        1. If we have a HF tokenizer, use _hf_encode_batch for the whole
           chunk at once — one forward pass, fast.
        2. Otherwise encode one-by-one and stack (always safe for custom models).
        """
        # ── Fast path: HF batch encoding (one forward pass) ──────────────
        if self._hf_tokenizer is not None and self._hf_encode_batch is not None:
            valid_texts = [t if t else "[UNK]" for t in texts]
            try:
                encoded = self._hf_encode_batch(valid_texts)  # plain dict of tensors
                return self._forward_dict(encoded)
            except Exception:
                pass  # fall through to sequential

        # ── Safe path: sequential (one text at a time) ────────────────────
        results = []
        for text in texts:
            if not text:
                results.append(None)
                continue
            encoded = self._encode_fn(text)
            proba   = self._forward_single(encoded)
            results.append(proba)

        # Fill any None (empty-text) slots with uniform distribution
        n_cls = next((r.shape[0] for r in results if r is not None), 2)
        return np.vstack([
            r if r is not None else np.full(n_cls, 1.0 / n_cls)
            for r in results
        ])

    def _forward_single(self, encoded) -> np.ndarray:
        """Forward a single encoded input → probability vector (n_classes,)."""
        self.model.eval()
        with torch.no_grad():
            # dict / BatchEncoding → unpack as **kwargs  (HF style)
            if isinstance(encoded, dict) or hasattr(encoded, "keys"):
                enc = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in encoded.items()
                }
                out = self.model(**enc)

            elif isinstance(encoded, torch.Tensor):
                x = encoded.to(self.device)
                if x.dim() == 1:
                    x = x.unsqueeze(0)        # (seq_len,) → (1, seq_len)
                out = self.model(x)

            else:
                # Fallback — pass as-is (user knows their model)
                out = self.model(encoded)

            return self._to_proba(out)

    def _forward_dict(self, encoded) -> np.ndarray:
        """Forward a batched dict / BatchEncoding → probability matrix (B, C)."""
        self.model.eval()
        with torch.no_grad():
            enc = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in encoded.items()
            }
            out = self.model(**enc)
        return self._to_proba_batch(out)

    # ── output normalisation ─────────────────────────────────────────────────

    def _unwrap(self, out):
        """Strip HF ModelOutput wrappers, tuples, lists."""
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, (tuple, list)):
            return out[0]
        return out

    def _to_proba(self, out) -> np.ndarray:
        """Single-sample output (1, C) or (C,) → ndarray (C,)."""
        tensor = self._unwrap(out).squeeze(0)   # → (C,)
        return self._apply_output_fn(tensor).cpu().numpy()

    def _to_proba_batch(self, out) -> np.ndarray:
        """Batched output (B, C) → ndarray (B, C)."""
        tensor = self._unwrap(out)              # → (B, C)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return self._apply_output_fn(tensor).cpu().numpy()

    def _apply_output_fn(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.output_type == "logits":
            return torch.softmax(tensor, dim=-1)
        if self.output_type == "log_proba":
            return torch.exp(tensor)
        return tensor   # already probabilities

    # ── misc ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _require_encode_fn(encode_fn):
        if encode_fn is None:
            raise ValueError(
                "encode_fn is required when tokenizer is not a HuggingFace "
                "tokenizer.  Provide a callable that maps a string to a "
                "Tensor or dict.\n\n"
                "Example:\n"
                "    encode_fn = lambda t: torch.tensor(\n"
                "        [vocab.get(w, 0) for w in t.split()])\n"
            )
        return encode_fn

    def __repr__(self) -> str:
        return (
            f"LOOTextExplainer("
            f"device={self.device}, "
            f"mask_strategy='{self.mask_strategy}', "
            f"output_type='{self.output_type}', "
            f"batch_size={self.batch_size})"
        )