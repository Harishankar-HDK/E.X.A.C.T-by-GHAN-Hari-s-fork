"""
Test file for LOOTextExplainer
==============================
Mirrors the same usage pattern as the LimeExplainer_Text test.

Run:
    python tests/loo_text_test.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from EXACT.explainers.loo_text_explainer import LOOTextExplainer


# ── Load the same pretrained sentiment model used in the LIME test ────────────

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model     = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ── Initialize the LOO explainer ──────────────────────────────────────────────
#
#   Just like the LIME test — pass the model, the tokenizer, and class names.
#   No encode_fn needed because we are passing a HuggingFace tokenizer directly.

explainer = LOOTextExplainer(
    model       = model,
    tokenizer   = tokenizer,
    class_names = ["negative", "positive"],
)


# ── Same test sentences used in the LIME test ─────────────────────────────────

test_texts = [
    "This movie was absolutely fantastic",        # pure positive — no 'but'
    "The acting was terrible and completely awful",  # pure negative — no 'but'  
    "I loved every single moment of this film",   # positive
    "Horrible waste of time I hated everything",
]


# ── Run explanations ──────────────────────────────────────────────────────────

for text in test_texts:

    print("\nInput Text:")
    print(text)

    explanation = explainer.explain(text)

    # Print results in the terminal  (same as explainer.visualize() in LIME test)
    explanation.show()

    # Save the bar chart to user_saves/
    explanation.visualize(save_png=True)