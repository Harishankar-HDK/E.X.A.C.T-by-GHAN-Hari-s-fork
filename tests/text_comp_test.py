"""
comparison_test.py
==================
Tests TextComparator with LIME and LOO.

Run:
    python -m tests.comparison_test
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from EXACT.explainers.lime_text_explainer  import LimeExplainer_Text
from EXACT.explainers.loo_text_explainer   import LOOTextExplainer
from EXACT.comparators.text_comp           import TextComparator
# from EXACT.explainers.shap_text_explainer  import ShapExplainer_Text


def test():

    # ---------------- Load model ----------------

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    # ---------------- Tokenizer wrapper for LIME ----------------

    def tokenize(texts):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


    # ---------------- Initialize explainers ----------------

    lime_explainer = LimeExplainer_Text(
        model       = model,
        tokenizer   = tokenize,
        class_names = ["negative", "positive"],
        num_samples = 3000,
    )

    loo_explainer = LOOTextExplainer(
        model       = model,
        tokenizer   = tokenizer,
        class_names = ["negative", "positive"],
    )

    # shap_explainer = ShapExplainer_Text(
    #     model = model,
    #     tokenizer = tokenizer,
    #     class_names = ["negative", "positive"],
    #     mask_token_id = tokenizer.mask_token_id,
    # )

    cmp = TextComparator(
        class_names = ["negative", "positive"],
        top_n       = 10,
    )


    # ---------------- Test sentences ----------------

    test_texts = [
        "This movie was absolutely fantastic and the acting was brilliant but the time duration was terrible and boring",
        "This product is terrible and completely useless but good and safe for health",
        "The story was good but the acting was horrible",
    ]


    # ---------------- Run comparison ----------------

    for i, text in enumerate(test_texts):

        print(f"\nInput Text:")
        print(text)

        # Step 1 — run each explainer
        lime_result = lime_explainer.explain(text)
        loo_result  = loo_explainer.explain(text)
        # shap_result = shap_explainer.explain(text)

        # Step 2 — compare
        results = cmp.compare(
            explanations = {
                "LIME" : lime_result,
                "LOO"  : loo_result,
                # "SHAP" : shap_result,
            },
            text = text,
        )

        # Step 3 — terminal output
        cmp.report(results)

        # Step 3 — PNG: bar plots side by side (word importance)
        cmp.plot_words(
            results,
            save_png = True,
            filename = f"words_sentence_{i + 1}",
        )

        # Step 3 — PNG: score comparison chart
        cmp.plot_scores(
            results,
            save_png = True,
            filename = f"scores_sentence_{i + 1}",
        )


if __name__ == "__main__":
    test()