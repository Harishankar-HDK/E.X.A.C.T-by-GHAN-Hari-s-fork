"""
examples/evaluate_explainer.py
==============================
Demonstrate all four EXACT evaluators on a single explainer result.
Each evaluator is independent — run only the ones you need.
"""

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from EXACT.explainers import GradCAM
from EXACT.evaluators import (
    FaithfulnessEvaluator,
    SharpnessEvaluator,
    StabilityEvaluator,
    LocalizationEvaluator,
)

# ── Setup ─────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"

import torchvision.models as models
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().to(device)

pil_img = Image.open("models/catexample.jpg").convert("RGB").resize((224, 224))
preprocess = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
img_np       = np.array(pil_img, dtype=np.float32) / 255.0

# ── Run explainer ──────────────────────────────────────────────────────────
gradcam_exp    = GradCAM(model)
gradcam_result = gradcam_exp.explain(input_tensor, method="gradcam", input_image=img_np)

# ── (Optional) ground-truth mask ──────────────────────────────────────────
gt_mask = np.zeros((224, 224), dtype=np.float32)
gt_mask[56:168, 56:168] = 1.0   # replace with a real mask

# ═══════════════════════════════════════════════════════════════════════════
# 1. FAITHFULNESS
# ═══════════════════════════════════════════════════════════════════════════
faith_ev     = FaithfulnessEvaluator(model, device=device, steps=10)
faith_result = faith_ev.evaluate(
    explainer_result=gradcam_result,
    input_tensor=input_tensor,
)
faith_ev.report(faith_result)
faith_ev.plot(faith_result, save_png=True, filename="gradcam_faithfulness.png")

# ═══════════════════════════════════════════════════════════════════════════
# 2. SHARPNESS  (no model needed)
# ═══════════════════════════════════════════════════════════════════════════
sharp_ev     = SharpnessEvaluator()
sharp_result = sharp_ev.evaluate(explainer_result=gradcam_result)
sharp_ev.report(sharp_result)
sharp_ev.plot(sharp_result, save_png=True, filename="gradcam_sharpness.png")

# ═══════════════════════════════════════════════════════════════════════════
# 3. STABILITY
# ═══════════════════════════════════════════════════════════════════════════
stab_ev     = StabilityEvaluator(runs=10, noise_std=0.05)
stab_result = stab_ev.evaluate(
    explainer_result=gradcam_result,
    explainer_obj=gradcam_exp,
    input_tensor=input_tensor,
    extra_kwargs={"method": "gradcam"},   # forwarded to gradcam_exp.explain()
)
stab_ev.report(stab_result)
stab_ev.plot(stab_result, save_png=True, filename="gradcam_stability.png")

# ── LIME stability example (reduce runs — LIME is expensive) ──────────────
# from EXACT.explainers import LimeExplainer
# lime_ev     = StabilityEvaluator(runs=3, noise_std=0.05)
# lime_result = lime_ev.evaluate(
#     explainer_result=lime_result,
#     explainer_obj=lime_exp,
#     input_tensor=input_tensor,
#     extra_kwargs={},
# )

# ═══════════════════════════════════════════════════════════════════════════
# 4. LOCALIZATION  (requires gt_mask)
# ═══════════════════════════════════════════════════════════════════════════
loc_ev     = LocalizationEvaluator(iou_threshold=0.5)
loc_result = loc_ev.evaluate(
    explainer_result=gradcam_result,
    gt_mask=gt_mask,
)
loc_ev.report(loc_result)
loc_ev.plot(loc_result, save_png=True, filename="gradcam_localization.png")

# ── Programmatic access ────────────────────────────────────────────────────
print("\nFaithfulness scores:", faith_result["scores"])
print("Sharpness grades:  ", sharp_result["grades"])
print("Stability overall: ", stab_result["overall"], stab_result["overall_grade"])
print("Localization IoU:  ", loc_result["scores"]["iou"])