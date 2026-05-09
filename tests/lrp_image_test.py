"""
test_lrp_image.py
-----------------
Test the LRPImageExplainer against model_1.pth.

Run from the project root:
    cd /home/nigerianrappet/E.X.A.C.T-by-GHAN
    python3 tests/test_lrp_image.py
"""

import sys, os, importlib.util, pathlib

# ── Load lrp_explainer directly, bypassing __init__.py so missing deps
# ── in other explainers (lime, grad-cam etc.) don't block this test.
_lrp_path = pathlib.Path(__file__).parent.parent / "src/EXACT/explainers/lrp_explainer.py"
_spec = importlib.util.spec_from_file_location("lrp_explainer", _lrp_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LRPImageExplainer = _mod.LRPImageExplainer

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms


# ── 1. Device ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 2. Model — exact architecture matching model_1.pth ───────────────────────

class TumorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Conv2d(in_channels=3,  out_channels=32,  kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=4),
        )

    def forward(self, x):
        return self.layer_stack(x)


weights_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_1.pth")
model = TumorModel()
model.load_state_dict(torch.load(weights_path, map_location=device))
model = model.to(device)
model.eval()
print("model_1.pth loaded")


# ── 3. Preprocessing — identical to GradCAM test ─────────────────────────────

tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

image_path = os.path.join(os.path.dirname(__file__), "..", "models", "Te-me_0010.jpg")


# ── 4. Explainer ─────────────────────────────────────────────────────────────

explainer = LRPImageExplainer(model, device=device, save_dir="user_saves/lrp_saves")
info = explainer.get_model_info()
print(f"\nModel type  : {info['model_type']}")
print(f"WHY         : {info['why'][:90]}...")
print(f"CONSTRAINT  : {info['constraint'][:90]}...")
print(f"\nRules used:")
for r in info['rules_used']:
    print(f"  [{r['layer_type']:>15}]  {r['rule']}")


# ── 5. Test A — real image, default alpha=1 beta=0 (conservative) ────────────

if os.path.exists(image_path):
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)
    p_img   = tf(img_pil).unsqueeze(0).to(device)

    result = explainer.explain(
        input_tensor=p_img,
        input_image=img_np,
        lrp_alpha=1.0,          # conservative — positive contributions only
        lrp_beta=0.0,
        save_png=True,
        class_name="tumor_a1b0",
    )
    print(f"\nTest A — real image  [alpha=1, beta=0  conservative]")
    print(f"  input_type        : {result['input_type']}")
    print(f"  target_class      : {result['target_class']}")
    print(f"  lrp_alpha/beta    : {result['lrp_alpha']} / {result['lrp_beta']}")
    print(f"  completeness_error: {result['completeness_error']:.4f}",
          " [OK]" if result['completeness_error'] < 0.05 else " [!!]")
    print(f"  WHY               : {result['why'][:80]}...")
    print(f"  WHY NOT           : {result['why_not'][:80]}...")
else:
    print(f"\n(skipping Test A — {image_path} not found)")


# ── 6. Test B — same image, alpha=2 beta=1 (balanced, shows inhibitory too) ──

if os.path.exists(image_path):
    img_pil = Image.open(image_path).convert("RGB")
    img_np  = np.array(img_pil)
    p_img   = tf(img_pil).unsqueeze(0).to(device)

    result_b = explainer.explain(
        input_tensor=p_img,
        input_image=img_np,
        lrp_alpha=2.0,          # balanced — surfaces both excitatory and inhibitory
        lrp_beta=1.0,
        save_png=True,
        class_name="tumor_a2b1",
    )
    print(f"\nTest B — real image  [alpha=2, beta=1  balanced]")
    print(f"  lrp_alpha/beta    : {result_b['lrp_alpha']} / {result_b['lrp_beta']}")
    print(f"  completeness_error: {result_b['completeness_error']:.4f}",
          " [OK]" if result_b['completeness_error'] < 0.05 else " [!!]")


# ── 7. Test C — synthetic normalised tensor ───────────────────────────────────

synthetic = torch.randn(1, 3, 128, 128).to(device)
result_c = explainer.explain(
    input_tensor=synthetic,
    lrp_alpha=1.0,
    lrp_beta=0.0,
    save_png=True,
    class_name="synthetic",
)
print(f"\nTest C — synthetic normalised tensor [1,3,128,128]")
print(f"  input_type        : {result_c['input_type']}")
print(f"  completeness_error: {result_c['completeness_error']:.4f}",
      " [OK]" if result_c['completeness_error'] < 0.05 else " [!!]")


# ── 8. Test D — invalid alpha/beta (conservation enforcement check) ───────────

print(f"\nTest D — invalid alpha=3 beta=0 (should auto-correct and warn)")
result_d = explainer.explain(
    input_tensor=synthetic,
    lrp_alpha=3.0,
    lrp_beta=0.0,           # violates alpha - beta = 1, should be corrected to 2.0
    save_png=False,
)
print(f"  corrected beta    : {result_d['lrp_beta']}  (expected 2.0)")


# ── 9. Rule log ───────────────────────────────────────────────────────────────

print(f"\n── Rule log ({len(result_c['rule_log'])} active layers) ──")
for entry in result_c["rule_log"]:
    print(f"  [{entry['layer']:>20}]  {entry['rule_name']}")
    print(f"    WHY    : {entry['why'][:75]}...")
    print(f"    WHY NOT: {entry['why_not'][:75]}...")

print("\nAll tests complete.")