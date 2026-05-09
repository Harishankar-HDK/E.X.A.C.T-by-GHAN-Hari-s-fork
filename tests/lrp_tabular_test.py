"""
lrp_tabular_test.py
-----------------
Test the LRPTabularExplainer using synthetic tabular data.

Run from the project root:
    python3 tests/lrp_tabular_test.py
"""

import sys, os, importlib.util, pathlib
import torch
import torch.nn as nn
import numpy as np

# ── Load lrp_tabular directly, bypassing __init__.py so missing deps
# ── in other explainers (lime, grad-cam etc.) don't block this test.
_lrp_path = pathlib.Path(__file__).parent.parent / "src/EXACT/explainers/lrp_tabular.py"
_spec = importlib.util.spec_from_file_location("lrp_tabular", _lrp_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LRPTabularExplainer = _mod.LRPTabularExplainer


# ── 1. Device ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── 2. Model — Simple MLP for tabular data ───────────────────────────────────

class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
        )

    def forward(self, x):
        return self.net(x)

model = MLPModel().to(device)
model.eval()
print("MLPModel initialized")

# ── 3. Data — Synthetic Tabular Data ──────────────────────────────────────────

input_tensor = torch.randn(1, 10).to(device)
training_data = np.random.randn(100, 10)
feature_names = [f"Feature_{i}" for i in range(10)]

# ── 4. Explainer ─────────────────────────────────────────────────────────────

explainer = LRPTabularExplainer(model, feature_names=feature_names, device=device)

# ── 5. Test A — synthetic tabular data ────────────────────────────────────────

print(f"\nTest A — synthetic tabular data")
result = explainer.explain(
    input_tensor=input_tensor,
    training_data=training_data,
    target_class=1
)

print(f"  target_class      : {result['target_class']}")
print(f"  completeness_error: {result['convergence_delta']:.4f}",
      " [OK]" if result['convergence_delta'] < 0.05 else " [!!]")

save_path = os.path.join("user_saves", "lrp_tabular_test.png")
os.makedirs("user_saves", exist_ok=True)
explainer.save_dashboard(result, save_path, class_name="Class 1")
print(f"  Saved dashboard to {save_path}")

print("\nAll tests complete.")
