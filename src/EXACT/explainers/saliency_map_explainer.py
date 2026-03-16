# explainers/saliency.py

"""
SaliencyMap
===========
Pixel-level attribution via gradient-based saliency methods.

Unlike GradCAM which produces coarse heatmaps from intermediate feature
layers, saliency methods backpropagate all the way to the input pixels.
The result is a full-resolution map showing exactly which pixels drove
the model's prediction — at the cost of sometimes being noisier.

Three methods are supported:

'vanilla'
    The gradient of the target class score with respect to each input
    pixel. Fast and exact, but can be noisy since a single backward pass
    captures local gradient information only.

'guided'
    Guided Backpropagation. Modifies the backward pass through ReLU so
    that only positive gradients flowing through positive activations are
    kept. This suppresses noise and produces sharper, more visually
    coherent maps. Note: has been shown to behave partly as an edge
    detector, so interpret with care.

'smoothgrad'
    Runs vanilla saliency over N copies of the input with small Gaussian
    noise added, then averages the results. The noise washes out
    irrelevant local gradients and the averaging reveals stable, meaningful
    attribution signal. Slower (N forward+backward passes) but noticeably
    cleaner than vanilla.

EXACT compatibility
-------------------
explain() returns a standardised result dict with a 'heatmap' key
containing a (H, W) float32 array in [0, 1]. This makes SaliencyMap
a drop-in for HeatmapComparator and all four evaluators.

    explainer = SaliencyMap(model)
    result    = explainer.explain(input_tensor, method="vanilla")

    cmp.compare(
        entries={
            "GradCAM":  (gradcam_result, gradcam_exp, {"method": "gradcam"}),
            "Vanilla":  (result,         explainer,   {"method": "vanilla"}),
            "Guided":   (guided_result,  explainer,   {"method": "guided"}),
        },
        ...
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


class SaliencyMap:
    """
    Pixel-level attribution via gradient-based saliency methods.

    Parameters
    ----------
    model : torch.nn.Module
        The model to explain.
    guided_activation_layer : torch.nn.Module, optional
        Activation class to hook for guided backpropagation. Default nn.ReLU.
        Change for other architectures:
            nn.GELU  — Vision Transformers (ViT, Swin)
            nn.SiLU  — EfficientNet
        Only relevant for method='guided'.
    save_dir : str, optional
        Directory for saved outputs. Default 'user_saves/saliency_saves'.

    Compatibility
    -------------
    Method      CNN (ReLU)   ViT/Swin (GELU)   Any other model
    ----------  -----------  ----------------  ---------------
    vanilla     yes          yes               yes
    smoothgrad  yes          yes               yes
    guided      yes          yes*              yes*
    * Pass the correct guided_activation_layer at construction time.
    """

    METHODS = ("vanilla", "guided", "smoothgrad")

    def __init__(
        self,
        model: torch.nn.Module,
        guided_activation_layer=nn.ReLU,
        save_dir: str = "user_saves/saliency_saves",
    ):
        self.model = model
        self.model.eval()
        self.guided_activation_layer = guided_activation_layer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._hooks = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def explain(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        method: str = "vanilla",
        n_samples: int = 50,
        noise_level: float = 0.1,
        input_image: Optional[np.ndarray] = None,
        overlay: bool = False,
        save_png: bool = False,
        tag: str = "",
    ) -> dict:
        """
        Compute a saliency map and return a standardised EXACT result dict.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed model input, shape (1, C, H, W).
        target_class : int, optional
            Class index to attribute. None = top predicted class.
        method : str, optional
            One of 'vanilla', 'guided', 'smoothgrad'. Default 'vanilla'.
        n_samples : int, optional
            Number of noisy samples for smoothgrad. Default 50.
        noise_level : float, optional
            Noise std as a fraction of input range for smoothgrad. Default 0.1.
        input_image : np.ndarray, optional
            Original image for overlay. Accepts uint8 [0,255], float [0,1],
            float [0,255], channel-first (C,H,W), or ImageNet-normalised arrays.
            If None, derived from input_tensor.
        overlay : bool, optional
            If True, blends the saliency map over the original image.
            Default False.
        save_png : bool, optional
            Whether to save the output. Default False.
        tag : str, optional
            Optional suffix for the saved filename. Default ''.

        Returns
        -------
        dict with keys:
            'heatmap'      : np.ndarray (H, W) float32 in [0, 1]
                             Grayscale saliency map — used by HeatmapComparator
                             and all EXACT evaluators.
            'visualization': np.ndarray (H, W, 3) float32 in [0, 1]
                             Colorized saliency map or overlay (RGB).
            'saliency_rgb' : np.ndarray (H, W, 3) float32 in [0, 1]
                             Colorized saliency map (always standalone, RGB).
            'filepath'     : Path or None
            'target_class' : int
            'method'       : str
        """
        method = method.lower()
        if method not in self.METHODS:
            raise ValueError(
                f"Method '{method}' not supported. Available: {list(self.METHODS)}"
            )

        # Resolve target class before any hooks alter the backward pass
        if target_class is None:
            with torch.no_grad():
                target_class = self.model(input_tensor).argmax(dim=1).item()

        # Compute raw gradient map (C, H, W)
        if method == "vanilla":
            gradient = self._vanilla_gradient(input_tensor, target_class)
        elif method == "guided":
            gradient = self._guided_backprop(input_tensor, target_class)
        else:
            gradient = self._smoothgrad(input_tensor, target_class, n_samples, noise_level)

        # Collapse to (H, W) float32 in [0, 1]
        heatmap     = self._to_grayscale(gradient)          # (H, W) float32 [0,1]
        saliency_rgb = self._colorize(heatmap)              # (H, W, 3) float32 [0,1]

        # Prepare display image — defensively normalised to float32 [0,1]
        img = _to_display_image(
            input_image if input_image is not None else input_tensor[0]
        )

        visualization = self._overlay(img, heatmap) if overlay else saliency_rgb

        filepath = None
        if save_png:
            suffix   = f"_{tag}" if tag else ""
            filepath = self.save_dir / f"saliency_{method}{suffix}.png"
            vis_u8   = np.uint8(visualization * 255)
            cv2.imwrite(str(filepath), cv2.cvtColor(vis_u8, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved: {filepath}")

        return {
            # ── EXACT standard keys ───────────────────────────────────
            "heatmap":      heatmap,        # (H,W) float32 [0,1] — for comparator/evaluators
            "visualization": visualization, # (H,W,3) float32 [0,1] RGB
            "filepath":      filepath,
            # ── Saliency-specific keys ────────────────────────────────
            "saliency_rgb":  saliency_rgb,  # standalone colorized map, always available
            "target_class":  target_class,
            "method":        method,
        }

    # ------------------------------------------------------------------
    # Gradient computation methods  (private)
    # ------------------------------------------------------------------

    def _vanilla_gradient(self, input_tensor: torch.Tensor, target_class: int):
        """Gradient of target class score w.r.t. input pixels."""
        x = input_tensor.clone().requires_grad_(True)
        output = self.model(x)
        self.model.zero_grad()
        output[0, target_class].backward()
        return x.grad.data[0].cpu()   # (C, H, W)

    def _guided_backprop(self, input_tensor: torch.Tensor, target_class: int):
        """
        Guided Backpropagation — hooks ReLU (or configured activation) to
        only propagate positive gradients through positive activations.
        Hooks are registered before and cleaned up after the pass.
        """
        self._register_guided_hooks()
        try:
            gradient = self._vanilla_gradient(input_tensor, target_class)
        finally:
            self._remove_hooks()
        return gradient

    def _smoothgrad(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        n_samples: int,
        noise_level: float,
    ):
        """
        Average vanilla gradients over n_samples noisy input copies.
        noise_std = noise_level * (input.max - input.min).
        """
        noise_std   = noise_level * (input_tensor.max() - input_tensor.min()).item()
        device      = input_tensor.device
        # Accumulate on CPU — _vanilla_gradient always returns a CPU tensor
        # via .cpu() so the accumulator must also live on CPU.
        accumulated = torch.zeros_like(input_tensor[0].cpu())

        for _ in range(n_samples):
            noise = torch.randn_like(input_tensor) * noise_std
            noisy = (input_tensor + noise).detach().to(device)
            accumulated += self._vanilla_gradient(noisy, target_class)

        return accumulated / n_samples   # (C, H, W) on CPU

    # ------------------------------------------------------------------
    # Guided backprop hook machinery  (private)
    # ------------------------------------------------------------------

    def _guided_relu_hook(self, module, grad_in, grad_out):
        """
        Only pass back gradients that are positive in both directions.
        Compatible with register_full_backward_hook — grad_in is a tuple
        of tensors, some of which may be None.
        """
        return tuple(
            torch.clamp(g, min=0.0) if g is not None else g
            for g in grad_in
        )

    def _register_guided_hooks(self):
        """
        Register guided backprop hooks and disable inplace operations.

        Inplace activations (e.g. ReLU(inplace=True), common in ResNet)
        conflict with full backward hooks — the hook's custom backward
        output gets modified before autograd reads it, causing a RuntimeError.
        We temporarily set inplace=False on all matching activation layers
        and restore them in _remove_hooks().
        """
        self._inplace_restored = []   # track layers we patched so we can restore

        for module in self.model.modules():
            if isinstance(module, self.guided_activation_layer):
                # Disable inplace if the layer supports it
                if hasattr(module, "inplace") and module.inplace:
                    module.inplace = False
                    self._inplace_restored.append(module)
                self._hooks.append(
                    module.register_full_backward_hook(self._guided_relu_hook)
                )

        if not self._hooks:
            warnings.warn(
                f"Guided backprop found no '{self.guided_activation_layer.__name__}' "
                f"layers in the model. Result will be identical to vanilla saliency. "
                f"If your model uses a different activation (e.g. GELU, SiLU), pass it "
                f"via guided_activation_layer=nn.GELU when constructing SaliencyMap.",
                UserWarning,
                stacklevel=3,
            )

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        # Restore inplace on all layers we patched
        for module in self._inplace_restored:
            module.inplace = True
        self._inplace_restored = []

    # ------------------------------------------------------------------
    # Visualisation helpers  (private, all output float32 [0,1] RGB)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_grayscale(gradient: torch.Tensor) -> np.ndarray:
        """
        Collapse (C, H, W) gradient tensor → (H, W) float32 in [0, 1].
        Takes the max absolute value across channels then normalises.
        """
        sal = gradient.abs().max(dim=0).values.numpy().astype(np.float32)
        mn, mx = sal.min(), sal.max()
        return (sal - mn) / (mx - mn + 1e-8)

    @staticmethod
    def _colorize(saliency: np.ndarray) -> np.ndarray:
        """
        Apply inferno colormap to a [0, 1] grayscale map.
        Returns RGB float32 [0, 1].
        """
        uint8   = np.uint8(saliency * 255)
        colored = cv2.applyColorMap(uint8, cv2.COLORMAP_INFERNO)   # BGR uint8
        rgb     = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)         # RGB uint8
        return rgb.astype(np.float32) / 255.0                      # RGB float32 [0,1]

    @staticmethod
    def _overlay(image: np.ndarray, saliency: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        """
        Blend colorized saliency over the display image.
        Both inputs must be float32 [0, 1]. Returns float32 [0, 1] RGB.
        """
        import matplotlib.cm as cm
        colored = cm.inferno(saliency)[..., :3].astype(np.float32)  # RGB float32 [0,1]
        return np.clip(alpha * colored + (1 - alpha) * image, 0, 1)


# ---------------------------------------------------------------------------
# Display image helper  (mirrors convention in comparator and IG explainer)
# ---------------------------------------------------------------------------

def _to_display_image(img) -> np.ndarray:
    """
    Convert any image input to RGB float32 in [0, 1].

    Handles:
      - torch.Tensor  (1,C,H,W) or (C,H,W) or (H,W,C)
      - np.ndarray    uint8 [0,255], float [0,255], float [0,1]
      - channel-first (C,H,W) arrays
      - ImageNet-normalised arrays with negative values
    """
    if isinstance(img, torch.Tensor):
        img = img[0] if img.ndim == 4 else img
        img = img.cpu().numpy()

    img = np.array(img, dtype=np.float32)

    if img.ndim == 3 and img.shape[0] == 3:       # (C,H,W) → (H,W,C)
        img = np.transpose(img, (1, 2, 0))

    if img.max() > 1.0:                            # [0,255] → [0,1]
        img = img / 255.0

    if img.min() < 0.0:                            # ImageNet-normalised → [0,1]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return np.clip(img, 0.0, 1.0)