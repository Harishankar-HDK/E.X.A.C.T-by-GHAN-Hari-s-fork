import warnings
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
#  Rule metadata  —  every rule carries its "why" and "why not" as data,
#  not just as comments.  This is what gets surfaced in explain() output.
# ──────────────────────────────────────────────────────────────────────────────

RULE_INFO = {
    "zB": {
        "name": "LRP-z^B  (Bounded Rule)",
        "applied_to": "First Conv2d layer (raw / normalised pixel input)",
        "why": (
            "Pixel values are bounded — either [0,1] after /255 scaling or "
            "roughly [-2.1, 2.6] after ImageNet normalisation. "
            "z^B incorporates the actual lower bound (l) and upper bound (h) "
            "of the input domain into the denominator, which cancels the bias "
            "term and preserves the completeness axiom (sum R_inputs = f(x)[class]) "
            "at the pixel layer."
        ),
        "why_not_elsewhere": (
            "Bounds are meaningless for deeper activations that are unbounded. "
            "Applying z^B past the first layer produces noisy, uninterpretable "
            "attributions and breaks conservation."
        ),
    },
    "alphabeta": {
        "name": "LRP-alphabeta  (Alpha-Beta Rule)",
        "applied_to": "All Conv2d layers except the first",
        "why": (
            "LRP-alphabeta separately weights positive contributions (alpha) and "
            "negative contributions (beta), where alpha + beta = 1. "
            "This gives explicit control over the excitatory vs inhibitory balance "
            "of the explanation. alpha=1 beta=0 (the default) is equivalent to the "
            "old z+ rule and gives perfect conservation with no stabiliser leak. "
            "Increasing beta surfaces which inputs actively suppress the prediction. "
            "Standard values: alpha=1 beta=0 (conservative), alpha=2 beta=1 (balanced)."
        ),
        "why_not_first_layer": (
            "Pixel values after mean-subtraction normalisation ARE signed (negative). "
            "The alpha path handles positive weights correctly but the beta path on "
            "signed pixel inputs can produce unstable attributions. "
            "z^B is the correct rule for the first layer touching raw pixels."
        ),
        "why_not_linear": (
            "Linear layers can have both positive and negative activations with no "
            "ReLU constraint, so separating positive/negative weight paths is not "
            "well-defined. LRP-epsilon is the correct rule for Linear layers."
        ),
    },
    "epsilon": {
        "name": "LRP-epsilon  (Epsilon Stabiliser Rule)",
        "applied_to": "All nn.Linear layers (classifier head + MLP layers)",
        "why": (
            "Linear-layer activations are not constrained to be non-negative, "
            "so z+ would discard real signal. "
            "A small epsilon added to the denominator prevents division-by-zero and "
            "sign-flipping without meaningfully violating conservation. "
            "This is the standard choice for fully-connected layers."
        ),
        "why_not_conv_input": (
            "For the first convolutional layer, epsilon does not account for the "
            "bounded pixel domain and can over-smooth low-level attributions. "
            "z^B is the correct rule there."
        ),
    },
    "alphabeta": {
        "name": "LRP-alphabeta  (Alpha-Beta Rule)",
        "applied_to": "All Conv2d layers (replaces z+ when alpha != 1 or beta != 0)",
        "why": (
            "LRP-alphabeta separates positive (alpha) and negative (beta) "
            "weight contributions and propagates them independently, weighted "
            "by alpha and beta respectively, where alpha + beta = 1. "
            "This gives the user direct control over how excitatory vs "
            "inhibitory evidence is balanced in the explanation. "
            "alpha=1, beta=0 recovers z+ exactly (positive only). "
            "alpha=2, beta=1 is the classic setting from Bach et al. 2015 "
            "which gives sharper, more localised attributions for image CNNs."
        ),
        "why_not_linear": (
            "Linear layers in the classifier head can have both positive and "
            "negative activations that are not ReLU-gated. Applying alphabeta "
            "there requires separating activations too (not just weights), "
            "which is more expensive and rarely improves interpretability. "
            "LRP-epsilon is the standard and sufficient choice for Linear layers."
        ),
        "why_not_first_layer": (
            "The first Conv2d touching raw pixel inputs uses z^B (bounded rule) "
            "regardless of alpha/beta setting. z^B accounts for the bounded "
            "pixel domain which alphabeta does not handle correctly at the input."
        ),
    },
    "passthrough": {
        "name": "Pass-through  (no rule applied)",
        "applied_to": "BatchNorm, Dropout, ReLU/activations, Pooling, Flatten, Identity",
        "why": (
            "These layers have no learnable weight matrix that meaningfully "
            "transforms relevance — they rescale, gate, or reshape it. "
            "Passing relevance through unchanged is the standard approximation "
            "used by all major LRP libraries (Zennit, iNNvestigate, Captum)."
        ),
        "why_not_fuse_bn": (
            "The principled alternative is to absorb BatchNorm parameters into "
            "the adjacent Conv2d weights before applying z+. That requires "
            "mutating the caller's model object, which is unacceptable in a "
            "library context. BN fusion is left as a user-side extension."
        ),
    },
}

# What we support and — critically — what we don't, and why
MODEL_SUPPORT = {
    "cnn": {
        "supported": True,
        "why": (
            "CNNs are sequential Conv2d -> activation -> pooling -> Linear stacks. "
            "LRP has a well-defined, theoretically grounded rule for each layer "
            "type (z^B for pixel input, z+ for conv, epsilon for linear). "
            "Completeness (sum R_pixels = f(x)[class]) holds for VGG-style and "
            "AlexNet-style architectures."
        ),
        "why_not": None,
        "constraint": (
            "Sequential topology only. Models with skip connections (ResNet, "
            "DenseNet, EfficientNet) cannot be correctly handled by layer-by-layer "
            "LRP — relevance at residual additions needs to be split across two "
            "paths, which requires full graph traversal, not a layer list. "
            "BatchNorm is approximated as a pass-through; for maximum accuracy, "
            "fuse BN into adjacent Conv2d weights before running LRP."
        ),
    },
    "mlp": {
        "supported": True,
        "why": (
            "A plain MLP is entirely Linear layers + activations. "
            "LRP-epsilon is correct and complete for every Linear layer. "
            "There are no convolutions, no skip connections, no attention — "
            "the architecture is the simplest possible case for LRP."
        ),
        "why_not": None,
        "constraint": (
            "The output is a flat relevance vector reshaped to the input tensor "
            "shape. For image inputs this means per-pixel-channel scores, not a "
            "spatial heatmap derived from feature maps. Visualisation quality is "
            "lower than CNN-LRP because there is no spatial hierarchy to exploit."
        ),
    },
    "unsupported": {
        "supported": False,
        "why": None,
        "why_not": (
            "Transformers and attention-based models are not supported. "
            "LRP-epsilon applied to softmax-attention layers has no valid theoretical "
            "interpretation: the softmax mixes token relevances in a non-linear "
            "way that the epsilon rule cannot correctly invert. "
            "String-matching attention class names is also unreliable across "
            "third-party libraries. "
            "Use IGImageExplainer for transformer / ViT architectures instead."
        ),
        "constraint": None,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
#  Model-type detection
# ──────────────────────────────────────────────────────────────────────────────

def _detect_model_type(model: nn.Module) -> str:
    """
    Returns 'cnn', 'mlp', or 'unsupported'.

    Conv2d is the decisive signal for CNN — checked first, wins even if
    Linear layers are also present (e.g. classifier head).
    Linear-only with no Conv2d -> MLP.
    Anything else -> unsupported.
    """
    has_conv   = any(isinstance(m, nn.Conv2d) for m in model.modules())
    has_linear = any(isinstance(m, nn.Linear) for m in model.modules())

    if has_conv:
        return "cnn"
    if has_linear:
        return "mlp"
    return "unsupported"


# ──────────────────────────────────────────────────────────────────────────────
#  Input-type detection and normalisation
# ──────────────────────────────────────────────────────────────────────────────

def _detect_input_type(x: torch.Tensor) -> str:
    """
    Inspect a tensor and return one of:
        'image_chw'  — [1, C, H, W]  standard image batch (C = 1 or 3)
        'image_hwc'  — [1, H, W, C]  channels-last image (auto-corrected)
        'sequence'   — [1, L, D]  text embedding / time-series
        'flat'       — [1, N]  flat feature vector (MLP input)
        'volume'     — [1, C, D, H, W]  3-D input (LRP runs, vis skipped)
        'unknown'    — anything else

    Decision logic
    --------------
    ndim=4 with C in (1,3) → standard image.  Any other 4-D → treated as
    spatial anyway (CNN accepts it).  ndim=3 → sequence (flattened to 2-D
    before LRP).  ndim=2 → flat MLP input.  ndim=5 → volumetric.
    The LRP math is shape-agnostic; this function only drives
    visualisation decisions and pixel bound inference.
    """
    ndim = x.dim()
    if ndim == 4:
        _, c, h, w = x.shape
        if h in (1, 3) and c > 1 and w > 1:
            return "image_hwc"      # channels last
        return "image_chw"          # everything else 4-D treated as CHW
    if ndim == 3:
        return "sequence"
    if ndim == 2:
        return "flat"
    if ndim == 5:
        return "volume"
    return "unknown"


def _normalise_input_tensor(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, float, float, str]:
    """
    Bring any supported input tensor into [1, C, H, W] or [1, N] and
    infer safe pixel_low / pixel_high bounds for the z^B rule.

    Handles
    -------
    - channels-last images  → transposed to channels-first
    - sequence tensors      → flattened to [1, L*D]
    - [0,1] float images    → bounds 0.0 / 1.0
    - [0,255] uint8 images  → bounds 0.0 / 255.0
    - signed/normalised     → bounds = actual batch min / max
    - volumetric            → passed through with a warning; vis skipped

    Returns
    -------
    tensor     : corrected tensor
    pixel_low  : inferred lower bound for z^B
    pixel_high : inferred upper bound for z^B
    input_type : one of the strings from _detect_input_type
    """
    input_type = _detect_input_type(x)

    if input_type == "image_hwc":
        x = x.permute(0, 3, 1, 2).contiguous()
        input_type = "image_chw"

    if input_type == "sequence":
        x = x.flatten(1)
        input_type = "flat"

    if input_type == "volume":
        warnings.warn(
            "LRPImageExplainer: volumetric input [1,C,D,H,W] detected. "
            "LRP pass will run but 2-D overlay visualisation is skipped. "
            "result['overlay_*'] keys will be None.",
            UserWarning,
            stacklevel=4,
        )

    vmin = float(x.min().item())
    vmax = float(x.max().item())

    if vmin >= 0.0 and vmax <= 1.0:
        pixel_low, pixel_high = 0.0, 1.0
    elif vmin >= 0.0 and vmax <= 255.0:
        pixel_low, pixel_high = 0.0, 255.0
    else:
        pixel_low, pixel_high = vmin, vmax   # signed / normalised

    return x, pixel_low, pixel_high, input_type


def _tensor_to_bgr(
    x: torch.Tensor,
    input_type: str,
) -> Optional[np.ndarray]:
    """
    Convert an input tensor to a BGR numpy array for overlay visualisation.
    Returns None for non-image inputs (flat, sequence, volume) — the
    dashboard will skip overlays gracefully in those cases.
    """
    if input_type not in ("image_chw", "image_hwc"):
        return None

    img = x[0].cpu().numpy()               # [C, H, W]
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)    # grayscale → 3-channel
    img = np.transpose(img, (1, 2, 0))     # [H, W, C]
    if img.max() <= 1.0:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────────────────
#  LRP rules  (stateless, functional)
# ──────────────────────────────────────────────────────────────────────────────

class _LRPRules:

    @staticmethod
    def epsilon_rule(
        layer: nn.Module,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        activation = activation.detach().requires_grad_(True)
        with torch.enable_grad():
            z = layer(activation)
            z = z + eps * z.sign()
            z = z + 1e-9
            s = (relevance / z).detach()
            (z * s).sum().backward()
        return (activation * activation.grad).detach()

    @staticmethod
    def alphabeta_rule(
        layer: nn.Module,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """
        LRP-alphabeta — used for all Conv2d layers except the first.

        Separates the forward pass into positive-weight and negative-weight
        contributions and weights them by alpha and beta respectively.

        Conservation guarantee: alpha - beta = 1 ensures sum of redistributed
        relevances equals incoming relevance exactly.  The code enforces
        beta = alpha - 1 if the caller supplies only alpha, and warns if
        both are supplied but do not satisfy the constraint.

        alpha=1, beta=0  →  equivalent to the old z+ rule (conservative,
                             only positive contributions, no inhibitory signal)
        alpha=2, beta=1  →  balanced, surfaces both excitatory and inhibitory
                             pixel contributions
        """
        # Enforce conservation constraint: alpha - beta must equal 1
        if abs((alpha - beta) - 1.0) > 1e-6:
            import warnings
            warnings.warn(
                f"LRP-alphabeta: alpha - beta = {alpha - beta:.4f}, should be 1.0. "
                f"Setting beta = alpha - 1 = {alpha - 1.0:.4f} to enforce conservation.",
                UserWarning,
                stacklevel=3,
            )
            beta = alpha - 1.0

        w      = layer.weight
        w_pos  = w.clamp(min=0)
        w_neg  = w.clamp(max=0)
        b      = layer.bias
        b_pos  = b.clamp(min=0) if b is not None else None
        b_neg  = b.clamp(max=0) if b is not None else None

        activation = activation.detach().requires_grad_(True)

        def _conv(x, wt, bt):
            return nn.functional.conv2d(
                x, wt, bt,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups,
            )

        with torch.enable_grad():
            # positive path: positive weights on positive activations
            z_pos = _conv(activation, w_pos, b_pos) + 1e-9
            s_pos = (relevance / z_pos).detach()
            (z_pos * s_pos).sum().backward(retain_graph=True)
            grad_pos = activation.grad.clone()
            activation.grad.zero_()

            # negative path: negative weights on activations
            z_neg = _conv(activation, w_neg, b_neg) - 1e-9
            s_neg = (relevance / z_neg).detach()
            (z_neg * s_neg).sum().backward()
            grad_neg = activation.grad.clone()

        return (activation * (alpha * grad_pos - beta * grad_neg)).detach()

    @staticmethod
    def alphabeta_rule(
        layer: nn.Module,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """
        LRP-alphabeta — tunable positive/negative contribution split.

        alpha controls how much positive weight contributions are propagated.
        beta  controls how much negative weight contributions are propagated.
        Constraint: alpha - beta = 1  (equivalently beta = alpha - 1).

        alpha=1, beta=0  →  identical to z+ (positive only, no negative)
        alpha=2, beta=1  →  classic Bach et al. 2015 setting (sharper maps)
        alpha=1, beta=0 is the default here to preserve backward compatibility
        with the existing z+ behaviour.

        Why separate positive and negative weights?
        -------------------------------------------
        Post-ReLU activations are non-negative, but weights can be positive
        or negative. Positive weights amplify the activation (excitatory).
        Negative weights suppress it (inhibitory). By weighting these
        contributions separately with alpha and beta, the user can tune
        whether the explanation emphasises what the network found (alpha)
        or what it suppressed (beta).

        Why alpha - beta = 1?
        ---------------------
        This constraint ensures relevance conservation: the redistributed
        relevances sum exactly to the incoming relevance. Violating it
        breaks the completeness axiom.
        """
        # enforce conservation constraint: beta = alpha - 1
        beta = alpha - 1.0

        w      = layer.weight
        w_pos  = w.clamp(min=0)
        w_neg  = w.clamp(max=0)
        b_pos  = layer.bias.clamp(min=0) if layer.bias is not None else None
        b_neg  = layer.bias.clamp(max=0) if layer.bias is not None else None

        activation = activation.detach().requires_grad_(True)

        with torch.enable_grad():
            # positive contribution path
            z_pos = nn.functional.conv2d(
                activation, w_pos, b_pos,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups,
            ) + 1e-9
            s_pos = (relevance / z_pos).detach()
            grad_pos = torch.autograd.grad(
                (z_pos * s_pos).sum(), activation, retain_graph=True
            )[0]

            # negative contribution path
            z_neg = nn.functional.conv2d(
                activation, w_neg, b_neg,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups,
            ) - 1e-9
            s_neg = (relevance / z_neg).detach()
            grad_neg = torch.autograd.grad(
                (z_neg * s_neg).sum(), activation
            )[0]

        return (activation * (alpha * grad_pos - beta * grad_neg)).detach()

    @staticmethod
    def zb_rule(
        layer: nn.Module,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        low: float = 0.0,
        high: float = 1.0,
    ) -> torch.Tensor:
        w     = layer.weight
        w_pos = w.clamp(min=0)
        w_neg = w.clamp(max=0)

        l = torch.full_like(activation, low).requires_grad_(True)
        h = torch.full_like(activation, high).requires_grad_(True)
        activation = activation.detach().requires_grad_(True)

        def _fwd(x, wt):
            b = layer.bias if layer.bias is not None else None
            return nn.functional.conv2d(
                x, wt, b,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups,
            )

        with torch.enable_grad():
            z  = _fwd(activation, w)
            zl = _fwd(l, w_pos)
            zh = _fwd(h, w_neg)
            z  = z - zl - zh + 1e-9
            s  = (relevance / z).detach()
            (z * s).sum().backward()

        return (
            activation * activation.grad
            - l * l.grad
            - h * h.grad
        ).detach()


# ──────────────────────────────────────────────────────────────────────────────
#  Main class
# ──────────────────────────────────────────────────────────────────────────────

class LRPImageExplainer:
    """
    Layer-wise Relevance Propagation (LRP) explainer for CNN and MLP PyTorch models.

    Supported architectures
    -----------------------
    CNN  (contains nn.Conv2d)
        Sequential topology only — VGG, AlexNet, simple custom CNNs.
        Rules: z^B on first Conv2d, z+ on all other Conv2d, epsilon on Linear.

    MLP  (nn.Linear only, no Conv2d)
        All Linear layers get LRP-epsilon. Output is per-pixel relevance.

    NOT supported
    -------------
    Transformers / ViT / attention-based models.
    ResNet / DenseNet / EfficientNet (skip connections break sequential LRP).
    Use IGImageExplainer for those.

    Quick start
    -----------
        explainer = LRPImageExplainer(model)
        result    = explainer.explain(input_tensor, input_image)
        explainer.save_dashboard(result, "lrp_out.png")

    Every result dict contains 'why' and 'why_not' keys explaining
    which rules were applied and what this model type cannot do.
    Call get_model_info() to see this without running an explanation.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        save_dir: str = "user_saves/lrp_saves",
    ):
        self.model    = model.eval()
        self.device   = device or next(model.parameters()).device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model_type = _detect_model_type(model)
        self._support   = MODEL_SUPPORT[self.model_type]

        if not self._support["supported"]:
            raise ValueError(
                f"LRPImageExplainer does not support this model.\n\n"
                f"WHY NOT:\n{self._support['why_not']}"
            )

        self._warn_if_skip_connections(model)
        self._layers: List[nn.Module] = self._extract_layers(model)

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def explain(
        self,
        input_tensor: torch.Tensor,
        input_image: Optional[np.ndarray] = None,
        target_class: Optional[int] = None,
        pixel_low: float = 0.0,
        pixel_high: float = 1.0,
        eps: float = 1e-6,
        lrp_alpha: float = 1.0,
        lrp_beta: float = 0.0,
        overlay_alpha: float = 0.5,
        class_name: str = "",
        save_png: bool = False,
    ) -> Dict:
        """
        Run LRP on any input type and return attributions + visualisations + why/why_not.

        Input auto-detection
        --------------------
        input_tensor can be ANY tensor the model accepts:
            [1, C, H, W]   — standard image (channels-first)
            [1, H, W, C]   — channels-last image (auto-transposed)
            [1, L, D]      — sequence / text embedding (flattened to [1, L*D])
            [1, N]         — flat feature vector (MLP input)
            [1, C, D, H, W]— volumetric (LRP runs, overlays skipped)

        pixel_low / pixel_high are inferred automatically from the actual value
        range of input_tensor.  You only need to set them manually if your
        preprocessing uses non-standard bounds.

        Parameters
        ----------
        input_tensor  : Any tensor the model accepts — shape is auto-detected.
        input_image   : Original BGR image [H, W, 3] for overlay visualisation.
                        If None and input is an image tensor, derived automatically.
                        If None and input is flat/sequence, overlays are skipped.
        target_class  : Class index to explain. None = predicted class.
        pixel_low     : Override lower pixel bound for z^B rule (default: auto).
        pixel_high    : Override upper pixel bound for z^B rule (default: auto).
        eps           : Stabiliser for LRP-epsilon on Linear layers.
        lrp_alpha     : Alpha for LRP-alphabeta rule on Conv2d layers.
                        Controls weight of positive contributions.
                        Default 1.0. Common values: 1 (conservative), 2 (balanced).
        lrp_beta      : Beta for LRP-alphabeta rule on Conv2d layers.
                        Controls weight of negative contributions.
                        Must satisfy alpha - beta = 1 (enforced automatically).
                        Default 0.0. Common values: 0 (conservative), 1 (balanced).
        overlay_alpha : Heatmap blend strength (0 = image only, 1 = heatmap only).
        class_name    : Human-readable label used in saved filenames.
        save_png      : If True, saves the dashboard PNG to save_dir.

        Returns
        -------
        dict:
            model_type          : 'cnn' or 'mlp'
            input_type          : detected input shape type
            target_class        : int
            lrp_alpha           : float  — alpha used
            lrp_beta            : float  — beta used  (alpha - beta = 1)
            completeness_error  : float  — should be < 0.05 for valid results
            overlay_magnitude   : BGR ndarray or None (None for non-image inputs)
            overlay_positive    : BGR ndarray or None
            overlay_negative    : BGR ndarray or None
            overlay_contour     : BGR ndarray or None
            filepath            : str or None
            why                 : str  — why LRP is valid for this model type
            why_not             : str  — what this model type cannot do / limits
            rule_log            : list[dict]  — per-rule why/why_not at each layer
        """
        input_tensor = input_tensor.to(self.device)

        # ── Auto-detect input type, normalise shape, infer pixel bounds ──────
        input_tensor, inferred_low, inferred_high, input_type = \
            _normalise_input_tensor(input_tensor)

        # Caller-supplied bounds always win; inferred bounds are the fallback
        pixel_low  = pixel_low  if pixel_low  != 0.0 else inferred_low
        pixel_high = pixel_high if pixel_high != 1.0 else inferred_high

        # ── Resolve visualisation image ───────────────────────────────────────
        # For non-image inputs (flat/sequence/volume) this returns None and
        # overlay generation is skipped — the result dict still contains
        # completeness_error, why, why_not, and rule_log.
        if input_image is None:
            input_image = _tensor_to_bgr(input_tensor, input_type)

        if target_class is None:
            with torch.no_grad():
                target_class = self.model(input_tensor).argmax(dim=1).item()

        # Enforce conservation constraint here so result dict shows actual values
        if abs((lrp_alpha - lrp_beta) - 1.0) > 1e-6:
            warnings.warn(
                f"LRP-alphabeta: alpha - beta = {lrp_alpha - lrp_beta:.4f}, "
                f"should be 1.0. Setting beta = alpha - 1 = {lrp_alpha - 1.0:.4f}.",
                UserWarning,
                stacklevel=2,
            )
            lrp_beta = lrp_alpha - 1.0

        relevance_map, completeness_err, rule_log = self._lrp_pass(
            input_tensor, target_class, pixel_low, pixel_high, eps,
            lrp_alpha, lrp_beta
        )

        mag_map = self._magnitude_map(relevance_map)
        pos_map = self._positive_map(relevance_map)
        neg_map = self._negative_map(relevance_map)

        if input_image is not None:
            overlays = {
                "overlay_magnitude": self._heatmap_overlay(mag_map, input_image, overlay_alpha, cv2.COLORMAP_JET),
                "overlay_positive":  self._heatmap_overlay(pos_map, input_image, overlay_alpha, cv2.COLORMAP_HOT),
                "overlay_negative":  self._heatmap_overlay(neg_map, input_image, overlay_alpha, cv2.COLORMAP_WINTER),
                "overlay_contour":   self._contour_overlay(mag_map, input_image),
            }
        else:
            # Non-image input (flat feature vector, etc.) — no spatial overlay
            overlays = {
                "overlay_magnitude": None,
                "overlay_positive":  None,
                "overlay_negative":  None,
                "overlay_contour":   None,
            }

        filepath = None
        if save_png:
            suffix   = f"_{class_name}" if class_name else f"_{target_class}"
            filepath = str(self.save_dir / f"lrp{suffix}.png")
            self._write_dashboard(overlays, completeness_err, target_class,
                                  class_name, filepath)
            print(f"Saved: {filepath}")

        return {
            "model_type":         self.model_type,
            "input_type":         input_type,
            "target_class":       target_class,
            "lrp_alpha":          lrp_alpha,
            "lrp_beta":           lrp_beta,
            "completeness_error": completeness_err,
            **overlays,
            "filepath":           filepath,
            "why":                self._support["why"],
            "why_not":            self._support["constraint"],
            "rule_log":           rule_log,
        }

    def visualize_and_save(
        self,
        result: Dict,
        save_path: str,
        class_name: Optional[str] = None,
        dpi: int = 150,
    ) -> str:
        """
        Save a 2x2 dashboard PNG from a result dict returned by explain().

        Parameters
        ----------
        result    : Dict returned by explain().
        save_path : Destination path, e.g. "lrp_out.png".
        class_name: Optional human-readable class label.
        dpi       : Output resolution.

        Returns
        -------
        save_path : str  — echoed back for convenience.
        """
        self._write_dashboard(
            result,
            result["completeness_error"],
            result["target_class"],
            class_name or "",
            save_path,
            dpi=dpi,
        )
        print(f"Saved: {save_path}")
        return save_path

    def save_dashboard(
        self,
        result: Dict,
        save_path: str,
        class_name: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Alias for visualize_and_save() — matches IGImageExplainer API."""
        self.visualize_and_save(result, save_path, class_name, dpi)

    def get_model_info(self) -> Dict:
        """
        Return a summary of what LRP can and cannot do for this model,
        without running an explanation. Useful for pre-flight checks.

        Returns
        -------
        dict:
            model_type : str
            supported  : bool
            why        : str or None  — why this architecture is supported
            why_not    : str or None  — why it is not (None if supported)
            constraint : str or None  — known limits even when supported
            rules_used : list[dict]   — which rule applies to which layer type
        """
        info = MODEL_SUPPORT[self.model_type]
        return {
            "model_type": self.model_type,
            "supported":  info["supported"],
            "why":        info["why"],
            "why_not":    info["why_not"],
            "constraint": info["constraint"],
            "rules_used": self._rules_summary(),
        }

    def get_model(self) -> nn.Module:
        return self.model

    # ──────────────────────────────────────────────────────────────────────
    #  LRP backward pass  (private)
    # ──────────────────────────────────────────────────────────────────────

    def _lrp_pass(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        pixel_low: float,
        pixel_high: float,
        eps: float,
        lrp_alpha: float = 1.0,
        lrp_beta: float = 0.0,
    ) -> Tuple[torch.Tensor, float, List[Dict]]:
        """
        Full forward + backward LRP pass.

        Returns
        -------
        relevance_map    : [1, C, H, W]
        completeness_err : |sum R_input - f(x)[class]| / |f(x)[class]|
        rule_log         : list of dicts in forward layer order, each with
                           keys: layer, rule_name, why, why_not
        """
        # 1. Collect activations and MaxPool switch indices
        # MaxPool indices record which input position "won" each pool window.
        # In the backward pass relevance is routed back ONLY through those
        # positions — this is the correct winner-take-all redistribution and
        # is what makes the completeness axiom hold across pooling layers.
        activations: List[torch.Tensor] = []
        pool_indices: Dict[int, torch.Tensor] = {}   # layer_idx -> indices
        x = input_tensor.clone().detach()
        activations.append(x)
        with torch.no_grad():
            for i, layer in enumerate(self._layers):
                if isinstance(layer, nn.MaxPool2d):
                    # return_indices gives us the argmax positions
                    pool = nn.MaxPool2d(
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        return_indices=True,
                    )
                    x, indices = pool(x)
                    pool_indices[i] = indices
                else:
                    x = layer(x)
                activations.append(x)

        # 2. Initialise relevance at output
        logits    = activations[-1]
        score     = logits[0, target_class].item()
        relevance = torch.zeros_like(logits)
        relevance[0, target_class] = logits[0, target_class]

        # 3. Identify first conv
        conv_indices   = [i for i, l in enumerate(self._layers) if isinstance(l, nn.Conv2d)]
        first_conv_idx = conv_indices[0] if conv_indices else -1

        # 4. Backward pass
        rule_log: List[Dict] = []
        for idx in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[idx]
            act   = activations[idx]

            # CNN -> Linear boundary (forward): flatten spatial activation
            if isinstance(layer, nn.Linear) and act.dim() > 2:
                act = act.flatten(1)

            relevance, log_entry = self._apply_rule(
                layer, act, relevance,
                is_first_conv=(idx == first_conv_idx),
                pixel_low=pixel_low,
                pixel_high=pixel_high,
                eps=eps,
                lrp_alpha=lrp_alpha,
                lrp_beta=lrp_beta,
                pool_indices=pool_indices.get(idx),
            )
            if log_entry:
                rule_log.insert(0, log_entry)   # keep forward order

        # 5. Reshape MLP flat output back to spatial
        if relevance.dim() == 2:
            relevance = relevance.view_as(input_tensor)

        # 6. Completeness error
        completeness_err = abs(relevance.sum().item() - score) / (abs(score) + 1e-9)

        return relevance.detach(), completeness_err, rule_log

    def _apply_rule(
        self,
        layer: nn.Module,
        activation: torch.Tensor,
        relevance: torch.Tensor,
        is_first_conv: bool,
        pixel_low: float,
        pixel_high: float,
        eps: float,
        lrp_alpha: float = 1.0,
        lrp_beta: float = 0.0,
        pool_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Select and apply the appropriate LRP rule.
        Returns (new_relevance, log_entry).
        log_entry is None for pass-through layers.
        """
        PASSTHROUGH = (
            nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm,
            nn.Dropout, nn.Dropout2d,
            nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU,
            nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
            nn.Flatten, nn.Identity,
        )

        layer_name = type(layer).__name__

        if isinstance(layer, PASSTHROUGH):
            info = RULE_INFO["passthrough"]

            # Flatten backward: relevance is flat [1,N], reshape back to the
            # spatial dims of the activation that fed INTO Flatten.
            if isinstance(layer, nn.Flatten) and relevance.dim() == 2:
                relevance = relevance.view(activation.shape)

            # MaxPool backward: route relevance back only through the positions
            # that were selected (the "winning" max positions) during the
            # forward pass.  This is winner-take-all redistribution.
            #
            # Why not nearest-neighbour upsample?
            # Upsample spreads each relevance value across the entire pool
            # window uniformly, which multiplies total relevance by kernel_size²
            # at every pooling layer.  With 3 MaxPool2d(2) layers that is an
            # 8x inflation per pass — exactly what produced completeness_error
            # of 22000+.  Winner-take-all routes relevance back through exactly
            # one position per window, so the total relevance sum is preserved.
            if isinstance(layer, nn.MaxPool2d) and pool_indices is not None:
                unpool = nn.MaxUnpool2d(
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                )
                relevance = unpool(relevance, pool_indices,
                                   output_size=activation.shape)

            elif isinstance(layer, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                # AvgPool: distribute relevance uniformly (correct for avg)
                if relevance.dim() == 4 and activation.dim() == 4                         and relevance.shape != activation.shape:
                    relevance = nn.functional.interpolate(
                        relevance.float(),
                        size=activation.shape[2:],
                        mode="nearest",
                    ) / (layer.kernel_size ** 2
                         if hasattr(layer, "kernel_size") else 1)

            return relevance, {
                "layer":     layer_name,
                "rule_name": info["name"],
                "why":       info["why"],
                "why_not":   info["why_not_fuse_bn"],
            }

        if isinstance(layer, nn.Conv2d):
            if is_first_conv:
                info = RULE_INFO["zB"]
                return (
                    _LRPRules.zb_rule(layer, activation, relevance,
                                      low=pixel_low, high=pixel_high),
                    {
                        "layer":     layer_name,
                        "rule_name": info["name"],
                        "why":       info["why"],
                        "why_not":   info["why_not_elsewhere"],
                    },
                )
            else:
                info = RULE_INFO["alphabeta"]
                return (
                    _LRPRules.alphabeta_rule(layer, activation, relevance,
                                             alpha=lrp_alpha, beta=lrp_beta),
                    {
                        "layer":     layer_name,
                        "rule_name": f"{info['name']}  [α={lrp_alpha}, β={lrp_beta}]",
                        "why":       info["why"],
                        "why_not":   info["why_not_first_layer"],
                    },
                )

        if isinstance(layer, nn.Linear):
            info = RULE_INFO["epsilon"]
            return (
                _LRPRules.epsilon_rule(layer, activation, relevance, eps=eps),
                {
                    "layer":     layer_name,
                    "rule_name": info["name"],
                    "why":       info["why"],
                    "why_not":   info["why_not_conv_input"],
                },
            )

        # Unrecognised layer — pass through silently
        return relevance, None

    # ──────────────────────────────────────────────────────────────────────
    #  Layer extraction  (private)
    # ──────────────────────────────────────────────────────────────────────

    def _extract_layers(self, model: nn.Module) -> List[nn.Module]:
        """
        Recursively flatten all nn.Sequential containers into a single
        ordered list of primitive layers.

        Why recursive?
        --------------
        Models like TumorModel wrap their entire stack inside self.layer_stack
        (a Sequential that is itself a child of the module). A single-level
        unroll sees layer_stack as one atomic block and never reaches Conv2d,
        MaxPool2d, Linear etc. individually — causing shape mismatches in the
        LRP backward pass because activations and layers go out of sync.

        Recursing into every Sequential (at any nesting depth) guarantees the
        layer list matches the actual forward execution order regardless of how
        deeply the user has nested their Sequential containers.

        Only nn.Sequential is unrolled. Custom nn.Module subclasses (e.g. a
        ResBlock) are kept atomic — their internals run via their own forward()
        and cannot be intercepted without hooks.
        """
        layers: List[nn.Module] = []

        def _flatten(module: nn.Module) -> None:
            for child in module.children():
                if isinstance(child, nn.Sequential):
                    _flatten(child)          # recurse into nested Sequentials
                else:
                    layers.append(child)     # primitive layer — keep as-is

        _flatten(model)
        return layers

    @staticmethod
    def _warn_if_skip_connections(model: nn.Module) -> None:
        name = type(model).__name__.lower()
        for keyword in ["resnet", "densenet", "efficientnet", "inception", "regnet"]:
            if keyword in name:
                warnings.warn(
                    f"LRPImageExplainer: '{type(model).__name__}' appears to use "
                    f"skip connections. Layer-by-layer LRP cannot split relevance "
                    f"at residual additions — completeness will be violated and "
                    f"attributions will be inaccurate. "
                    f"Use IGImageExplainer for this architecture.",
                    UserWarning,
                    stacklevel=3,
                )
                return

    def _rules_summary(self) -> List[Dict]:
        if self.model_type == "cnn":
            return [
                {"layer_type": "First Conv2d", "rule": RULE_INFO["zB"]["name"],
                 "why": RULE_INFO["zB"]["why"]},
                {"layer_type": "Other Conv2d", "rule": RULE_INFO["alphabeta"]["name"],
                 "why": RULE_INFO["alphabeta"]["why"]},
                {"layer_type": "Linear",       "rule": RULE_INFO["epsilon"]["name"],
                 "why": RULE_INFO["epsilon"]["why"]},
                {"layer_type": "BN/Pool/Act",  "rule": RULE_INFO["passthrough"]["name"],
                 "why": RULE_INFO["passthrough"]["why"]},
            ]
        return [
            {"layer_type": "Linear",      "rule": RULE_INFO["epsilon"]["name"],
             "why": RULE_INFO["epsilon"]["why"]},
            {"layer_type": "Activations", "rule": RULE_INFO["passthrough"]["name"],
             "why": RULE_INFO["passthrough"]["why"]},
        ]

    # ──────────────────────────────────────────────────────────────────────
    #  Attribution maps  (private)
    # ──────────────────────────────────────────────────────────────────────

    def _magnitude_map(self, attr: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(attr.squeeze(0)), dim=0)

    def _positive_map(self, attr: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.clamp(attr.squeeze(0), min=0), dim=0)

    def _negative_map(self, attr: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.sum(torch.clamp(attr.squeeze(0), max=0), dim=0))

    # ──────────────────────────────────────────────────────────────────────
    #  Visualisation helpers  (private)
    # ──────────────────────────────────────────────────────────────────────

    def _to_uint8(self, attr_map: torch.Tensor, clip_pct: float = 99.0) -> np.ndarray:
        arr   = attr_map.cpu().numpy()
        upper = np.percentile(arr, clip_pct)
        arr   = np.clip(arr, 0, upper)
        arr   = arr - arr.min()
        arr   = arr / (arr.max() + 1e-8)
        return np.uint8(255 * arr)

    def _resize(self, heatmap: np.ndarray, image: np.ndarray) -> np.ndarray:
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        return heatmap

    def _heatmap_overlay(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,
        blend: float,
        colormap: int,
    ) -> np.ndarray:
        heatmap = self._to_uint8(attr_map)
        heatmap = self._resize(heatmap, image)
        colored = cv2.applyColorMap(heatmap, colormap)
        return cv2.addWeighted(image, 1 - blend, colored, blend, 0)

    def _contour_overlay(
        self,
        attr_map: torch.Tensor,
        image: np.ndarray,
        threshold_pct: float = 90.0,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        heatmap = self._to_uint8(attr_map)
        heatmap = self._resize(heatmap, image)
        heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
        threshold = int(np.percentile(heatmap, threshold_pct))
        _, binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            min_area = image.shape[0] * image.shape[1] * 0.005
            contours = [c for c in contours if cv2.contourArea(c) > min_area]
        overlay = image.copy()
        cv2.drawContours(overlay, contours, -1, color, thickness)
        return overlay

    def _write_dashboard(
        self,
        overlays: Dict,
        completeness_err: float,
        target_class: int,
        class_name: str,
        save_path: str,
        dpi: int = 150,
    ) -> None:
        cls_str = f"Class: {class_name}" if class_name else f"Class: {target_class}"
        quality = "[OK]" if completeness_err < 0.05 else "[!!] Check architecture / pixel bounds"

        panels = [
            ("Magnitude  (Overall Importance)",    "overlay_magnitude"),
            ("Positive   (Supports Prediction)",   "overlay_positive"),
            ("Negative   (Suppresses Prediction)", "overlay_negative"),
            ("Contour    (Important Region)",       "overlay_contour"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor="#111122")
        fig.suptitle(
            f"LRP Explanation  [{self.model_type.upper()}]  —  {cls_str}\n"
            f"Completeness error = {completeness_err:.4f}   {quality}",
            color="white", fontsize=12, fontweight="bold", y=0.98,
        )
        for ax, (title, key) in zip(axes.flat, panels):
            ax.imshow(cv2.cvtColor(overlays[key], cv2.COLOR_BGR2RGB))
            ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)