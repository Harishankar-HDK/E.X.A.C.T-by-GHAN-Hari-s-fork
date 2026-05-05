import numpy as np
import torch
import shap


class ShapExplainer_Text:
    """
    SHAP Text Explainer for EXACT Library  (PyTorch-specific)

    Responsibilities:
        - Generate SHAP explanations for ANY PyTorch text classification model
        - Return per-token SHAP attribution values
        - Provide visualization utilities (text_plot, bar_plot, summary_plot)

    Explainer used:
        KernelExplainer — model-agnostic black-box approach.
        Works with ALL PyTorch text model types:
            - Transformers (BERT, DistilBERT, RoBERTa, etc.)  [including HF dict-input models]
            - RNN / LSTM / GRU
            - Embedding + Linear (bag-of-words style)
            - Any custom PyTorch nn.Module that accepts token IDs
            - Multi-input models (attention_mask, token_type_ids, etc.)

    How it works for text:
        1. Input text is tokenized into tokens using the provided tokenizer.
        2. A binary masking matrix is built — each row is a coalition where
           some tokens are "present" (original) and some are "absent" (masked).
        3. Absent tokens are replaced with mask_token_id (default: 0 = padding).
        4. The model is evaluated on all coalitions via _predict_proba.
        5. KernelExplainer fits a weighted linear model with Shapley-correct
           weights to get per-token SHAP values.

    Supported model interfaces:
        1. Simple tensor input:
              model(input_ids)   -> logits

        2. HuggingFace-style dict input:
              model(input_ids=..., attention_mask=...)  -> output with .logits

        3. Fully custom via forward_fn:
              forward_fn(input_ids_tensor) -> logits_tensor
              (user supplies any custom wrapping logic)

    Notes:
        - Model is automatically set to eval() mode.
        - tokenizer accepts ALL formats:
              HuggingFace tokenizer object  (recommended — pass directly)
              callable returning a BatchEncoding / dict
              callable returning a tensor   shape (1, seq_len) or (seq_len,)
              callable returning a list     [101, 2054, ...]
          The explainer extracts input_ids automatically from any of these.
        - For HuggingFace tokenizers, pass the HF tokenizer object directly;
          the class handles input_ids extraction and token decoding automatically.
        - max_seq_len controls truncation for long texts.
        - All tensor/numpy conversions are handled internally.

    Limitations:
        - Tested with: Embedding+Linear, LSTM, BERT-style transformers.
        - For BERT-style models, prefer mask_token_id=103 over 0 to avoid
          all-PAD background issues.
        - For models requiring extra inputs beyond input_ids, use the
          forward_fn parameter.
    """

    # PAD token strings produced by common tokenizer families
    _PAD_TOKEN_VARIANTS = {"<pad>", "[pad]", "<PAD>", "[PAD]", "<|pad|>"}

    def __init__(
        self,
        model,
        tokenizer,
        class_names=None,
        mask_token_id=0,
        max_seq_len=128,
        nsamples=500,
        id2token=None,
        hf_tokenizer=None,
        forward_fn=None,
        pad_token=None,
    ):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained PyTorch text classification model.
            Must accept input of shape (batch_size, seq_len) as LongTensor
            (token IDs) and return logits of shape (batch_size, num_classes),
            OR accept keyword arguments (HuggingFace style) and return an
            object with a .logits attribute.

        tokenizer : callable or HuggingFace tokenizer object
            Converts raw text to token IDs. Accepts ALL of the following:

            1. HuggingFace tokenizer object passed directly:
                  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                  # Just pass it — no wrapping needed.

            2. Callable returning a BatchEncoding or dict:
                  tokenizer = lambda text: hf_tok(text, return_tensors="pt")
                  # BatchEncoding / {"input_ids": tensor([[101, ...]]), ...}

            3. Callable returning a tensor:
                  tokenizer = lambda text: hf_tok(text, return_tensors="pt")["input_ids"]
                  # tensor([[101, 2054, ...]])

            4. Callable returning a plain list:
                  tokenizer = lambda text: [vocab[w] for w in text.split()]
                  # [101, 2054, ...]

            In all cases, input_ids are extracted automatically.

        class_names : list[str], optional
            Names of output classes (used in visualization only).
            If None, auto-labeled as ['Class_0', 'Class_1', ...].

        mask_token_id : int
            Token ID used to replace "absent" tokens in SHAP coalitions.
            - Use 0 for padding-based models.
            - Use your tokenizer's [MASK] token ID for BERT-style models
              (typically 103 for BERT, 50264 for RoBERTa).
            Default: 0.

        max_seq_len : int
            Maximum number of tokens to explain.
            Longer sequences are truncated to this length.
            Lower values = faster computation. Default: 128.

        nsamples : int
            Number of masked samples KernelExplainer evaluates per explanation.
            Higher -> more accurate but slower.
            Recommended: 200-1000. Default: 500.

        id2token : dict[int, str], optional
            Reverse vocabulary mapping: token_id -> token_string.
            Used for token decoding when a full HuggingFace tokenizer is not
            available. Example: {0: "<PAD>", 1: "hello", 2: "world"}.

        hf_tokenizer : HuggingFace tokenizer object, optional
            Explicit HuggingFace tokenizer for attention_mask building and
            token decoding. Only needed if `tokenizer` is a plain lambda
            that does NOT have HuggingFace methods (convert_ids_to_tokens etc.).
            If `tokenizer` is already an HuggingFace tokenizer object,
            you do NOT need to pass this separately.

        forward_fn : callable, optional
            Custom forward function: forward_fn(input_ids_tensor) -> logits_tensor.
            Use this when your model requires special preprocessing, multiple
            inputs beyond attention_mask, or non-standard output formats.
            When provided, this takes precedence over both the simple tensor
            path and the HuggingFace dict-input path.

            Example (model needing token_type_ids):
                def forward_fn(input_ids):
                    token_type_ids = torch.zeros_like(input_ids)
                    out = model(input_ids=input_ids,
                                token_type_ids=token_type_ids)
                    return out.logits

        pad_token : str, optional
            The string representation of the PAD token in your vocabulary.
            Used to filter PAD tokens from SHAP visualizations.
            If not provided, auto-detected from hf_tokenizer.pad_token or
            matched against known PAD variants ("<PAD>", "[PAD]", "<pad>", etc.).
        """
        self.model         = model
        self.tokenizer     = tokenizer
        self.class_names   = class_names
        self.mask_token_id = mask_token_id
        self.max_seq_len   = max_seq_len
        self.nsamples      = nsamples
        self.id2token      = id2token
        self.forward_fn    = forward_fn

        # If tokenizer itself is an HF tokenizer object, use it as hf_tokenizer too
        # unless the user explicitly passed a separate one
        if hf_tokenizer is not None:
            self.hf_tokenizer = hf_tokenizer
        elif hasattr(tokenizer, "convert_ids_to_tokens"):
            # tokenizer is already an HF tokenizer object — use it directly
            self.hf_tokenizer = tokenizer
        else:
            self.hf_tokenizer = None

        # Resolve PAD token string for visualization filtering
        if pad_token is not None:
            self._pad_token_str = pad_token
        elif self.hf_tokenizer is not None and hasattr(self.hf_tokenizer, "pad_token"):
            self._pad_token_str = self.hf_tokenizer.pad_token  # e.g. "[PAD]" or "<pad>"
        else:
            self._pad_token_str = None  # will fall back to _PAD_TOKEN_VARIANTS set

        # Resolve device from model parameters
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        # Always eval mode — critical for correct, deterministic attributions
        self.model.eval()

    # ─────────────────────────────────────────────────────────────────────
    # Internal: tokenizer output normalisation
    # ─────────────────────────────────────────────────────────────────────

    def _extract_token_ids(self, raw):
        """
        Normalise any tokenizer output to a flat Python list of int token IDs.

        Handles every format a user might pass:

            BatchEncoding  {"input_ids": tensor([[101,...]]), ...}  HuggingFace default
            dict           {"input_ids": [[101, ...]], ...}         HuggingFace, no tensors
            tensor         shape (1, seq_len) or (seq_len,)         direct tensor output
            list           [[101, ...]] or [101, ...]               nested or flat list

        BatchEncoding is HuggingFace's return type from tokenizer.__call__().
        It is NOT a plain dict subclass in all versions, so we check for the
        presence of an 'input_ids' attribute rather than isinstance(raw, dict).

        Parameters
        ----------
        raw : BatchEncoding | dict | torch.Tensor | list
            Direct output of self.tokenizer(text).

        Returns
        -------
        list[int]   flat list of integer token IDs, no batch dimension.
        """
        # ── Case 1: BatchEncoding or dict  (HuggingFace __call__ or wrapped lambda) ──
        # hasattr(raw, "input_ids") catches BatchEncoding which may not be a plain dict
        if isinstance(raw, dict) or hasattr(raw, "input_ids"):
            ids = raw["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.squeeze().tolist()          # (1, seq_len) -> [int, ...]
            elif isinstance(ids, (list, tuple)):
                ids = ids[0] if (ids and isinstance(ids[0], (list, tuple))) else ids
            return [int(i) for i in ids]

        # ── Case 2: torch.Tensor returned directly ──
        if isinstance(raw, torch.Tensor):
            return raw.squeeze().cpu().tolist()       # (1, N) or (N,) -> [int, ...]

        # ── Case 3: list or tuple ──
        if isinstance(raw, (list, tuple)):
            # Unwrap one level of batch nesting if present
            if raw and isinstance(raw[0], (list, tuple)):
                raw = raw[0]
            return [int(i) for i in raw]

        raise TypeError(
            f"[EXACT] ShapExplainer_Text: tokenizer returned an unrecognised type "
            f"'{type(raw).__name__}'. Expected BatchEncoding, dict, torch.Tensor, or list.\n"
            f"If your tokenizer is unusual, pass a lambda that returns a plain "
            f"list of integer token IDs:\n"
            f"    tokenizer=lambda text: my_tok(text)['input_ids'][0].tolist()"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: forward pass  (handles all model interface variants)
    # ─────────────────────────────────────────────────────────────────────

    def _forward(self, input_ids_tensor):
        """
        Run the model forward pass for a batch of token ID sequences.

        Handles three interface variants:
            1. Custom forward_fn (user-supplied, highest priority)
            2. HuggingFace dict-input (detected via hf_tokenizer presence)
            3. Simple tensor input (default bare nn.Module)

        Parameters
        ----------
        input_ids_tensor : torch.Tensor
            Shape: (batch_size, seq_len), dtype: torch.long.

        Returns
        -------
        torch.Tensor   shape: (batch_size, num_classes) — raw logits.
        """
        # ── Path 1: User-supplied custom forward function ──
        if self.forward_fn is not None:
            output = self.forward_fn(input_ids_tensor)
            return self._extract_logits(output)

        # ── Path 2: HuggingFace-style dict input ──
        if self.hf_tokenizer is not None:
            # Build attention_mask: 1 where token != pad, 0 where pad
            attention_mask = (
                input_ids_tensor != self.mask_token_id
            ).long().to(self.device)

            output = self.model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask,
            )
            return self._extract_logits(output)

        # ── Path 3: Simple bare tensor input ──
        output = self.model(input_ids_tensor)
        return self._extract_logits(output)

    def _extract_logits(self, output):
        """
        Extract a plain logit tensor from any model output format.

        Handles:
            - Plain torch.Tensor  (most custom PyTorch models)
            - Object with .logits attribute  (HuggingFace ModelOutput)
            - Tuple / list — first element assumed to be logits

        Parameters
        ----------
        output : any   Raw model output.

        Returns
        -------
        torch.Tensor   shape: (batch_size, num_classes)
        """
        if isinstance(output, torch.Tensor):
            return output

        if hasattr(output, "logits"):
            return output.logits

        if isinstance(output, (tuple, list)) and len(output) > 0:
            if isinstance(output[0], torch.Tensor):
                return output[0]

        raise RuntimeError(
            f"Unrecognised model output type: {type(output)}. "
            "Please provide a `forward_fn` that returns a plain logits tensor."
        )

    # ─────────────────────────────────────────────────────────────────────
    # Internal: predict proba  (text-specific, keeps token IDs as long)
    # ─────────────────────────────────────────────────────────────────────

    def _predict_proba(self, input_tensor):
        """
        Run a forward pass on the model and return softmax probabilities.

        Kept self-contained inside the text explainer because text models
        receive LongTensor (token IDs), not FloatTensor like tabular or
        image models.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Shape: (batch_size, seq_len), dtype: torch.long.

        Returns
        -------
        np.ndarray   shape: (batch_size, num_classes) — softmax probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            logits = self._forward(input_tensor)

            if logits.ndim == 1:
                logits = logits.unsqueeze(0)

            if logits.ndim == 2 and logits.shape[-1] == 1:
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    # ─────────────────────────────────────────────────────────────────────
    # Internal: masked prediction function  (used by KernelExplainer)
    # ─────────────────────────────────────────────────────────────────────

    def _make_predict_fn(self, token_ids):
        """
        Build the masked prediction function for a specific input.

        mask = 1 -> token PRESENT  (use original token_id)
        mask = 0 -> token ABSENT   (replace with mask_token_id)

        Parameters
        ----------
        token_ids : np.ndarray   shape: (seq_len,)

        Returns
        -------
        predict_fn : callable
            (n_coalitions, seq_len) -> (n_coalitions, num_classes)
        """
        def predict_fn(mask_matrix):
            n_coalitions = mask_matrix.shape[0]
            seq_len      = len(token_ids)

            masked_inputs = np.full(
                (n_coalitions, seq_len),
                fill_value=self.mask_token_id,
                dtype=np.int64,
            )
            for i in range(n_coalitions):
                present = mask_matrix[i].astype(bool)
                masked_inputs[i, present] = token_ids[present]

            input_tensor = torch.tensor(
                masked_inputs, dtype=torch.long
            ).to(self.device)

            return self._predict_proba(input_tensor)

        return predict_fn

    # ─────────────────────────────────────────────────────────────────────
    # Core explanation logic
    # ─────────────────────────────────────────────────────────────────────

    def explain(self, text):
        """
        Explain a single text input — generate SHAP values per token.

        Parameters
        ----------
        text : str   Raw input text string to explain.

        Returns
        -------
        explanation : dict with keys:
            'shap_values'    : np.ndarray or list[np.ndarray]
                               For multi-class: list of (1, seq_len), one per class.
                               For binary (1 output neuron): (1, seq_len).
            'expected_value' : float or list[float]
            'tokens'         : list[str]
            'token_ids'      : np.ndarray  shape: (seq_len,)
            'text'           : str
        """
        # ── Step 1: Tokenize ──
        # Accepts any return type: BatchEncoding, dict, tensor, list
        raw       = self.tokenizer(text)
        ids_list  = self._extract_token_ids(raw)           # always flat list[int]
        token_ids = np.array(ids_list, dtype=np.int64)[:self.max_seq_len]
        seq_len   = len(token_ids)

        if seq_len == 0:
            raise ValueError(
                "Tokenizer returned an empty token list for the input text. "
                "Check that your tokenizer handles this input correctly."
            )

        # ── Step 2: Decode tokens to strings for visualization ──
        tokens = self._decode_tokens(token_ids)

        # ── Step 3: Background — all tokens absent (pure masked baseline) ──
        background = np.zeros((1, seq_len), dtype=np.float64)

        # ── Step 4: Build masked prediction function ──
        predict_fn = self._make_predict_fn(token_ids)

        # ── Step 5: KernelExplainer ──
        kernel_explainer = shap.KernelExplainer(predict_fn, background)
        shap_values      = kernel_explainer.shap_values(
            np.ones((1, seq_len), dtype=np.float64),
            nsamples=self.nsamples,
        )

        return {
            "shap_values"    : shap_values,
            "expected_value" : kernel_explainer.expected_value,
            "tokens"         : tokens,
            "token_ids"      : token_ids,
            "text"           : text,
        }

    # ─────────────────────────────────────────────────────────────────────
    # Internal: token decoding
    # ─────────────────────────────────────────────────────────────────────

    def _decode_tokens(self, token_ids):
        """
        Convert integer token IDs to human-readable token strings.

        Priority:
            1. id2token dict
            2. hf_tokenizer.convert_ids_to_tokens()
            3. tokenizer.convert_ids_to_tokens()
            4. tokenizer.decode() per token
            5. Fallback: str(id)
        """
        ids = token_ids.tolist()

        if self.id2token is not None:
            return [self.id2token.get(int(tid), str(tid)) for tid in ids]

        if self.hf_tokenizer is not None and hasattr(
            self.hf_tokenizer, "convert_ids_to_tokens"
        ):
            return self.hf_tokenizer.convert_ids_to_tokens(ids)

        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            return self.tokenizer.convert_ids_to_tokens(ids)

        if hasattr(self.tokenizer, "decode"):
            return [self.tokenizer.decode([tid]) for tid in ids]

        return [str(tid) for tid in ids]

    # ─────────────────────────────────────────────────────────────────────
    # Internal: PAD token detection
    # ─────────────────────────────────────────────────────────────────────

    def _is_pad_token(self, token_str):
        if self._pad_token_str is not None:
            return token_str == self._pad_token_str
        return token_str in self._PAD_TOKEN_VARIANTS

    # ─────────────────────────────────────────────────────────────────────
    # Internal: SHAP value extraction
    # ─────────────────────────────────────────────────────────────────────

    def _extract_values(self, shap_values, class_index):
        """
        Normalise shap_values to a flat 1D numpy array for a given class.

        KernelExplainer returns:
            - list of arrays (one per class): multi-class
            - 3D array (1, seq_len, n_classes): some SHAP versions
            - 2D array (1, seq_len): binary / single-output
        """
        if isinstance(shap_values, list):
            arr = np.array(shap_values[class_index])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            arr = shap_values[0, :, class_index]
        else:
            arr = np.array(shap_values)

        return arr.flatten()

    # ─────────────────────────────────────────────────────────────────────
    # Raw explanation data
    # ─────────────────────────────────────────────────────────────────────

    def get_explanation_data(self, explanation, class_index=0, num_tokens=10):
        """
        Return sorted (token, shap_value) list with PAD tokens filtered out.

        Parameters
        ----------
        explanation : dict   output of explain()
        class_index : int    which class. Default: 0.
        num_tokens  : int    how many top tokens. Default: 10.

        Returns
        -------
        list of (str, float)   sorted by |shap_value| descending.
        """
        values = self._extract_values(explanation["shap_values"], class_index)
        tokens = explanation["tokens"]

        paired  = [
            (tok, float(val))
            for tok, val in zip(tokens, values.tolist())
            if not self._is_pad_token(tok)
        ]
        sorted_ = sorted(paired, key=lambda x: abs(x[1]), reverse=True)

        return sorted_[:num_tokens]

    # ─────────────────────────────────────────────────────────────────────
    # Console visualization
    # ─────────────────────────────────────────────────────────────────────

    def visualize(self, explanation, class_index=0, num_tokens=10):
        """
        Print SHAP token attributions in readable console format.

        Parameters
        ----------
        explanation : dict   output of explain()
        class_index : int    which class. Default: 0.
        num_tokens  : int    how many top tokens. Default: 10.

        Returns
        -------
        list of (str, float)
        """
        token_scores = self.get_explanation_data(
            explanation,
            class_index=class_index,
            num_tokens=num_tokens,
        )

        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"Class_{class_index}"
        )

        print(f"\nSHAP Text Explanation  [Class: {class_label}]")
        print("-" * 42)
        for token, score in token_scores:
            sign = "+" if score >= 0 else "-"
            bar  = "|" * min(int(abs(score) * 80), 20)
            print(f"  {token:<28s}  {sign}  {abs(score):.6f}  {bar}")
        print("-" * 42)

        return token_scores

    # ─────────────────────────────────────────────────────────────────────
    # Console text plot
    # ─────────────────────────────────────────────────────────────────────

    def text_plot(self, explanation, class_index=0):
        """
        Print an inline ASCII text plot showing per-token SHAP attributions.

        Parameters
        ----------
        explanation : dict   output of explain()
        class_index : int    which class. Default: 0.
        """
        values = self._extract_values(explanation["shap_values"], class_index)
        tokens = explanation["tokens"]

        class_label = (
            self.class_names[class_index]
            if self.class_names is not None
            else f"Class_{class_index}"
        )

        pairs = [
            (tok, float(val))
            for tok, val in zip(tokens, values.tolist())
            if not self._is_pad_token(tok)
        ]

        if not pairs:
            print("No non-PAD tokens to display.")
            return

        max_val = max(abs(v) for _, v in pairs) or 1.0

        print(f"\nSHAP Text Plot  [Class: {class_label}]")
        print("─" * 60)

        token_line = ""
        score_line = ""

        for token, val in pairs:
            bar_len = max(int((abs(val) / max_val) * 4), 0)
            if val > 0.001:
                marker = "+" * bar_len
            elif val < -0.001:
                marker = "-" * bar_len
            else:
                marker = ""
            cell   = f"[{marker}{token}{marker}]"
            score  = f" {val:+.3f} "
            width  = max(len(cell), len(score)) + 1
            token_line += cell.center(width)
            score_line += score.center(width)

        print(token_line)
        print(score_line)
        print("─" * 60)
        print("[+++ word +++] = pushes toward this class")
        print("[--- word ---] = pushes away from this class")

    # ─────────────────────────────────────────────────────────────────────
    # SHAP summary plot
    # ─────────────────────────────────────────────────────────────────────

    def summary_plot(self, explanation, class_index=0):
        """
        SHAP summary plot — shows SHAP value distribution per token.
        PAD tokens filtered before plotting.

        Parameters
        ----------
        explanation : dict   output of explain()
        class_index : int    which class. Default: 0.
        """
        values = self._extract_values(explanation["shap_values"], class_index)
        tokens = explanation["tokens"]

        filtered = [
            (tok, val)
            for tok, val in zip(tokens, values.tolist())
            if not self._is_pad_token(tok)
        ]
        if not filtered:
            print("No non-PAD tokens to display.")
            return

        filtered_tokens, filtered_vals = zip(*filtered)
        values_2d = np.array(filtered_vals).reshape(1, -1)

        shap.summary_plot(values_2d, feature_names=list(filtered_tokens))

    # ─────────────────────────────────────────────────────────────────────
    # SHAP bar plot
    # ─────────────────────────────────────────────────────────────────────

    def bar_plot(self, explanation, class_index=0):
        """
        SHAP bar plot — shows mean absolute SHAP value per token.
        PAD tokens filtered before plotting.

        Parameters
        ----------
        explanation : dict   output of explain()
        class_index : int    which class. Default: 0.
        """
        values = self._extract_values(explanation["shap_values"], class_index)
        tokens = explanation["tokens"]

        filtered = [
            (tok, val)
            for tok, val in zip(tokens, values.tolist())
            if not self._is_pad_token(tok)
        ]
        if not filtered:
            print("No non-PAD tokens to display.")
            return

        filtered_tokens, filtered_vals = zip(*filtered)
        values_2d = np.array(filtered_vals).reshape(1, -1)

        shap.summary_plot(values_2d, feature_names=list(filtered_tokens), plot_type="bar")