class ActivationCache:
    """Context manager that hooks into transformer layers and caches hidden states."""

    def __init__(self, model, layers: list, full_sequence: bool = False):
        self.model = model
        self.layers = layers
        self.full_sequence = full_sequence
        self._cache: dict = {}
        self._handles: list = []

    def __enter__(self):
        for layer_idx in self.layers:
            # Capture layer_idx by default arg to avoid closure over loop variable
            def make_hook(idx):
                def hook_fn(module, input, output):
                    # output is a tuple; first element is the hidden states tensor.
                    # Shape is (batch, seq_len, hidden_dim) when batch > 1 or in
                    # some transformers versions, but Gemma-2 with batch=1 may
                    # return (seq_len, hidden_dim) — handle both gracefully.
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    if self.full_sequence:
                        if hidden.dim() == 3:
                            self._cache[idx] = hidden[0].detach().cpu().float().numpy()
                        elif hidden.dim() == 2:
                            self._cache[idx] = hidden.detach().cpu().float().numpy()
                        else:
                            self._cache[idx] = hidden.detach().cpu().float().numpy()
                    elif hidden.dim() == 3:
                        # (batch, seq_len, hidden_dim) — standard case
                        self._cache[idx] = hidden[0, -1, :].detach().cpu().float().numpy()
                    elif hidden.dim() == 2:
                        # (seq_len, hidden_dim) — batch dim already squeezed
                        self._cache[idx] = hidden[-1, :].detach().cpu().float().numpy()
                    else:
                        # Scalar or 1-D edge case — store as-is
                        self._cache[idx] = hidden.detach().cpu().float().numpy()
                return hook_fn

            handle = self.model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False  # do not suppress exceptions

    def get_activations(self, layer_idx):
        return self._cache.get(layer_idx, None)

    def clear(self):
        self._cache = {}
