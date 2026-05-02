import torch
import numpy as np

# Module-level SAE cache — avoids reloading the same SAE multiple times
_sae_cache: dict = {}


def load_sae(layer_idx: int, release: str, sae_id: str):
    """Load a Gemma Scope SAE for the given layer, with in-process caching.

    Args:
        layer_idx: layer number (used as cache key)
        release:   SAELens release name, e.g. "gemma-scope-2b-pt-res"
        sae_id:    full SAE ID, e.g. "layer_10/width_16k/average_l0_70"
    """
    if layer_idx in _sae_cache:
        return _sae_cache[layer_idx]

    from sae_lens import SAE

    print(f"Loading SAE: release={release}, sae_id={sae_id}")
    sae, cfg_dict, _ = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
    )

    # SAELens loads in bfloat16 by default; cast to float32 for MPS compatibility
    sae = sae.to(torch.float32)
    # Keep SAE on CPU — hidden states will be moved to CPU before encoding
    sae = sae.cpu()

    _sae_cache[layer_idx] = sae
    return sae


def get_feature_activations(sae, hidden_state_numpy: np.ndarray) -> np.ndarray:
    """
    Encode a single hidden state through the SAE encoder.

    Args:
        sae: loaded SAE (CPU, float32)
        hidden_state_numpy: numpy array of shape (hidden_dim,), float32

    Returns:
        numpy array of shape (n_features,) with ReLU-activated feature values
    """
    tensor = torch.from_numpy(hidden_state_numpy).float()  # CPU float32

    with torch.no_grad():
        # Linear encoder pass: W_enc shape (hidden_dim, n_features), b_enc shape (n_features,)
        pre_act = tensor @ sae.W_enc + sae.b_enc
        feature_acts = torch.nn.functional.relu(pre_act)

    return feature_acts.numpy()  # shape (n_features,)
