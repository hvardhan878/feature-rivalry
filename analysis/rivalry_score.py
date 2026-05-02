import numpy as np


def compute_rivalry_score(correlation_upper_triangle: np.ndarray) -> float:
    """
    Summarize the rivalry in a correlation distribution.

    Uses the 5th percentile of pairwise correlations:
    more negative = stronger rivalry (more anti-correlated feature pairs).
    """
    return float(np.percentile(correlation_upper_triangle, 5))


def find_top_rival_pairs(
    corr_matrix: np.ndarray,
    active_indices: np.ndarray,
    top_k: int,
) -> list:
    """
    Return the top_k most anti-correlated feature pairs from corr_matrix.

    Args:
        corr_matrix: (n_active, n_active) symmetric correlation matrix
        active_indices: global SAE feature indices corresponding to corr_matrix rows/cols
        top_k: number of pairs to return

    Returns:
        List of dicts with keys feature_i, feature_j, correlation — sorted ascending
        (most negative first).
    """
    n = corr_matrix.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    corr_values = corr_matrix[rows, cols]

    sorted_idx = np.argsort(corr_values)  # ascending: most negative first
    top_idx = sorted_idx[:top_k]

    pairs = []
    for idx in top_idx:
        pairs.append(
            {
                "feature_i": int(active_indices[rows[idx]]),
                "feature_j": int(active_indices[cols[idx]]),
                "correlation": float(corr_values[idx]),
            }
        )
    return pairs
