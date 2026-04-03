"""
crise.py — Contrastive RISE (CRISE-ID) for 1:N face identification.

Extends rise.py by replacing the cosine similarity weight with a softmax
probability over all gallery embeddings:

    sims = cosine_sim(masked_embedding, G)   # shape (n_gallery,)
    w    = softmax(sims / TAU)[true_id_idx]

This weight is high only when the masked image confidently identifies as
the true identity *relative to all gallery alternatives*, reflecting the
actual competitive decision structure of 1:N identification.

Evaluation uses identification margin (same as RISE baseline), so the
weighting and evaluation quantities are different — this is a genuine
faithfulness test, not a self-consistency check.

Usage:
    from crise import run_crise, TAU

    out = run_crise(
        true_id="John_Doe",
        probe_path="data/lfw-deepfunneled/John_Doe/John_Doe_0002.jpg",
        G=G,
        id_to_index=id_to_index,
        rec=rec,
        app=app,
        run_seed=1230000,
        out_dir="results/crise_maps",
    )
"""

import numpy as np
from rise import run_rise

# ---------------------------------------------------------------------------
# Temperature — top-level constant for easy ablation
# ---------------------------------------------------------------------------

TAU = 0.1   # ablation values: [0.05, 0.1, 0.2, 0.5]


# ---------------------------------------------------------------------------
# Softmax weight function
# ---------------------------------------------------------------------------

def softmax_weight(
    embedding: np.ndarray,
    G: np.ndarray,
    true_id_idx: int,
    tau: float = TAU,
) -> float:
    """
    Softmax probability of true identity over the full gallery.

    Parameters
    ----------
    embedding   : (512,) float32 L2-normalized embedding of the masked chip
    G           : (n_gallery, 512) float32 gallery embeddings
    true_id_idx : index of the true identity in G
    tau         : softmax temperature (lower = sharper)

    Returns
    -------
    float in (0, 1): confidence assigned to true identity under this mask
    """
    sims = G @ embedding                        # (n_gallery,) cosine sims
    logits = sims / tau
    logits -= logits.max()                      # numerical stability
    exp_l = np.exp(logits)
    return float(exp_l[true_id_idx] / exp_l.sum())


# ---------------------------------------------------------------------------
# run_crise: thin wrapper over run_rise
# ---------------------------------------------------------------------------

def run_crise(
    true_id: str,
    probe_path: str,
    G: np.ndarray,
    id_to_index: dict,
    rec,
    app,
    run_seed: int,
    out_dir: str = "results/crise_maps",
    tau: float = TAU,
    N: int = 1000,
    s: int = 8,
    p1: float = 0.5,
    batch_save: int = 50,
) -> dict:
    """
    Run CRISE for a single probe image.

    Parameters mirror run_rise; see rise.py for full documentation.
    tau controls softmax sharpness (default TAU = 0.1).

    Saliency maps are cached under out_dir with prefix crise_tau{tau}.
    Returns the same dict as run_rise.
    """
    weight_fn = lambda e, G_, idx: softmax_weight(e, G_, idx, tau=tau)

    tau_str = f"{tau:.3g}".replace(".", "p")   # e.g. 0.1 -> "0p1"
    prefix = f"crise_tau{tau_str}"

    return run_rise(
        true_id=true_id,
        probe_path=probe_path,
        G=G,
        id_to_index=id_to_index,
        rec=rec,
        app=app,
        weight_fn=weight_fn,
        run_seed=run_seed,
        out_dir=out_dir,
        N=N,
        s=s,
        p1=p1,
        batch_save=batch_save,
        run_name_prefix=prefix,
    )
