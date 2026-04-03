"""
rise.py — RISE saliency module for CRISE-ID.

Core entry point: run_rise()

weight_fn signature:
    (embedding: np.ndarray, G: np.ndarray, true_id_idx: int) -> float

    embedding   : (512,) float32 L2-normalized embedding of the masked chip
    G           : (n_gallery, 512) float32 gallery embeddings
    true_id_idx : index of the true identity in G

Baseline weight function (cosine similarity to true identity):
    weight_fn = lambda e, G, idx: float(np.dot(e, G[idx]))

CRISE weight function (softmax probability — defined in crise.py):
    weight_fn = lambda e, G, idx: softmax_weight(e, G, idx, tau=TAU)

Model objects (app, rec) are passed in by the caller — not instantiated here.
"""

import os
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_mask_rise(H: int, W: int, s: int = 8, p1: float = 0.5,
                       rng: np.random.Generator = None) -> np.ndarray:
    """Return a (H, W) float32 upsampled random binary mask."""
    if rng is None:
        rng = np.random.default_rng()

    grid = (rng.random((s, s)) < p1).astype(np.float32)

    cell_h = int(np.ceil(H / s))
    cell_w = int(np.ceil(W / s))
    up_h = H + cell_h
    up_w = W + cell_w

    mask_up = cv2.resize(grid, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
    dy = int(rng.integers(0, cell_h))
    dx = int(rng.integers(0, cell_w))

    m = mask_up[dy:dy + H, dx:dx + W]
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def apply_mask_black(img_bgr: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Zero out pixels where mask is 0. Returns uint8 BGR."""
    return (img_bgr.astype(np.float32) * m[..., None]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Face alignment and embedding
# ---------------------------------------------------------------------------

# Standard ArcFace 5-point landmark destinations (112×112)
_DST_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


def build_aligned_chip_112(img_bgr: np.ndarray, app) -> np.ndarray:
    """
    Detect the largest face in img_bgr and affine-warp to a 112×112 aligned chip.
    Raises ValueError if no face or no landmarks are found.
    """
    faces = app.get(img_bgr)
    if len(faces) == 0:
        raise ValueError("No face detected")

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    if not hasattr(face, "kps") or face.kps is None:
        raise ValueError("No 5-point landmarks found")

    M, _ = cv2.estimateAffinePartial2D(
        face.kps.astype(np.float32), _DST_LANDMARKS, method=cv2.LMEDS
    )
    if M is None:
        raise ValueError("Failed to estimate alignment transform")

    return cv2.warpAffine(img_bgr, M, (112, 112),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def get_embedding_from_chip(chip_bgr: np.ndarray, rec) -> np.ndarray:
    """
    Run ArcFace recognition on a (112,112,3) BGR aligned chip.
    Returns a (512,) float32 L2-normalized embedding.
    """
    assert chip_bgr.shape == (112, 112, 3), \
        f"Expected (112,112,3), got {chip_bgr.shape}"
    feat = np.asarray(rec.get_feat(chip_bgr)).reshape(-1).astype(np.float32)
    return feat / (float(np.linalg.norm(feat)) + 1e-12)


# ---------------------------------------------------------------------------
# Core RISE loop
# ---------------------------------------------------------------------------

def run_rise(
    true_id: str,
    probe_path: str,
    G: np.ndarray,
    id_to_index: dict,
    rec,
    app,
    weight_fn,
    run_seed: int,
    out_dir: str,
    N: int = 1000,
    s: int = 8,
    p1: float = 0.5,
    batch_save: int = 50,
    run_name_prefix: str = "rise_baseline",
) -> dict:
    """
    Run RISE (or any variant) for a single probe image.

    Parameters
    ----------
    true_id     : identity string (must be a key in id_to_index)
    probe_path  : path to the raw probe image
    G           : (n_gallery, 512) float32 gallery embeddings
    id_to_index : dict mapping identity string -> row index in G
    rec         : InsightFace ArcFaceONNX recognition model
    app         : InsightFace FaceAnalysis app (used for chip alignment)
    weight_fn   : callable(embedding, G, true_id_idx) -> float
                  Baseline: lambda e, G, idx: float(np.dot(e, G[idx]))
                  CRISE:    defined in crise.py
    run_seed    : integer seed for the mask RNG
    out_dir     : directory for cached .npy and state files
    N           : number of masks
    s           : RISE grid size
    p1          : mask cell activation probability
    batch_save  : save accumulator every this many masks

    Returns
    -------
    dict with keys: true_id, probe_path, run_seed, failures, w_clean,
                    w_black, saliency_path, chip_bgr, sal_norm
    """
    img = cv2.imread(probe_path)
    if img is None:
        raise ValueError(f"Could not read {probe_path}")

    chip_bgr = build_aligned_chip_112(img, app)
    H, W = chip_bgr.shape[:2]
    true_id_idx = id_to_index[true_id]

    rng = np.random.default_rng(run_seed)

    safe_id = true_id.replace("/", "_")
    safe_probe = os.path.splitext(os.path.basename(probe_path))[0]
    run_name = f"{run_name_prefix}_{safe_id}_{safe_probe}_N{N}_s{s}_p{p1}_seed{run_seed}"

    state_path = os.path.join(out_dir, f"{run_name}_state.json")
    accum_path = os.path.join(out_dir, f"{run_name}_sal_accum.npy")
    final_path = os.path.join(out_dir, f"{run_name}_saliency_norm.npy")

    # --- fast path: already complete ---
    if os.path.exists(final_path):
        print(f"[skip]   {true_id} | {os.path.basename(probe_path)} already finished")
        sal_norm = np.load(final_path).astype(np.float32)
        e_clean = get_embedding_from_chip(chip_bgr, rec)
        w_clean = float(np.dot(e_clean, G[true_id_idx]))
        black_chip = np.zeros_like(chip_bgr, dtype=np.uint8)
        e_black = get_embedding_from_chip(black_chip, rec)
        w_black = float(np.dot(e_black, G[true_id_idx]))
        return dict(true_id=true_id, probe_path=probe_path, run_seed=run_seed,
                    failures=0, w_clean=w_clean, w_black=w_black,
                    saliency_path=final_path, chip_bgr=chip_bgr, sal_norm=sal_norm)

    # --- resume or start ---
    start_i = 0
    failures = 0
    sal_accum = np.zeros((H, W), dtype=np.float64)

    if os.path.exists(state_path) and os.path.exists(accum_path):
        with open(state_path) as f:
            st = json.load(f)
        start_i = int(st["i"])
        failures = int(st["failures"])
        sal_accum = np.load(accum_path).astype(np.float64)
        print(f"[resume] {true_id} | {os.path.basename(probe_path)} at i={start_i}, fail={failures}")
    else:
        print(f"[start]  {true_id} | {os.path.basename(probe_path)}")

    t0 = time.time()
    for i in range(start_i, N):
        m = generate_mask_rise(H, W, s=s, p1=p1, rng=rng)
        masked_chip = apply_mask_black(chip_bgr, m)

        try:
            e = get_embedding_from_chip(masked_chip, rec)
            w = weight_fn(e, G, true_id_idx)
            sal_accum += w * m
        except Exception:
            failures += 1

        if (i + 1) % batch_save == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            done = i + 1
            rate = (done - start_i) / elapsed if elapsed else 0.0
            eta = (N - done) / rate if rate else float("inf")
            np.save(accum_path, sal_accum.astype(np.float32))
            with open(state_path, "w") as f:
                json.dump({"i": done, "failures": failures}, f, indent=2)
            print(f"  [{done}/{N}] {rate:.2f} masks/s | ETA {eta / 60:.2f} min | fail {failures}")

    # --- normalize ---
    effective_N = N - failures
    sal = (sal_accum / (max(1, effective_N) * p1)).astype(np.float32)
    sal_norm = sal - float(sal.min())
    sal_norm = sal_norm / (float(sal_norm.max()) + 1e-8)
    np.save(final_path, sal_norm.astype(np.float32))

    # --- sanity scores ---
    e_clean = get_embedding_from_chip(chip_bgr, rec)
    w_clean = float(np.dot(e_clean, G[true_id_idx]))
    black_chip = np.zeros_like(chip_bgr, dtype=np.uint8)
    e_black = get_embedding_from_chip(black_chip, rec)
    w_black = float(np.dot(e_black, G[true_id_idx]))

    return dict(true_id=true_id, probe_path=probe_path, run_seed=run_seed,
                failures=failures, w_clean=w_clean, w_black=w_black,
                saliency_path=final_path, chip_bgr=chip_bgr, sal_norm=sal_norm)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_overlay_images(
    true_id: str,
    probe_path: str,
    sal_path: str,
    w_clean: float,
    w_black: float,
    fig_dir: str,
    N: int,
    s: int,
    p1: float,
    app,
) -> tuple[str, str, str]:
    """
    Save three PNGs for a probe: aligned chip, saliency heatmap, overlay.

    Returns (chip_out, heat_out, overlay_out) paths.
    """
    img = cv2.imread(probe_path)
    if img is None:
        raise ValueError(f"Could not read {probe_path}")

    chip_bgr = build_aligned_chip_112(img, app)
    chip_rgb = cv2.cvtColor(chip_bgr, cv2.COLOR_BGR2RGB)
    sal = np.load(sal_path).astype(np.float32)

    os.makedirs(fig_dir, exist_ok=True)
    safe_id = true_id.replace("/", "_")
    probe_file = os.path.splitext(os.path.basename(probe_path))[0]
    base = f"{safe_id}_{probe_file}_N{N}_s{s}_p{p1}"

    chip_out = os.path.join(fig_dir, f"{base}_chip.png")
    cv2.imwrite(chip_out, cv2.cvtColor(chip_rgb, cv2.COLOR_RGB2BGR))

    heat_out = os.path.join(fig_dir, f"{base}_saliency.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(sal, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(heat_out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

    overlay_out = os.path.join(fig_dir, f"{base}_overlay.png")
    plt.figure(figsize=(4, 4))
    plt.imshow(chip_rgb)
    plt.imshow(sal, alpha=0.50, vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"{true_id}\nw_clean={w_clean:.3f} w_black={w_black:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(overlay_out, dpi=200, bbox_inches="tight")
    plt.close()

    return chip_out, heat_out, overlay_out


# ---------------------------------------------------------------------------
# Baseline weight function (import or use directly)
# ---------------------------------------------------------------------------

def cosine_weight(embedding: np.ndarray, G: np.ndarray, true_id_idx: int) -> float:
    """Vanilla RISE weight: cosine similarity to true identity gallery embedding."""
    return float(np.dot(embedding, G[true_id_idx]))
