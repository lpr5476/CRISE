"""
synth_gen.py — Shared utilities for synthetic probe generation.

Provides:
  face_swap_affine()        : affine warp + Poisson seamless clone
  get_embedding_from_image(): detect → align → embed (None on failure)
  select_target_identities(): reproducible 50-identity selection
  select_sources_for_target(): reproducible source-probe selection per target

All random operations use deterministic seeds; sorted lists are used before
any sampling to ensure reproducibility across Python sessions.
"""

import os
import random
import numpy as np
import cv2

from rise import _DST_LANDMARKS, build_aligned_chip_112, get_embedding_from_chip


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYNTH_SEED      = 42          # independent of RISE MASTER_SEED=123
N_TARGET_IDS    = 50
N_PROBES_PER_ID = 3

OUT_BASE = os.path.join("data", "synthetic_probes")

# Default path for inswapper_128.onnx (InsightFace model zoo)
INSWAPPER_DEFAULT_PATH = os.path.expanduser(
    os.path.join("~", ".insightface", "models", "inswapper_128.onnx")
)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def get_embedding_from_image(img_path: str, app, rec) -> np.ndarray | None:
    """
    Read image → detect face → align 112×112 → embed.
    Returns (512,) float32 L2-normalised embedding, or None if face detection fails.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    try:
        chip = build_aligned_chip_112(img, app)
        return get_embedding_from_chip(chip, rec)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core swap: affine warp + Poisson seamless clone
# ---------------------------------------------------------------------------

def face_swap_affine(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    app,
) -> np.ndarray | None:
    """
    Swap the face from *source_bgr* onto the background of *target_bgr*.

    Pipeline:
      1. Detect largest face in both images.
      2. Estimate partial affine transform (source kps → target kps).
      3. Warp the full source image into target's coordinate frame.
      4. Build an elliptical mask around the target face bounding box.
      5. Poisson seamless clone warped source onto target;
         fall back to alpha blend if cv2.seamlessClone raises an error.

    Returns a BGR image the same size as *target_bgr*, or None on any failure.
    """
    src_faces = app.get(source_bgr)
    tgt_faces = app.get(target_bgr)
    if not src_faces or not tgt_faces:
        return None

    src_face = max(src_faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    tgt_face = max(tgt_faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    if src_face.kps is None or tgt_face.kps is None:
        return None

    H, W = target_bgr.shape[:2]

    # Affine: map source landmarks → target landmarks
    M, _ = cv2.estimateAffinePartial2D(
        src_face.kps.astype(np.float32),
        tgt_face.kps.astype(np.float32),
        method=cv2.LMEDS,
    )
    if M is None:
        return None

    warped = cv2.warpAffine(source_bgr, M, (W, H))

    # Elliptical mask around target face bbox
    x1, y1, x2, y2 = tgt_face.bbox.astype(int)
    x1, x2 = int(np.clip(x1, 0, W - 1)), int(np.clip(x2, 0, W - 1))
    y1, y2 = int(np.clip(y1, 0, H - 1)), int(np.clip(y2, 0, H - 1))
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax = max((x2 - x1) // 2, 10)
    ay = max((y2 - y1) // 2, 10)

    binary_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.ellipse(binary_mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

    # Poisson clone; fall back to alpha blend on cv2.error
    try:
        result = cv2.seamlessClone(
            warped, target_bgr, binary_mask, (cx, cy), cv2.NORMAL_CLONE
        )
    except cv2.error:
        m = binary_mask.astype(np.float32)[:, :, None] / 255.0
        result = np.clip(
            m * warped.astype(np.float32) + (1.0 - m) * target_bgr.astype(np.float32),
            0, 255,
        ).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Neural swap: InsightFace inswapper_128
# ---------------------------------------------------------------------------

def load_inswapper(model_path: str = None):
    """
    Load InsightFace inswapper_128.onnx.

    model_path: explicit path to inswapper_128.onnx.
                Defaults to ~/.insightface/models/inswapper_128.onnx.

    Raises FileNotFoundError with download instructions if the file is missing.
    """
    import insightface

    path = model_path or INSWAPPER_DEFAULT_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"inswapper_128.onnx not found at: {path}\n\n"
            "Download it manually:\n"
            "  https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx\n"
            f"Then place it at: {path}\n"
            "(Create the directory if it does not exist.)"
        )
    return insightface.model_zoo.get_model(path, download=False, download_zip=False)


def face_swap_neural(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    app,
    swapper,
) -> np.ndarray | None:
    """
    Neural face swap using InsightFace inswapper_128.

    Transfers the *source* person's facial identity onto the *target* image's
    face region.  Uses the neural model rather than geometric warp, producing
    qualitatively different (and typically more realistic) artifacts vs
    face_swap_affine().

    Returns a BGR image the same size as *target_bgr*, or None on any failure.
    """
    src_faces = app.get(source_bgr)
    tgt_faces = app.get(target_bgr)
    if not src_faces or not tgt_faces:
        return None

    src_face = max(src_faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    tgt_face = max(tgt_faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    result = target_bgr.copy()
    try:
        result = swapper.get(result, tgt_face, src_face, paste_back=True)
    except Exception:
        return None

    return result


# ---------------------------------------------------------------------------
# Face morphing: affine-align then pixel blend (Method 3)
# ---------------------------------------------------------------------------

MORPH_ALPHAS = [0.3, 0.5, 0.7]   # alpha = weight on target; primary = 0.5


def face_morph_blend(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    app,
    alpha: float = 0.5,
) -> np.ndarray | None:
    """
    Morphing attack: affine-align source to target, then pixel-level blend.

        morph = cv2.addWeighted(target, alpha, aligned_source, 1-alpha, 0)

    Alignment is performed first so faces are in the same spatial position
    before blending — avoids the ghosting that raw addWeighted produces on
    unaligned images.

    alpha : weight on *target_bgr* (1-alpha on source).
            alpha=0.5 → equal blend.  alpha=0.7 → 70% target-like.

    Returns a BGR image the same size as *target_bgr*, or None on failure.
    """
    src_faces = app.get(source_bgr)
    tgt_faces = app.get(target_bgr)
    if not src_faces or not tgt_faces:
        return None

    src_face = max(src_faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    tgt_face = max(tgt_faces,
                   key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    if src_face.kps is None or tgt_face.kps is None:
        return None

    H, W = target_bgr.shape[:2]

    # Align source into target's coordinate frame
    M, _ = cv2.estimateAffinePartial2D(
        src_face.kps.astype(np.float32),
        tgt_face.kps.astype(np.float32),
        method=cv2.LMEDS,
    )
    if M is None:
        return None

    aligned_source = cv2.warpAffine(source_bgr, M, (W, H))

    return cv2.addWeighted(target_bgr, alpha, aligned_source, 1.0 - alpha, 0)


# ---------------------------------------------------------------------------
# Identity selection helpers
# ---------------------------------------------------------------------------

def select_target_identities(
    completed_ids: list[str],
    n: int = N_TARGET_IDS,
    seed: int = SYNTH_SEED,
) -> list[str]:
    """
    Deterministically sample *n* target identities from *completed_ids*.
    Input is sorted before sampling so results are stable across sessions.
    """
    rng = random.Random(seed)
    return rng.sample(sorted(completed_ids), min(n, len(completed_ids)))


def select_probes_for_identity(
    identity: str,
    exp_i: int,
    split: dict,
    n: int = N_PROBES_PER_ID,
    seed: int = SYNTH_SEED,
) -> list[str]:
    """
    Pick n probe images from the identity's own probe set (for SD img2img).
    Deterministic given (exp_i, seed).
    """
    probes = sorted(split["probes"][identity])
    rng = random.Random(seed * 1000 + exp_i)
    return rng.sample(probes, min(n, len(probes)))


def select_sources_for_target(
    target_id: str,
    exp_i: int,
    split: dict,
    n: int = N_PROBES_PER_ID,
    seed: int = SYNTH_SEED,
) -> list[tuple[str, str]]:
    """
    For a target identity, return *n* (source_id, source_probe_path) pairs.
    Sources are drawn from other identities that have probe images in the split.
    All selections are deterministic given (exp_i, seed).

    Seed formula mirrors RISE: seed*1000 + exp_i*100 + k
    """
    all_probe_ids = sorted(split["probes"].keys())
    candidates = [pid for pid in all_probe_ids if pid != target_id]

    rng_src = random.Random(seed * 1000 + exp_i)
    source_ids = rng_src.sample(candidates, n)

    pairs = []
    for k, src_id in enumerate(source_ids):
        probes = sorted(split["probes"][src_id])
        rng_img = random.Random(seed * 1000 + exp_i * 100 + k)
        pairs.append((src_id, rng_img.choice(probes)))

    return pairs
