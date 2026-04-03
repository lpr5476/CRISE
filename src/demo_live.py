"""
demo_live.py — Live webcam face identification demo using ArcFace.

Runs a real-time face recognition system against the LFW gallery + any
personally enrolled identities. Use this to demonstrate the adversarial
patch live: hold the printed patch over your face and watch rank-1
identification fail.

Usage:
    python demo_live.py
    python demo_live.py --your-name your_name_here   # include personal enrollment
    python demo_live.py --camera 1                   # use camera index 1

Controls (while running):
    Q  — quit
    S  — save current frame to results/demo_patch/screenshots/
    P  — toggle patch overlay on the aligned chip display
"""

import argparse
import os
import time
import numpy as np
import cv2
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GALLERY_EMB   = "cache/G.npy"
GALLERY_IDS   = "cache/gallery_ids.npy"
SCREENSHOT_DIR = "results/demo_patch/screenshots"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--your-name", type=str, default=None,
                    help="Name used during personal enrollment (demo_personal.ipynb)")
parser.add_argument("--camera", type=int, default=0,
                    help="Webcam device index (default: 0)")
parser.add_argument("--det-size", type=int, default=640,
                    help="InsightFace detection resolution")
args = parser.parse_args()

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# InsightFace setup
# ---------------------------------------------------------------------------
print("Loading InsightFace...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
rec = app.models["recognition"]

# ---------------------------------------------------------------------------
# Load gallery
# ---------------------------------------------------------------------------
print("Loading gallery embeddings...")
G           = np.load(GALLERY_EMB).astype(np.float32)
gallery_ids = np.load(GALLERY_IDS, allow_pickle=True).tolist()

# Optionally add personal enrollment
if args.your_name:
    personal_emb_path = f"results/crise_maps/crise_tau0p1_{args.your_name}_"
    # Find the gallery chip from demo_personal output
    gallery_chip_path = f"data/demo_identity/{args.your_name}"
    if os.path.isdir(gallery_chip_path):
        photos = sorted([
            os.path.join(gallery_chip_path, f)
            for f in os.listdir(gallery_chip_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        if photos:
            from rise import build_aligned_chip_112, get_embedding_from_chip
            img = cv2.imread(photos[0])
            try:
                chip = build_aligned_chip_112(img, app)
                emb  = get_embedding_from_chip(chip, rec).reshape(1, -1)
                G           = np.vstack([G, emb]).astype(np.float32)
                gallery_ids = gallery_ids + [args.your_name]
                print(f"Enrolled: {args.your_name} (gallery size: {len(gallery_ids)})")
            except Exception as e:
                print(f"[warn] Could not enroll {args.your_name}: {e}")
    else:
        print(f"[warn] Personal photo dir not found: {gallery_chip_path}")

id_to_index = {gid: i for i, gid in enumerate(gallery_ids)}
print(f"Gallery ready: {len(gallery_ids)} identities")

# ---------------------------------------------------------------------------
# Optional: load personal saliency patch for overlay
# ---------------------------------------------------------------------------
patch_mask = None
if args.your_name:
    patch_path = f"results/demo_patch/{args.your_name}/{args.your_name}_mean_saliency.npy"
    if os.path.exists(patch_path):
        sal = np.load(patch_path).astype(np.float32)
        k   = int(112 * 112 * 0.15)
        top_k = np.argsort(sal.ravel())[-k:]
        patch_mask = np.zeros((112, 112), dtype=np.uint8)
        ys, xs = np.unravel_index(top_k, (112, 112))
        patch_mask[ys, xs] = 255
        print(f"Patch loaded: {patch_path}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_DST_LANDMARKS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

COLOURS = {
    "hit":  (0, 220, 0),    # green — recognised
    "miss": (0, 0, 220),    # red   — not recognised
    "you":  (220, 180, 0),  # gold  — it's you
}


def get_aligned_chip(img_bgr, face):
    if face.kps is None:
        return None
    M, _ = cv2.estimateAffinePartial2D(
        face.kps.astype(np.float32), _DST_LANDMARKS, method=cv2.LMEDS
    )
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, (112, 112),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def identify(chip_bgr):
    feat = np.asarray(rec.get_feat(chip_bgr)).reshape(-1).astype(np.float32)
    emb  = feat / (np.linalg.norm(feat) + 1e-12)
    sims = G @ emb
    idx  = int(np.argmax(sims))
    return gallery_ids[idx], float(sims[idx])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {args.camera}")

print("\nRunning — press Q to quit, S to screenshot, P to toggle patch overlay")

show_patch   = False
fps_t        = time.time()
frame_count  = 0
screenshot_n = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display = frame.copy()

    faces = app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)

        chip = get_aligned_chip(frame, face)
        if chip is None:
            continue

        name, sim = identify(chip)

        # Colour: gold if it's you, green if recognised, red otherwise
        if args.your_name and name == args.your_name:
            colour = COLOURS["you"]
        elif sim > 0.35:
            colour = COLOURS["hit"]
        else:
            colour = COLOURS["miss"]
            name = "Unknown"

        # Bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)

        # Label
        label = f"{name}  {sim:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(display, (x1, y1 - lh - 8), (x1 + lw + 4, y1), colour, -1)
        cv2.putText(display, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Aligned chip inset (top-right corner) with optional patch overlay
        chip_rgb = cv2.cvtColor(chip, cv2.COLOR_BGR2RGB)
        chip_disp = chip.copy()
        if show_patch and patch_mask is not None:
            chip_disp[patch_mask == 255] = (0, 0, 0)

        inset = cv2.resize(chip_disp, (100, 100))
        ih, iw = 100, 100
        fx, fy = display.shape[1] - iw - 10, 10
        display[fy:fy+ih, fx:fx+iw] = inset
        cv2.rectangle(display, (fx, fy), (fx+iw, fy+ih), colour, 2)

    # FPS counter
    if frame_count % 30 == 0:
        elapsed = time.time() - fps_t
        fps = 30 / elapsed
        fps_t = time.time()
        fps_label = f"FPS: {fps:.1f}"
    else:
        fps_label = ""

    if fps_label:
        cv2.putText(display, fps_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Patch overlay indicator
    if show_patch and patch_mask is not None:
        cv2.putText(display, "[PATCH OVERLAY ON]", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 220), 1)

    cv2.imshow("CRISE-ID Live Demo", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        screenshot_n += 1
        path = os.path.join(SCREENSHOT_DIR, f"screenshot_{screenshot_n:04d}.jpg")
        cv2.imwrite(path, display)
        print(f"Saved: {path}")
    elif key == ord("p"):
        show_patch = not show_patch
        print(f"Patch overlay: {'ON' if show_patch else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Demo ended.")
