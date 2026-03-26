# CLAUDE.md — CRISE-ID Project

Read this at the start of every session before taking any action.

---

## Project Goal

Build **CRISE-ID** (Contrastive RISE for Identification), a forensic auditing toolkit for 1:N face recognition with two tightly coupled contributions:

1. **Better saliency maps** — CRISE-ID produces more faithful saliency maps than vanilla RISE by using softmax probability over the full gallery as the mask weight, reflecting the actual competitive decision structure of identification rather than raw similarity to the true identity alone.

2. **Deepfake forensics** — Use those saliency maps to answer the central research question: *when a deepfake successfully fools ArcFace, is it because it replicated genuine identity features, or because it exploited generative artifacts and non-facial regions?* Vanilla confidence scores cannot answer this. CRISE-ID can.

### Deliverables Split

**Research paper (primary deliverable):** The two contributions above. Concludes that face recognition systems cannot be trusted in forensic/legal contexts because they can be fooled by deepfakes that don't replicate genuine identity features, and this failure is invisible without explainability tools like CRISE-ID.

**Capstone demo (separate deliverable, not in the paper):** A physical adversarial patch — a printable occlusion pattern derived from CRISE saliency maps targeting high-saliency facial regions for a specific identity — demonstrated live against a running ArcFace system. The paper references this in one future work paragraph only.

> **Future work paragraph (use verbatim or adapt):** CRISE-ID saliency maps directly identify the facial regions a recognition system depends on per identity. A natural extension is translating these maps into physical adversarial patches — printable occlusions targeting the exact high-saliency regions identified for a given individual. Preliminary experiments suggest this approach is viable as a principled privacy countermeasure grounded in model-specific decision evidence, and we leave rigorous evaluation of physical-world patch effectiveness to future work.

---

## Stack

- **Python 3.10**, Jupyter notebooks
- **InsightFace** (`buffalo_l` model): face detection, 5-point landmark alignment, ArcFace recognition (512-dim L2-normalized embeddings)
- **ONNX Runtime** with CUDA: GPU inference for ArcFace
- **OpenCV**: image I/O, affine warp for chip alignment
- **NumPy / Pandas / Matplotlib**: numerics, results tables, figures
- Model weights live at `~/.insightface/models/buffalo_l/` (not in repo)

---

## Dataset: LFW-deepfunneled

| Stat | Value |
|---|---|
| Raw images | 13,233 |
| Raw identities | 5,749 |
| Valid identities (≥2 images) | 1,680 |
| Gallery size (1 image/identity) | 1,680 |
| Total probe images | 7,484 |
| Bad probes (no face detected) | 23 |
| Bad gallery entries (zero embedding) | 2 (`Emile_Lahoud`, `John_Paul_II`) |
| Split seed | 123 |

Images are 250×250 JPEGs. ArcFace operates on 112×112 aligned chips produced by affine-warping to 5 standard landmark positions.

---

## Repository Structure

```
CRISE/
├── data_prep.ipynb          # Build gallery/probe split from LFW
├── embedding.ipynb          # ArcFace embeddings for gallery + all probes
├── Rise_Baseline.ipynb      # Vanilla RISE saliency (aligned-chip, N=1000)
├── eval.ipynb               # Insertion/deletion evaluation (margin-based AUC)
├── RESEARCH_NOTES.md        # Full design spec — read before implementing anything
│
├── data/
│   ├── lfw-deepfunneled/    # Raw LFW images (13,233 JPEGs)
│   ├── *.csv                # Original LFW metadata files
│   └── synthetic_probes/    # (not yet created)
│       ├── insightface_swap/
│       │   └── {identity}/{identity}_swap_{i}.jpg
│       ├── simswap/
│       │   └── {identity}/{identity}_simswap_{i}.jpg
│       ├── morphing/
│       │   └── {identity}/{identity}_morph_{alpha}_{i}.jpg
│       └── metadata.csv     # All methods; columns: identity, generation_method,
│                            # source_identity, blend_alpha, output_path,
│                            # embedding_ok, arcface_similarity, rank1_match,
│                            # saliency_cosine_sim, saliency_l1, case_label
│
├── splits/
│   └── lfw_1N_split.json    # Gallery + probe assignments (seed=123)
│
├── cache/
│   ├── G.npy                # Gallery embeddings (1680×512 float32)
│   ├── gallery_ids.npy      # Ordered identity names for G.npy rows
│   ├── probe_embeds.npy     # Probe embeddings (7484×512 float32)
│   ├── probe_meta.json      # Per-probe: true_id, img_path, ok flag
│   └── probe_cache_state.json  # Resume state (idx=7484, complete)
│
└── results/
    ├── baseline_arcface_lfw_1N.json          # Rank-1/5 accuracy
    ├── rise_alignedchip_baseline/            # Early single-probe cosine runs (ignore)
    │   └── rise_baseline_cosine_N{n}_*.npy/json
    └── rise_alignedchip_baseline_multi/
        ├── *_saliency_norm.npy               # 1674 completed saliency maps
        ├── *_sal_accum.npy                   # Intermediate accumulators
        ├── *_state.json                      # Per-probe resume state
        ├── summary_K5_N1000_s8_p0.5_MASTERSEED123.csv  # Per-probe summary
        ├── figures/                          # Chip/saliency/overlay PNGs per probe
        │   └── {id}_N1000_s8_p0.5_{chip|saliency|overlay}.png
        ├── eval_margin_auc_multi/
        │   └── eval_margin_auc_multi_steps50_black.csv  # Full AUC results
        └── margin_eval/                      # Early 5-probe test run — ignore
```

---

## Pipeline (Execution Order)

1. `data_prep.ipynb` → `splits/lfw_1N_split.json`
2. `embedding.ipynb` → `cache/G.npy`, `cache/probe_embeds.npy`, etc.
3. `Rise_Baseline.ipynb` → `results/rise_alignedchip_baseline_multi/*`
4. `eval.ipynb` → `results/rise_alignedchip_baseline_multi/eval_margin_auc_multi/*`

Steps 1–4 are **complete** for the baseline.

---

## Baseline Results (Vanilla RISE)

| Metric | Value |
|---|---|
| ArcFace Rank-1 accuracy | 0.9436 (7,462 probes) |
| ArcFace Rank-5 accuracy | 0.9672 |
| RISE probes evaluated | 1,597 / 1,674 completed |
| **Deletion AUC (margin)** | **0.035** |
| **Insertion AUC (margin)** | **0.280** |
| RISE hyperparams | N=1000, s=8, p=0.5, seed=123 |
| Insertion baseline | black (zeros) |
| Evaluation steps | 50 |

The evaluation metric is **identification margin**: `sim(true_id) - max(sim(all_impostors))`. Higher is better for insertion; lower is better for deletion.

---

## Known Issues

1. ~~**77 saliency maps excluded from eval**~~: **Fixed.** Replaced `split("_N")[0]` with a regex anchor and exact `pid + "_" + stub` matching in `rebuild_df_from_saliency_dir_robust`. All 1674 completed maps should now be evaluable.

2. **6 missing RISE runs**: RISE was configured for 1680 identities but only 1674 completed. The missing 6 likely failed face detection on the probe image.

3. ~~**Two zero gallery embeddings**~~: **Fixed.** `Emile_Lahoud` and `John_Paul_II` now excluded via `BAD_GALLERY_IDS` in eval.ipynb.

4. **`margin_eval/` is a 5-probe leftover**: Ignore for analysis purposes.

5. **`rise_alignedchip_baseline/`**: Early single-probe cosine runs at N=1000/2000/20000. Not part of main evaluation — ignore.

---

## Current Goals (What Is Not Yet Built)

In priority order:

1. ~~Fix eval.ipynb bugs~~ — **Done**.

2. **Refactor `Rise_Baseline.ipynb` → `rise.py`** — Extract the RISE loop into a module with the weighting function as a parameter, so CRISE is a clean extension, not a copy.

3. **Implement `crise.py`** — Extend `rise.py` with softmax weighting:
   ```python
   sims = cosine_sim(f(x * m), G_all)   # shape (1680,)
   w = softmax(sims / tau)[true_id_idx]  # tau=0.1 default
   ```
   Cache results to `results/crise_maps/`.

4. **Unified eval: RISE vs. CRISE** — Run insertion/deletion on CRISE maps using the same margin metric. Include sanity checks (CRISE insertion AUC > RISE, deletion AUC < RISE) and a stability test (two seeds, compare map distance).

5. **Synthetic probe generation** — Three methods, 50 identities per method, 3 probes per identity (~450 total). See Methodology Notes below.

6. **Run CRISE on synthetic probes** — Cache saliency maps to `results/crise_maps/` alongside real-probe maps.

7. **Deepfake forensics analysis** — Four-case stratification (A/B/C/D), per-region importance profiles, saliency divergence metrics, 8+ figures. Core contribution. See Methodology Notes below.

8. **(Extra) Digital validation for capstone demo** — Time permitting. See Methodology Notes below.

---

## Methodology Notes

### Softmax Weighting (CRISE Core)

Baseline RISE uses cosine similarity to the true identity for **both** mask weighting and evaluation — measuring self-consistency, not faithfulness to the decision boundary. CRISE uses softmax probability for weighting and margin for evaluation. These are different quantities; the evaluation is a genuine test of faithfulness, not a tautology. State this explicitly in the paper.

- **Softmax probability** (`w = softmax(sims / tau)[true_id]`): probabilistic interpretation, connects to softmax cross-entropy literature, most defensible to reviewers. **Primary choice.**
- **Margin** (`w = s_true - max(s_impostors)`): can go negative, internally consistent with evaluation metric. Implement as a secondary comparison.
- **Rank-gated** (`w = s_true * 1[rank==1]`): conservative and noisy, sanity check only.

### Temperature Tau Ablation

- Default: **tau = 0.1**
- Ablation values: `[0.05, 0.1, 0.2, 0.5]`
- `TAU` must be a top-level constant in `crise.py` for easy ablation
- Report insertion/deletion AUC at each tau in a sensitivity table

### Synthetic Probe Generation: Three Methods

Using three methods lets you ask whether the forensic case distribution (A/B/C/D) is method-specific or consistent across generation paradigms — a much stronger claim than a single method.

**Method 1 — InsightFace Face Swap:** Already in the stack. Affine warp + blend. Source: random probe from a different identity; Target: gallery image of true identity.

**Method 2 — SimSwap (GAN-based):** Dedicated neural face swap with identity injection. Install via SimSwap GitHub repo. Same source/target convention. Produces different artifact signatures; expected to yield more Case A instances.

**Method 3 — Face Morphing:** Pixel-level weighted blend using OpenCV. No generative model required. Primary analysis at blend ratio 0.5; secondary ablation at [0.3, 0.5, 0.7]. Directly relevant to morphing attacks documented in passport/border control FR literature.
```python
morph = cv2.addWeighted(target_gallery_img, alpha, source_img, 1 - alpha, 0)
```

**Targets:** 50 identities × 3 probes × 3 methods = ~450 synthetic probes total.

**Cross-method comparison table** (primary new result):

| Generation Method | Case A % | Case B % | Case C % | Case D % | Rank-1 Rate |
|---|---|---|---|---|---|
| InsightFace Swap | | | | | |
| SimSwap | | | | | |
| Face Morphing (α=0.5) | | | | | |

### Deepfake Forensics: Four-Case Stratification

Every synthetic probe is classified on two binary axes (fooled ArcFace? / CRISE map similar to real probe?):

| Case | Fooled ArcFace? | Saliency similar? | Interpretation |
|---|---|---|---|
| A | Yes | Yes | Fooled for right reasons — genuine identity features replicated |
| B | Yes | No | **Fooled for wrong reasons** — exploiting artifacts or non-identity regions |
| C | No | Yes | Correct features but insufficient identity transfer strength |
| D | No | No | Complete failure |

**Case B is the headline finding.** Direct evidence ArcFace can be spoofed without replicating genuine identity features. Case A validates some deepfakes are genuine high-quality identity transfers. Distinguishing A from B is the forensic contribution.

**Confound control:** Stratify by match confidence. Only compare real/synthetic pairs where both achieve rank-1 AND ArcFace similarity > 0.3. Report saliency divergence per similarity bracket (0.2–0.3, 0.3–0.4, 0.4+).

**Saliency similarity threshold:** Established empirically from the cosine similarity distribution across all real/synthetic pairs. Report the full distribution, not just the binary cutoff.

**Per-region analysis:** Break each map into 5 zones via InsightFace 5-point landmarks (eye zone, nose zone, mouth zone, jaw/chin, forehead/upper face). Report mean fractional saliency weight per region per case. Expected for Case B: elevated weight in skin texture, hairline, or background.

**Divergence metrics per real/synthetic pair:** cosine distance (flattened maps), L1 distance, per-region importance divergence. Visual inspection: 2+ examples per case (8 figures minimum).

### Path A: Digital Validation (Extra — Supports Capstone Demo, Not in Paper)

Use CRISE saliency maps to identify high-importance pixels, replace with mean face value, measure rank-1 accuracy drop. Pixel budgets: `[0.05, 0.10, 0.20, 0.30]` of 112×112 = 12,544 pixels. Four conditions on one figure:

1. **CRISE-guided** (primary)
2. **Baseline RISE-guided** ← most important for reviewers; proves CRISE finds better attack surface
3. **Random pixel perturbation**
4. **Bottom-k CRISE pixels** (low-saliency control)

---

## Paper Framing Notes

- **Primary novelty:** The 1:N vs. 1:1 distinction — S-RISE and CorrRISE are verification methods, CRISE-ID addresses identification. Emphasize this.
- **Circularity argument:** Worth a paragraph in the methodology section — our evaluation is cleaner than baseline because weighting and evaluation use different quantities.
- **Narrative spine:** The four-case taxonomy (A/B/C/D) structures the synthetic probe section. Case B is the headline.
- **Quantitative payoff:** Per-region importance profiles make the deepfake findings publishable, not just qualitative ("30% more saliency weight shifted to non-geometric regions in Case B").
- **Closing argument:** ArcFace can be fooled by deepfakes that don't replicate genuine identity features, invisible without explainability tools → FR systems cannot be trusted in forensic/legal contexts.
- **Future work:** Use the verbatim paragraph above. One paragraph only; do not over-promise on the physical demo.

---

## Important Invariants

- **Never regenerate ArcFace embeddings** unless explicitly necessary. `cache/` is the source of truth.
- All saliency maps are cached as `.npy` files. Evaluation figures must be reproducible from cached files without re-running RISE or CRISE.
- All random operations must use deterministic seeds. RISE seeds follow `MASTER_SEED * 10_000 + exp_i * 100 + k`.
- Working directory is assumed to be the repo root for all relative paths.
- `TAU` must be a top-level constant in `crise.py`.
- Synthetic probe `metadata.csv` is the source of truth for case labels — keep it updated as analysis runs.
