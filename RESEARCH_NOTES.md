# CRISE-ID Research Notes
**For Claude Code sessions -- read this before any task**

---

## Project Goal and Scope Split

### Research Paper (primary deliverable)
Two clean contributions:

**Contribution 1 -- CRISE-ID:** A contrastive explainability framework for 1:N facial recognition that produces more faithful saliency maps than baseline RISE by incorporating the full gallery decision structure via softmax-normalized weighting. Primary novelty: prior methods (S-RISE, CorrRISE) address 1:1 verification; CRISE-ID addresses 1:N identification.

**Contribution 2 -- Deepfake Forensics:** Using CRISE-ID as a forensic tool to diagnose why deepfakes succeed or fail at fooling ArcFace, at the level of individual facial regions. Central research question: when a deepfake fools a face recognition system, is it because it replicated genuine identity features, or because it exploited generative artifacts and non-facial regions? This distinction cannot be made from confidence scores alone -- it requires saliency-level analysis.

The paper concludes that face recognition systems cannot be trusted in legal or forensic contexts because (a) they can be fooled by deepfakes that do not even replicate genuine identity features, and (b) the failure modes are invisible without explainability tools like CRISE-ID.

### Capstone Demo (separate deliverable, not in paper)
A personal digital forensics demonstration using `demo_personal.ipynb`. Enroll your own face, run CRISE-ID, and produce a four-panel forensics report:
- **Panel A** — Saliency overview (chip / map / overlay per probe + mean)
- **Panel B** — Per-region importance (5 anatomical zones, fractional saliency mass)
- **Panel C** — Deletion experiment (recognition confidence vs. % of most-salient pixels masked; 3 conditions: CRISE-guided, random, reverse-CRISE)
- **Panel D** — Cross-probe consistency (pairwise cosine similarity across your probe maps)

**Framing — informational asymmetry:** When you enroll in a FR system, the system knows which parts of your face it owns. You don't. CRISE-ID closes that gap. Live demo narrative: *"I enrolled in this system. Here is the audit of what it knows about me that I didn't know before running this tool."* This is the civil liberties argument made concrete and personal.

The paper references the broader implications in future work with one paragraph.

### Future Work Paragraph (use verbatim or adapt)
> CRISE-ID saliency maps directly identify the facial regions a recognition system depends on per identity. A natural extension is translating these maps into physical adversarial patches -- printable occlusions targeting the exact high-saliency regions identified for a given individual. Preliminary experiments suggest this approach is viable as a principled privacy countermeasure grounded in model-specific decision evidence, and we leave rigorous evaluation of physical-world patch effectiveness to future work.

The surveillance disruption framing is intentional and central throughout. This is a forensic auditing toolkit, not just an XAI paper.

---

## Current State

### What is complete
- Baseline RISE pipeline on LFW 1:N identification
- ArcFace embeddings via InsightFace (512-dim, aligned 112x112 chips)
- Gallery: 1680 identities, Probes: 7484 images
- Insertion/deletion evaluation using identification margin: `margin = s(true) - max(s(impostors))`
- Baseline results: **deletion AUC = 0.035, insertion AUC = 0.280**, over 1597 valid probes
- Codebase: data_prep.ipynb, embedding.ipynb, rise_baseline.ipynb, eval.ipynb

### What is NOT yet implemented
- CRISE-ID (contrastive weighting -- the core contribution)
- Synthetic probe generation
- Saliency divergence analysis (real vs synthetic)
- Path A: saliency-guided adversarial perturbation experiment
- Unified evaluation comparing RISE vs CRISE side by side

---

## CRISE Weighting Function: The Core Change

### Why baseline RISE is insufficient for 1:N identification

Baseline RISE weights each mask by cosine similarity to the true identity alone. This ignores the competitive structure of identification. A pixel can increase similarity to the true identity while equally increasing similarity to impostors -- meaning it carries no discriminative information for the actual rank-1 decision.

### The CRISE estimand

Instead of:
```
w(masked_img) = cosine_sim(f(x * m), gallery[true_id])
```

Use softmax-normalized probability over the full gallery:
```python
sims = cosine_sim(f(x * m), all_gallery_embeddings)  # shape (1680,)
softmax_sims = softmax(sims / tau)
w = softmax_sims[true_id]
```

The weight is now high only when the masked image confidently identifies as the true identity *relative to all 1680 gallery identities*. This reflects the actual decision boundary.

### Temperature parameter tau

- tau controls sharpness of the softmax distribution
- Too low (e.g. 0.01): near one-hot behavior, unstable gradients
- Too high (e.g. 1.0): flat distribution, loses discriminative signal
- Default starting point: **tau = 0.1**
- Worth ablating: [0.05, 0.1, 0.2, 0.5] and reporting sensitivity


**Primary recommendation: Option A (softmax probability).** Has probabilistic interpretation, connects to softmax cross-entropy literature, reviewers will find it most defensible.

---

## Evaluation Design

### Primary metrics (faithfulness)
- Mean insertion AUC (higher = better)
- Mean deletion AUC (lower = better)
- Both use identification margin as the y-axis, same as baseline

### Expected result
CRISE should outperform baseline RISE on both metrics. If it does not, the contrastive weighting is not capturing decision-relevant information -- revisit tau or weighting function.

### Avoiding circularity
Baseline RISE uses cosine similarity to true identity for BOTH mask weighting and evaluation. CRISE uses softmax probability for weighting but margin for evaluation. This is better practice -- state it explicitly in the paper. You are not measuring self-consistency, you are measuring faithfulness to the true decision boundary.

### Additional evaluation axes

**Sanity checks:**
- Does CRISE insertion AUC > RISE insertion AUC? (should be yes)
- Does CRISE deletion AUC < RISE deletion AUC? (should be yes)
- If neither holds, something is wrong before moving on

**Rank sensitivity:**
- Correlate pixel importance scores with identification margin across probes
- CRISE saliency should be a better predictor of margin than RISE saliency

**Stability:**
- Run CRISE twice on the same probe with different random seeds
- Compute L1 or cosine distance between the two saliency maps
- Low variance = reliable estimator, worth reporting as a table

**Qualitative coherence:**
- Do saliency maps concentrate on eyes, nose bridge, jaw? (expected for ArcFace)
- Flag any probes where saliency concentrates on background or hair -- these are failure cases

---

## Deepfake Forensics: The Core Experiment

This is the primary contribution of the synthetic probe work. The goal is not just to compare saliency maps between real and synthetic probes -- it is to use CRISE-ID as a forensic tool to diagnose *why* a deepfake succeeded or failed at fooling ArcFace, at the level of individual facial regions.

### The Central Research Question

When a deepfake successfully achieves rank-1 match against a gallery identity:
- Did it succeed because it replicated the genuine identity features ArcFace relies on (eyes, nose bridge, jaw geometry)?
- Or did it succeed by exploiting generative artifacts, texture statistics, or non-facial regions that ArcFace incorrectly treats as identity signals?

Vanilla confidence scores cannot answer this. CRISE-ID can.

### Four-Case Stratification

Every synthetic probe falls into one of four forensic cases based on two binary dimensions: whether the deepfake fooled ArcFace (rank-1 match) and whether its CRISE saliency map is structurally similar to the real probe's map.

| Case | Fooled ArcFace? | Saliency similar to real? | Forensic interpretation |
|---|---|---|---|
| A | Yes | Yes | Fooled for right reasons -- genuine identity features replicated |
| B | Yes | No | Fooled for wrong reasons -- exploiting artifacts or non-identity regions |
| C | No | Yes | Correct features but insufficient identity transfer strength |
| D | No | No | Complete failure -- neither identity nor feature regions transferred |

**Case B is the most important finding.** It means ArcFace is being fooled by something other than genuine facial identity. This is direct evidence the system is unreliable as a forensic or legal tool -- it can be spoofed without actually replicating the target's identity features.

**Case A is the second most important.** It validates that some deepfakes are genuinely high-quality identity transfers, not just statistical artifacts. Distinguishing A from B is the forensic contribution of this work.

### Saliency Similarity Threshold

To classify a probe as "saliency similar" vs "saliency divergent", compute cosine similarity between the flattened CRISE maps of the real and synthetic probes for the same identity. Establish a threshold empirically from the distribution across all pairs. Probes above the threshold are Case A or C; below are Case B or D. Report the full distribution, not just binary classification.

### Generation Approach: Three Methods

Using a single generation method is a significant reviewers will flag -- findings would only generalize to that one approach. Three methods lets you ask whether the forensic case distribution (A/B/C/D) is method-specific or consistent across generation paradigms. If Case B appears across all methods, that is a strong claim about ArcFace's general vulnerability. If it is method-specific, that is equally interesting -- it identifies which generation approaches produce artifact-driven matches vs genuine identity transfer.

**Method 1 -- InsightFace Face Swap**
Already in the stack. Transfers identity from a source face onto a target face structure via affine warping and blending. Fast and controllable.
- Source: random probe from a different identity (provides face geometry)
- Target: gallery image of the true identity (provides identity)
- Output: blended face that should resemble the target identity

**Method 2 -- SimSwap (GAN-based dedicated face swap)**
A dedicated neural face swap model trained specifically for identity preservation. Produces different artifact signatures than InsightFace swap because it uses a different architecture (encoder-decoder with identity injection). Install via the SimSwap GitHub repo. Use the same source/target convention as Method 1. This represents a higher-quality neural transfer baseline and will likely produce more Case A instances than Method 1.

**Method 3 -- Face Morphing**
Blend two real faces at the pixel level using a weighted average. No generative model required -- implementable with OpenCV. Morph the gallery image of the target identity with a randomly selected source face at blend ratios [0.3, 0.5, 0.7] (target weight). This is conceptually distinct from the swap methods and directly relevant to the legal context: morphing attacks are a documented real-world threat to passport and border control FR systems, connecting the forensics findings to existing literature.

```python
# Simple morphing implementation
morph = cv2.addWeighted(target_gallery_img, alpha, source_img, 1-alpha, 0)
```

Generate at blend ratio 0.5 for the primary analysis. Use [0.3, 0.5, 0.7] for a secondary ablation showing how blend ratio affects rank-1 success rate and case distribution.

**Generation targets:** 50 identities per method, 3 synthetic probes per identity per method. Total: ~450 synthetic probes.

**Directory structure:**
```
data/synthetic_probes/
    insightface_swap/
        {identity_name}/{identity_name}_swap_{i}.jpg
    simswap/
        {identity_name}/{identity_name}_simswap_{i}.jpg
    morphing/
        {identity_name}/{identity_name}_morph_{alpha}_{i}.jpg
    metadata.csv   # single CSV covering all methods, with generation_method column
```

**Metadata CSV columns:** identity, generation_method, source_identity, blend_alpha (morphing only), output_path, embedding_ok, arcface_similarity, rank1_match, saliency_cosine_sim, saliency_l1, case_label.

### Cross-Method Comparison Table

The primary new result enabled by three methods is a method comparison table:

| Generation Method | Case A % | Case B % | Case C % | Case D % | Rank-1 Rate |
|---|---|---|---|---|---|
| InsightFace Swap | | | | | |
| SimSwap | | | | | |
| Face Morphing (α=0.5) | | | | | |

Expected pattern: SimSwap produces more Case A (genuine identity transfer, higher quality); morphing produces more Case B at low alpha (identity partially transferred but saliency driven by blending artifacts); InsightFace swap somewhere in between. Report the actual distribution without assuming this pattern -- the finding is whatever the data shows.

### Critical Confound to Control

Saliency divergence could simply reflect weak identity transfer rather than anything meaningful about generative artifacts. A poorly generated deepfake will have a diffuse saliency map for trivial reasons.

**Control strategy:** Stratify all analysis by match confidence. Only compare real/synthetic pairs where both achieve rank-1 match AND ArcFace similarity is above 0.3. Report saliency divergence as a function of similarity score bracket (e.g. 0.2-0.3, 0.3-0.4, 0.4+). This lets you make the claim: *even among deepfakes that match with comparable confidence to real probes, CRISE maps reveal structurally different decision regions in Case B instances.*

### Per-Region Saliency Analysis

Do not only compare full saliency maps. Break each map into facial regions using InsightFace 5-point landmarks (extended to approximate regions: eye zone, nose zone, mouth zone, jaw/chin, forehead/upper face) and compute per-region importance as the fraction of total saliency weight in that region.

For each case (A, B, C, D), report the mean per-region importance profile. The expected finding for Case B: elevated saliency weight in skin texture regions, hairline, or background relative to Case A. This is the publishable quantitative result -- not just "maps look different" but "successful artifact-driven deepfakes shift 30% more saliency weight to non-geometric regions."

### Saliency Divergence Metrics

For each real/synthetic probe pair of the same identity:
- **Cosine distance** on flattened saliency maps: captures structural pattern difference regardless of magnitude
- **L1 distance**: captures absolute magnitude difference region by region
- **Per-region importance divergence**: for each facial region, absolute difference in fractional saliency weight between real and synthetic
- **Visual inspection**: side-by-side saliency map figures for at least 2 examples from each of the four cases (8 figures minimum)

### Expected Findings and Paper Claims

If the experiment works as designed, you can make the following claims:
1. CRISE-ID can classify deepfake success/failure into mechanistically distinct categories that confidence scores cannot distinguish
2. A measurable fraction of successful deepfakes (Case B) fool ArcFace via non-identity features, indicating the system is vulnerable to artifact-driven spoofing
3. The per-region saliency profile of Case B deepfakes is quantitatively distinct from genuine matches, providing a potential detection signal
4. This has direct implications for the reliability of face recognition systems in forensic and legal contexts

---

## Demographic Saliency Analysis (New Capstone Experiment)

### Research Question
Does ArcFace rely on different facial regions for different demographic groups?

### Method
1. Run InsightFace's built-in gender/age estimator on the 1,680 gallery images — these attributes are produced by `buffalo_l` alongside detection; they have just not been used yet
2. Split the existing real-probe CRISE saliency maps by estimated gender (and optionally by age bracket: <35, 35–55, 55+)
3. Compute mean per-region importance profile for each group using the 5-zone framework: Forehead, Eye zone, Nose, Mouth, Jaw/chin
4. Compare profiles across groups — both as bar charts and as significance tests (Mann-Whitney U or permutation test on per-region fractions)

**Zero new saliency computation required.** Uses the 1,680+ maps already cached in `results/crise_maps/`.

### Expected Finding
The system may over-rely on skin texture regions (forehead, jaw) for one group and on geometric landmark regions (eye zone, nose) for another. If confirmed, this is structural bias in feature extraction — not just an accuracy disparity, but evidence the model uses a fundamentally different decision basis across groups.

### Why This Matters
Current bias research in FR reports accuracy gaps (e.g. "5% higher error rate for Group X"). CRISE-ID goes one level deeper: *why* is the error rate higher? If the model relies on less stable or less geometrically grounded regions for a given group, the accuracy gap has a mechanistic explanation. That is a much stronger and more actionable finding.

### Societal Framing
This is the second invisible failure mode: not just that deepfakes can fool the system without replicating identity features, but that the system uses different evidence standards for different people. Both are invisible from confidence scores alone. Both require CRISE-ID to detect.

### Combined Conclusion for Capstone
Two distinct societal contributions:
1. **Deepfake forensics** — FR can be fooled without replicating genuine identity features (Case B); invisible without saliency analysis
2. **Demographic bias** — FR relies on structurally different facial evidence for different demographic groups; invisible without saliency analysis

Both failures share the same root: you cannot trust or audit these systems without an explainability layer. CRISE-ID provides that layer.

### Output
- Mean per-region importance bar chart per gender group (side-by-side)
- Significance table (per-region p-values)
- Sample saliency maps from each group to support qualitative claims
- One paragraph in paper conclusions or future work if finding is strong enough

---

## Path A: Digital Validation (Extra -- Time Permitting, Supports Capstone Demo)

Not required for the paper. If time permits, this provides the digital validation step that grounds the capstone demo. The demo itself is built separately after the paper is complete.

### The digital experiment
Use CRISE saliency maps to identify the highest-importance pixels for a set of probes, replace them with mean face value at increasing budgets, and measure rank-1 accuracy drop. This validates that CRISE saliency correctly identifies the attack surface before committing to a physical print.

### The four conditions
1. **CRISE-guided perturbation** (primary)
2. **Baseline RISE-guided perturbation** (proves CRISE identifies better attack surface)
3. **Random pixel perturbation** (obvious baseline)
4. **Bottom-k CRISE pixels** (low-saliency control)

### Perturbation budgets
[0.05, 0.10, 0.20, 0.30] fraction of pixels. Plot rank-1 accuracy vs budget, all four conditions on one figure.

---

## Capstone Demo Prep (Separate from Paper -- demo_personal.ipynb)

### Goal
A live forensics demonstration using your own enrolled face. The narrative: informational asymmetry — the FR system knows which parts of your face it owns; you don't; CRISE-ID reveals that.

### Steps
1. Place 5-10 photos of your face in `data/demo_identity/{YOUR_NAME}/`
2. Set `YOUR_NAME` in `demo_personal.ipynb` config cell and run top to bottom
3. The notebook produces four forensics panels automatically (see Capstone Demo section above)
4. Present Panel C (deletion experiment) live: show recognition breaking as most-salient pixels are progressively masked

### What makes this compelling
The forensics report is personalized. You are not showing an abstract result from LFW — CRISE-ID is auditing which parts of *your* face *this specific model* relies on, in real time, against a gallery of 1,681 identities. That is the accountability argument made personal.

---

## Implementation Order

1. Fix known bugs in eval.ipynb (substring matching issue, zero gallery embeddings) ✓
2. Refactor rise_baseline.ipynb into rise.py module (weighting function as parameter) ✓
3. Implement crise.py extending rise.py with softmax weighting ✓
4. Run CRISE on 1597 probes, cache results to results/crise_maps/ ✓
5. Unified eval: RISE vs CRISE insertion/deletion side by side, sanity checks, stability test ✓
6. Generate synthetic probes: 3-5 per identity, 50-100 identities, metadata CSV ✓
7. Run CRISE on synthetic probes, cache saliency maps  ← current
8. Deepfake forensics analysis: four-case stratification, per-region importance, saliency divergence metrics, figures
9. Demographic saliency analysis: gender/age estimation on gallery, split existing maps, per-region comparison across groups, significance tests
10. Capstone demo: run demo_personal.ipynb with personal photos, present four forensics panels live
11. (Extra) Path A: four-condition perturbation experiment if time permits

---

## Notes on Code Architecture

- Keep weighting function as a parameter in rise.py so CRISE is a clean extension, not a copy
- Cache all saliency maps as .npy files -- regenerating is expensive
- Never regenerate ArcFace embeddings unless explicitly necessary, they are already cached
- Temperature tau should be a top-level constant in crise.py, easy to change for ablation
- All evaluation figures must be reproducible from cached .npy files without re-running RISE/CRISE
- Synthetic probe metadata CSV is the source of truth for case labels -- keep it updated as analysis runs

---

## Paper Framing Notes

- The 1:N vs 1:1 distinction is the primary novelty claim -- emphasize that S-RISE and CorrRISE are verification methods, not identification methods
- The softmax weighting has a clean probabilistic interpretation -- connect to softmax cross-entropy in related work
- The circularity argument (why our evaluation is cleaner than baseline) is worth a paragraph in the methodology section
- The four-case deepfake taxonomy (A/B/C/D) is the narrative spine of the synthetic probe section -- Case B is the headline finding
- The per-region importance analysis is what makes the deepfake findings quantitative and publishable rather than qualitative
- The paper's closing argument: ArcFace can be fooled by deepfakes that do not even replicate genuine identity features, and this is invisible without explainability tools -- therefore FR systems cannot be trusted in legal or forensic contexts
- Future work paragraph is already written above -- use it verbatim or adapt, one paragraph only, do not over-promise on the physical demo
- Frame the overall contribution as: CRISE-ID is a forensic auditing tool that reveals why face recognition systems succeed or fail, including under adversarial synthetic inputs
- If the demographic analysis produces a strong finding, add one paragraph to the paper conclusion or future work section: "Beyond deepfake forensics, CRISE-ID reveals that ArcFace relies on structurally different facial evidence for different demographic groups — a form of bias that confidence scores cannot expose."
- The two societal contributions share the same conclusion: FR systems cannot be trusted or audited without an explainability layer. CRISE-ID provides that layer and reveals two previously invisible failure modes.