# Project context — handoff (written 2026-11-22)

Read this first, then `papers/INDEX.md`. This file is the full state of the project so far, so work can resume
on another machine (one with a GPU) without the original chat.

## 1. Goal

Critics often say generative models (LLMs, diffusion) can't produce original or tasteful content. The aim is to
**give a generative model taste**. The plan is to first **learn a critic** that captures the taste of *specific individual
experts*, and only later use it as the reward for a generator.

- The **end goal is design generation**, so the domain is images. Chess was dropped.
- **Step 1 (current): learn and verify a pairwise critic.** Given two items, it predicts which one a specific person prefers.
  Then check whether its scores match that person's.
- **Step 2 (deferred, do not start until the user asks):** use the critic as the RL / fine-tuning reward for a generator
  (e.g. diffusion, à la ImageReward/ReFL).
- The user wants to model a **few experts individually**, not a consensus. Averaging raters erases taste (Artificial Hivemind 2025).
- **The critic must be verified before any training of a generator.**
- Everything must be **backed by top papers**. Keep the literature list **lean**, because the user dislikes noisy reading lists.

## 2. How we got here

1. **Origin:** Feng et al. 2025, *Generating Creative Chess Puzzles* (DeepMind, NeurIPS 2025, `papers/Feng2025_...pdf`).
   They RL-train a puzzle generator with a **hand-designed scorer** (engine-based uniqueness + counter-intuitiveness, weights tuned
   by grid search on a tiny 84+41 "Golden Set"). Key facts we took from it:
   - their score correlates with puzzle *difficulty* but has **~0 correlation with user popularity** (Table 6), so it doesn't capture liking;
   - optimising it caused reward hacking (entropy collapse, extra pieces, engine exploits);
   - aesthetics was only filtered afterwards with weak rule-based detectors.
   → Motivation: in domains like design you *can't* write such rules, so learn the critic from people instead.
2. The first idea was to replicate this in chess with a learned critic. It was dropped because there is no per-expert chess data,
   expert labels are expensive, and the end goal is design anyway.
3. A synthetic-experts step was proposed (hidden scoring functions as fake experts, Gao-2022-style), then dropped together with chess.
   It can come back as a sanity check on images if it's useful.
4. Domain chosen: **personalized image aesthetics (PIAA)**.
   - **PARA** (CVPR 2022) would be ideal: 438 raters, ~1.8k images each, **repeated images inside each session, so per-rater test–retest
     consistency is available**, plus photographic/artistic experience per rater. But access needs registration at
     cv-datasets.institutecv.com (terms unread, no public mirror). The user may request it later.
   - **LAPIS was chosen** (user decision): openly downloadable, artworks curated with art historians, closer to design.

## 3. Dataset: LAPIS

- Hugging Face: `asmaerten/LAPIS` (ungated, **CC BY-NC 4.0**, non-commercial). Paper: arXiv 2504.07670 (`papers/Maerten2025_LAPIS.pdf`).
- Cleaned release: **281,174 individual ratings, 11,723 artworks, ~24 ratings/image (range 13–67)**. 552 Prolific raters
  (~500 ratings each on average). Images are split into `figurative/` and `abstract/`.
- Files:
  - `piaa/piaa.csv` (~84 MB): one row per rating, with `rating`, `response time`, `participant_id`, rater demographics,
    **art-interest score**, `imgs_known`. Read it with `pd.read_csv(..., keep_default_na=False, na_values=[""])`, otherwise `imgs_known="None"` turns into NaN.
  - `giaa/{train,validation,test}.csv`: per-image aggregates (`mean_response`, `std_response`, number of raters…). This is the official split.
  - `image_attributes/image_attributes.csv`: low-level visual stats. `image_attributes/image_prompts.csv`: a generated caption per image.
  - `images/` (~716 MB): join on the `path` column (e.g. `figurative/<file>.jpg`) or on `image_filename`.
- **There are no repeated images in LAPIS**, so the per-rater ceiling has to come from split-half reliability or from how well
  other raters predict a rater (see §5). This is the main weakness compared with PARA.
- In the paper, raters with more art interest rate differently (e.g. novices score abstract art lower). Use art interest to pick
  "expert" raters.

## 4. Literature (full notes in `papers/INDEX.md`)

| Role | Paper |
|---|---|
| Data | Maerten 2025 **LAPIS**; Yang 2022 **PARA** (fallback / later) |
| Personalization baseline | Ren 2017 *Personalized Image Aesthetics* (generic model + small learned per-user residual; active learning) |
| Pairwise critic method | Christiano 2017 (Bradley–Terry reward model); Stiennon 2020 (judge the RM against the inter-labeler ceiling) |
| Pairwise → scales, study design | Pérez-Ortiz & Mantiuk 2017 |
| Image critics + bridge to diffusion | Kirstain 2023 PickScore (CLIP + BT head); Xu 2023 ImageReward (expert comparisons → ReFL) |
| Risks / motivation | Gao 2022 RM overoptimization; Jiang 2025 Artificial Hivemind; Feng 2025 (origin) |

`papers/chess_archive/` holds the set-aside chess papers (Zaidi 2024, Maia 2020, Ruoss 2024).

**A key warning from the literature:** in PARA's benchmark, 10-shot personalization ≈ no personalization. Most of an individual's
rating is consensus. **The personalization gap is small, and showing it is the whole point.**

## 5. Pre-registered success criterion (decided before any training)

For a chosen rater A, on A's held-out images/pairs:

1. `agree(critic_A, A)` > `agree(critic_consensus, A)`, i.e. the personal critic beats one trained on everyone;
2. `agree(critic_A, A)` > `agree(critic_B, A)` for other raters B, i.e. it captured A specifically;
3. both measured against **A's ceiling**. With no repeats in LAPIS, estimate it from split-half reliability and from how well the
   mean of the other raters predicts A (leave-one-rater-out consensus);
4. also check the critic's **top-ranked** items against A's ratings (best-of-n), not only pairwise accuracy (Gao 2022).

Metrics: pairwise accuracy on within-rater pairs (ties dropped), Spearman (SROCC) per rater. Report everything per rater, with
CIs, for the few selected "experts" plus a larger rater pool for statistics.

## 6. Steps (Phase A done 2026-11-22; Phase B not started)

**Phase A: data verification, no model training.**
1. Download the LAPIS tables (and later the images) into `data/lapis/` (keep the HF folder layout).
2. Per-rater statistics: number of ratings, rating scale and usage (does the rater use the whole scale?), response times,
   art-interest distribution, how many images pairs of raters share.
3. Build within-rater pairs from absolute ratings (drop ties; Pérez-Ortiz guide).
4. **Measure the ceilings:** for each rater, the agreement of the leave-one-out consensus with that rater, and split-half reliability.
   This tells us how much *individual* taste exists to learn before we train anything. If it's ≈0 for everyone, stop and rethink.
5. Select the candidate "experts": high art interest, many ratings, and **low agreement with consensus while staying internally consistent**.

**Phase A decisions (user, 2026-11-22)** — supersede the steps above where they differ:
- "Expert" = art education/profession, but LAPIS has no such field (only general `demo_edu`, `Art Interest VAIAK`, `imgs_known`).
  → **Select raters purely on distinct + consistent taste**; report VAIAK/`imgs_known` alongside, not as a filter. The user will later
  pick whom to mimic.
- Goal of Phase A: show that some raters *have* a taste that departs from consensus consistently, and find at least one for sure.
  Eventual pool ≈ 100 raters. If there is enough signal, stop and move to Phase B.
- The critic always predicts A's *full* preference; Phase A only tests whether A departs from consensus consistently.
- Every rating difference is a preference (only exact ties dropped). Use all data, no rater filtering at first (filter only if
  results look bad). All images (abstract + figurative).
- Phase A uses **only ratings + metadata** (genre/style labels, `image_attributes.csv` stats); no embeddings, no neural models.
- Method (final): leave-one-out consensus c; generic metadata prediction g of each image's panel mean (cross-fitted ridge);
  baseline = "consensus critic" = OLS of A's rating on [c, g]. A's deviation = residual from that baseline. Taste = the deviation is
  predictable from metadata out-of-sample across 20 random half-splits (ridge, λ=1000), permutation null (1000 perms, Gaussian-fitted
  p), BH-FDR 5% over raters. Effect size = gain in A's held-out pairwise accuracy of baseline+personal ridge over the baseline alone;
  report counts of significant raters at several gain thresholds. This is a **lower bound** (taste metadata can't describe is
  invisible). Validated by planted-taste synthetic raters and by pseudo-raters.
- Why the method changed during development (keep, so it isn't repeated): (1) λ=10 overfit (≈100 features, ≈240 images per half):
  no-taste synthetic raters *lost* ~5 pts; λ tuned on synthetic raters only (30…3000), 1000 best. (2) With c alone as baseline,
  95% of raters looked significant partly because the 23-rater LOO consensus is noisy and metadata helps predict the consensus
  itself → added g to the baseline (and later a non-linear baseline check) so only *personal* deviation counts.
- Found on load: 63 raters rated some images twice (614 repeated pairs) → small real test–retest check. Duplicates are averaged
  for the main analysis.

**§6b Phase A results (2026-11-22)** — `outputs/phase_a/{raters.csv,summary.json,checks.json,self_test.csv,phase_a.png}`
- Setup: 555 raters (553 testable, ≥40 ratings), 86 metadata features, ridge λ=1000 (tuned on synthetic raters only), 20 half-splits,
  1000 permutations, BH-FDR 5%. Baseline ("consensus critic") = OLS of A's rating on LOO consensus + generic metadata prediction.
- Noise context: consensus reliability (full panel, Spearman–Brown) 0.84; test–retest on 614 repeated pairs r=0.72, mean |diff| 13.5 pts.
  Median pairwise agreement of a rater with LOO consensus 0.67 (IQR 0.61–0.73).
- Validation: synthetic no-taste raters 1% p<.05, 0% FDR; planted 6/10-pt taste found 61%/94%. **Pseudo-raters** (each rating
  borrowed from a random other rater) 0% FDR-significant → the signal needs a consistent individual.
- **Result: 530/553 raters have significant consistent taste** beyond consensus+generic metadata. Held-out pairwise gain of the
  simple personal model over the baseline: median +2.7 pts (IQR 1.4–5.3, max 24.8). Raters significant with gain ≥1/2/3/5 pts:
  460/335/249/151. Survives a non-linear baseline (95% sig, median +2.4). Abstract/figurative label alone: 56% sig but median +0.17 →
  taste is broader than that axis. Taste is unrelated to art interest (Spearman VAIAK vs taste z = −0.08).
- Caveat: the largest gains come from raters who agree with consensus at chance (~0.50) yet are well predicted by metadata (0.73–0.79),
  often with coarse scale use (e.g. #1381: 12 distinct values). Consistent, but possibly simple heuristics rather than refined taste.
  141 raters are significant with gain ≥5 pts and pass basic quality flags (median RT ≥1.5 s, <30% of ratings at one value).

**Status / next step (2026-11-22):** Phase A says there is enough signal → per the user's rule, move to Phase B when the user says so.
Suggested before Phase B: look at a few top raters' highest/lowest-rated images to decide which kind of taste to mimic (see caveat).
Nothing committed to git yet; `outputs/` is not gitignored.

Rerun Phase A (≈1 min + ≈5 min checks):
```bash
docker compose run --rm taste python src/phase_a.py          # → outputs/phase_a/{raters.csv,summary.json,self_test.csv,phase_a.png}
docker compose run --rm taste python src/phase_a_checks.py   # → outputs/phase_a/checks.json (pseudo-raters, non-linear, abstract-only)
docker compose run --rm taste python src/phase_a.py --self-test-only --lam 1000   # λ tuning on synthetic raters
```

**Phase B: critics (GPU machine).**
6. Embed every image once with a frozen encoder (CLIP ViT-L/14 or SigLIP; DINOv2 as an alternative). Cache the embeddings to disk.
7. Consensus critic: BT/regression head on the embeddings, trained on all raters (official giaa split).
8. Personal critics: Ren-2017 style, consensus + small per-rater residual head, trained with BT loss on A's pairs. Vary the number of
   ratings per rater (10/50/100/all) to get a label-efficiency curve.
9. Evaluate with the §5 criterion.

**Phase B decisions (user, 2026-11-22)** — supersede steps 6–9 where they differ:
- **Pairwise only, everywhere.** In real use people will only answer "this or that", never give a score. LAPIS scores are used only
  to build within-rater pairs (exact ties dropped); no critic is ever trained on scores; all metrics are pairwise.
- Encoders: embed with CLIP ViT-L/14, SigLIP 2 and DINOv2 (frozen, cached). Pick by consensus-critic validation pairwise accuracy; report all.
- Consensus critic: Bradley–Terry head on embeddings, trained on within-rater pairs pooled over all raters. When evaluating rater A,
  **retrain with A left out** and evaluate only on A's held-out images, which are also excluded from consensus training (as in the Phase A LOO).
- Personal critic: Ren-2017 style. Frozen consensus score + per-rater linear residual on the embedding, BT loss, L2 toward 0.
- Raters: **a handful (5–10), chosen to be diverse in taste direction** (Phase A residual directions point different ways).
  Heuristic raters (§6b caveat) are kept, flagged and reported separately.
- Label efficiency is measured in **number of pairs** (e.g. 25/50/100/250/500/1000/all, random pairs from A's train images, several seeds).
- Train on all non-tied pairs; report accuracy **split by score gap** (e.g. <10, 10–30, >30 pts).
- Deliverables: `src/phase_b_*.py`, `outputs/phase_b/` (JSON/CSV/plots), CONTEXT §6c with the §5 verdict per rater, and a **visual sanity
  page** showing each selected rater's critic top/bottom images next to the consensus top images.

**§6c Phase B results (2026-11-22)** — `outputs/phase_b/{verdict.csv,summary.json,label_efficiency.csv,personal_curve.csv,cross_critics.csv,
best_of_n.csv,oof_scores.csv,selected.csv,consensus_select.csv,phase_b.png,sanity.html}`; sanity page https://claude.ai/artifact/TJQr2SvwPeWw4kdvqmUX9y (private).
- Scripts: `src/phase_b_embed.py` (embeddings → `embeddings/*.npz`), `src/phase_b_critics.py --stage all` (25 min on the 5070 Ti),
  `src/phase_b_page.py`. Checkpoints in `checkpoints/phase_b/` (gitignored).
- Encoders (consensus BT critic, giaa val/test, mean per-rater pairwise acc): CLIP-L/14 MLP 0.662/0.664, SigLIP2-SO400M MLP 0.662/0.663,
  DINOv2-L MLP 0.656/0.658; linear heads ~0.003–0.004 lower. **Chosen: CLIP ViT-L/14 (frozen) + MLP 768→512→1**, 4000 steps.
- Implementation details beyond the decisions: per-fold consensus excludes *all* 18 evaluated raters and all their test images (one consensus
  per fold, clean for criterion 2 too); inner CV for λ splits **by image** (splitting by pair leaked: the head memorises per-image utilities);
  noise ceiling = agreement of A's ratings with A's simulated *true* scores using the global noise var (191.9, from the 614 repeats).
- Raters: 8 selected (greedy farthest-point on metadata taste direction among 138 candidates: sig, Phase A gain ≥5, RT/scale flags, ≥300 ratings)
  + 10 random controls (≥300 ratings, not chosen on taste). Heuristic flag = acc with consensus <0.58 or <20 distinct values.
- **§5 verdict: all 8 selected and all 10 controls pass criteria 1, 2 and 4.**
  1. Personal vs consensus (all pairs, 5-fold, 95% bootstrap CI): selected median **+15.7 pts** (range +6.8 [201] to +33.0 [24]); controls
     median **+6.9** (+1.6 to +20.9). Every CI excludes 0.
  2. Own critic ranks 1st of 18 on its own rater for all 18 raters (e.g. 201: 0.844 own vs 0.713 best other).
  3. Fraction of the consensus→ceiling gap closed: median 0.77 (selected), 0.75 (controls). The ceiling is rough: control #6 exceeds it (0.893 vs
     0.857), i.e. the global noise level is too pessimistic for consistent raters.
  4. Best-of-n (top 10% of A's held-out images): mean A-z of personal picks 0.86–1.55 vs consensus picks −0.10–0.98 (selected); oracle 1.1–1.9.
- Label efficiency (median gain, selected / controls): 25 pairs +1.5/+0.6, 100 +6.6/+1.5, 250 +12.2/+4.0, 1000 +14.5/+6.4, all +15.7/+6.9.
  Most of the gain arrives by ~250–500 pairs; CV falls back to consensus-only at 25 pairs in ~40% of fits.
- By score gap (selected, consensus → personal): <10 pts 0.52→0.56, 10–30 0.56→0.70, >30 0.63→0.86.
- Readings / caveats: (a) embeddings expose far more individual taste than Phase A metadata (controls +6.9 vs Phase A median +2.7);
  (b) anti-consensus raters (24, 1303) get β≈0–0.2, i.e. their critic ignores consensus rather than inverting it; (c) the selected group was
  chosen on Phase A results from the same data → some optimism; controls are the unbiased estimate; (d) the personal model is linear on frozen
  CLIP; gains plateau after ~1000 pairs with ~25% of the ceiling gap left, so a richer per-rater head (MLP / shared low-rank) is the obvious
  next test; (e) consensus training-loss curves were not saved (only val acc).

**Phase C decisions (user, 2026-11-22): shared low-rank taste model.** Goal: better sample efficiency (Phase B gains only
+0.6/+1.5 pts at 25 pairs) and higher accuracy (Phase B plateaus after ~1000 pairs with ~25% of the ceiling gap left).
- Architecture: **low-rank taste basis** (LoRe 2025 / PAL style). `s_u(x) = c(x) + z_u · V(f(x)) + w_u · f(x)`, where f = image
  encoder, c = consensus head, V = shared MLP → K taste axes trained on many raters, z_u = K per-person weights, and w_u = a free
  per-person residual with a strong prior toward 0 (hierarchical). With few pairs only z_u moves; with many, w_u can capture taste
  outside the basis (needed because the eventual experts are not LAPIS raters).
- Encoder: **frozen cached CLIP-L/14 first**, then **LoRA-tune the shared encoder** on the training raters as a second step.
- Rater split: shared trunk/basis trained on **all raters except the 18 Phase B raters** (and without their test images). Validation
  raters for model selection are taken from those ~537 raters, never the 18. Evaluation uses the same 18 raters and folds as Phase B,
  so the comparison is direct.
- K: sweep {1,2,4,8,16,32,64}, pick on validation raters, and report the whole curve (it also measures how many taste dimensions LAPIS has).
- Pair selection: **random pairs first** (same budgets as Phase B), **then active learning** (choose the most informative pair for z_u,
  e.g. expected information gain under BT).
- Metrics: report all-pairs accuracy for continuity, but the **headline is pairs with score gap ≥10**; re-estimate the ceiling
  **per rater** (the global-noise ceiling is too pessimistic: control #6 exceeds it).
- Literature added to `papers/INDEX.md` (2026-11-22): LoRe 2025, PAL (ICLR 2025), Zhu 2020 (no PDF), ED-PBRL 2025 (ETH; chosen as
  the active-learning paper, builds on Bıyık/Sadigh), plus PAM∃LA (Maerten 2026, LAPIS authors: closest prior work).

**§6d Phase C results (2026-11-22, frozen CLIP, random pairs)** — `outputs/phase_c/{ksweep.csv,ksweep_val_grid.csv,kstar.json,
curve.csv,per_rater_budget.csv,verdict.csv,chosen_cfg.csv,cross_critics.csv,best_of_n.csv,summary.json,phase_c.png}`;
script `src/phase_c_lowrank.py --stage all` (~70 min on the 5070 Ti). Checkpoints `checkpoints/phase_c/shared_K8_fold*.pt` (U, Z, train history).
- Implementation notes: validation = 40 raters (≥300 ratings, not in the 18) evaluated on one shared pool of 1000 held-out images
  (per-rater 20% splits removed ~1/3 of all images from training, so they were dropped). Shared model per fold: 497 raters, ~182k ratings
  (Phase B consensus also had the 40 val raters). MLP 768→512→K+1, 6000 steps, early stop on 5% held-out images. New user: MAP of z under
  N(μ, Σ) estimated from training raters' Z (prior scaled by τ), residual w with L2 λ_w; τ and λ_w picked per budget on val raters.
  Same folds/test pairs as Phase B (asserted).
- Training: BT loss 0.54→0.47; held-out-image accuracy plateaus at 0.73–0.74 by step ~2000, no overfitting.
- **K sweep (val raters):** held-out accuracy of training raters rises to K=8 (0.742) then flat (K=64: 0.744); few-shot val score best at
  **K*=8** (16–64 slightly worse). → LAPIS taste is ~8-dimensional on frozen CLIP. K=0 (consensus + residual only) is clearly worse at small budgets.
- **Few-shot (headline: pairs with gap ≥10, median over raters, Phase B → Phase C):**
  controls 25 pairs 0.721→0.744, 50 0.742→0.769, 100 0.754→0.781, 250 0.779→0.789, all 0.799→0.801 (consensus 0.669);
  selected 25 0.642→0.697, 50 0.675→0.715, 100 0.707→0.727, 250 0.749→0.755, all 0.785→0.776 (consensus 0.612).
  Phase C at 50–100 pairs ≈ Phase B at 250–500 pairs → **~3–5× fewer pairs**. Improvement holds for 9–10/10 controls at ≤1000 pairs.
- **No gain at full data**: controls +0.1 pts, selected −0.45 (991, 1357, 1411 lose 1.5–2.5 pts; λ_w picked on average val raters is
  too strong for very idiosyncratic raters). The plateau is not broken → frozen features are the likely limit → next: LoRA (decided step).
- Criterion 2 still holds (own critic rank 1 for 18/18); best-of-n similar to Phase B.
- Training-loss curves of the K sweep were not saved in this run (fixed in the script for future runs: `ksweep_train_curves.csv`).
- External reference (not comparable protocol): PAM∃LA (Maerten 2026) unseen users 0.663 pairwise all pairs / ~0.80 without near-ties,
  15 context images, on its own AI-image dataset. LAPIS paper PIAA baselines report pooled SROCC (0.70 seen / 0.28 unseen users), not
  per-user accuracy. No SOTA claim possible without running a shared protocol.

**§6e Encoder fine-tuning — tried and dropped (2026-11-23).** LoRA (rank 16, last 8 of 24 CLIP blocks, 2.1M params) inside the
Phase C model. From scratch the head was undertrained; with linear-probe-then-fine-tune (head on frozen features first, then LoRA +
head) held-out-image accuracy rose only +0.2 pts for 1–2 epochs (fold 0 0.7416→0.7435, fold 1 0.7341→0.7359, fold 2 +0.0) and then
overfit; on the 18 raters (fold 0) LoRA tied frozen within ±1 pt. ~10k images is too little to improve CLIP for this. **User decision:
stopped, code deleted; do not retry without much more data.** The frozen encoder is not the bottleneck.

**§6f Active pair selection (2026-11-23)** — `src/active.py --seeds 5`; `outputs/active/{curve_C.csv,curve_B.csv,per_rater.csv,
summary.json,active.png}` (~42 min). Simulated "this or that" sessions: 18 raters × 5 folds × 5 seeds, pool = the rater's training images
(tied pairs not askable), same 5 random start pairs for every strategy, batches up to 250 pairs, refit after each batch, same test images
as Phase B/C. Strategies: random, uncertainty (P≈0.5), infogain (greedy D-optimal on the Laplace posterior of the person's parameters:
score p(1−p)·dᵀH⁻¹d). Phase C uses its fold checkpoints and val-chosen τ/λ_w; Phase B retrains its consensus per fold and picks λ by
inner CV at 10/25/50/100/250 pairs.
- **Phase C + infogain is best at every budget.** Gap≥10, median, controls: random/infogain 10 pairs 0.731/0.746, 25 0.755/0.769,
  50 0.768/0.782, 100 0.781/0.791; selected 25 0.692/0.722, 50 0.711/0.735. Infogain beats random for 10/10 controls at 10–25 pairs and
  8/8 selected at 15–35; reaches random's 250-pair accuracy at 150 pairs; never worse.
- **Phase B gains little from infogain** (selected +1–2 pts at 10–35 pairs, controls ≈0) and loses at 250 pairs (−1.5 / −2.5): with
  769 parameters a few chosen pairs cannot cover the space, and the actively chosen (non-representative) pairs seem to hurt Phase B's
  λ cross-validation.
- **Uncertainty sampling hurts both models** (asks near-ties).
- Overall: Phase C + infogain at 25 pairs ≈ Phase B + random at ~150–200 pairs → **~7–8× fewer questions** than the Phase B start.
- Also: `src/axes_page.py` → `outputs/phase_c/axes.html`, top/bottom LAPIS images per shared axis (fold 0), published privately at
  https://claude.ai/artifact/WEVBtxXDxtwHtZWR25qw58. Each axis is one scalar per image (MLP 768→512→9); a person = 9 weights.

**Status / next step (2026-11-23):** Remaining decided step: per-rater ceiling (how much accuracy is left on LAPIS). Commit 1ca9cc8 and
later commits are not pushed: no GitHub credentials on this machine (SSH key not on the account, no gh CLI).

## 7. Environment (GPU machine, set up 2026-11-22)

- Machine: RTX 5070 Ti (16 GB, Blackwell, driver 580), 60 GB RAM. Docker 29 with the NVIDIA runtime.
- **Everything runs in Docker** (user decision, for reproducibility across machines and an easy port to Euler via Apptainer). No `.venv`.
  - `Dockerfile`: `pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime` + `requirements.txt` (pinned pandas, numpy, scipy, matplotlib,
    open_clip_torch, huggingface_hub). The build fails if pip replaces the base image's CUDA torch.
  - `compose.yaml`: bind-mounts the repo at `/workspace` (so `data/` and outputs stay on the host) and `~/.cache/huggingface`
    for model weights; runs as the host user, all GPUs.
  - Usage: `docker compose run --rm taste` (shell) or `docker compose run --rm taste python src/<script>.py ...`.
  - One-time per machine: `printf 'UID=%s\nGID=%s\n' $(id -u) $(id -g) > .env` (gitignored) and `mkdir -p ~/.cache/huggingface`.
- Verified in the container: torch 2.15.0+cu132 sees the GPU and runs a test calculation; `data/` and the HF cache are readable/writable.
- **Data in `data/lapis/`** (HF layout): all tables downloaded complete (`piaa.csv` 83.6 MB). Images: all **11,723** downloaded (707 MB); every `path` in `piaa.csv` exists and opens with PIL (checked in the container).
- **Papers:** all 14 PDFs fetched with `bash papers/download.sh`.
- **Two corrections to §3 found on load:** `piaa.csv` has **555** distinct `participant_id`s (paper says 552), and ratings are
  on a **0–100** scale (mean 48.5, sd 26.9), not 1–5.
- Phase A done 2026-11-22: `src/phase_a.py` (main) + `src/phase_a_checks.py` (robustness); outputs in `outputs/phase_a/`. See §6b.

Fetching the data again on a new machine:
```bash
mkdir -p data/lapis && cd data/lapis
for f in piaa/piaa.csv giaa/train.csv giaa/validation.csv giaa/test.csv image_attributes/image_attributes.csv image_attributes/image_prompts.csv; do
  mkdir -p $(dirname $f); curl -sSfL -o $f "https://huggingface.co/datasets/asmaerten/LAPIS/resolve/main/$f"; done
cd ../.. && docker compose run --rm taste python -c "from huggingface_hub import snapshot_download as s; s('asmaerten/LAPIS', repo_type='dataset', allow_patterns='images/*', local_dir='data/lapis')"
```
Paper PDFs are not in git: run `bash papers/download.sh` to fetch them all.
Keep analysis scripts outside `data/` (e.g. in `src/`), and pass data paths as arguments.

## 7b. "This or That" web app (2026-11-23) — `app/`, export `src/app_export.py`

iPhone-first static web app: 25 picks (5 random + 20 infogain) fit a new person's z (9 numbers) on the frozen Phase C model
(fold 0, K=8, z-only, tau=0.5 = validation choice at 25 pairs), then 10 random pairs where the frozen model's guess is fixed
before the pair is shown, then a reveal with "model score for you" and "crowd" percentiles. Results: score, crowd-model baseline
on the same pairs, top-3 taste traits with example images, model picks, share card (PNG via Web Share API) and a challenge
link (`#c=` carries name, score and z → friend sees "taste match %" = correlation of the personal score parts).
- User decisions: all 11,723 LAPIS images; no skip button; 10 random test pairs; model frozen during the test; shareable result;
  local first, Vercel later; data storage later (sessions are kept in localStorage for now, `saveSession()` is the hook).
- Presentation (from the literature check): position effects in simultaneous choice are real but inconsistent (Bar-Hillel 2015);
  swipe direction has a valence association (rightward = positive in app conventions) → **tap to choose**, images **stacked**
  in portrait (full width each; side by side in landscape), sides randomised every pair, a per-person side-bias term b in the
  fit (prior sd 0.5) so side preference cannot leak into z, both images appear only when both are decoded, taps < 450 ms ignored.
- Axis pole names (read from top/bottom images): 0 crowd-pleaser/against the grain, 1 vivid sensual painting/quiet minimalism,
  2 raw gestural/polished detail, 3 old-master portraits/the human body, 4 open landscapes/intimate figures, 5 baroque drama/clean
  colour fields, 6 busy geometry/near-empty canvases, 7 playful naive figures/explosive abstraction, 8 lush florals/stark B&W.
- 873 images flagged as nudity from the captions: never on the share card; `?sfw=1` removes them from the pool.
- Checks: JS MAP fit = Python `fit_users` to 6e-7; simulated users from the population prior: infogain 0.857 / random 0.823 /
  consensus 0.745 agreement with the true preference; full flow tested in WebKit (iPhone 15 emulation), light/dark, landscape.
- Run: `python3 app/serve.py` → open `http://<LAN IP>:8000` on the phone (same Wi-Fi). Re-export: `docker compose run --rm taste
  python src/app_export.py`. Vercel plan: app as static files; images in Vercel Blob (set `imgBase` in `app/config.js`).

## 8. Working preferences of the user
- Decisions are made step by step. Verify before training. Do not start step 2 (RL/generator) unless asked.
- Science-backed choices only, from the best papers. Keep the reading list small and update `papers/INDEX.md` when it changes.
- The user prefers to choose the direction (e.g. picked LAPIS over PARA, dropped chess). Propose a recommendation and let them decide.
