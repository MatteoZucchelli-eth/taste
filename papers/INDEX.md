# Literature index: learning an individual's taste as a pairwise critic

**Research question.** Using only one person's own judgments, can we learn a critic that predicts *that person's*
preference between two images better than a consensus model does, approaching how consistent the person is with themselves?
End goal: design generation. Later step (not started): use the critic as the reward for a generator (diffusion).

**Success criterion (pre-registered, before any training).** On held-out pairs for rater A:
`agree(critic_A, A)` > `agree(critic_consensus, A)` and > `agree(critic_B, A)`, measured against A's
self-consistency (test–retest) as the ceiling. Check the critic's *top-ranked* items too, not only pairwise accuracy (Gao 2022).

Selection rule: one paper per question we actually have to answer. If a paper stops being useful, move it out.

---

## 1. Data: personalized aesthetics datasets

### Yang et al. 2022 — *PARA: Personalized Image Aesthetics Assessment with Rich Attributes* (CVPR) · arXiv 2203.16754
`Yang2022_PARA.pdf`
31,220 photos and 438 raters. Each image is scored by ~25 raters, so each rater scores ~1.8k images on average (446 sessions × 80 images).
Scores are 1–5 for aesthetics plus attributes (composition, light, color, depth of field, content, "I like the content", willingness to share),
along with rater info: age, education, Big-Five personality, **artistic and photographic experience**.
**Each session has 5 repeated images**, so per-rater test–retest consistency is built in. That is our ceiling.
**Official PIAA protocol:** 40 test raters, fine-tune a generic model on 10 or 100 of the rater's images, evaluate on 50 (SROCC/PLCC).
**Key warning (Table 3):** 10-shot personalization ≈ no personalization. Most of an individual's score is
consensus. The personalization gap is the thing we have to show, and it is small.
**Use:** main dataset. "Experts" = raters with high photographic/artistic experience.
**Access:** request through https://cv-datasets.institutecv.com (registration; terms not yet read). No public mirror found.

### Maerten et al. 2025 — *LAPIS: Leuven Art Personalized Image Set* (CVPR-W) · arXiv 2504.07670
`Maerten2025_LAPIS.pdf`
11,723 **artworks**, curated with art historians (styles from abstract to portrait). 552 Prolific raters, ~24 per image (~500 per rater).
Per-rater **art-interest score** plus demographics; 31 image attributes. Novices and art-interested raters disagree in systematic ways (e.g. on abstract art).
**Use:** closer to design than PARA's photos. **Open download now:** `huggingface.co/datasets/asmaerten/LAPIS` (ungated, CC BY-NC).
No repeated images, so the ceiling has to come from split-half / inter-rater agreement.

## 2. Personalization method and protocol

### Ren et al. 2017 — *Personalized Image Aesthetics* (ICCV)
`Ren2017_PersonalizedImageAesthetics.pdf` (CVF open access, not on arXiv)
The founding PIAA paper. Introduces FLICKR-AES (40k images, 210 AMT raters, 5 per image) and the standard idea
**personal score = generic score + small learned residual per user**, from a few of the user's ratings. Also active learning to choose which images to ask a user about.
**Use:** the baseline architecture (generic backbone + per-person head) and the few-shot protocol. Active learning matters
once we label with real experts, whose time is scarce.

## 3. Learning a critic from pairwise comparisons (the core method)

### Christiano et al. 2017 — *Deep RL from Human Preferences* (NeurIPS) · arXiv 1706.03741
`Christiano2017_DeepRLHumanPreferences.pdf`
The foundation for learning a reward from pairwise judgments. P(a ≻ b) = σ(r(a) − r(b)) (Bradley–Terry), cross-entropy loss,
ensembles, disagreement-based query selection.
**Use:** the loss and the critic template.

### Stiennon et al. 2020 — *Learning to Summarize from Human Feedback* (NeurIPS) · arXiv 2009.01325
`Stiennon2020_LearningToSummarizeHF.pdf`
The clearest study of whether a learned pairwise reward model agrees with humans. It compares reward-model accuracy against
**inter-labeler agreement as the ceiling** and studies scaling with data and model size.
**Use:** our evaluation framing. Always report the critic next to the human ceiling.

### Pérez-Ortiz & Mantiuk 2017 — *A Practical Guide and Software for Analysing Pairwise Comparison Experiments* · arXiv 1712.03686
`PerezOrtiz2017_PairwiseComparisonGuide.pdf`
How to turn comparisons into scales (Thurstone / Bradley–Terry), with confidence intervals, outlier raters and few-observer priors,
and why comparisons are more reliable than absolute ratings.
**Use:** (a) PARA and LAPIS give absolute 1–5 scores. Convert them to within-rater pairs (ties dropped) the principled way.
(b) Design the later expert labelling study (how many pairs).

## 4. Learned taste critics for generative images (and the bridge to diffusion)

### Kirstain et al. 2023 — *Pick-a-Pic / PickScore* (NeurIPS) · arXiv 2305.01569
`Kirstain2023_PickAPic_PickScore.pdf`
Real users' pairwise preferences over text-to-image outputs. A CLIP backbone plus a Bradley–Terry objective predicts preferences
**better than individual humans do** and tracks human rankings better than FID.
**Use:** proof that a frozen or fine-tuned CLIP plus a pairwise head works. This is the consensus-critic baseline.

### Xu et al. 2023 — *ImageReward* (NeurIPS) · arXiv 2304.05977
`Xu2023_ImageReward.pdf`
A reward model trained on 137k expert comparisons, then used to fine-tune diffusion directly (ReFL).
**Use:** the expert-annotation pipeline, and the template for step 2 (critic → diffusion).

## 5. Risks and motivation

### Gao, Schulman & Hilton 2022 — *Scaling Laws for Reward Model Overoptimization* (ICML 2023) · arXiv 2210.10760
`Gao2022_RewardModelOveroptimization.pdf`
Optimising a learned proxy too hard lowers true quality, and the effect scales predictably.
**Use:** now, evaluate the critic's top picks (best-of-n against the rater). Later, KL and early stopping in step 2.

### Jiang et al. 2025 — *Artificial Hivemind* (NeurIPS 2025) · arXiv 2510.22954
`Jiang2025_ArtificialHivemind.pdf`
LM outputs are homogeneous. LMs, reward models and LM judges are miscalibrated exactly where individual human preferences diverge.
**Use:** motivation. A consensus critic erases taste, which is why we model individuals.

### Feng et al. 2025 — *Generating Creative Chess Puzzles* (NeurIPS 2025) · arXiv 2510.23881 · `Feng2025_GeneratingCreativeChessPuzzles.pdf`
The original inspiration. RL with a **hand-designed** creativity scorer. Their scorer tracked difficulty but had ~0 correlation
with user popularity (Table 6), and optimising it led to reward hacking (Sec 4, App. H).
**Use:** the motivating contrast. A hand-written scorer misses taste; we try to learn it. RL tricks (KL, diversity filters) are for step 2.

---

## Moved out
`chess_archive/`: Zaidi & Guerzhoy 2024 (move brilliance), Maia 2020, Ruoss 2024. Chess-specific, set aside after we
dropped chess (2026-11-22). Bring them back only if we return to chess.

## Referenced, no PDF
- **Bradley & Terry 1952**, Biometrika. The model behind every pairwise loss above.

## Considered and left out
NIMA, AVA (aggregate scores only, no per-rater IDs) · FLICKR-AES data (5 raters/image, too sparse for rater-vs-rater analysis; PARA supersedes it) ·
RewardBench / LLM-as-a-judge (add only if we test a VLM judge as a baseline) · HPSv2 (redundant) · DPO / RL algorithms (step 2).

## Open question for later
The end goal is *design* (layouts, graphics), not photos or paintings. We will need a design-specific preference dataset, or our own
expert labels, once the method is validated on PARA/LAPIS. Not researched yet.
