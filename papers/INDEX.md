# Literature index: learning an individual's taste as a pairwise critic

**Research question.** Using only one person's own judgments, can we learn a critic that predicts *that person's*
preference between two images better than a consensus model does, approaching how consistent the person is with themselves?
End goal: design generation. Step 2 (started 2026-11-23, see §6): use the critic as the reward for a generator (diffusion), as a simple proof of concept with strong KL regularisation.

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

### Maerten et al. 2026 — *Personalizing Text-to-Image Generation to Individual Taste (PAM∃LA)* · arXiv 2604.07427
`Maerten2026_PAMELA.pdf`
From the LAPIS authors (Tübingen + KU Leuven). A dataset of 5k AI-generated images × 205 raters, plus a personalised reward model:
frozen SigLIP2, a learnable embedding per user, demographics, and a small fusion transformer, trained jointly on PAM∃LA + **LAPIS** + PARA.
Unseen users are handled by interpolating the embeddings of nearest training users from 15 rated images. User pairwise accuracy is 0.663;
it reaches ~0.80 once near-ties are excluded with a margin. It then steers FLUX through prompt optimisation, and a user study finds people prefer images steered by their own model.
**Use:** the closest prior work, and the external reference point for our few-shot numbers. It also backs our "headline on gap ≥ 10" choice and is a template for step 2.

### Zhu et al. 2020 — *Personalized Image Aesthetics Assessment via Meta-Learning with Bilevel Gradient Optimization* (IEEE T-Cybernetics)
No PDF (not open access).
The standard few-shot PIAA baseline: meta-learn an initialisation of the aesthetic model across users, then adapt it to a new user with a few
gradient steps from 10/100 images.
**Use:** positions our low-rank model against the meta-learning family (the alternative we did not choose for Phase C).

### Bose et al. 2025 — *LoRe: Personalizing LLMs via Low-Rank Reward Modeling* · arXiv 2504.14439
`Bose2025_LoRe.pdf`
Each user's reward is a weighted combination of a few shared basis reward functions learned across users. A new user only fits
the low-dimensional weights from a few comparisons. Shows few-shot generalisation to unseen users.
**Use:** the template for the Phase C architecture (shared basis + per-user weights).

### Chen et al. 2024 — *PAL: Pluralistic Alignment Framework for Learning from Heterogeneous Preferences* (ICLR 2025) · arXiv 2406.08469
`Chen2024_PAL.pdf`
Ideal-point / mixture model: shared mapping and K prototypes, with each user a mixture of them. Trained on frozen foundation-model embeddings with a
small MLP, and evaluated on seen vs **unseen** users. Pick-a-Pic results use CLIP-H embeddings. With too few pairs per user, extra prototypes hurt,
and the pairs a new user needs grow with K.
**Use:** backs frozen embeddings + MLP + per-user weights, the K sweep, and the unseen-user protocol.

### Schacht, …, Krause 2025 — *Efficient Personalization of Generative Models via Optimal Experimental Design (ED-PBRL)* · arXiv 2512.19057
`Schacht2025_EDPBRL.pdf`
Chooses preference queries that maximise information about a latent BT reward (optimal experimental design, convex formulation,
guarantees). Personalises a text-to-image model to user styles with fewer queries than random. Builds on Sadigh 2017 / Bıyık 2020 active preference learning.
**Use:** backs the active pair-selection step for the per-user weights.

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

## 6. Step 2: RL fine-tuning of a diffusion generator against the critic (added 2026-11-23)

Step 2 is a proof of concept. A pretrained text-to-image model is the actor, the personal critic (§2–3) is the reward, and a
**strong KL penalty toward the base model** limits how far the actor can move. The objective in every paper below has the same form:
`max E[r(x)] − β·KL(p_θ ‖ p_base)`, whose optimum is the "tilted" distribution `p_base(x)·exp(r(x)/β)`.
Each paper answers a different design question. ImageReward/ReFL (Xu 2023, §4) and Gao 2022 (§5) also apply here.

### Uehara, Zhao, Biancalani & Levine 2024 — *Understanding RL-Based Fine-Tuning of Diffusion Models: A Tutorial and Review* · arXiv 2407.13734
`Uehara2024_RLDiffusionTutorial.pdf`
A survey that puts every family into one KL-regularised framework: PPO-style policy gradient, direct reward backprop, reward-weighted MLE,
value-weighted sampling, and path consistency. For each it covers the target distribution, when the reward must be differentiable, and the strengths and weaknesses.
**Use:** the map of the design space, and the shared notation for the KL-regularised objective.

### Black, Janner, Du, Kostrikov & Levine 2023 — *Training Diffusion Models with RL (DDPO)* (ICLR 2024) · arXiv 2305.13301
`Black2023_DDPO.pdf`
Treats the denoising chain as a multi-step MDP (each step is a Gaussian action with a tractable log-probability) and applies policy gradient
(REINFORCE or a PPO-style clipped ratio). Uses SD 1.4 with LoRA. Rewards are compressibility, the **LAION aesthetic predictor (a linear head on CLIP
embeddings, which has the same shape as our critic)**, and VLM-judged prompt alignment. Works across hundreds of prompts and generalises to unseen ones.
It **has no KL term and leaves overoptimization out of scope** (App. A).
**Use:** the reference online-RL formulation. The reward only needs to be evaluated, not differentiated.

### Fan, Watkins, Du, …, Lee 2023 — *DPOK: RL for Fine-tuning Text-to-Image Diffusion Models* (NeurIPS 2023) · arXiv 2305.16381
`Fan2023_DPOK.pdf`
Online policy gradient plus **KL regularisation to the pretrained model**, using an upper bound of the KL that splits into a sum of per-step KLs
between Gaussian denoising steps (cheap to compute). Compares KL-regularised RL with KL-regularised supervised fine-tuning, with ImageReward as the reward on SD 1.5 + LoRA.
KL keeps image quality from collapsing while the reward goes up.
**Use:** the template for how to add and tune the KL term (β) in policy-gradient diffusion RL.

### Clark, Vicol, Swersky & Fleet 2023 — *Directly Fine-Tuning Diffusion Models on Differentiable Rewards (DRaFT)* (ICLR 2024) · arXiv 2309.17400
`Clark2023_DRaFT.pdf`
Backpropagates the reward gradient through the sampling chain: full chain, the **last K steps only (DRaFT-K)**, or K=1 with lower variance
(DRaFT-LV). It is more sample-efficient than RL by orders of magnitude when the reward is differentiable. Uses SD 1.4 + LoRA, with aesthetic and HPSv2 rewards.
It reports reward hacking (diversity collapse). As a regulariser, **scaling the LoRA weights interpolates between base and fine-tuned models
and worked better than KL or early stopping** in their tests.
**Use:** the gradient-based alternative. It applies because our critic (frozen CLIP + MLP) is differentiable end to end.

### Wallace, Dang, Rafailov, …, Naik 2023 — *Diffusion Model Alignment Using Direct Preference Optimization (Diffusion-DPO)* (CVPR 2024) · arXiv 2311.12908
`Wallace2023_DiffusionDPO.pdf`
DPO adapted to diffusion through the ELBO. It trains directly on **pairwise preferences** (Pick-a-Pic, 851k pairs) with no explicit reward model and
no online sampling. β is the implicit KL strength toward the reference model. Used on SD 1.5 and SDXL.
**Use:** the offline / pairwise alternative to online RL. Pairs could come from the critic ranking the base model's samples, or straight from the person.

### Liu, Liu, Liang, …, Ouyang 2025 — *Flow-GRPO: Training Flow Matching Models via Online RL* (NeurIPS 2025) · arXiv 2505.05470
`Liu2025_FlowGRPO.pdf`
GRPO (group-relative advantages, no value network) for **flow-matching** models such as SD3.5-M. It needs (1) an ODE→SDE conversion with the same marginals so that
sampling is stochastic and has log-probabilities, and (2) "denoising reduction": few steps during training, the full step count at inference. It shows that
**a KL constraint prevents reward hacking and is not equivalent to early stopping**. GenEval goes from 63% to 95%, and it also reports PickScore alignment. Code is public.
**Use:** the policy-gradient recipe for a modern flow-matching base model, with its KL ablation.

### Domingo-Enrich, Drozdzal, Karrer & Chen 2024 — *Adjoint Matching* (ICLR 2025) · arXiv 2409.08861
`DomingoEnrich2024_AdjointMatching.pdf`
Frames KL-regularised reward fine-tuning as stochastic optimal control. It proves that **naively adding a KL term does not reach the tilted distribution**
(the "initial value function bias") unless fine-tuning uses a specific *memoryless* noise schedule, and it gives a regression-style algorithm. Works for both
flow matching and diffusion. Reports better realism, sample diversity and generalisation to unseen reward models than prior methods.
**Use:** the theory of what "strong KL regularisation" actually converges to, and why the noise schedule matters for that.

### Zhang, Zhang, Zhan, Luo, Wen & Tao 2024 — *Confronting Reward Overoptimization for Diffusion Models (TDPO-R)* (ICML 2024) · arXiv 2402.08552
`Zhang2024_TDPO_RewardOveroptimization.pdf`
Diffusion-specific reward overoptimization. It optimises one learned reward and measures others it never trained on (cross-reward evaluation), uses per-timestep rewards (temporal
inductive bias), and resets active neurons in the critic.
**Use:** a protocol for detecting overoptimization in diffusion: score the actor's images with held-out rewards that were not optimised (Gao 2022 transposed to images).

---

## Moved out
`chess_archive/`: Zaidi & Guerzhoy 2024 (move brilliance), Maia 2020, Ruoss 2024. Chess-specific, set aside after we
dropped chess (2026-11-22). Bring them back only if we return to chess.

## Referenced, no PDF
- **Bradley & Terry 1952**, Biometrika. The model behind every pairwise loss above.

## Considered and left out
NIMA, AVA (aggregate scores only, no per-rater IDs) · FLICKR-AES data (5 raters/image, too sparse for rater-vs-rater analysis; PARA supersedes it) ·
RewardBench / LLM-as-a-judge (add only if we test a VLM judge as a baseline) · HPSv2 (redundant) · AlignProp (Prabhudesai 2023, 2310.03739: reward backprop, which DRaFT covers) · Uehara et al. 2024 entropy-regularised control (2402.15194: the tutorial covers it).

## Open question for later
The end goal is *design* (layouts, graphics), not photos or paintings. We will need a design-specific preference dataset, or our own
expert labels, once the method is validated on PARA/LAPIS. Not researched yet.
