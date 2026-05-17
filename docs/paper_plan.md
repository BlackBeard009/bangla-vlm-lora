# Paper Plan: LoRA Fine-Tuning of Vision-Language Models for Bangla Image Captioning

**Target:** Q1 journal (IEEE Access / ACM TALLIP / Multimedia Tools and Applications)
**Compute budget:** Google Colab free tier (T4 GPU primary; TPU as future work)
**Date:** 2026-05-17

---

## 1. Confirmed Research Gaps (verified by literature survey)

| # | Gap | Evidence |
|---|---|---|
| G1 | No peer-reviewed paper applies LoRA to an end-to-end generative VLM for Bangla image captioning. Closest is a 2025 ResearchGate manuscript on CLIP (encoder-only, contrastive) and BiCap 2025 (ResNet+LSTM). | aclanthology.org/2025.banglalp-1.6; ResearchGate 392102021 |
| G2 | GiT + PEFT integration is documented as broken (PEFT's `task_type` does not cleanly support GiT; naive LoRA "doesn't learn"). No published fix. | huggingface/peft discussions #1958 |
| G3 | Tokenizer fertility and embedding initialization for Bangla on VLMs is unstudied. Existing multilingual VLM extension work targets Hindi/Urdu/Tamil at scale, never Bangla in a PEFT setting. | Survey returned zero hits |
| G4 | No Bangla captioning human-judgment dataset exists (no Flickr8k-Expert or Polaris equivalent). | Survey returned zero hits |
| G5 | Multilingual CLIPScore and learned metrics (Polos, FLEUR) are unvalidated for Bangla. Both are English-trained; correlation with Bangla human ratings is unknown. | Polos arXiv:2402.18091; FLEUR ACL 2024 |
| G6 | All prior Bangla captioning work evaluates only with n-gram metrics (BLEU/METEOR/CIDEr/ROUGE). | Bornon, BiCap, BAN-Cap baselines |

---

## 2. Proposed Contributions (four pillars)

**C1 — Method.** A LoRA-based adaptation recipe for end-to-end VLMs to Bangla, including a vocabulary/embedding bridge (tokenizer extension + smart embedding initialization). Ablate across init strategies: random / mean / FOCUS / WECHSEL / multilingual-aligned.

**C2 — Engineering artifact.** A working GiT + PEFT integration with a fix for the documented `task_type` issue (#1958), released as code. Converts a known community blocker into a reproducible asset.

**C3 — Evaluation framework.**
- A Bangla caption human-judgment benchmark: 200 images × ~5 system outputs × 3 annotators on adequacy + fluency Likert, with Krippendorff α reported.
- First Bangla validation of M-CLIPScore, BERTScore-bn, and Polos / FLEUR against human ratings.
- Show metric disagreement and recommend a Bangla evaluation protocol.

**C4 — Empirical study.** Comprehensive results across BAN-Cap, BanglaLekha, BNATURE, and the Bengali split of XM3600 (held-out, native-annotated). Baselines: CNN-LSTM (Chittron-style), transformer (Bornon-style), zero-shot multilingual VLM (Qwen2-VL-2B or PaliGemma-3B), and our LoRA-adapted models.

The combination — PEFT + embedding bridge + new evaluation framework — is what makes this Q1-defensible rather than just "yet another LoRA paper."

---

## 3. Model Strategy (Colab-feasible)

| Role | Model | Params | Why | Fits T4? |
|---|---|---|---|---|
| Primary | GiT-base | 347M | Small; no Bangla tokens → makes embedding-bridge contribution meaningful; documented PEFT issue lets us claim C2 | Yes, easily |
| Secondary | BLIP-base | 224M | Cross-architecture confirmation that the recipe generalizes | Yes |
| Strong baseline | PaliGemma-3B (or Qwen2-VL-2B) + QLoRA | 3B / 2B | Tokenizer already has Bangla → isolates the value of our embedding bridge | Yes with QLoRA |
| Zero-shot baseline | Same as above, no fine-tuning | — | Establishes lower bound | n/a |

**Compute notes:**
- Free Colab TPU has patchy HF support for GiT-style models. Plan around T4 GPU as primary path.
- GiT-base + LoRA on BAN-Cap ≈ < 1 hour/epoch on T4.
- PaliGemma-3B + QLoRA fits in ~12–14 GB VRAM.
- Whole experimental matrix is feasible in tens of GPU-hours.

---

## 4. Datasets

**Training / evaluation:**
- BAN-Cap (8,091 images, native annotation — most rigorous Bangla resource)
- BanglaLekhaImageCaptions (9,154 images, 2 refs/img — note CIDEr weakness)
- BNATURE (8,000 images, 5 refs)

**Held-out test:**
- Bengali split of XM3600 (Thapliyal et al., EMNLP 2022) — gold native references; never used by prior Bangla captioning work as held-out. Free novelty boost.

**Human-judgment benchmark (C3):**
- 200 images sampled from BAN-Cap test set
- Generated captions from 5 systems: CNN-LSTM, Bornon, zero-shot PaliGemma, ours-GiT-LoRA, ours-PaliGemma-LoRA
- 3 annotators per (image, caption) pair, 1–5 Likert on adequacy and fluency
- Report Krippendorff α for inter-annotator agreement

---

## 5. Paper Structure

1. **Introduction** — three gaps, four contributions.
2. **Related Work** — Bangla captioning (Chittron → Bornon → BiCap); PEFT on VLMs (PALO, Chitrarth, Chitranuvad); captioning metrics critique (Kilickaya 2017, Sai et al. 2022, Polos, FLEUR).
3. **Tokenizer Fertility Audit** — fertility of GiT, BLIP, Florence, PaliGemma, Qwen2-VL tokenizers on Bangla text from BAN-Cap. Motivates the bridge.
4. **Method** — (a) vocab extension + 4 embedding init strategies; (b) LoRA placement (q/k/v/o + cross-attn + LM head); (c) GiT-PEFT fix.
5. **Experiments** — datasets, baselines, hyperparameters, ablations: LoRA rank ∈ {4, 8, 16, 32}, target-modules ablation, embedding-init ablation.
6. **Evaluation Framework** — n-gram + BERTScore-bn + M-CLIPScore + Polos / FLEUR; build the human-judgment benchmark; report per-metric correlation with human ratings; recommend a Bangla protocol.
7. **Efficiency Analysis** — trainable params, VRAM, wall-clock; reinforces low-resource-accessibility narrative.
8. **Discussion / Limitations / Ethical Considerations** — translation artifacts, dialectal coverage, annotator demographics.
9. **Conclusion + Code/Data Release.**

---

## 6. Venue Ranking (realistic)

1. **IEEE Access** — most realistic Q1; broad scope; fast review.
2. **ACM TALLIP** — perfect topical fit (low-resource Asian language NLP); Q1.
3. **Multimedia Tools and Applications (Springer)** — common venue for non-English captioning.
4. **Expert Systems with Applications** — possible if framed as a deployable system; strict reviewers.
5. Stretch: **Information Processing & Management** if the evaluation framework dominates the story.

Skip TPAMI / TMM for first submission — they want architectural novelty beyond adaptation.

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| GiT-PEFT fix may turn out trivial | Fallback: present as engineering reproducibility note; lean on evaluation framework |
| Human-annotation effort (3,000 ratings) | 3 volunteer Bangla speakers ≈ 1 week; pilot with 20 images first |
| Translation-quality artifacts in BanglaLekha / Flickr8k-BN | Use BAN-Cap and XM3600-bn as gold references; report translation-heavy results separately |
| Reviewers ask for bigger model | PaliGemma-3B + QLoRA covers that |
| TPU compatibility issues | Don't promise TPU; report T4 wall-clock; mention TPU as future work |

---

## 8. Next Steps (in order)

1. Retrieve and read the 2025 ResearchGate paper (publication 392102021) to lock down novelty claim.
2. Read Chitrarth (arXiv:2502.15392) carefully to confirm whether they did LoRA or full FT for Bengali — closest precedent.
3. Reproduce HF's GiT fine-tune Colab on a tiny BanglaLekha slice to confirm baseline runs.
4. Audit tokenizer fertility (~1 day of work; forms Section 3 of the paper).
5. Pilot human-evaluation protocol with 20 images before scaling.

---

## 9. Key References

### Bangla image captioning
- BAN-Cap — arXiv:2205.14462
- Bornon — arXiv:2109.05218
- Chittron — Procedia CS, doi:10.1016/j.procs.2019.05.071
- CNN-Transformer Bengali — arXiv:2110.12442
- BiCap — aclanthology.org/2025.banglalp-1.6
- Visual-attention Bengali (PLOS One 2025) — PMC11825021
- Bengali Vision Encoder-Decoder — IEEE 10441125

### LoRA / PEFT on VLMs
- LoRA — arXiv:2106.09685
- QLoRA — arXiv:2305.14314
- PALO — arXiv:2402.14818
- Chitrarth — arXiv:2502.15392
- Chitranuvad — arXiv:2502.20420
- mBLIP — arXiv:2307.06930
- GiT — arXiv:2205.14100
- GiT + PEFT issue — huggingface/peft discussions #1958

### Tokenizer / embedding extension
- Vocab expansion with 0.01 GB target text — arXiv:2406.11477
- English-Centric to Bilingual — aclanthology.org/2025.unlp-1.1
- FOCUS / WECHSEL — referenced via above

### Metrics
- Kilickaya 2017 (EACL) — aclanthology.org/E17-1019
- Sai, Mohankumar, Khapra — arXiv:2008.12009 (ACM Computing Surveys 2022)
- CLIPScore — arXiv:2104.08718
- PAC-S — Sarto et al., CVPR 2023
- Polos — arXiv:2402.18091
- FLEUR — aclanthology.org/2024.acl-long.205
- BERTScore — ICLR 2020
- XM3600 — arXiv:2205.12522
