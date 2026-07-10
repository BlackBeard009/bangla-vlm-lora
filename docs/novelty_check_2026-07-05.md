# Novelty check — gap G1 and Chitrarth precedent (2026-07-05)

Closes paper-plan §8 items 1–2 (flagged since 2026-05-17). Two independent
literature sweeps: (A) ResearchGate 392102021 + broad 2024–2026 sweep for
anything killing G1; (B) deep read of Chitrarth + Chitranuvad.

## Headline verdict

**G1 is still defensible as of July 2026, but the wording must be
tightened.** No peer-reviewed paper applies LoRA/PEFT to a generative VLM
*specifically for Bangla image captioning*. But two peer-reviewed
multilingual papers combine LoRA + generative VLM + Bengali-among-many
and would let a reviewer attack the current literal wording.

### Recommended G1 re-wording (use in the paper)

> To our knowledge, no peer-reviewed work performs a **dedicated,
> Bangla-specific** study of parameter-efficient (LoRA) adaptation of an
> end-to-end generative vision-language model for Bangla image
> captioning. Massively multilingual VLMs such as mBLIP and PALO apply
> LoRA and include Bengali among dozens of languages, but rely on
> machine-translated data and provide no Bangla-focused adaptation
> methodology, benchmark analysis, or human-quality evaluation.

## ResearchGate 392102021 — resolved

- Title: **"Fine-tuning Vision-Language Models for Bengali Captions"**,
  posted ~2025-05-26. ResearchGate-only: no arXiv/DOI/DBLP/Semantic
  Scholar record → **not peer-reviewed**.
- Correction to the paper plan's prior belief: it is *not* CLIP
  encoder-only — indexed snippets describe a multilingual encoder-decoder
  that generates captions. But the task is **Bengali–English
  code-switched** captioning (social-media style), and **no snippet
  mentions LoRA/PEFT** (method is "hybrid loss" fine-tuning).
- Verdict: **does not kill G1.** Cite as a non-peer-reviewed preprint on
  the distinct code-switched task.
- Residual TODO: RG blocks all automated access — a colleague with an RG
  account should skim the PDF once to confirm no LoRA.

## Threat matrix (2024 – July 2026)

| Paper | Venue | LoRA? | Generative captioning? | Bangla-specific? | G1 impact |
|---|---|---|---|---|---|
| **mBLIP** (arXiv:2307.06930) | ALVR@ACL 2024 ✔ peer-rev | **Yes** (LLM decoder) | Yes, 96 langs incl. bn (MT data) | No | **Weakens literal wording — biggest threat. Must cite + differentiate.** |
| **PALO** (arXiv:2402.14818) | WACV 2025 ✔ | **Yes** (LM) | Instruction-following, not captioning benchmarks | No | Weakens moderately. Cite + differentiate. |
| **BanglaGPT ClipCap-style** (RG 387295591) | ICCIT 2024 ✔ (IEEE) | No (full FT) | Yes — Bangla-specific ClipCap + BanglaGPT | **Yes** | Closest peer-reviewed Bangla generative work. **Must cite**; differentiate: stitched pipeline vs. end-to-end VLM, full FT vs. LoRA. |
| Chitrarth (arXiv:2502.15392) | preprint | No (full FT of Krutrim LLM) | VQA-centric, MT data | No (10 Indic langs) | No overlap. |
| Chitranuvad (arXiv:2502.20420) | WAT2024 sys paper | Yes — but for multimodal *translation*; found LoRA < full FT | Bengali captioning track: no numbers reported | No | No direct overlap; see rebuttal risk below. |
| Align Where the Words Look (arXiv:2509.18369) | preprint 2025-09 | No | Yes (MaxViT+mBART, novel loss) | Yes | Cite as recent related work; no LoRA overlap. |
| BiCap (BLP-2025) | ✔ | No | ResNet+LSTM | Yes | No overlap. Full BLP-2025 proceedings checked: no LoRA-VLM-captioning paper. |
| BanglaVerse (arXiv:2603.21165) | preprint 2026-03 | No (zero-shot eval) | benchmark | Yes | Supports our motivation (VLMs weak on Bangla captioning). |
| ChitroJera (ECML PKDD 2025) | ✔ | No | VQA | Yes | No overlap. |

## Chitrarth deep-read findings (for Related Work §2)

- Architecture: Krutrim LLM (natively multilingual incl. Bengali) +
  CLIP-ViT-L/SigLIP + MLP projector, LLaVA-style.
- Training: two-stage **full fine-tuning** (projector, then projector +
  LLM). **No LoRA/PEFT anywhere in the paper.**
- Bengali data: 100% machine-translated (IndicTrans2). No native data.
- **No tokenizer/vocabulary extension** — Indic coverage inherited from
  the backbone's pretraining. → **No collision with our C1 embedding
  bridge.**
- No Bengali captioning benchmark; BharatBench LLaVA-Bench-Bengali 53.7.
- Suggested positioning sentence:
  > Chitrarth extends visual grounding to ten Indic languages, including
  > Bengali, by fully fine-tuning a proprietary multilingual LLM
  > (Krutrim) on IndicTrans2-translated instruction data, and its sibling
  > system Chitranuvad explored LoRA for multimodal translation at
  > WAT2024 (finding it inferior to full fine-tuning on that task); in
  > contrast, we adapt an openly available English-centric VLM to Bangla
  > image captioning using parameter-efficient LoRA alone, without a
  > natively multilingual backbone, and evaluate directly on Bengali
  > captioning benchmarks.

## Rebuttal risk to prepare for

Chitranuvad §5.4: "full fine-tuning consistently outperforms LoRA"
(echoing Biderman et al. 2024). Preempt in Discussion: their comparison
was (a) on multimodal *translation*, not captioning, and (b) on a
backbone already fluent in Bengali — LoRA had no new language knowledge
to inject cheaply. Our setting is the opposite: cross-lingual adaptation
where the vocabulary bridge + LoRA injects capability the base model
lacks entirely (as our stages 1–6 motivation chain demonstrates).

## Obligations checklist for the paper

- [ ] Re-word G1 as above (Introduction + Related Work).
- [ ] Cite and differentiate mBLIP, PALO, ICCIT-2024 BanglaGPT paper.
- [ ] Cite RG 392102021 as non-peer-reviewed preprint, code-switched task.
- [ ] Cite Align-Where-the-Words-Look, BanglaVerse (motivation), BiCap.
- [ ] Human check of RG 392102021 PDF (no-LoRA confirmation).
- [ ] Preempt the Chitranuvad/Biderman LoRA-inferiority rebuttal.
- [ ] Native-data differentiator: our BAN-Cap refs are native, mBLIP /
      PALO / Chitrarth Bengali data is machine-translated. Say so.

## Search coverage (what "absence of evidence" means here)

Google/web indices, arXiv, ACL Anthology (BLP-2025 complete 63-paper
list, ALVR, WAT), Semantic Scholar, OpenAlex, DBLP, IEEE-indexed venues.
Query families: {Bangla, Bengali} × {image captioning, caption
generation} × {LoRA, QLoRA, PEFT, parameter-efficient, vision-language
model, BLIP, LLaVA, PaliGemma, Qwen-VL}. BLP-2026 proceedings do not
exist yet (workshop @ AACL-IJCNLP upcoming).
