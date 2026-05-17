# LoRA target-module ablation

Reference caption (held image): `একটি লাল গাড়ি রাস্তায় চলছে।`

| Variant | Targets | Trainable | % of base | Loss start → end | Wall (s) | Generated |
|---|---|---|---|---|---|---|
| `v1_attn` | attn (q,k,v) | 221,184 | 0.125% | 12.348 → 10.060 | 7.5 | `part of a wall` |
| `v2_attn_head` | attn + LM head (fully retrained) | 41,414,970 | 18.995% | 12.348 → 3.458 | 8.0 | `##াাাাাাাাাাাাাাাাাাাাাাাাাাাাাা` |
| `v3_attn_head_emb` | attn + LM head + word embeddings (fully retrained) | 64,855,866 | 26.858% | 12.348 → 3.388 | 8.9 | `##াাাাাাাাাাাাাাাাাাাাাাাাাাাাাা` |
