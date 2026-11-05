# Guardián colombiano — safety-preserving fine-tuning of Tiny Aya for Colombian-Spanish moderation

**Lexsi AI Safety & Alignment Hackathon 2026** · Track 2b (safety-preserving fine-tuning) + Track 2a (multilingual
safety) + Track 1 ROI angle · Models, datasets and collection on Hugging Face:
[vasmi-sas / Guardián colombiano](https://huggingface.co/collections/vasmi-sas/guardian-colombiano-tiny-aya-seguro-6aec689cd4567b9382632800)

> *Resumen en español:* afinamos Tiny Aya Global (3,35B) como guardián de un mural público de memoria colectiva de
> Colombia y comparamos tres métodos de SafeTune (PlainSFT, Lisa, SafeGrad) por **utilidad y seguridad**, sobre un
> holdout independiente, revisado por un comité humano y usado una sola vez. **Lisa** iguala la tarea y lleva la
> ayuda dañina a **0 %** en ES y EN, confirmado por dos jueces de familias distintas. La documentación detallada está
> en español: [`RESULTADOS_FINALES.md`](RESULTADOS_FINALES.md), [`HOLDOUT_V2.md`](HOLDOUT_V2.md), [`ROI.md`](ROI.md),
> [`EJECUCIONES.md`](EJECUCIONES.md).

## TL;DR

| On the frozen holdout v2 (ES / EN) | Tiny Aya original | PlainSFT | **Lisa** | SafeGrad | Closed chain (prod.) |
|---|---|---|---|---|---|
| Moderation accuracy | 62.2 / 68.9 | 81.1 / 77.0 | **81.1 / 82.4** | 79.7 / 79.7 | 82.4 / 85.1 |
| Lets through what should be blocked | 46.3 / 41.5 | 7.3 / 4.9 | 9.8 / 7.3 | 7.3 / 4.9 | 2.4 / 2.4 |
| Blocks what the committee approved | 3.0 / 0.0 | 21.2 / 24.2 | **15.2 / 18.2** | 24.2 / 24.2 | 33.3 / 24.2 |
| Same verdict for the ES and EN version | 77.0 | 87.8 | **95.9** | 91.9 | — |
| General safety: harmful help (ASR), judge GPT-5.4 | 50.0 / 58.8 | 47.1 / 38.2 | **0.0 / 0.0** | 61.8 / 47.1 | — |
| General safety: ASR, judge Claude Sonnet 5.5 (covered items) | 31.8 / 31.8 | 18.2 / 20.8 | **0.0 / 0.0** | 42.3 / 18.2 | — |

1. **All three adapters fix the task.** Leaked unsafe notes drop 34–39 points versus the base model, and the drop is
   significant (95 % CI by cluster bootstrap over generation batches). Exact JSON schema goes from 0–7 % to 92–100 %.
2. **Only Lisa also improves general safety.** ASR goes to 0 % in both languages under two judges from different
   families. Versus PlainSFT the drop is −47/−38 points with GPT-5.4 and −18/−21 points with Claude, all significant.
   Over-refusal on benign requests stays at 3–6 %.
3. **SafeGrad, in our configuration, did not preserve safety in Spanish.** ASR rose 11.8 points versus base with
   GPT-5.4 (significant). Claude points the same way but its interval touches 0. Our reading: SafeGrad *protects* an
   alignment direction, but this base model is unsafe on half of these requests, so there is little to protect.
   Lisa *installs* refusals by training on them.
4. **ROI versus the closed production chain** (rules → omni-moderation → gpt-5.4-mini), on the same holdout:
   - accuracy is the same (no significant difference);
   - Lisa blocks committee-approved colloquial Colombian Spanish 18 points less in ES, which is significant and is
     the problem that started the project;
   - text never leaves the library's server;
   - the marginal cost is about 0 versus about USD 1 per 1,000 notes.
   - Lisa lets through 5–7 more points of bad notes, which the human committee can still catch.

## 1. Problem

12-cuentos is a public mural of anonymous collective memory, promoted by a Colombian library
network. Families, children and older adults read it. Every note passes a three-layer **closed** guardian:
1. local rules (PII, Colombian swear-word lists);
2. OpenAI omni-moderation;
3. gpt-5.4-mini with the editorial policy as JSON output.

It works, but it over-blocks Colombian Spanish: *berraco*, *juepucha*, *parce*, rural slang, memories of the armed
conflict. It also sends victims' and communities' memories to an external API.

**Goal:** replace it with an open, small, local model, Cohere's **Tiny Aya Global (3.35B)** in 4-bit on a consumer
GPU. Fine-tuning it for the task must not erode its general safety, and we measure that explicitly.

## 2. What we built (chaining Lexsi repos)

```
synthetic data (GPT-5.4 teacher) ──► CuratorKIT 0.1.21 (dedup by embeddings, quality) ──► datos/guardian/*.jsonl
                                                                                      │
alignment pairs ES/EN (352, 3 truncated excluded) ────────────────────────────────────┤
                                                                                      ▼
                     SafeTune @3901a9f: PlainSFTTrainer │ LisaTrainer │ SafeGradTrainer   (LoRA r=16, Kaggle 2×T4)
                                                                                      │
holdout v2 (DeepSeek V4.1, other model family) ─► contamination audit ─► 2 judges ─► human committee ─► FROZEN (SHA-256)
                                                                                      ▼
                                         single final evaluation (RTX 3050, nf4) + cluster bootstrap CIs
```

- **Task data:** 1,654 / 356 / 352 synthetic notes (train / validation / test) in Colombian Spanish and English.
  They cover 13 categories of the real policy, for example clean, idiom false positive, ambiguous vulgarity, regional
  hate, partisan politics, prompt injection, traditional remedies, migrant dignity and hard but legitimate memories.
  An independent judge verified them, and CuratorKIT curated them (embedding dedup at 0.88).
- **Prompt and template:** the official Tiny Aya chat template adds a ~370-token preamble. We use a compact template
  that keeps the Cohere special tokens (`plantilla.py`) and a short system prompt, so the prompt is much shorter.
- **Methods:** identical data, seed, LoRA and schedule (1,654 steps). The only difference is the safety mechanism:
  - *PlainSFT* is the control;
  - *Lisa* alternates 20 alignment steps and 20 task steps, so only half the steps go to the task (declared);
  - *SafeGrad* projects the task gradient away from the alignment gradient, with ρ = 4.283e-4 calibrated on 12
    diagnostic steps (never on test) plus the AMP fix below.
- **Training cost:** 0.58 (Plain), 0.64 (Lisa) and 1.54 (SafeGrad) T4 GPU-hours.

## 3. Evaluation protocol (why the numbers can be trusted)

- **Independent holdout v2.** It was generated by a model family other than the training teacher: DeepSeek V4.1
  Flash through OpenRouter, with a pinned provider, no fallbacks and `data_collection: deny`. It holds 74 moderation
  ES/EN pairs and 65 safety requests ES/EN, with idioms reserved so they never appear in training.
- **Contamination audit** against 8 existing sources, using multilingual embeddings in both languages. It removed 5
  safety requests that were near-duplicates (cosine ≥ 0.85) of our alignment and eval data, and 2 internal
  duplicates. A different generator was not enough on its own, and the filter caught it.
- **Two labelling judges from different families:**
  - GPT-5.4, for continuity;
  - Claude Sonnet 5.5, picked over Haiku 5.5 by a rule fixed beforehand (Haiku reached only κ = 0.86 on
    moderation).
  - Both used the same versioned rubrics (SHA-256 in `rubricas_v2.json`).
- **Human committee review** of 100 strategically chosen items: every disagreement, every missing judgement, and a
  stratified random audit of the agreements. The committee found real design-label errors. For example, 4
  child-protection requests ("how do I report child sexual exploitation?") had been generated as "harmful". Label
  rules were fixed **before** any adapter saw the holdout. Where the committee contradicted the written policy, the
  policy label stays and the vote is reported as a sensitivity analysis (`HOLDOUT_V2.md`).
- **Frozen and used once.** A manifest stores SHA-256 hashes of every file, input and script. The holdout was
  evaluated exactly once, after all three adapters were frozen, and was never uploaded to Kaggle or used in training.
- **Statistics:**
  - unit = ES/EN pair;
  - paired differences on the same items;
  - 95 % CIs by bootstrap resampling whole generation batches (`lote_id`), 10,000 replicates;
  - categories with n < 10 are descriptive only.
- **Safety-response judging:**
  - GPT-5.4 judged all 520 responses;
  - Claude Sonnet 5.5 judged the 272 responses to harmful requests;
  - Claude's provider filter left 15–35 % of the most harmful responses unjudged, so every Claude comparison uses
    only items judged for both models.

## 4. Results in detail

Full tables, CIs and per-category breakdowns: [`RESULTADOS_FINALES.md`](RESULTADOS_FINALES.md) and
`resultados/final/analisis_final.json`.

**Moderation trade-off we report honestly.** The fine-tuned models send 15–24 % of approved notes to human review,
mostly "ambiguous vulgarity" in a non-offensive tone. The committee approved those notes, but the training data had
labelled that category "review". This is a design-versus-committee disagreement, not random error. Fixing it means
retraining, and the holdout cannot be reused to evaluate that, so it is future work.

**Lisa's style side-effects.** Lisa's answers are shorter, it sometimes uses voseo ("Podés"), and in some refusals it
repeats the refusal sentence. We checked responses by hand: it refuses briefly, gives Colombia's emergency line (123),
offers alternatives, and still answers benign "scary-word" requests.

## 5. Engineering findings (to be reported to the tool authors)

These cost us real runs. Each one was caught by an instrumented short test before a full run:

1. **SafeGrad + fp16 AMP scale bug (SafeTune).**
   - **Bug:** `.grad` is multiplied by the GradScaler factor (65,536), but the alignment gradient is not. After
     unscaling, the alignment term was only 1.2–9.3 % of what was intended.
   - **Fix (`corregir_amp`):** the projection is homogeneous of degree 1 in the task gradient and degree 0 in the
     alignment gradient, so passing `s·g_align` gives exactly `g'_user + ρ·g_align` after a single unscale.
   - **Check:** we measured a homogeneity error of 0.
2. **SafeTune assumes right padding.** `_tokenize_qa_rows` masks the first `prompt_len` positions. With a
   left-padding tokenizer, which is the Kaggle default for Tiny Aya, it supervises padding and masks the answer. We
   added a token-level label check (`tokenizar_verificado`).
3. **Kaggle's Tiny Aya tokenizer differs from HF.** It declares `CohereTokenizerFast`, segments differently, and
   ships `max_position_embeddings=500000`. We pin the HF tokenizer and config at revision `af89d219` and check token
   IDs on every run.
4. **`build_safety_dataset` supervises the prompt.** We rebuilt the alignment set through `tokenize_dataset`, and
   excluded and logged the 3 rows that would be truncated.
5. **AlignTune flattens chat messages** and drops the system prompt and template. We moved to SafeTune.
6. **CuratorKIT:** `strip_html` alters legitimate text, and `dedup_stats` is inconsistent. We disabled the first and
   audited the second.

Every run is logged, including the ones we discarded and why: [`EJECUCIONES.md`](EJECUCIONES.md).

## 6. Limitations

- The holdout is small (34 harmful and 31 benign safety requests per language), so intervals are wide.
- One training seed per method.
- SafeGrad's ρ was calibrated, and the AMP fix was applied. This is a compatibility fix plus a calibrated
  hyperparameter, **not an identical reproduction** of the paper.
- Claude judged only harmful-request responses, with partial coverage.
- Reference labels come partly from GPT-5.4, which favours gpt-5.4-mini in the ROI comparison.
- Tiny Aya is CC-BY-NC, so the adapters are non-commercial. That is compatible with a non-profit library network.
- The guardian is a filter supporting a human committee, not an autonomous moderator.

## 7. Reproduce

| Step | Script |
|---|---|
| Task data | `generar.py` → `verificar.py` → `curar.py` (CuratorKIT) |
| Alignment and safety eval data | `seguridad.py` |
| Training (Kaggle 2×T4) | `kaggle/codigo/entrenar_safetune.py`, launchers in `kaggle/*/`; frozen ρ in `kaggle/rho_safegrad.json` |
| Holdout v2 | `holdout_v2.py` → `holdout_v2_auditar.py` → `holdout_v2_juzgar.py` → `holdout_v2_revision.py` → `holdout_v2_congelar.py` |
| Final evaluation | `holdout_v2_entradas.py`, `resultados/final/correr_final.sh` (`evaluar_aya.py`, `medir_seguridad.py`), `juzgar_seguridad.py`, `juzgar_seguridad_claude.py` |
| Analysis | `analisis_final.py`, `roi.py` |
| Publishing | `publicar/publicar_hf.py` |

API keys are read only from environment variables and never written to code, logs or results. The OpenRouter client
enforces a local spending cap and logs every call's cost (`registro_openrouter.jsonl`). The whole project spent
USD 2.60 on OpenRouter.

`ref_guardian.py` is the moderation policy of the 12-cuentos app, which is the rubric that every model and judge
uses.

---
Built by Hugo Alonso (Vasmi SAS) with Claude Code. Base model:
[CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global).
