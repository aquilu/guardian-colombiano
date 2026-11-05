<p align="center">
  <img src="docs/img/banner.svg" alt="Guardián colombiano — safe, local moderation for Colombian Spanish" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/aquilu/guardian-colombiano"><img alt="Live demo" src="https://img.shields.io/badge/🤗%20demo-live%20on%20ZeroGPU-f2c230?style=for-the-badge"></a>
  <a href="https://huggingface.co/collections/vasmi-sas/guardian-colombiano-tiny-aya-seguro-6aec689cd4567b9382632800"><img alt="Models and datasets" src="https://img.shields.io/badge/🤗%20models%20%26%20data-vasmi--sas-2a5db0?style=for-the-badge"></a>
  <img alt="Lexsi AI Safety Hackathon 2026" src="https://img.shields.io/badge/Lexsi%20Hackathon-Track%202b%20%2B%202a-c8323c?style=for-the-badge">
</p>
<p align="center">
  <img alt="Code: MIT" src="https://img.shields.io/badge/code-MIT-informational">
  <img alt="Models: CC-BY-NC 4.0" src="https://img.shields.io/badge/models%20%26%20data-CC--BY--NC%204.0-lightgrey">
  <img alt="Base: Tiny Aya Global 3.35B" src="https://img.shields.io/badge/base-Tiny%20Aya%20Global%203.35B-39594d">
  <img alt="Runs on a 6 GB GPU" src="https://img.shields.io/badge/runs%20on-6%20GB%20consumer%20GPU-1baf7a">
</p>

# 🧶 Guardián colombiano

**A guardian for every library's mural. It runs local, speaks *parcero* and won't be talked into harm.**

Guardián colombiano is a fine-tuned **Tiny Aya Global (3.35B)** that moderates the anonymous post-its of
*12-cuentos*, a public mural of Colombian collective memory run with a library network. It replaces a closed chain of
three cloud services, rules → OpenAI omni-moderation → gpt-5.4-mini, with **one small open model on a 6 GB consumer
GPU**. The memories of victims and communities never leave the library's server.

The twist: fine-tuning usually erodes a model's general safety. We trained the same guardian three ways (**PlainSFT,
Lisa, SafeGrad**) with Lexsi's [SafeTune](https://github.com/Lexsi-Labs/SafeTune) and measured **both** the task
**and** general safety, on an independent holdout used exactly once.
**Lisa wins:** it matches the task and drives harmful help to **0 %** in Spanish and English, confirmed by two judges
from different model families.

> ¿Español? Toda la documentación detallada está en español:
> [`RESULTADOS_FINALES.md`](RESULTADOS_FINALES.md) · [`HOLDOUT_V2.md`](HOLDOUT_V2.md) · [`ROI.md`](ROI.md) ·
> [`EJECUCIONES.md`](EJECUCIONES.md)

---

## ⚡ TL;DR

<p align="center"><img src="docs/img/safety_asr.png" alt="Attack success rate per method and judge" width="95%"></p>

| Frozen holdout v2 (ES / EN) | Tiny Aya original | PlainSFT | **Lisa** | SafeGrad | Closed chain (prod.) |
|---|---|---|---|---|---|
| 🧱 Moderation accuracy | 62.2 / 68.9 | 81.1 / 77.0 | **81.1 / 82.4** | 79.7 / 79.7 | 82.4 / 85.1 |
| 🚪 Lets through what should be blocked | 46.3 / 41.5 | 7.3 / 4.9 | 9.8 / 7.3 | 7.3 / 4.9 | 2.4 / 2.4 |
| 🧱 Blocks what the committee approved | 3.0 / 0.0 | 21.2 / 24.2 | **15.2 / 18.2** | 24.2 / 24.2 | 33.3 / 24.2 |
| 🌎 Same verdict for the ES and EN version | 77.0 | 87.8 | **95.9** | 91.9 | — |
| 🛡️ Harmful help (ASR), judge GPT-5.4 | 50.0 / 58.8 | 47.1 / 38.2 | **0.0 / 0.0** | 61.8 / 47.1 | — |
| 🛡️ Harmful help (ASR), judge Claude Sonnet 5.5 | 31.8 / 31.8 | 18.2 / 20.8 | **0.0 / 0.0** | 42.3 / 18.2 | — |
| 💸 Cost per 1,000 notes | — | ~$0.01 power | **~$0.01 power** | ~$0.01 power | ~$1.02 API |

All differences called significant below have a 95 % CI (cluster bootstrap over generation batches, 10,000
replicates) that excludes 0. Power cost assumes 150 W and USD 0.22/kWh (see [`ROI.md`](ROI.md)).

## ✨ Highlights

- **🛡️ Zero harmful help.** Under GPT-5.4, Lisa refuses all 34 harmful requests per language. Under Claude Sonnet 5.5,
  its ASR is also 0 %. Versus PlainSFT that is −47/−38 points (GPT) and −18/−21 points (Claude), all significant.
- **🗣️ Speaks Colombian.** Lisa blocks approved colloquial notes (*berraco*, *juepucha*, *parce*) **18 points less**
  than the production chain in Spanish (significant), with the same accuracy.
- **🧱 Fixes the task.** All three adapters let through 34–39 points less unsafe content than the base model. Exact
  JSON schema goes from 0–7 % to 92–100 %.
- **🌎 Bilingual by construction.** The same note gets the same verdict in ES and EN 96 % of the time with Lisa,
  versus 77 % for the base model.
- **🏠 Local and sovereign.** It runs in 4-bit on an RTX 3050 at about 1.2 s per note, with no API keys at inference.
- **🔬 Built to be trusted.** The holdout comes from another model family, passed a contamination audit, was labelled
  by two judges and reviewed by a human committee, then frozen with SHA-256 and used once.
- **🐛 Found real bugs upstream.** Five issues reported to Lexsi's tools, including a fp16 AMP scale bug that silently
  disables SafeGrad's alignment term ([details](#-upstream-findings)).
- **🤝 Honest about trade-offs.** The adapters over-block 15–24 % of approved notes, SafeGrad got worse in Spanish,
  and Lisa's style shifted. All of it is in the tables, not under the rug.

## 🧠 How it works (short)

```
             ┌──────────────── training (Kaggle 2× T4, LoRA r=16) ────────────────┐
 GPT-5.4     │                                                                    │
 teacher ──► CuratorKIT ──► 1,654 notes ES/EN ─┐                                  │
             │ (dedup, QA)                     ├──► SafeTune ──┬─► PlainSFT       │
 alignment pairs (harmful→refuse, benign→help) ┘    trainers   ├─► Lisa      ◄── winner
             │                                                 └─► SafeGrad       │
             └────────────────────────────────────────────────────────────────────┘

 DeepSeek V4.1 ──► contamination ──► GPT-5.4 + Claude ──► human ──► FROZEN ──► one final evaluation
 (other family)    audit (8 sources)  judges               committee  SHA-256    + cluster-bootstrap CIs

 inference:  post-it ──► Tiny Aya + Lisa (nf4, 6 GB GPU) ──► {"veredicto": "aprobado|revision|rechazado",
                                                              8 policy flags, one-line explanation}
                                                       └──► human committee reads "revision"
```

```mermaid
flowchart LR
  A[Post-it ES/EN] --> B[Tiny Aya Global + Lisa LoRA<br/>4-bit, local]
  B -->|aprobado| C[🟢 published on the mural]
  B -->|revision| D[🟡 committee reads it first]
  B -->|rechazado| E[🔴 not shown, kept for audit]
```

## 🚀 Quickstart

<p align="center"><a href="https://huggingface.co/spaces/aquilu/guardian-colombiano"><img src="docs/img/demo.png" alt="Live demo: a regionally discriminatory note is rejected in 6.9 s on ZeroGPU" width="90%"></a></p>

**Try it:** [huggingface.co/spaces/aquilu/guardian-colombiano](https://huggingface.co/spaces/aquilu/guardian-colombiano).
It has an ES/EN switch, the mural moderator, and a side-by-side general-safety test of Lisa against the original
Tiny Aya.

**Use the adapter:**

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE, REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
tok = AutoTokenizer.from_pretrained("vasmi-sas/guardian-tiny-aya-lisa")          # compact chat template
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(BASE, revision=REV, quantization_config=q, device_map="auto"),
    "vasmi-sas/guardian-tiny-aya-lisa").eval()

system = ("Eres el guardián editorial del mural «12 Cuentos» (memorias anónimas de Colombia, público familiar). "
          "Evalúa el post-it y responde solo JSON con: veredicto (aprobado | revision | rechazado), "
          "las 8 banderas booleanas y una explicación de una frase para el comité.")
user = 'Región declarada: Andina. Tema declarado: Amor y Nostalgia.\nPost-it:\n"""\n¡Qué frío tan berraco en Tunja!\n"""'
ids = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                              add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**ids, max_new_tokens=160, do_sample=False)
print(json.loads(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)))
# {'veredicto': 'aprobado', 'vulgaridad': False, ..., 'explicacion': 'Es una memoria personal y nostálgica…'}
```

The English system prompt and the user format are in `curar.py` (`SISTEMA`). Tiny Aya is gated: accept its license
on Hugging Face first.

## 📊 Results

<p align="center"><img src="docs/img/moderation_tradeoff.png" alt="Moderation trade-off: leaks versus over-blocking" width="75%"></p>

- **Task:** every adapter moves from "lets almost half the bad notes through" to the low-leak corner. Lisa sits closest
  to *better*: it over-blocks less than PlainSFT, SafeGrad and the closed chain.
- **Why the adapters over-block:** most of it is "ambiguous vulgarity" in a friendly tone. The committee approved
  those notes, but our training data had labelled them "review". It is a design-versus-committee disagreement, not
  random error. The holdout cannot be reused to evaluate a retrained model, so fixing it is future work.
- **General safety:** the official chat template is used, without the mural prompt. Lisa installs refusals: it
  refuses briefly, gives Colombia's emergency line (123), offers a safe alternative, and still answers benign
  "scary-word" requests (*¿cómo mato el tiempo en el bus?*). Over-refusal stays at 3–6 %.
- **SafeGrad in Spanish:** ASR +11.8 points versus base under GPT-5.4 (significant). Claude points the same way, but
  its interval touches 0. Our reading: SafeGrad *protects* an existing alignment direction, and this base model is
  unsafe on half of these requests, so there is little to protect. Lisa *trains* on refusals.
- **Every number, CI and per-category breakdown:** [`RESULTADOS_FINALES.md`](RESULTADOS_FINALES.md) and
  `resultados/final/analisis_final.json`.

## 🏗️ Architecture

| Component | What it is | Where |
|---|---|---|
| **Policy** | The 12-cuentos editorial guardian: 8 flags and 3 verdicts, the rubric for every model and judge | `ref_guardian.py`, `rubricas_v2.json` |
| **Task data** | 1,654 / 356 / 352 synthetic notes covering 13 policy categories, with regional slang, homoglyphs, emoji, spanglish and prompt injections | `generar.py` → `verificar.py` → `curar.py` |
| **Alignment data** | 352 ES/EN pairs (harmful → brief refusal, benign → help); 3 truncated rows excluded and logged | `seguridad.py` |
| **Compact template** | The Cohere special tokens without the ~370-token default preamble | `plantilla.py` |
| **Trainer** | One script for SafeTune's PlainSFT, Lisa and SafeGrad, with token-level label checks, the AMP fix and diagnostics | `kaggle/codigo/entrenar_safetune.py` |
| **Holdout v2** | Generated by DeepSeek, audited, judged twice, reviewed by the committee, frozen | `holdout_v2*.py`, `datos/holdout_v2/` |
| **Evaluation** | 4-bit inference, two response judges, cluster-bootstrap analysis | `evaluar_aya.py`, `medir_seguridad.py`, `juzgar_seguridad*.py`, `analisis_final.py`, `roi.py` |
| **Demo** | A Gradio app on ZeroGPU that streams the guardian JSON | `space/app.py` |

## 🔐 Security model

- **Data sovereignty.** At inference the note goes to a local model only. There is no API call and no third-party
  retention.
- **Human in the loop.** The guardian is a filter that supports a moderation committee, not an autonomous moderator.
  Doubt always goes to `revision`, never to `aprobado`.
- **Safety measured, not assumed.** We track general-assistant safety (ASR, refusals of harmful requests,
  over-refusal of benign ones, helpfulness) next to the task, under two judges.
- **Evaluation integrity:**
  - the holdout comes from a different model family;
  - near-duplicates were filtered against 8 sources;
  - label rules were fixed before any adapter saw the holdout;
  - a SHA-256 manifest covers every file, input and script;
  - it was used exactly once and never uploaded to Kaggle or used in training.
- **Secrets hygiene:**
  - API keys come only from environment variables;
  - they never appear in code, logs, results, Git or Kaggle;
  - the git history was scanned before publishing.
- **Spend guards.** The OpenRouter client pins provider and model, has no fallbacks, sets `data_collection: deny`,
  enforces a local USD cap, logs every call's cost even on failure, and limits retries. Total OpenRouter spend: USD 2.60.
- **Licensing.** Tiny Aya is CC-BY-NC, so the adapters and data are non-commercial too. That is compatible with a
  non-profit library network.

## 🛠️ Development / From source

```bash
git clone https://github.com/aquilu/guardian-colombiano && cd guardian-colombiano
uv sync                                   # generation, judging, analysis
# training and 4-bit inference: torch + transformers 5.x + peft + bitsandbytes (see kaggle/lanzador.py.tpl)

export OPENAI_API_KEY=...                 # teacher and judges (never committed)
export OPENROUTER_API_KEY=...             # holdout generator and Claude judge
```

### 🧰 CLI

Every script documents its flags in its docstring.

| Step | Command |
|---|---|
| Generate, verify and curate the task data | `generar.py` → `verificar.py` → `curar.py` |
| Alignment and safety eval data | `seguridad.py` |
| Train on Kaggle (2× T4) | `kaggle kernels push -p kaggle/plain-final` · `kaggle kernels push -p kaggle/harden-final` |
| Moderation eval | `python evaluar_aya.py --modelo CohereLabs/tiny-aya-global --adaptador modelos/guardian-lisa-global --prompt corto --datos <jsonl> --salida <out>` |
| General-safety responses | `python medir_seguridad.py --adaptador modelos/guardian-lisa-global --datos <jsonl> --salida <out>` |
| Judge responses | `uv run juzgar_seguridad.py <out>.jsonl` · `uv run juzgar_seguridad_claude.py` |
| Build and freeze holdout v2 | `holdout_v2.py` → `holdout_v2_auditar.py` → `holdout_v2_juzgar.py` → `holdout_v2_revision.py` → `holdout_v2_congelar.py` |
| Final evaluation, run once | `python holdout_v2_entradas.py && bash resultados/final/correr_final.sh` |
| Analysis, ROI and figures | `python analisis_final.py` · `python roi.py` · `uv run --with matplotlib python docs/figuras.py` |
| Publish to Hugging Face | `python publicar/publicar_hf.py --org vasmi-sas` |

## 🔌 Integrations

- **Lexsi Labs:**
  - [SafeTune](https://github.com/Lexsi-Labs/SafeTune) @ `3901a9f`: PlainSFTTrainer, LisaTrainer, SafeGradTrainer;
  - [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) 0.1.21: data curation;
  - [AlignTune](https://github.com/Lexsi-Labs/aligntune) @ `c3d648f`: evaluated, then replaced (see findings).
- **Cohere Labs:** [Tiny Aya Global](https://huggingface.co/CohereLabs/tiny-aya-global), pinned at revision `af89d219`.
- **Hugging Face:**
  - [models, datasets and collection](https://huggingface.co/collections/vasmi-sas/guardian-colombiano-tiny-aya-seguro-6aec689cd4567b9382632800);
  - [ZeroGPU demo](https://huggingface.co/spaces/aquilu/guardian-colombiano).
- **Teachers and judges:**
  - OpenAI GPT-5.4 and gpt-5.4-mini (also the production baseline);
  - Claude Sonnet 5.5 via OpenRouter (Google Vertex);
  - DeepSeek V4.1 Flash via OpenRouter (DeepInfra).
- **Compute:** Kaggle 2× T4 for training, a local RTX 3050 6 GB for evaluation.

## 🐛 Upstream findings

Each of these was caught by an instrumented short test before it could ruin a full run:

| # | Finding | Issue |
|---|---|---|
| 1 | **SafeGrad under fp16 AMP:** `.grad` is loss-scaled (×65,536) but `g_align` is not, so `ρ·g_align` ends up divided by the scale. Exact fix: pass `s·g_align`, since the projection is homogeneous (measured error 0) | [SafeTune#37](https://github.com/Lexsi-Labs/SafeTune/issues/37) |
| 2 | `build_safety_dataset` supervises the prompt, i.e. the harmful request, not only the refusal | [SafeTune#38](https://github.com/Lexsi-Labs/SafeTune/issues/38) |
| 3 | `_tokenize_qa_rows` assumes right padding; a left-padding tokenizer supervises part of the prompt | [SafeTune#39](https://github.com/Lexsi-Labs/SafeTune/issues/39) |
| 4 | AlignTune SFT flattens `messages` to `Human:/Assistant:`, dropping the system turn and the chat template | [aligntune#54](https://github.com/Lexsi-Labs/aligntune/issues/54) |
| 5 | CuratorKIT `strip_html` (on by default) silently strips tag-based prompt injections from safety data | [CuratorKIT#42](https://github.com/Lexsi-Labs/CuratorKIT/issues/42) |
| 6 | Kaggle's Tiny Aya copy ships a different tokenizer (left padding, other segmentation) and `max_position_embeddings=500000` | pinned HF tokenizer + token-ID check |

Every run, including the discarded ones and why, is in [`EJECUCIONES.md`](EJECUCIONES.md).

## ⚠️ Limitations

- The holdout is small (34 harmful and 31 benign safety requests per language), so intervals are wide.
- One training seed per method.
- SafeGrad's ρ was calibrated on 12 diagnostic steps (never on test), and the AMP fix was applied. That makes it a
  faithful *compatible* variant, not an identical reproduction of the paper.
- Claude judged only the harmful-request responses, with 65–85 % coverage because of its provider's filter.
- Reference labels come partly from GPT-5.4, which favours gpt-5.4-mini in the ROI comparison.
- Lisa's answers are shorter, it sometimes uses voseo ("Podés"), and in some refusals it repeats the refusal sentence.

## 🗺️ Roadmap

- [ ] Retrain with the committee's labels for ambiguous vulgarity, and evaluate on a **new** holdout v3 (v2 is spent).
- [ ] More seeds, and a larger safety holdout to tighten the intervals.
- [ ] SafeGrad with per-token KL normalisation and ρ sweeps; Lisa with a better task/alignment ratio.
- [ ] Deploy in 12-cuentos as the first layer, with the closed chain as fallback during an A/B period.
- [ ] Indigenous and creole languages (Wayuunaiki, Palenquero, San Andrés creole) as their own evaluation slice.
- [ ] CircuitKIT probes: *where* does Lisa's refusal live?

## 📚 Docs

| Document | What's inside (Spanish) |
|---|---|
| [`RESULTADOS_FINALES.md`](RESULTADOS_FINALES.md) | Final evaluation: every metric, paired difference and CI, second judge |
| [`HOLDOUT_V2.md`](HOLDOUT_V2.md) | Provenance, contamination, judges, committee review, freeze |
| [`ROI.md`](ROI.md) | Closed chain versus Tiny Aya + Lisa: quality, cost, latency, privacy |
| [`EJECUCIONES.md`](EJECUCIONES.md) | Every training and evaluation run, including the discarded ones |
| [`reportes/`](reportes/) | The upstream bug reports, as filed |

## 🙏 Special thanks

- **[Lexsi Labs](https://github.com/Lexsi-Labs)** for SafeTune, CuratorKIT and AlignTune, and for a hackathon that
  asks for safety *and* usefulness.
- **[Cohere Labs](https://cohere.com/research)** for Tiny Aya: small, open and genuinely multilingual.
- **[Hugging Face](https://huggingface.co)** for the Hub, ZeroGPU and the community.
- **Kaggle** for the T4s.
- **The moderation committee** that reviewed 100 holdout items by hand and caught real labelling errors.
- **Every person who writes a memory on the mural.** This guardian exists to keep their voice, slang and all.

<p align="center"><sub>Built in Colombia by Hugo Alonso (<a href="https://vasmi.co">Vasmi SAS</a>) with Claude Code ·
Code MIT · Models & data CC-BY-NC 4.0</sub></p>
