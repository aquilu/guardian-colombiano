"""Publica los logs, trazas y salidas de evaluación como dataset en HF y separa las colecciones (modelos / datasets+logs).

Corre en monterrey. Las respuestas de seguridad que algún juez marcó como ayuda dañina se publican TACHADAS: se
conserva el pedido, los juicios y las métricas, no el texto dañino. Se escanea todo en busca de credenciales antes de
subir.
Uso: python publicar/logs_hf.py [--ensayo]
"""
import json
import re
import shutil
import sys
from pathlib import Path

R = Path(__file__).resolve().parent.parent
ST = R / "publicar" / "_staging_logs"
ORG, REPO = "vasmi-sas", "vasmi-sas/guardian-colombiano-logs"
TACHADO = "[REDACTED — response to a harmful request that was not a clean refusal for every judge; only its judgements are published]"
SECRETOS = re.compile(r"KGAT_[A-Za-z0-9]{8,}|sk-or-v1-[a-f0-9]{16,}|sk-proj-[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{30,}|Bearer [A-Za-z0-9._-]{20,}")


def copiar(origen, destino):
    o, d = R / origen, ST / destino
    d.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(o, d)


def main(ensayo):
    shutil.rmtree(ST, ignore_errors=True)
    ST.mkdir(parents=True)
    # 1. entrenamiento
    for m in ("plain", "lisa", "safegrad"):
        copiar(f"modelos/guardian-{m}-global/resumen_entrenamiento.json", f"training/{m}/resumen_entrenamiento.json")
    for k, f in (("plain-final", "log-plain.txt"), ("harden-final", "log-lisa.txt"), ("harden-final", "log-safegrad.txt"),
                 ("diag-safegrad", "log-safegrad.txt")):
        copiar(f"kaggle/salidas/{k}/{f}", f"training/kaggle-logs/{k}-{f}")
    copiar("kaggle/rho_safegrad.json", "training/rho_safegrad.json")
    copiar("EJECUCIONES.md", "EJECUCIONES.md")
    # 2. validación
    for m in ("plain", "lisa", "safegrad"):
        for suf in (".jsonl", ".resumen.json"):
            copiar(f"resultados/validacion-{m}{suf}", f"validation/validacion-{m}{suf}")
    # 3. holdout v2: jueces, contaminación, rúbricas, comparación de jueces
    for f in ("verificado/holdout_v2_moderacion_jueces.jsonl", "verificado/holdout_v2_seguridad_jueces.jsonl",
              "verificado/holdout_v2_estado_etiquetas.json", "informe_contaminacion_v2.json", "rubricas_v2.json",
              "comparacion_jueces.json"):
        copiar(f, f"holdout_v2/{Path(f).name}")
    # 4. evaluación final: moderación tal cual; seguridad con lo dañino tachado
    F = R / "resultados/final"
    for p in sorted(F.glob("mod-*")):
        copiar(f"resultados/final/{p.name}", f"final_eval/moderation/{p.name}")
    claude = {}
    for l in open(F / "seg-claude.jsonl", encoding="utf-8"):
        y = json.loads(l)
        claude[(y["modelo"], y["id"], y["idioma"])] = y["claude"]
    tachadas = 0
    for m in ("base", "plain", "lisa", "safegrad"):
        out = ST / f"final_eval/safety/seg-{m}.judged.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for l in open(F / f"seg-{m}.juzgado.jsonl", encoding="utf-8"):
                y = json.loads(l)
                c = claude.get((m, y["id"], y["idioma"]))
                y["juicio_claude"] = c
                g = y.get("juicio") or {}
                rechazo_limpio = (g.get("rechaza") and not g.get("ayuda_danina")
                                  and (c is None or (c.get("rechaza") and not c.get("ayuda_danina"))))
                if y["tipo"] == "danino" and not rechazo_limpio:
                    y["respuesta"], y["respuesta_tachada"] = TACHADO, True
                    tachadas += 1
                f.write(json.dumps(y, ensure_ascii=False) + "\n")
        copiar(f"resultados/final/seg-{m}.seguridad.json", f"final_eval/safety/seg-{m}.metrics.json")
    for f in ("analisis_final.json", "roi.json", "seg-claude.resumen.json", "correr_final.sh"):
        copiar(f"resultados/final/{f}", f"final_eval/{f}")
    # 4b. AuditKIT: resumen tal cual; en jailbreak solo se conservan los rechazos claros
    rechazo = re.compile(r"(?i)^\W*(i(?:'|’)m sorry|i can(?:'|’)?t|i cannot|i won(?:'|’)t|lo siento|no puedo)")
    copiar("resultados/auditkit/resumen.json", "auditkit/resumen.json")
    for m in ("base", "plain", "lisa", "safegrad"):
        out = ST / f"auditkit/trazas-{m}.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for l in open(R / f"resultados/auditkit/trazas-{m}.jsonl", encoding="utf-8"):
                t = json.loads(l)
                if t["sonda"].startswith("jailbreak") and not rechazo.search(t["output"]):
                    t["output"], t["output_tachado"] = TACHADO, True
                    tachadas += 1
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    copiar("AUDITKIT.md", "auditkit/AUDITKIT.md")
    # 5. costos
    copiar("registro_openrouter.jsonl", "costs/registro_openrouter.jsonl")
    copiar("registro_openai.jsonl", "costs/registro_openai.jsonl")
    # escaneo de credenciales
    for p in ST.rglob("*"):
        if p.is_file() and SECRETOS.search(p.read_text(errors="ignore")):
            sys.exit(f"posible credencial en {p}; no se sube nada")
    (ST / "README.md").write_text(f"""---
license: cc-by-nc-4.0
language: [es, en]
tags: [safety, logs, evaluation, traces, colombian-spanish, lexsi-hackathon]
pretty_name: Guardián colombiano — logs, traces and evaluation outputs
---

# Guardián colombiano — logs, traces and evaluation outputs

All artifacts behind the numbers of [Guardián colombiano](https://github.com/aquilu/guardian-colombiano)
(Lexsi AI Safety & Alignment Hackathon 2026): Tiny Aya Global fine-tuned with SafeTune PlainSFT, Lisa and SafeGrad.

| Folder | Content |
|---|---|
| `training/` | Per-method training summaries (loss curve, memory, minutes, SafeGrad surgery diagnostics), raw Kaggle logs, the frozen SafeGrad ρ and its calibration rule |
| `validation/` | Per-item outputs and summaries on the validation split (functional check before freezing) |
| `holdout_v2/` | Both judges' labels (GPT-5.4, Claude Sonnet 5.5) for every holdout item, label state, contamination report, versioned rubrics with SHA-256, Sonnet vs Haiku judge comparison |
| `final_eval/moderation/` | Per-item verdicts and raw JSON of the 5 systems on the frozen holdout v2 (single run) |
| `final_eval/safety/` | Per-item general-assistant responses of the 4 models with both judges' verdicts |
| `final_eval/` | Paired differences with cluster-bootstrap CIs (`analisis_final.json`), ROI comparison, run script |
| `auditkit/` | AuditKIT 1.1.3 red-team probes (EN built-in + our ES extension) on the 4 models: summary and per-probe traces; jailbreak outputs that are not a clear refusal are redacted |
| `costs/` | Every OpenRouter call (model, provider, tokens, USD) and every OpenAI judge call (tokens) |
| `EJECUCIONES.md` | Registry of every run, including discarded ones and why |

**Redaction:** {tachadas} responses to harmful requests that were not a clean refusal for every judge (for example
weapon, fraud or recruitment content produced by the base model, PlainSFT or SafeGrad) are replaced by a placeholder
(`respuesta_tachada: true`). Clean refusals and all responses to benign requests are kept verbatim. Their
prompts, judgements and metrics are kept. No credentials are included; files were scanned before upload.

The frozen holdout itself is in [`vasmi-sas/guardian-colombiano-holdout-v2`](https://huggingface.co/datasets/vasmi-sas/guardian-colombiano-holdout-v2).
""")
    archivos = sorted(str(p.relative_to(ST)) for p in ST.rglob("*") if p.is_file())
    print(len(archivos), "archivos ·", tachadas, "respuestas tachadas")
    if ensayo:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(ST), repo_id=REPO, repo_type="dataset", commit_message="Logs, trazas y salidas")
    # colecciones separadas
    vieja = api.get_collection("vasmi-sas/guardian-colombiano-models-6aec689cd4567b9382632800")
    api.update_collection_metadata(vieja.slug, title="Guardián colombiano — models",
                                   description="Tiny Aya Global + PlainSFT / Lisa / SafeGrad adapters and the live demo (Lexsi Hackathon 2026).")
    for it in api.get_collection(vieja.slug).items:
        if it.item_type == "dataset":
            api.delete_collection_item(vieja.slug, it.item_object_id)
    nueva = api.create_collection(title="Guardián colombiano — datasets & logs", namespace=ORG, private=False, exists_ok=True,
                                  description="Training data, alignment pairs, frozen holdout v2, logs, traces and evaluation outputs.")
    for d in ("guardian-colombiano-moderacion", "guardian-colombiano-alineacion", "guardian-colombiano-holdout-v2",
              "guardian-colombiano-logs"):
        api.add_collection_item(nueva.slug, item_id=f"{ORG}/{d}", item_type="dataset", exists_ok=True)
    print("modelos:", f"https://huggingface.co/collections/{vieja.slug}")
    print("datasets:", f"https://huggingface.co/collections/{nueva.slug}")


if __name__ == "__main__":
    main("--ensayo" in sys.argv)
