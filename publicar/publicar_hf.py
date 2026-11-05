"""Publica en Hugging Face (organización ORG) los 3 adaptadores, los 3 datasets y una colección pública.

Corre en monterrey (tiene los adaptadores y el token de HF). Arma copias en publicar/_staging/, corrige
base_model_name_or_path del adaptador (la ruta de Kaggle → CohereLabs/tiny-aya-global) y sube con HfApi.
Uso: python publicar/publicar_hf.py --org vasmi [--ensayo]   (--ensayo arma todo y no sube)
"""
import argparse
import json
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ST = RAIZ / "publicar" / "_staging"
BASE, REV = "CohereLabs/tiny-aya-global", "af89d219b53ed9b13b8a4645f9c8028973510324"
METODOS = {
    "plain": ("PlainSFT", "fine-tuning supervisado simple (control)"),
    "lisa": ("Lisa", "alternancia de 20 pasos de alineación y 20 de tarea (SafeTune `LisaTrainer`)"),
    "safegrad": ("SafeGrad", "cirugía de gradiente contra la dirección de alineación, con corrección AMP y ρ calibrado (SafeTune `SafeGradTrainer`)"),
}
# moderación (exactitud ES/EN, se cuela ES/EN, bloquea lo aprobado ES/EN) y seguridad (ASR GPT ES/EN, ASR Claude ES/EN)
RES = {"plain": ("81,1 / 77,0", "7,3 / 4,9", "21,2 / 24,2", "47,1 / 38,2", "18,2 / 20,8"),
       "lisa": ("81,1 / 82,4", "9,8 / 7,3", "15,2 / 18,2", "0,0 / 0,0", "0,0 / 0,0"),
       "safegrad": ("79,7 / 79,7", "7,3 / 4,9", "24,2 / 24,2", "61,8 / 47,1", "42,3 / 18,2")}


def tarjeta_modelo(org, m):
    nombre, desc = METODOS[m]
    r = RES[m]
    return f"""---
license: cc-by-nc-4.0
base_model: {BASE}
library_name: peft
language: [es, en]
pipeline_tag: text-generation
tags: [safety, content-moderation, colombian-spanish, lora, safetune, {m}, lexsi-hackathon]
datasets: [{org}/guardian-colombiano-moderacion, {org}/guardian-colombiano-alineacion, {org}/guardian-colombiano-holdout-v2]
---

# Guardián colombiano — Tiny Aya Global + {nombre} (LoRA)

Adaptador LoRA de [Tiny Aya Global]({"https://huggingface.co/" + BASE}) (3,35B) que modera los post-its de
**12-cuentos**, un mural público de memoria colectiva de Colombia promovido por una red de bibliotecas. Reemplaza una
cadena cerrada (reglas → omni-moderation → gpt-5.4-mini) que sobre-bloqueaba el español colombiano coloquial.
Método: **{nombre}**, {desc}. Proyecto del *Lexsi AI Safety & Alignment Hackathon* (Track 2b + 2a).

*LoRA adapter for Tiny Aya Global that moderates Colombian-Spanish / English community memory notes, trained with
{nombre} to preserve general safety while fine-tuning.*

## Uso

El modelo responde un JSON con `veredicto` (`aprobado` | `revision` | `rechazado`) y 8 banderas de la política.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{org}/guardian-tiny-aya-{m}")       # incluye la plantilla compacta
base = AutoModelForCausalLM.from_pretrained("{BASE}", revision="{REV}", device_map="auto")
modelo = PeftModel.from_pretrained(base, "{org}/guardian-tiny-aya-{m}")
```
El prompt de sistema corto (ES/EN) y el formato del usuario están en `curar.py` (`SISTEMA`) del repositorio del proyecto.

## Resultados en el holdout v2 (evaluación final única, 74 pares de moderación y 65 de seguridad, ES / EN)

| | Este modelo | Tiny Aya original |
|---|---|---|
| Exactitud de moderación (%) | {r[0]} | 62,2 / 68,9 |
| Publica lo indebido (%) | {r[1]} | 46,3 / 41,5 |
| Bloquea lo aprobado (%) | {r[2]} | 3,0 / 0,0 |
| Seguridad general: ayuda dañina, juez GPT-5.4 (%) | {r[3]} | 50,0 / 58,8 |
| Seguridad general: ayuda dañina, juez Claude Sonnet 5.5 (%, ítems cubiertos) | {r[4]} | 31,8 / 31,8 |

IC 95 % por bootstrap de lotes, protocolo y límites en el write-up del proyecto. En resumen:
**Lisa** es el único método que mantiene la tarea y lleva la ayuda dañina a 0 % con los dos jueces; SafeGrad (en
esta configuración) no preservó la seguridad en ES.

## Entrenamiento

- 1.654 ejemplos de moderación sintéticos (GPT-5.4, curados con CuratorKIT), plantilla compacta del chat de Cohere,
  etiquetas del prompt enmascaradas, LoRA r=16, α=32, dropout 0,05; Kaggle 2× T4, fp16.
- Lisa y SafeGrad usan 349 ejemplos de alineación en ES y EN (pedido dañino → rechazo; pedido límite → ayuda), tras excluir 3 que se truncaban.
- `resumen_entrenamiento.json` trae las pérdidas, la memoria y el diagnóstico.

## Límites

Una sola semilla; holdout pequeño; el modelo hereda la licencia **CC-BY-NC** de Tiny Aya (uso no comercial). Es un
filtro de apoyo a un comité humano, no un moderador autónomo.
"""


TARJETAS_DATOS = {
    "guardian-colombiano-moderacion": ("""Post-its sintéticos del mural 12-cuentos (ES de Colombia y EN) con el veredicto y las banderas de la
política del guardián. 13 categorías (limpio, modismo falso positivo, vulgaridad ambigua, odio regional, sexual
explícito, violencia gráfica, política partidista, publicidad, datos de terceros, inyección, cura tradicional,
migrante digno, memoria dura legítima). Generados con GPT-5.4, verificados por un juez y curados con CuratorKIT
(deduplicación por embeddings 0,88).""", ["train.jsonl", "validation.jsonl", "test.jsonl"], "datos/guardian"),
    "guardian-colombiano-alineacion": ("""Pares de alineación de seguridad general en español colombiano e inglés: pedidos dañinos con rechazo breve y
pedidos límite inofensivos (modismos que asustan a un filtro) con ayuda. `alineacion.jsonl` se usó para entrenar Lisa
y SafeGrad; `eval.jsonl` es la evaluación exploratoria de deriva de seguridad.""", ["alineacion.jsonl", "eval.jsonl"], "datos/seguridad"),
    "guardian-colombiano-holdout-v2": ("""Holdout independiente y **congelado** de la evaluación final (usado una sola vez). Generado con DeepSeek V4.1
Flash (otra familia que la de los datos de entrenamiento), con auditoría de contaminación por embeddings, dos jueces
(GPT-5.4 y Claude Sonnet 5.5) y revisión humana de 100 ítems por el comité. 74 pares de moderación y 65 pedidos de
seguridad. `MANIFIESTO.json` guarda el SHA-256 de cada archivo y de las entradas y las reglas de etiqueta.
No entrenes con este conjunto si quieres comparar con nuestros resultados.""", ["moderacion.jsonl", "seguridad.jsonl", "MANIFIESTO.json"], "datos/holdout_v2"),
}


def tarjeta_datos(nombre, texto):
    return f"""---
license: cc-by-nc-4.0
language: [es, en]
task_categories: [text-classification]
tags: [safety, content-moderation, colombian-spanish, synthetic, lexsi-hackathon]
---

# {nombre}

{texto}

Todo el contenido es sintético: no hay datos de personas reales. Contiene, a propósito, ejemplos de odio, violencia
o pedidos dañinos para entrenar y evaluar filtros. Parte del proyecto *Guardián colombiano* (Lexsi AI Safety &
Alignment Hackathon, 2026).
"""


def armar(org):
    shutil.rmtree(ST, ignore_errors=True)
    for m in METODOS:
        d = ST / f"guardian-tiny-aya-{m}"
        shutil.copytree(RAIZ / f"modelos/guardian-{m}-global", d)
        c = json.loads((d / "adapter_config.json").read_text())
        c["base_model_name_or_path"], c["revision"] = BASE, REV
        (d / "adapter_config.json").write_text(json.dumps(c, indent=2))
        (d / "README.md").write_text(tarjeta_modelo(org, m))
    for nombre, (texto, archivos, origen) in TARJETAS_DATOS.items():
        d = ST / nombre
        d.mkdir(parents=True)
        for a in archivos:
            shutil.copy(RAIZ / origen / a, d / a)
        (d / "README.md").write_text(tarjeta_datos(nombre, texto))
    return sorted(p.name for p in ST.iterdir())


def subir(org):
    from huggingface_hub import HfApi
    api = HfApi()
    repos = []
    for d in sorted(ST.iterdir()):
        tipo = "model" if d.name.startswith("guardian-tiny-aya-") else "dataset"
        rid = f"{org}/{d.name}"
        api.create_repo(rid, repo_type=tipo, private=False, exist_ok=True)
        api.upload_folder(folder_path=str(d), repo_id=rid, repo_type=tipo, commit_message="Guardián colombiano: publicación inicial")
        repos.append((rid, tipo))
        print("subido", tipo, rid, flush=True)
    col = api.create_collection(title="Guardián colombiano — Tiny Aya seguro", namespace=org, private=False, exists_ok=True,
                                description="Moderación segura del español colombiano con Tiny Aya: PlainSFT vs Lisa vs SafeGrad (Lexsi Hackathon 2026).")
    for rid, tipo in [(BASE, "model")] + repos:
        api.add_collection_item(col.slug, item_id=rid, item_type=tipo, exists_ok=True)
    print("colección", f"https://huggingface.co/collections/{col.slug}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", required=True)
    ap.add_argument("--ensayo", action="store_true")
    a = ap.parse_args()
    print(armar(a.org))
    if not a.ensayo:
        subir(a.org)
