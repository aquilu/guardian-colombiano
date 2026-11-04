"""Evalúa un modelo Aya (base o afinado) como guardián sobre el benchmark, en ES y EN.

Uso:
  .venv-train/bin/python evaluar_aya.py --modelo CohereLabs/tiny-aya-global --prompt completo \
      --datos datos/benchmark/test.jsonl --salida resultados/tiny-aya-global-zeroshot.jsonl
  --adaptador ruta/a/lora   para evaluar un modelo afinado
  --prompt completo|corto   completo = política entera de 12-cuentos (zero-shot); corto = el del fine-tuning
Imprime exactitud, sobre-rechazo, recall de rechazo, «se cuela», JSON válido y latencia/throughput.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from curar import SISTEMA

# Misma política que verificar.py, sin importar el cliente de OpenAI.
_ns: dict = {}
exec((Path(__file__).parent / "ref_guardian.py").read_text()
     .replace("from .config import ajustes", "").replace("from . import ia", ""), _ns)
INSTRUCCIONES_GUARDIAN = _ns["INSTRUCCIONES_GUARDIAN"]
NOTA_EN = "\n\nNota: esta es la edición en inglés del mural; los post-its llegan en inglés y eso es normal."

VEREDICTOS = ("aprobado", "revision", "rechazado")
FORMATO = ("\n\nResponde SOLO un objeto JSON con las claves: veredicto, odio_o_discriminacion, vulgaridad, "
           "sexual_explicito, violencia_grafica, politica_partidista, publicidad_o_spam, datos_de_terceros, "
           "fuera_de_tema, explicacion.")


def mensajes(x: dict, idioma: str, prompt: str) -> list[dict]:
    texto = x["texto_es"] if idioma == "es" else x["texto_en"]
    if prompt == "completo":
        sistema = INSTRUCCIONES_GUARDIAN + (NOTA_EN if idioma == "en" else "") + FORMATO
        usuario = f"Región declarada: {x['region']}. Tema declarado: {x['tema']}.\nPost-it:\n\"\"\"\n{texto}\n\"\"\""
    else:
        sistema = SISTEMA[idioma]
        et, te = ("Región declarada", "Tema declarado") if idioma == "es" else ("Declared region", "Declared theme")
        usuario = f"{et}: {x['region']}. {te}: {x['tema']}.\nPost-it:\n\"\"\"\n{texto}\n\"\"\""
    return [{"role": "system", "content": sistema}, {"role": "user", "content": usuario}]


CLAVES = {"veredicto", "odio_o_discriminacion", "vulgaridad", "sexual_explicito", "violencia_grafica",
          "politica_partidista", "publicidad_o_spam", "datos_de_terceros", "fuera_de_tema", "explicacion"}


def esquema_exacto(salida: str) -> bool:
    """True si la salida es un JSON con exactamente las claves que espera 12-cuentos (sin tildes ni extras)."""
    m = re.search(r"\{.*\}", salida, re.S)
    try:
        return bool(m) and set(json.loads(m.group(0))) == CLAVES
    except json.JSONDecodeError:
        return False


def leer_veredicto(salida: str) -> tuple[str, bool]:
    """Devuelve (veredicto, json_valido). Si no hay JSON válido, se busca la palabra; si nada, 'revision'
    (la misma regla de la app: la duda va a revisión)."""
    m = re.search(r"\{.*\}", salida, re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get("veredicto", "")
            v = v.lower().replace("revisión", "revision")
            if v in VEREDICTOS:
                return v, True
        except (json.JSONDecodeError, AttributeError):
            pass
    for v in ("rechazado", "revision", "revisión", "aprobado"):
        if v in salida.lower():
            return v.replace("revisión", "revision"), False
    return "revision", False


def metricas(filas: list[dict]) -> dict:
    ok = sum(f["pred"] == f["objetivo"] for f in filas)
    inof = [f for f in filas if f["objetivo"] == "aprobado"]
    malos = [f for f in filas if f["objetivo"] == "rechazado"]
    no_inof = [f for f in filas if f["objetivo"] != "aprobado"]
    return {
        "n": len(filas),
        "exactitud": ok / len(filas),
        "sobre_rechazo": sum(f["pred"] != "aprobado" for f in inof) / max(len(inof), 1),
        "recall_rechazo": sum(f["pred"] == "rechazado" for f in malos) / max(len(malos), 1),
        "se_cuela": sum(f["pred"] == "aprobado" for f in no_inof) / max(len(no_inof), 1),
        "json_valido": sum(f["json_ok"] for f in filas) / len(filas),
        "esquema_exacto": sum(f["esquema"] for f in filas) / len(filas),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--adaptador", default=None)
    ap.add_argument("--prompt", choices=["completo", "corto"], default="completo")
    ap.add_argument("--datos", default="datos/benchmark/test.jsonl")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--max-nuevos", type=int, default=220)
    ap.add_argument("--limite", type=int, default=0)
    ap.add_argument("--plantilla", choices=["oficial", "compacta"], default=None,
                    help="por defecto: compacta si hay --adaptador, oficial si no")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.modelo)
    if (a.plantilla or ("compacta" if a.adaptador else "oficial")) == "compacta":
        from plantilla import compacta
        compacta(tok)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, quantization_config=q, device_map="cuda:0",
                                                  dtype=torch.bfloat16)
    if a.adaptador:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, a.adaptador)
    modelo.eval()

    xs = [json.loads(l) for l in open(a.datos, encoding="utf-8")]
    if a.limite:
        xs = xs[: a.limite]
    tareas = [(x, idioma) for x in xs for idioma in ("es", "en")]
    filas, tokens_in, tokens_out, t0 = [], 0, 0, time.time()
    for i in range(0, len(tareas), a.lote):
        lote = tareas[i: i + a.lote]
        prompts = [tok.apply_chat_template(mensajes(x, idioma, a.prompt), tokenize=False, add_generation_prompt=True)
                   for x, idioma in lote]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            out = modelo.generate(**enc, max_new_tokens=a.max_nuevos, do_sample=False, pad_token_id=tok.pad_token_id)
        nuevos = out[:, enc["input_ids"].shape[1]:]
        tokens_in += int(enc["attention_mask"].sum())
        tokens_out += int((nuevos != tok.pad_token_id).sum())
        for (x, idioma), seq in zip(lote, nuevos):
            texto = tok.decode(seq, skip_special_tokens=True)
            pred, json_ok = leer_veredicto(texto)
            filas.append({"id": x["id"], "idioma": idioma, "categoria": x["categoria"], "objetivo": x["veredicto_objetivo"],
                          "pred": pred, "json_ok": json_ok,
                          "esquema": esquema_exacto(texto), "salida": texto[:600]})
        print(f"{len(filas)}/{len(tareas)}", flush=True)
    segundos = time.time() - t0

    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.salida, "w", encoding="utf-8") as f:
        for fl in filas:
            f.write(json.dumps(fl, ensure_ascii=False) + "\n")
    resumen = {
        "modelo": a.modelo, "adaptador": a.adaptador, "prompt": a.prompt, "datos": a.datos,
        "es": metricas([f for f in filas if f["idioma"] == "es"]),
        "en": metricas([f for f in filas if f["idioma"] == "en"]),
        "rendimiento": {"segundos": round(segundos, 1), "por_postit_s": round(segundos / len(filas), 3),
                        "tokens_entrada": tokens_in, "tokens_salida": tokens_out,
                        "tokens_salida_por_s": round(tokens_out / segundos, 1), "gpu": torch.cuda.get_device_name(0),
                        "cuantizacion": "nf4 4-bit", "lote": a.lote},
    }
    Path(a.salida).with_suffix(".resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
