"""Deriva de seguridad, paso 1: genera las respuestas del modelo como asistente general (sin el prompt del mural).

Uso: .venv-train/bin/python medir_seguridad.py --modelo CohereLabs/tiny-aya-global [--adaptador ruta | --fusionado ruta]
       --salida resultados/seguridad-base.jsonl
Luego: uv run juzgar_seguridad.py resultados/seguridad-base.jsonl
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adaptador", default=None, help="LoRA de PEFT sobre --modelo")
    ap.add_argument("--fusionado", default=None, help="checkpoint completo (p. ej. el que guarda SafeTune)")
    ap.add_argument("--datos", default="datos/seguridad/eval.jsonl")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--max-nuevos", type=int, default=200)
    a = ap.parse_args()

    ruta = a.fusionado or a.modelo
    tok = AutoTokenizer.from_pretrained(a.modelo)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    modelo = AutoModelForCausalLM.from_pretrained(ruta, quantization_config=q, device_map="cuda:0", dtype=torch.bfloat16)
    if a.adaptador:
        from peft import PeftModel
        modelo = PeftModel.from_pretrained(modelo, a.adaptador)
    modelo.eval()

    xs = [json.loads(l) for l in open(a.datos, encoding="utf-8")]
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.salida, "w", encoding="utf-8") as f:
        for i in range(0, len(xs), a.lote):
            lote = xs[i: i + a.lote]
            prompts = [tok.apply_chat_template([{"role": "user", "content": x["prompt"]}], tokenize=False,
                                               add_generation_prompt=True) for x in lote]
            enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
            with torch.no_grad():
                out = modelo.generate(**enc, max_new_tokens=a.max_nuevos, do_sample=False, pad_token_id=tok.pad_token_id)
            for x, seq in zip(lote, out[:, enc["input_ids"].shape[1]:]):
                f.write(json.dumps({**x, "respuesta": tok.decode(seq, skip_special_tokens=True),
                                    "modelo": ruta, "adaptador": a.adaptador}, ensure_ascii=False) + "\n")
            print(f"{min(i + a.lote, len(xs))}/{len(xs)}", flush=True)


if __name__ == "__main__":
    main()
