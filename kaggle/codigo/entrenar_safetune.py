"""Entrenamiento del guardián con SafeTune, idéntico para los tres brazos del experimento (Track 2b):

  plain     PlainSFTTrainer  — SFT sin defensa (control)
  lisa      LisaTrainer      — bi-state proximal: alterna lotes de la tarea y de alineación (Faithful)
  safegrad  SafeGradTrainer  — gradient surgery + KL a la referencia congelada (Faithful)

Configuración común: Tiny Aya Global, plantilla compacta, LoRA r=16 α=32 (q,k,v,o,gate,up,down), lote 2 sin
acumulación (SafeGrad no la admite), lr 1e-4 coseno, warmup 5 %, 2 épocas, max_len 384, fp16 en T4, semilla 42.
Datos de la tarea: train.jsonl (1.654 filas ES+EN). Alineación (Lisa, SafeGrad): alineacion.jsonl (352 filas ES+EN).
Se guarda solo el adaptador LoRA. Uso: python entrenar_safetune.py --metodo lisa --modelo <dir> --datos <dir> --salida <dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import safetune
import safetune.runner.harden._gradient_surgery as gs
from safetune.runner.harden import LisaTrainer, SafeGradTrainer
from safetune.runner.harden._gradient_surgery import PlainSFTTrainer

sys.path.insert(0, str(Path(__file__).parent))
from plantilla import compacta  # noqa: E402

LORA_MODULOS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Comun:
    """Hiperparámetros comunes sobre los argumentos que arma SafeTune, y guardado del adaptador sin fusionar."""

    def _ajustar(self, args):
        args.gradient_accumulation_steps = 1
        args.gradient_checkpointing = True
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        args.lr_scheduler_type = "cosine"
        args.warmup_ratio = 0.05
        args.max_grad_norm = 0.3
        args.save_strategy = "no"
        args.report_to = []
        args.seed = 42
        return args

    def _training_args(self, out_dir):           # PlainSFT y SafeGrad
        return self._ajustar(super()._training_args(out_dir))

    def _configure_args(self, config, out_dir):  # Lisa
        return self._ajustar(super()._configure_args(config, out_dir))

    def _save_merged(self, model, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        return out_dir


class Plain(Comun, PlainSFTTrainer):
    pass


class Lisa(Comun, LisaTrainer):
    pass


class SafeGrad(Comun, SafeGradTrainer):
    pass


def referencia_4bit(ruta, **kw):
    """SafeGrad carga una referencia congelada; en 4-bit para que quepan las dos copias en 15 GB."""
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(ruta, quantization_config=q, device_map="cuda:0", dtype=torch.float16)
    m.to = lambda *a, **k: m      # SafeGrad hace .to(dev): ya está ahí, y bnb no admite .to
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodo", choices=["plain", "lisa", "safegrad"], required=True)
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--datos", required=True, help="carpeta con train.jsonl y alineacion.jsonl")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--max-muestras", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    safetune.configure(datasets={"safety_refusals": f"{a.datos}/alineacion.jsonl"}, max_len=384, seed=42,
                       lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=LORA_MODULOS)
    tok = compacta(AutoTokenizer.from_pretrained(a.modelo))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, device_map="cuda:0", dtype=torch.float16)
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modelo.enable_input_require_grads()

    filas = [{"messages": json.loads(l)["messages"]} for l in open(f"{a.datos}/train.jsonl", encoding="utf-8")]
    if a.max_muestras:
        filas = filas[: a.max_muestras]
    comun = dict(epochs=a.epocas, batch_size=a.lote, lr=a.lr, bf16=False, fp16=True,
                 optimizer="paged_adamw_8bit", logging_steps=10)
    if a.metodo == "plain":
        trainer = Plain(modelo, tok, **comun)
    elif a.metodo == "lisa":
        trainer = Lisa(modelo, tok, lisa_rho=0.1, **comun)
    else:
        gs._load_pretrained_lm = referencia_4bit
        trainer = SafeGrad(modelo, tok, reference_model_path=a.modelo, **comun)
    ruta = trainer.train(Dataset.from_list(filas), out_dir=a.salida)
    resumen = {"metodo": a.metodo, "gpu": torch.cuda.get_device_name(0), "minutos": round((time.time() - t0) / 60, 1),
               "filas": len(filas), **{k: v for k, v in vars(a).items() if k not in ("modelo", "datos", "salida")},
               "notas": "fp16, LoRA r16 α32, lote 2 sin acumulación, coseno, warmup 5 %, max_len 384, plantilla compacta"
                        + (", referencia SafeGrad en 4-bit nf4" if a.metodo == "safegrad" else "")}
    Path(ruta, "resumen_entrenamiento.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
