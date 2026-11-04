"""Kaggle (GPU T4 16 GB): fine-tuning endurecido con SafeTune · SafeGrad (gradient surgery + KL a una referencia).

Mismos datos y presupuesto que el SFT plano y Lisa en monterrey (LoRA r=16, lote efectivo 16, 2 épocas, seq 704).
Diferencias forzadas por el hardware, documentadas en el write-up:
  - el modelo que se entrena va en fp16 sin cuantizar (la T4 no tiene bf16);
  - la referencia congelada de SafeGrad se carga en 4-bit nf4 para que quepan las dos copias en 16 GB.
Salida: /kaggle/working/guardian-safegrad-global (solo el adaptador LoRA) + log.
"""
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
SAFETUNE_COMMIT = "3901a9f48a88984a181c8c9493be2d3a20ab160f"   # el mismo que se usa en monterrey
def _hay_internet() -> bool:
    import urllib.request
    try:
        urllib.request.urlopen("https://pypi.org/simple/pip/", timeout=10)
        return True
    except Exception as e:
        print("sin internet:", e)
        return False


INTERNET = _hay_internet()
print("internet:", INTERNET)
# SafeTune viaja en el dataset privado (git archive del mismo commit), así no dependemos de GitHub.
# Kaggle descomprime el .tar.gz al montar el dataset: se instala desde la carpeta que tenga pyproject.toml.
import glob
_raiz = next(iter(glob.glob("/kaggle/input/**/guardian-colombiano-datos", recursive=True)), "/kaggle/input")
_pp = sorted(glob.glob(f"{_raiz}/**/pyproject.toml", recursive=True), key=len)
_fuente = os.path.dirname(_pp[0]) if _pp else f"{_raiz}/safetune.tar.gz"
print("SafeTune desde:", _fuente)
_cmd = [sys.executable, "-m", "pip", "install", "-q", _fuente]
if INTERNET:
    _cmd += ["bitsandbytes>=0.45.5", "peft>=0.18"]
else:
    _cmd += ["--no-deps", "--no-build-isolation"]
r = subprocess.run(_cmd, capture_output=True, text=True)
print(r.stdout[-2000:], r.stderr[-4000:])
r.check_returncode()

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import safetune
import safetune.runner.harden._gradient_surgery as gs
from safetune.runner.harden import SafeGradTrainer

# Kaggle monta en /kaggle/input/datasets/<usuario>/<slug> y /kaggle/input/models/<org>/<modelo>/<framework>/<variante>/<versión>
MODELO = os.path.dirname(sorted(glob.glob("/kaggle/input/**/tiny-aya/transformers/global/**/config.json", recursive=True), key=len)[0])
DATOS = _raiz
SALIDA = "/kaggle/working/guardian-safegrad-global"


def referencia_4bit(ruta, **kw):
    """Reemplaza el cargador de la referencia de SafeGrad: misma red, congelada, en 4-bit."""
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                           bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(ruta, quantization_config=q, device_map="cuda:0", dtype=torch.float16)
    m.to = lambda *a, **k: m          # SafeGrad hace .to(dev): ya está en el dispositivo, y bnb no admite .to
    return m


gs._load_pretrained_lm = referencia_4bit


class SafeGradLoRA(SafeGradTrainer):
    def _training_args(self, out_dir):
        args = super()._training_args(out_dir)
        args.gradient_accumulation_steps = 16
        args.gradient_checkpointing = True
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        args.lr_scheduler_type = "cosine"
        args.warmup_ratio = 0.05
        args.max_grad_norm = 0.3
        args.save_strategy = "no"
        args.report_to = []
        return args

    def _save_merged(self, model, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
        return out_dir


def main() -> None:
    t0 = time.time()
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.device_count())
    safetune.configure(datasets={"safety_refusals": f"{DATOS}/alineacion.jsonl"}, max_len=704, seed=42,
                       lora_r=16, lora_alpha=32, lora_dropout=0.05,
                       lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    tok = AutoTokenizer.from_pretrained(MODELO)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    modelo = AutoModelForCausalLM.from_pretrained(MODELO, device_map="cuda:0", dtype=torch.float16)
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modelo.enable_input_require_grads()

    filas = [{"messages": json.loads(l)["messages"]} for l in open(f"{DATOS}/train.jsonl", encoding="utf-8")]
    trainer = SafeGradLoRA(modelo, tok, epochs=2, batch_size=1, lr=2e-4, bf16=False, fp16=True,
                           optimizer="paged_adamw_8bit", logging_steps=5, reference_model_path=MODELO)
    ruta = trainer.train(Dataset.from_list(filas), out_dir=SALIDA)
    resumen = {"metodo": "SafeTune SafeGrad", "safetune_commit": SAFETUNE_COMMIT, "gpu": torch.cuda.get_device_name(0),
               "minutos": round((time.time() - t0) / 60, 1), "salida": ruta,
               "notas": "modelo fp16 (T4 sin bf16), referencia 4-bit nf4, LoRA r16, lote efectivo 16, 2 épocas"}
    json.dump(resumen, open(os.path.join(SALIDA, "resumen_kaggle.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
