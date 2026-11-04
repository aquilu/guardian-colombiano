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
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import safetune
import safetune.runner.harden._gradient_surgery as gs
from safetune.runner.harden import LisaTrainer, SafeGradTrainer
from safetune.runner.harden._gradient_surgery import PlainSFTTrainer

sys.path.insert(0, str(Path(__file__).parent))
from plantilla import compacta  # noqa: E402

LORA_MODULOS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


KL_REGISTRO: list = []
CONFIG_HF = None   # config.json de HF (revisión fijada); el de Kaggle Models trae max_position_embeddings=500000


def registrar_logs() -> list:
    """Copia cada log del HF Trainer (pérdida, lr, …) a una lista, sin cambiar nada del entrenamiento."""
    from transformers import Trainer
    registro, original = [], Trainer.log

    def log(self, logs, *args, **kwargs):
        registro.append(dict(logs))
        return original(self, logs, *args, **kwargs)

    Trainer.log = log
    return registro


def alineacion_corregida(ruta: str, tok, max_len: int):
    """Tokeniza alineacion.jsonl con la ruta validada de la tarea (tokenize_dataset: pérdida solo en la última
    respuesta) y excluye las filas que no caben completas en max_len. Devuelve (dataset, excluidas)."""
    from safetune.runner.utils.data_utils import tokenize_dataset
    filas, excluidas = [], []
    for i, l in enumerate(open(ruta, encoding="utf-8")):
        x = json.loads(l)
        msgs = [{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["response"]}]
        n = len(tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"])
        ident = f"alin-{i:03d}-{hashlib.sha1(x['prompt'].encode()).hexdigest()[:8]}"
        if n > max_len:
            excluidas.append({"id": ident, "linea": i, "tokens": n, "tipo": x.get("tipo"),
                              "categoria": x.get("categoria"), "idioma": x.get("idioma")})
            continue
        filas.append({"messages": msgs})
    return tokenizar_verificado(filas, tok, max_len, "alineación"), excluidas


def tokenizar_verificado(filas: list, tok, max_len: int, nombre: str):
    """tokenize_dataset (ruta de SafeTune) + comprobación fila por fila: relleno a la derecha, prompt y relleno en
    -100, respuesta supervisada que cierra con <|END_OF_TURN_TOKEN|>. Falla si alguna fila no cumple."""
    from safetune.runner.utils.data_utils import tokenize_dataset
    assert tok.padding_side == "right", tok.padding_side
    ds = tokenize_dataset(Dataset.from_list(filas), tok, max_len=max_len)
    sup_total = 0
    for e in ds:
        ids, lab, att = e["input_ids"], e["labels"], e["attention_mask"]
        assert att[0] == 1, "relleno a la izquierda"
        sup = tok.decode([t for t, y in zip(ids, lab) if y != -100])
        assert "<|USER_TOKEN|>" not in sup and "<|SYSTEM_TOKEN|>" not in sup, f"prompt en la pérdida: {sup[:160]}"
        assert sup.rstrip().endswith("<|END_OF_TURN_TOKEN|>"), f"sin cierre: {sup[-160:]}"
        assert all(y == -100 for y, m in zip(lab, att) if m == 0), "relleno supervisado"
        sup_total += sum(y != -100 for y in lab)
    print(f"[{nombre}] {len(ds)} filas verificadas · tokens supervisados promedio {sup_total / len(ds):.1f}")
    return ds


def vigilar_kl() -> None:
    """Envuelve la KL de SafeGrad: registra forma, dispositivo y dtype del primer cálculo y exige un valor finito."""
    import safetune.interventions.harden.safegrad as sg
    original = sg.safegrad_kl_alignment_loss

    def kl(student_logits, aligned_logits, *args, **kwargs):
        valor = original(student_logits, aligned_logits, *args, **kwargs)
        if len(KL_REGISTRO) < 3:
            KL_REGISTRO.append({"estudiante": [list(student_logits.shape), str(student_logits.device), str(student_logits.dtype)],
                                "referencia": [list(aligned_logits.shape), str(aligned_logits.device), str(aligned_logits.dtype)],
                                "kl": float(valor)})
        if not torch.isfinite(valor):
            raise FloatingPointError(f"KL no finita: {valor}")
        return valor

    sg.safegrad_kl_alignment_loss = kl


def probar_generacion(modelo, tok, ruta: str, n: int) -> list:
    """Prueba de formato: genera n respuestas del guardián y revisa JSON y esquema exacto."""
    claves = {"veredicto", "odio_o_discriminacion", "vulgaridad", "sexual_explicito", "violencia_grafica",
              "politica_partidista", "publicidad_o_spam", "datos_de_terceros", "fuera_de_tema", "explicacion"}
    modelo.eval()
    salidas = []
    for l in list(open(ruta, encoding="utf-8"))[:n]:
        msgs = json.loads(l)["messages"]
        enc = tok(tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True),
                  return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            out = modelo.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.pad_token_id)
        texto = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            j = json.loads(texto)
            ok, exacto = True, set(j) == claves
        except json.JSONDecodeError:
            ok, exacto = False, False
        salidas.append({"json_valido": ok, "esquema_exacto": exacto, "salida": texto[:300]})
    return salidas


class Comun:
    """Hiperparámetros comunes sobre los argumentos que arma SafeTune, y guardado del adaptador sin fusionar."""

    max_pasos = -1   # solo para pruebas cortas

    def _ajustar(self, args):
        if self.max_pasos > 0:
            args.max_steps = self.max_pasos
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
    m = AutoModelForCausalLM.from_pretrained(ruta, config=CONFIG_HF, quantization_config=q, device_map="cuda:0",
                                             dtype=torch.float16)
    m.to = lambda *a, **k: m      # SafeGrad hace .to(dev): ya está ahí, y bnb no admite .to
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metodo", choices=["plain", "lisa", "safegrad"], required=True)
    ap.add_argument("--modelo", required=True, help="pesos (idénticos byte a byte a los de HF)")
    ap.add_argument("--hf-dir", required=True, help="tokenizer y config de HF en la revisión fijada, con tokens_esperados.json")
    ap.add_argument("--datos", required=True, help="carpeta con train.jsonl y alineacion.jsonl")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lote", type=int, default=2)
    ap.add_argument("--max-muestras", type=int, default=0)
    ap.add_argument("--max-pasos", type=int, default=-1, help="prueba corta: corta el entrenamiento en N pasos")
    ap.add_argument("--probar", type=int, default=0, help="prueba corta: genera N respuestas del guardián al final")
    a = ap.parse_args()
    t0 = time.time()
    registro = registrar_logs()

    safetune.configure(datasets={"safety_refusals": f"{a.datos}/alineacion.jsonl"}, max_len=384, seed=42,
                       lora_r=16, lora_alpha=32, lora_dropout=0.05, lora_target_modules=LORA_MODULOS)
    # Tokenizer y config de HF, no los de Kaggle Models: con transformers 5.20 el tokenizer de Kaggle carga como
    # CohereTokenizerFast y segmenta distinto (293 frente a 276 tokens en la misma fila), y rellena a la izquierda.
    global CONFIG_HF
    CONFIG_HF = AutoConfig.from_pretrained(a.hf_dir)
    tok = compacta(AutoTokenizer.from_pretrained(a.hf_dir))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # SafeTune enmascara las primeras prompt_len posiciones: solo es correcto con relleno a la derecha. El tokenizer
    # de Kaggle Models (transformers 5.20) rellena a la izquierda por defecto y dejaba el prompt dentro de la pérdida.
    tok.padding_side = "right"
    esperado = json.load(open(f"{a.hf_dir}/tokens_esperados.json"))
    msgs = json.loads(open(f"{a.datos}/train.jsonl", encoding="utf-8").readline())["messages"]
    ids = tok(tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False)["input_ids"]
    x = json.loads(open(f"{a.datos}/alineacion.jsonl", encoding="utf-8").readline())
    ids_al = tok(tok.apply_chat_template([{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["response"]}],
                                         tokenize=False), add_special_tokens=False)["input_ids"]
    if ids != esperado["tarea"] or ids_al != esperado["alineacion"]:
        raise SystemExit(f"la tokenización no coincide con la de referencia (HF {esperado['hf_revision'][:8]}): "
                         f"{len(ids)} vs {len(esperado['tarea'])} tokens")
    print(f"tokenización idéntica a la referencia: tarea {len(ids)} tokens, alineación {len(ids_al)} tokens · {type(tok).__name__}")
    modelo = AutoModelForCausalLM.from_pretrained(a.modelo, config=CONFIG_HF, device_map="cuda:0", dtype=torch.float16)
    modelo.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    modelo.enable_input_require_grads()

    filas = [{"messages": json.loads(l)["messages"]} for l in open(f"{a.datos}/train.jsonl", encoding="utf-8")]
    if a.max_muestras:
        filas = filas[: a.max_muestras]
    comun = dict(epochs=a.epocas, batch_size=a.lote, lr=a.lr, bf16=False, fp16=True,
                 optimizer="paged_adamw_8bit", logging_steps=10)
    if a.max_pasos > 0:
        Comun.max_pasos = a.max_pasos
    extra, alineacion = {}, None
    if a.metodo in ("lisa", "safegrad"):
        # Alineación corregida, idéntica para Lisa y SafeGrad: pérdida solo en la respuesta (prompt = -100) y sin filas
        # truncadas. Sustituye a build_safety_dataset, que supervisa también el pedido dañino del usuario.
        alineacion, excluidas = alineacion_corregida(f"{a.datos}/alineacion.jsonl", tok, max_len=384)
        Path(a.salida).mkdir(parents=True, exist_ok=True)
        Path(a.salida, "alineacion_excluidas.json").write_text(json.dumps(excluidas, indent=2, ensure_ascii=False))
        print(f"alineación corregida: {len(alineacion)} filas · excluidas {len(excluidas)}:",
              [e["id"] for e in excluidas])
        extra["safety_dataset"] = alineacion
    if a.metodo == "plain":
        trainer = Plain(modelo, tok, **comun)
    elif a.metodo == "lisa":
        trainer = Lisa(modelo, tok, lisa_rho=0.1, **comun)
    else:
        gs._load_pretrained_lm = referencia_4bit
        vigilar_kl()
        trainer = SafeGrad(modelo, tok, reference_model_path=a.modelo, **comun)
    torch.cuda.reset_peak_memory_stats()
    tarea = tokenizar_verificado(filas, tok, 384, "tarea")
    ruta = trainer.train(tarea, out_dir=a.salida, **extra)
    perdidas = [r["loss"] for r in registro if "loss" in r]
    resumen = {"metodo": a.metodo, "gpu": torch.cuda.get_device_name(0), "minutos": round((time.time() - t0) / 60, 1),
               "memoria_max_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "perdidas": perdidas, "perdidas_finitas": all(map(math.isfinite, perdidas)) if perdidas else None,
               "kl": KL_REGISTRO[:3] if a.metodo == "safegrad" else None,
               "filas": len(filas), "filas_alineacion": len(alineacion) if alineacion is not None else None,
               **{k: v for k, v in vars(a).items() if k not in ("modelo", "datos", "salida")},
               "notas": "fp16, LoRA r16 α32, lote 2 sin acumulación, coseno, warmup 5 %, max_len 384, plantilla compacta"
                        + ("; alineación corregida (pérdida solo en la respuesta)" if alineacion is not None else "")
                        + ("; referencia SafeGrad en 4-bit nf4; la KL de SafeTune cubre todas las posiciones"
                           if a.metodo == "safegrad" else "")}
    if a.probar:
        resumen["prueba_generacion"] = probar_generacion(modelo, tok, f"{a.datos}/train.jsonl", a.probar)
    Path(ruta, "resumen_entrenamiento.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
