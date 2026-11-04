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


CIRUGIA_REGISTRO: list = []


def diagnosticar_cirugia(trainer_hf_cls, pasos: int = 12) -> None:
    """Solo diagnóstico (no cambia el método): antes de cada cirugía registra la escala del GradScaler, las normas de
    g_user (tal como está en .grad) y de g_align, su producto y si hay proyección; y después, la norma de g_final."""
    original = trainer_hf_cls._apply_global_surgery

    def envuelta(self, model, g_align):
        if len(CIRUGIA_REGISTRO) < pasos:
            params = self._trainable_params(model)
            gu = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).float().reshape(-1) for p in params])
            ga = torch.cat([g.float().reshape(-1).to(gu.device) for g in g_align])
            scaler = getattr(self.accelerator, "scaler", None)
            fila = {"paso": int(self.state.global_step), "escala": float(scaler.get_scale()) if scaler is not None else None,
                    "norma_g_user_en_grad": float(gu.norm()), "norma_g_align": float(ga.norm()),
                    "producto": float((gu * ga).sum()), "proyecta": bool((gu * ga).sum() < 0)}
        original(self, model, g_align)
        if len(CIRUGIA_REGISTRO) < pasos:
            gf = torch.cat([p.grad.float().reshape(-1) for p in self._trainable_params(model)])
            fila["norma_g_final"] = float(gf.norm())
            if fila["escala"]:
                fila["norma_g_user_desescalada"] = fila["norma_g_user_en_grad"] / fila["escala"]
                fila["norma_g_align_tras_desescalar"] = fila["norma_g_align"] / fila["escala"]
            CIRUGIA_REGISTRO.append(fila)
            print("cirugia", json.dumps(fila), flush=True)

    trainer_hf_cls._apply_global_surgery = envuelta


def _vec_cpu(tensores) -> torch.Tensor:
    return torch.cat([t.detach().float().reshape(-1).cpu() for t in tensores])


def _proyectar(u: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Eq. 4 de SafeGrad: proyecta u solo si u·a < 0."""
    d = (u * a).sum()
    return u - (d / ((a * a).sum() + 1e-12)) * a if d < 0 else u


def corregir_amp(trainer_hf_cls) -> None:
    """Compatibilidad de SafeGrad con AMP fp16. El HF Trainer deja en .grad el gradiente de la tarea multiplicado por
    la escala s del GradScaler; g_align (torch.autograd.grad sobre la KL) no lleva esa escala. Se pasa s·g_align a la
    función original de SafeTune: la proyección es invariante a la escala de g_align (grado 0) y el combinado queda
    s·(g'_user + ρ·g_align), que el Trainer desescala una sola vez → exactamente la fórmula de SafeGrad (Eq. 3–5)."""
    original = trainer_hf_cls._apply_global_surgery

    def corregida(self, model, g_align):
        scaler = getattr(self.accelerator, "scaler", None)
        if scaler is not None and scaler.is_enabled():
            s = scaler.get_scale()
            g_align = [g * s for g in g_align]
        return original(self, model, g_align)

    trainer_hf_cls._apply_global_surgery = corregida


def diagnosticar_cirugia(trainer_hf_cls, pasos: int = 12) -> None:
    """Solo diagnóstico (no cambia el método; cálculos en CPU para no gastar VRAM). Antes de la cirugía: escala s,
    normas de g_user (en .grad, escalado) y de g_align tal como llega, coseno y proyección. Comprueba la homogeneidad
    (g'(s·u)/s == g'(u)). Después: contribución relativa ‖ρ·a‖/‖g'_user‖ en unidades desescaladas."""
    original = trainer_hf_cls._apply_global_surgery

    def envuelta(self, model, g_align):
        if len(CIRUGIA_REGISTRO) >= pasos:
            return original(self, model, g_align)
        params = self._trainable_params(model)
        u_esc = _vec_cpu([(p.grad if p.grad is not None else torch.zeros_like(p)) for p in params])
        a_llega = _vec_cpu(g_align)
        scaler = getattr(self.accelerator, "scaler", None)
        s = float(scaler.get_scale()) if scaler is not None and scaler.is_enabled() else 1.0
        u = u_esc / s
        coseno = float((u * a_llega).sum() / (u.norm() * a_llega.norm() + 1e-12))
        homog = float((_proyectar(u_esc, a_llega) / s - _proyectar(u, a_llega)).norm() / (_proyectar(u, a_llega).norm() + 1e-12))
        original(self, model, g_align)
        g_final = _vec_cpu([p.grad for p in params]) / s
        u_proy = _proyectar(u, a_llega)
        aporte = g_final - u_proy                     # lo que quedó del término ρ·g_align tras desescalar
        fila = {"paso": int(self.state.global_step), "escala": s, "norma_g_user": float(u.norm()),
                "norma_g_align_llega": float(a_llega.norm()), "coseno": coseno, "proyecta": bool((u * a_llega).sum() < 0),
                "error_homogeneidad": homog, "norma_g_user_proyectado": float(u_proy.norm()),
                "norma_aporte_alineacion": float(aporte.norm()),
                "aporte_relativo": float(aporte.norm() / (u_proy.norm() + 1e-12))}
        CIRUGIA_REGISTRO.append(fila)
        print("\ncirugia " + json.dumps(fila), flush=True)

    trainer_hf_cls._apply_global_surgery = envuelta


LISA_REGISTRO: dict = {"estados": [], "generaciones": []}


def diagnosticar_lisa(trainer_hf_cls, tok, ruta: str, n: int = 3, max_eventos: int = 4) -> None:
    """Solo diagnóstico: registra el estado (alineación/tarea) de cada paso y, en cada cambio de fase, genera n
    respuestas del guardián para ver si conserva el esquema JSON tras alternar."""
    original = trainer_hf_cls._check_mode

    def envuelta(self, inputs):
        antes = self._lisa.status if getattr(self, "_lisa_bistate_enabled", False) else "tarea"
        salida = original(self, inputs)
        despues = self._lisa.status if getattr(self, "_lisa_bistate_enabled", False) else "tarea"
        LISA_REGISTRO["estados"].append(despues)
        if antes != despues and len(LISA_REGISTRO["generaciones"]) < max_eventos:
            estaba = self.model.training
            gen = probar_generacion(self.model, tok, ruta, n)
            if estaba:
                self.model.train()
            LISA_REGISTRO["generaciones"].append({"paso": int(self.state.global_step), "tras_fase": antes,
                                                  "json_valido": sum(g["json_valido"] for g in gen),
                                                  "esquema_exacto": sum(g["esquema_exacto"] for g in gen), "n": n,
                                                  "muestra": gen[0]["salida"][:200]})
            print("\nlisa " + json.dumps(LISA_REGISTRO["generaciones"][-1], ensure_ascii=False), flush=True)
        return salida

    trainer_hf_cls._check_mode = envuelta


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
    ap.add_argument("--diagnostico-cirugia", type=int, default=0, help="solo SafeGrad: registra la cirugía en N pasos")
    ap.add_argument("--diagnostico-lisa", action="store_true", help="solo Lisa: estados por paso y generación en cada cambio")
    ap.add_argument("--safegrad-rho", type=float, default=None, help="ρ de SafeGrad (por defecto, el de SafeTune: 1.0)")
    ap.add_argument("--safegrad-corregir-amp", action="store_true", help="pasa s·g_align a la cirugía (fp16 con GradScaler)")
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
        if a.diagnostico_lisa:
            import safetune.harden as HARD
            diagnosticar_lisa(HARD.LisaHFTrainer, tok, f"{a.datos}/train.jsonl")
        trainer = Lisa(modelo, tok, lisa_rho=0.1, **comun)
    else:
        import safetune.harden as HARD
        gs._load_pretrained_lm = referencia_4bit
        vigilar_kl()
        if a.safegrad_corregir_amp:
            corregir_amp(HARD.SafeGradHFTrainer)            # primero la corrección…
        if a.diagnostico_cirugia:
            diagnosticar_cirugia(HARD.SafeGradHFTrainer, pasos=a.diagnostico_cirugia)   # …y el diagnóstico la envuelve
        rho = {} if a.safegrad_rho is None else {"rho": a.safegrad_rho}
        trainer = SafeGrad(modelo, tok, reference_model_path=a.modelo, **rho, **comun)
    torch.cuda.reset_peak_memory_stats()
    tarea = tokenizar_verificado(filas, tok, 384, "tarea")
    ruta = trainer.train(tarea, out_dir=a.salida, **extra)
    perdidas = [r["loss"] for r in registro if "loss" in r]
    resumen = {"metodo": a.metodo, "gpu": torch.cuda.get_device_name(0), "minutos": round((time.time() - t0) / 60, 1),
               "memoria_max_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "perdidas": perdidas, "perdidas_finitas": all(map(math.isfinite, perdidas)) if perdidas else None,
               "kl": KL_REGISTRO[:3] if a.metodo == "safegrad" else None,
               "cirugia": CIRUGIA_REGISTRO or None,
               "lisa": ({"pasos_alineacion": LISA_REGISTRO["estados"].count("alignment"),
                         "pasos_tarea": len(LISA_REGISTRO["estados"]) - LISA_REGISTRO["estados"].count("alignment"),
                         "secuencia": "".join("A" if e == "alignment" else "T" for e in LISA_REGISTRO["estados"]),
                         "generaciones": LISA_REGISTRO["generaciones"]} if LISA_REGISTRO["estados"] else None),
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
