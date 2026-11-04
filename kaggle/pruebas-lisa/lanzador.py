"""Kaggle: instala SafeTune (commit fijado, desde el dataset privado) y entrena {"lisa": 1, "safegrad": 0} en paralelo, uno por GPU."""
import glob, json, os, shutil, subprocess, sys

EXTRA = ["--max-pasos", "60", "--probar", "3"]   # prueba corta de Lisa: 20 alineación + 20 tarea + 20 alineación
METODOS = {"lisa": 0}   # {metodo: gpu}
raiz = next(iter(glob.glob("/kaggle/input/**/guardian-colombiano-datos", recursive=True)))
modelo = os.path.dirname(sorted(glob.glob("/kaggle/input/**/tiny-aya/transformers/global/**/config.json", recursive=True), key=len)[0])
fuente = os.path.dirname(sorted(glob.glob(f"{raiz}/**/pyproject.toml", recursive=True), key=len)[0])
shutil.copytree(fuente, "/kaggle/working/_safetune_src", dirs_exist_ok=True)    # /kaggle/input es de solo lectura
shutil.copytree(f"{raiz}/codigo", "/kaggle/working/codigo", dirs_exist_ok=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "/kaggle/working/_safetune_src",
                    "bitsandbytes>=0.45.5", "peft>=0.18"], capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-3000:]); r.check_returncode()
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)  # peft exige >0.16
print("modelo:", modelo, "| datos:", raiz)

import threading
porgpu = {}
for metodo, gpu in METODOS.items():
    porgpu.setdefault(gpu, []).append(metodo)
codigos = {}


def cadena(gpu, metodos):
    for metodo in metodos:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        with open(f"/kaggle/working/log-{metodo}.txt", "w") as log:
            p = subprocess.run([sys.executable, "/kaggle/working/codigo/entrenar_safetune.py", "--metodo", metodo,
                                "--modelo", modelo, "--hf-dir", f"{raiz}/hf_tiny_aya", "--datos", raiz, "--salida", f"/kaggle/working/prueba-{metodo}"] + EXTRA,
                               env=env, stdout=log, stderr=subprocess.STDOUT)
        codigos[metodo] = p.returncode


hilos = [threading.Thread(target=cadena, args=(g, ms)) for g, ms in porgpu.items()]
[h.start() for h in hilos]
[h.join() for h in hilos]
procesos = {m: (type("P", (), {"returncode": c, "wait": lambda self: None})(), open(os.devnull, "w")) for m, c in codigos.items()}
fallos = []
for metodo, (p, log) in procesos.items():
    p.wait(); log.close()
    print(f"===== {metodo}: código {p.returncode}")
    print(open(f"/kaggle/working/log-{metodo}.txt").read()[-3000:])
    if p.returncode:
        fallos.append(metodo)
shutil.rmtree("/kaggle/working/_safetune_src", ignore_errors=True)
if fallos:
    raise SystemExit(f"fallaron: {fallos}")
