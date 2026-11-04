"""Kaggle: instala SafeTune (commit fijado, desde el dataset privado) y entrena {"plain": 0, "lisa": 1} en paralelo, uno por GPU."""
import glob, json, os, shutil, subprocess, sys

METODOS = {"plain": 0, "lisa": 1}   # {metodo: gpu}
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

procesos = {}
for metodo, gpu in METODOS.items():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    log = open(f"/kaggle/working/log-{metodo}.txt", "w")
    procesos[metodo] = (subprocess.Popen([sys.executable, "/kaggle/working/codigo/entrenar_safetune.py", "--metodo", metodo,
                                          "--modelo", modelo, "--datos", raiz,
                                          "--salida", f"/kaggle/working/guardian-{metodo}-global"],
                                         env=env, stdout=log, stderr=subprocess.STDOUT), log)
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
