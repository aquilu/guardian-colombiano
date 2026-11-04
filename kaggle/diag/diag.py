import os, importlib, subprocess, sys
for raiz, dirs, files in os.walk("/kaggle/input"):
    nivel = raiz.count(os.sep) - 2
    if nivel <= 4:
        print(raiz, "|", files[:6], "| dirs:", dirs[:6])
for m in ["torch", "transformers", "peft", "bitsandbytes", "trl", "datasets", "accelerate", "safetensors", "torchao"]:
    try:
        print(m, importlib.import_module(m).__version__)
    except Exception as e:
        print(m, "NO:", type(e).__name__, str(e)[:80])
import torch
print("cuda", torch.cuda.is_available(), torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
