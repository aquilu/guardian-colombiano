import glob, os, json, transformers
from transformers import AutoTokenizer
m = os.path.dirname(sorted(glob.glob("/kaggle/input/**/tiny-aya/transformers/global/**/config.json", recursive=True), key=len)[0])
t = AutoTokenizer.from_pretrained(m)
print("transformers", transformers.__version__, "| modelo", m)
print("padding_side:", t.padding_side, "| truncation_side:", t.truncation_side, "| pad:", t.pad_token)
cfg = json.load(open(os.path.join(m, "tokenizer_config.json")))
print("tokenizer_config padding_side:", cfg.get("padding_side"), "| claves:", sorted(k for k in cfg if "side" in k or "pad" in k))
print(t("hola", padding="max_length", max_length=6)["input_ids"])
