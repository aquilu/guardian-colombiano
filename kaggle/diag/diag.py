import glob, json, os, sys, transformers
m = os.path.dirname(sorted(glob.glob("/kaggle/input/**/tiny-aya/transformers/global/**/config.json", recursive=True), key=len)[0])
print("TCFG", open(os.path.join(m, "tokenizer_config.json")).read().replace("\n", "\\n"))
print("MCFG", open(os.path.join(m, "config.json")).read().replace("\n", "\\n"))
raiz = next(iter(glob.glob("/kaggle/input/**/guardian-colombiano-datos", recursive=True)))
sys.path.insert(0, f"{raiz}/codigo")
from transformers import AutoTokenizer
from plantilla import compacta
t = compacta(AutoTokenizer.from_pretrained(m))
msgs = json.loads(open(f"{raiz}/train.jsonl").readline())["messages"]
texto = t.apply_chat_template(msgs, tokenize=False)
ids = t(texto, add_special_tokens=False)["input_ids"]
print("TOK", transformers.__version__, len(ids), json.dumps(ids))
