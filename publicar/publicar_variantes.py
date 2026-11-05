"""Publica los adaptadores de Lisa sobre tiny-aya-water y tiny-aya-fire (ejecución #22) y los agrega a la colección.

Corrige base_model_name_or_path (ruta de Kaggle → CohereLabs/tiny-aya-<variante>) y fija la revisión de HF.
Uso (monterrey, OAuth de aquilu con acceso a vasmi-sas): python publicar/publicar_variantes.py [--ensayo]
"""
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ST = RAIZ / "publicar/_staging_variantes"
ORG = "vasmi-sas"
COLECCION = "vasmi-sas/guardian-colombiano-models-6aec689cd4567b9382632800"
VAR = {"water": ("41a99216d12b3ef4881df762934186e5e4d2bbc1", "European and Asia-Pacific languages, Spanish included",
                 "81,1 / 75,7", "12,1 / 18,2", "91,9", "50,0 / 47,1"),
       "fire": ("73500c80fcb6c8bea3e830c21d5d2920249a0e1a", "South Asian languages (off-region control)",
                "78,4 / 78,4", "21,2 / 27,3", "89,2", "47,1 / 47,1")}


def tarjeta(v, rev, foco, acc, bloq, cons, asr):
    base = f"CohereLabs/tiny-aya-{v}"
    return f"""---
license: cc-by-nc-4.0
base_model: {base}
library_name: peft
language: [es, en]
pipeline_tag: text-generation
tags: [safety, content-moderation, colombian-spanish, lora, safetune, lisa, lexsi-hackathon, regional-variant]
datasets: [{ORG}/guardian-colombiano-moderacion, {ORG}/guardian-colombiano-alineacion, {ORG}/guardian-colombiano-holdout-v2]
---

# Guardián colombiano — Tiny Aya {v.capitalize()} + Lisa (LoRA) · run #22

**Exploratory, pre-registered variant.** It uses the same frozen Lisa recipe as
[`{ORG}/guardian-tiny-aya-lisa`](https://huggingface.co/{ORG}/guardian-tiny-aya-lisa): the same data, alignment pairs,
hyperparameters and seed, nothing tuned per variant. The base is [{base}](https://huggingface.co/{base}) (revision
`{rev[:8]}`), which focuses on {foco}. It was trained to answer a Lexsi mentor's question: *does the base model's
regional focus matter?*

| Frozen holdout v2 (ES / EN) | This adapter | global + Lisa (recommended) |
|---|---|---|
| Moderation accuracy | {acc} | 81,1 / 82,4 |
| Blocks approved notes | {bloq} | 15,2 / 18,2 |
| Same verdict ES/EN | {cons} % | 95,9 % |
| Harmful help (ASR), GPT-5.4 judge: base → Lisa | {asr} → **0,0 / 0,0** | 50,0 / 58,8 → **0,0 / 0,0** |

Lisa's safety effect holds on every base. For deployment use the `global` adapter, which has the best ES/EN
consistency and English accuracy. Details:
[VARIANTES.md](https://github.com/aquilu/guardian-colombiano/blob/master/VARIANTES.md).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{ORG}/guardian-tiny-aya-lisa-{v}")
base = AutoModelForCausalLM.from_pretrained("{base}", revision="{rev}", device_map="auto")
modelo = PeftModel.from_pretrained(base, "{ORG}/guardian-tiny-aya-lisa-{v}")
```

License CC-BY-NC 4.0, inherited from Tiny Aya. It supports a human moderation committee; it is not an autonomous
moderator. Project: [github.com/aquilu/guardian-colombiano](https://github.com/aquilu/guardian-colombiano).
"""


def main(ensayo):
    shutil.rmtree(ST, ignore_errors=True)
    for v, (rev, *resto) in VAR.items():
        d = ST / f"guardian-tiny-aya-lisa-{v}"
        shutil.copytree(RAIZ / f"modelos/guardian-lisa-{v}", d)
        c = json.loads((d / "adapter_config.json").read_text())
        c["base_model_name_or_path"], c["revision"] = f"CohereLabs/tiny-aya-{v}", rev
        (d / "adapter_config.json").write_text(json.dumps(c, indent=2))
        (d / "README.md").write_text(tarjeta(v, rev, *resto))
        print(d.name, sorted(p.name for p in d.iterdir()))
    if ensayo:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    for v in VAR:
        rid = f"{ORG}/guardian-tiny-aya-lisa-{v}"
        api.create_repo(rid, private=False, exist_ok=True)
        api.upload_folder(folder_path=str(ST / f"guardian-tiny-aya-lisa-{v}"), repo_id=rid,
                          commit_message="Lisa sobre tiny-aya-" + v + " (ejecución #22)")
        api.add_collection_item(COLECCION, item_id=rid, item_type="model", exists_ok=True,
                                note=f"Run #22: same Lisa recipe on Tiny Aya {v.capitalize()} (regional-variant check)")
        print("subido", rid)


if __name__ == "__main__":
    main("--ensayo" in sys.argv)
