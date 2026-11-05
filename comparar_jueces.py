"""¿Hace diferencia Sonnet 5.5 frente a Haiku 5.5 como segundo juez? Se decide sobre datos que NO son del holdout v2:
40 post-its del holdout v1 (exploratorio) y 20 pedidos de datos/seguridad/eval.jsonl, muestra estratificada fija.

Regla de decisión (fijada antes de mirar resultados): si Haiku coincide con Sonnet en ≥ 95 % de los veredictos y
κ de Cohen ≥ 0,90 en ambos conjuntos, se usa Haiku; si no, Sonnet. Se reporta también el acuerdo de cada uno con GPT-5.4.
Uso: OPENROUTER_API_KEY=… uv run comparar_jueces.py
"""
import asyncio
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import openrouter_cliente as oc
from holdout_v2_juzgar import ESQUEMA_MOD, ESQUEMA_SEG, RUBRICA_MOD, RUBRICA_SEG, gpt

oc.MODELOS["juez_haiku"] = ("anthropic/claude-haiku-5.5", ["Anthropic"], False)
SEMILLA = 20261107


def muestra(xs, clave, n, azar):
    grupos = defaultdict(list)
    for x in xs:
        grupos[x[clave]].append(x)
    out = []
    while len(out) < n:
        for g in sorted(grupos):
            if grupos[g] and len(out) < n:
                out.append(grupos[g].pop(azar.randrange(len(grupos[g]))))
    return out


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


async def main() -> None:
    azar = random.Random(SEMILLA)
    orq = oc.OpenRouter()
    sem = asyncio.Semaphore(6)
    v1 = [json.loads(l) for l in open("datos/holdout/holdout_v1.jsonl", encoding="utf-8")]
    seg = [json.loads(l) for l in open("datos/seguridad/eval.jsonl", encoding="utf-8") if json.loads(l)["idioma"] == "es"]
    mod_m, seg_m = muestra(v1, "categoria", 40, azar), muestra(seg, "categoria", 20, azar)

    async def j_mod(x):
        u = f"Región declarada: {x['region']}. Tema declarado: {x['tema']}.\nPost-it:\n\"\"\"\n{x['texto_es']}\n\"\"\""
        s, h = await asyncio.gather(
            orq.json("juez_claude", f"cmp-sonnet-{x['id']}", RUBRICA_MOD, u, ESQUEMA_MOD, "juicio", 600, 0),
            orq.json("juez_haiku", f"cmp-haiku-{x['id']}", RUBRICA_MOD, u, ESQUEMA_MOD, "juicio", 600, 0))
        return {"id": x["id"], "gpt": x["veredicto_juez"], "sonnet": (s or {}).get("veredicto"), "haiku": (h or {}).get("veredicto")}

    async def j_seg(x):
        u = f"PEDIDO:\n\"\"\"\n{x['prompt']}\n\"\"\""
        s, h, g = await asyncio.gather(
            orq.json("juez_claude", f"cmp-sonnet-{x['id']}", RUBRICA_SEG, u, ESQUEMA_SEG, "juicio", 300, 0),
            orq.json("juez_haiku", f"cmp-haiku-{x['id']}", RUBRICA_SEG, u, ESQUEMA_SEG, "juicio", 300, 0),
            gpt(f"cmp-gpt-{x['id']}", "gpt-5.4", RUBRICA_SEG, u, ESQUEMA_SEG, sem))
        return {"id": x["id"], "gpt": (g or {}).get("clase"), "sonnet": (s or {}).get("clase"), "haiku": (h or {}).get("clase")}

    res = {"moderacion": await asyncio.gather(*map(j_mod, mod_m)), "seguridad": await asyncio.gather(*map(j_seg, seg_m))}
    informe = {"regla": "Haiku si acuerdo con Sonnet ≥ 95 % y κ ≥ 0,90 en ambos conjuntos; si no, Sonnet", "conjuntos": {}}
    usar_haiku = True
    for c, rs in res.items():
        rs = [r for r in rs if r["sonnet"] and r["haiku"] and r["gpt"]]
        s, h, g = [r["sonnet"] for r in rs], [r["haiku"] for r in rs], [r["gpt"] for r in rs]
        d = {"n": len(rs), "acuerdo_sonnet_haiku": sum(a == b for a, b in zip(s, h)) / len(rs), "kappa_sonnet_haiku": kappa(s, h),
             "acuerdo_sonnet_gpt": sum(a == b for a, b in zip(s, g)) / len(rs), "kappa_sonnet_gpt": kappa(s, g),
             "acuerdo_haiku_gpt": sum(a == b for a, b in zip(h, g)) / len(rs), "kappa_haiku_gpt": kappa(h, g),
             "desacuerdos_sonnet_haiku": [r for r in rs if r["sonnet"] != r["haiku"]]}
        usar_haiku &= d["acuerdo_sonnet_haiku"] >= 0.95 and d["kappa_sonnet_haiku"] >= 0.90
        informe["conjuntos"][c] = d
    informe["decision"] = "anthropic/claude-haiku-5.5" if usar_haiku else "anthropic/claude-sonnet-5.5"
    informe["gasto_openrouter_acumulado_usd"] = round(orq.gastado, 4)
    Path("comparacion_jueces.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False))
    for c, d in informe["conjuntos"].items():
        print(c, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items() if k != "desacuerdos_sonnet_haiku"},
              "| desacuerdos S/H:", [(r["sonnet"], r["haiku"]) for r in d["desacuerdos_sonnet_haiku"]])
    print("DECISIÓN:", informe["decision"], "| gasto acumulado", informe["gasto_openrouter_acumulado_usd"], "USD")


if __name__ == "__main__":
    asyncio.run(main())
