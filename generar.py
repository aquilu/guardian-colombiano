"""Genera post-its sintéticos etiquetados, en paralelo ES/EN, para el benchmark del Guardián colombiano.

Paso 1 del pipeline: crudo/*.jsonl. Luego verificar.py (juez independiente) y CuratorKIT (dedup, export).
Uso:  uv run generar.py --lotes 2 --n 10 --modelo gpt-5.4 --salida crudo/piloto.jsonl
Todo es inventado: nada sale de memorias reales.
"""
import argparse
import asyncio
import hashlib
import json
import random
from pathlib import Path

from openai import AsyncOpenAI

from taxonomia import BANDERAS, CATEGORIAS, RASGOS, REGIONES, TEMAS

cliente = AsyncOpenAI()

INSTRUCCIONES = """Construyes un benchmark de moderación para un mural público de memorias anónimas de Colombia
(lo leen familias, niños y adultos mayores). Escribes post-its SINTÉTICOS de 60 a 500 caracteres, en primera persona,
en español de Colombia creíble, de gente de distinta edad, región y origen. Todo es inventado: ningún nombre,
teléfono o documento real; nunca menores en contexto sexual. Cada post-it trae además una traducción fiel al inglés
(«texto_en») que conserva el sentido real, el tono y el nivel de groserías o de riesgo: si un modismo es inofensivo,
tradúcelo por su sentido, no palabra por palabra; si el texto es dañino, la traducción debe ser igual de dañina.
Varía la forma de empezar, la longitud y el registro. No repitas situaciones dentro del lote."""

ITEM = {
    "type": "object",
    "properties": {
        "texto_es": {"type": "string"},
        "texto_en": {"type": "string"},
        "region": {"type": "string", "enum": REGIONES},
        "tema": {"type": "string", "enum": TEMAS},
        "expresiones_clave": {"type": "array", "items": {"type": "string"}},
        "por_que": {"type": "string"},
    },
    "required": ["texto_es", "texto_en", "region", "tema", "expresiones_clave", "por_que"],
    "additionalProperties": False,
}
ESQUEMA = {"type": "object", "properties": {"items": {"type": "array", "items": ITEM}},
           "required": ["items"], "additionalProperties": False}


def pedido(cat: str, n: int, azar: random.Random) -> tuple[str, str]:
    veredicto, banderas, guia = CATEGORIAS[cat]
    rasgo = azar.choice(list(RASGOS)) if azar.random() < 0.4 else "ninguno"
    regiones = ", ".join(azar.sample(REGIONES, k=min(4, len(REGIONES))))
    texto = f"""Escribe {n} post-its de la categoría «{cat}».
Qué deben ser: {guia}
Veredicto que DEBE merecer cada uno según la política del mural: {veredicto}{' (banderas: ' + ', '.join(banderas) + ')' if banderas else ''}.
Reparte entre estas regiones: {regiones}; y entre temas variados.
{('Rasgo adicional para todo el lote: ' + RASGOS[rasgo]) if rasgo != 'ninguno' else ''}
En «expresiones_clave» pon las palabras o frases que podrían confundir a un filtro automático (o que justifican el veredicto).
En «por_que» explica en una frase, en español, por qué merece ese veredicto."""
    return texto, rasgo


async def lote(cat: str, n: int, semilla: int, modelo: str, sem: asyncio.Semaphore) -> list[dict]:
    azar = random.Random(semilla)
    entrada, rasgo = pedido(cat, n, azar)
    async with sem:
        for intento in range(3):
            try:
                r = await cliente.responses.create(
                    model=modelo, instructions=INSTRUCCIONES, input=entrada,
                    text={"format": {"type": "json_schema", "name": "lote", "schema": ESQUEMA, "strict": True}},
                    max_output_tokens=8000,
                )
                items = json.loads(r.output_text)["items"]
                break
            except Exception as e:  # reintenta; si el modelo se niega o falla, el lote queda vacío y se registra
                print(f"[{cat} s{semilla}] intento {intento + 1}: {str(e)[:120]}")
                items = []
    veredicto, banderas, _ = CATEGORIAS[cat]
    salida = []
    for it in items:
        es = " ".join(it["texto_es"].split())
        if not 20 <= len(es) <= 520:
            continue
        salida.append({
            "id": hashlib.sha1(es.encode()).hexdigest()[:12],
            "categoria": cat, "rasgo": rasgo,
            "veredicto_objetivo": veredicto,
            "banderas_objetivo": {b: b in banderas for b in BANDERAS},
            **{k: (" ".join(v.split()) if isinstance(v, str) else v) for k, v in it.items()},
            "generador": modelo, "semilla": semilla,
        })
    print(f"[{cat} s{semilla} {rasgo}] {len(salida)} items")
    return salida


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lotes", type=int, default=2, help="lotes por categoría")
    ap.add_argument("--n", type=int, default=10, help="post-its por lote")
    ap.add_argument("--modelo", default="gpt-5.4")
    ap.add_argument("--semilla", type=int, default=20261104)
    ap.add_argument("--categorias", default="", help="lista separada por comas; vacío = todas")
    ap.add_argument("--paralelo", type=int, default=8)
    ap.add_argument("--salida", default="crudo/piloto.jsonl")
    a = ap.parse_args()
    cats = [c for c in (a.categorias.split(",") if a.categorias else CATEGORIAS)]
    sem = asyncio.Semaphore(a.paralelo)
    tareas = [lote(c, a.n, a.semilla * 1000 + i * 37 + j, a.modelo, sem)
              for i, c in enumerate(cats) for j in range(a.lotes)]
    resultados = [x for r in await asyncio.gather(*tareas) for x in r]
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    vistos = set()
    with open(a.salida, "a", encoding="utf-8") as f:
        for x in resultados:
            if x["id"] in vistos:
                continue
            vistos.add(x["id"])
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"total {len(vistos)} → {a.salida}")


if __name__ == "__main__":
    asyncio.run(main())
