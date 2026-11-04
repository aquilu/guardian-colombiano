"""Paso 2: verifica y etiqueta cada post-it sintético con la política real de 12-cuentos.

Por cada ítem calcula:
  - reglas       capa 1 de 12-cuentos (reglas locales, sin IA), sobre texto_es
  - mod_es/en    capa 2: omni-moderation-latest con los umbrales de la app, en español y en inglés
  - mini_es/en   capa 3 actual: gpt-5.4-mini con el prompt del guardián (lo que corre hoy en producción)
  - juez_es      el mismo prompt con un modelo fuerte (gpt-5.4): es el control de calidad de la etiqueta
  - sistema_es/en  veredicto combinado de las tres capas, igual que guardian.evaluar()
El ítem entra al conjunto oro si juez_es coincide con el veredicto objetivo del generador.
Uso: uv run verificar.py crudo/piloto.jsonl verificado/piloto.jsonl
"""
import asyncio
import json
import sys
from pathlib import Path

from openai import AsyncOpenAI

cliente = AsyncOpenAI()

# Se carga el guardián de 12-cuentos tal cual (sin sus imports de la app) para no reescribir la política.
_FUENTE = Path(__file__).parent / "ref_guardian.py"
_ns: dict = {}
exec(_FUENTE.read_text().replace("from .config import ajustes", "").replace("from . import ia", ""), _ns)
reglas_locales = _ns["reglas_locales"]
INSTRUCCIONES_GUARDIAN = _ns["INSTRUCCIONES_GUARDIAN"]
ESQUEMA_GUARDIAN = _ns["ESQUEMA_GUARDIAN"]
CATEGORIAS_RECHAZO = _ns["CATEGORIAS_RECHAZO"]
ORDEN = _ns["ORDEN"]
UMBRAL_REVISION = 0.15  # config de 12-cuentos

# Para el texto en inglés el guardián recibe la misma política; se le avisa que el post-it llega traducido
# para que «texto en otro idioma» no cuente como fuera de tema.
NOTA_EN = "\n\nNota: esta es la edición en inglés del mural; los post-its llegan en inglés y eso es normal."


async def moderar(texto: str) -> dict:
    r = await cliente.moderations.create(model="omni-moderation-latest", input=texto)
    res = r.results[0]
    cats = [k for k, v in res.categories.model_dump().items() if v]
    pts = {k: round(v, 4) for k, v in res.category_scores.model_dump().items() if v and v >= 0.01}
    maximo = max(pts.values(), default=0.0)
    if res.flagged and any(c in CATEGORIAS_RECHAZO for c in cats):
        v = "rechazado"
    elif res.flagged or maximo >= UMBRAL_REVISION:
        v = "revision"
    else:
        v = "aprobado"
    return {"veredicto": v, "marcado": res.flagged, "categorias": cats, "puntajes": pts, "maximo": maximo}


async def guardian(texto: str, region: str, tema: str, modelo: str, en: bool = False) -> dict:
    for intento in range(3):
        try:
            r = await cliente.responses.create(
                model=modelo, instructions=INSTRUCCIONES_GUARDIAN + (NOTA_EN if en else ""),
                input=f"Región declarada: {region}. Tema declarado: {tema}.\nPost-it:\n\"\"\"\n{texto}\n\"\"\"",
                text={"format": {"type": "json_schema", "name": "guardian", "schema": ESQUEMA_GUARDIAN, "strict": True}},
                max_output_tokens=600,
            )
            return {**json.loads(r.output_text), "modelo": modelo}
        except Exception as e:
            err = str(e)[:200]
    return {"veredicto": "revision", "error": err, "modelo": modelo}


def combinar(*capas: dict) -> str:
    return max((c["veredicto"] for c in capas), key=ORDEN.get)


async def procesar(x: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        reg = reglas_locales(x["texto_es"])
        reg_en = reglas_locales(x["texto_en"])
        mod_es, mod_en, mini_es, mini_en, juez_es = await asyncio.gather(
            moderar(x["texto_es"]), moderar(x["texto_en"]),
            guardian(x["texto_es"], x["region"], x["tema"], "gpt-5.4-mini"),
            guardian(x["texto_en"], x["region"], x["tema"], "gpt-5.4-mini", en=True),
            guardian(x["texto_es"], x["region"], x["tema"], "gpt-5.4"),
        )
    # Igual que guardian.evaluar(): un rechazo de cualquier capa manda; si no, el más severo.
    sistema_es = combinar(reg, mod_es, mini_es)
    sistema_en = combinar(reg_en, mod_en, mini_en)
    return {**x, "reglas": reg, "mod_es": mod_es, "mod_en": mod_en, "mini_es": mini_es, "mini_en": mini_en,
            "juez_es": juez_es, "sistema_es": sistema_es, "sistema_en": sistema_en,
            "oro": juez_es["veredicto"] == x["veredicto_objetivo"]}


async def main() -> None:
    entrada, salida = sys.argv[1], sys.argv[2]
    hechos = set()
    if Path(salida).exists():
        hechos = {json.loads(l)["id"] for l in open(salida, encoding="utf-8")}
    items = [json.loads(l) for l in open(entrada, encoding="utf-8")]
    items = [x for x in items if x["id"] not in hechos]
    sem = asyncio.Semaphore(12)
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "a", encoding="utf-8") as f:
        for i, fut in enumerate(asyncio.as_completed([procesar(x, sem) for x in items]), 1):
            r = await fut
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(items)}")
    print(f"listo: {len(items)} nuevos → {salida}")


if __name__ == "__main__":
    asyncio.run(main())
