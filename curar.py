"""Paso 3: curado con CuratorKIT y partición sin fugas.

1. Toma solo el oro (juez = objetivo) y quita las negativas del generador.
2. CuratorKIT: exact + MinHash + embedding dedup (multilingüe) y limpieza sobre texto_es, con manifiesto.
3. Parte por post-it (ES y EN del mismo post-it en la misma partición), estratificado por categoría.
4. Escribe datos/guardian/{train,validation,test}.jsonl (formato messages, ES+EN) y datos/benchmark/test.jsonl
   con todas las líneas base, para AuditKIT.
Uso: ../.venv-curator/bin/python curar.py verificado/piloto.jsonl verificado/v1.jsonl
"""
import json
import random
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

from taxonomia import BANDERAS

SEMILLA = 20261104
PARTES = {"train": 0.70, "validation": 0.15, "test": 0.15}
NEGATIVA = re.compile(r"^(no (puedo|ayudo)|si quieres|opci[oó]n segura|lo siento)", re.I)
ETAPA = Path("curado")

# Prompt corto a propósito: el modelo afinado aprende la política; así cada llamada cuesta menos tokens.
SISTEMA = {
    "es": ("Eres el guardián editorial del mural «12 Cuentos» (memorias anónimas de Colombia, público familiar). "
           "Evalúa el post-it y responde solo JSON con: veredicto (aprobado | revision | rechazado), "
           "las 8 banderas booleanas y una explicación de una frase para el comité."),
    "en": ("You are the editorial guardian of the «12 Cuentos» mural (anonymous memories from Colombia, family "
           "audience). Evaluate the post-it and answer only JSON with: veredicto (aprobado | revision | rechazado), "
           "the 8 boolean flags and a one-sentence explanation for the committee."),
}


def cargar(rutas: list[str]) -> list[dict]:
    vistos, xs = set(), []
    for r in rutas:
        for l in open(r, encoding="utf-8"):
            x = json.loads(l)
            if x["id"] in vistos or not x["oro"] or NEGATIVA.search(x["texto_es"]):
                continue
            vistos.add(x["id"])
            xs.append(x)
    return xs


def _clave(t: str) -> str:
    """Clave robusta al limpiador de CuratorKIT (unicode, comillas, espacios)."""
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"[^\w]+", "", t)


def curatorkit(xs: list[dict]) -> set[str]:
    """Corre CuratorKIT sobre texto_es; devuelve los ids que sobreviven."""
    ETAPA.mkdir(exist_ok=True)
    entrada = ETAPA / "entrada_alpaca.jsonl"
    with open(entrada, "w", encoding="utf-8") as f:
        for x in xs:
            f.write(json.dumps({"instruction": x["texto_es"], "input": "", "output": x["veredicto_objetivo"],
                                "id": x["id"], "categoria": x["categoria"]}, ensure_ascii=False) + "\n")
    yaml = f"""name: guardian_colombiano_curado
version: "1.0.0"
readers:
  - type: jsonl
    path: {entrada.resolve()}
    format: alpaca
normalizers:
  - type: text_cleaner
    transforms:            # strip_html apagado: las etiquetas falsas son parte de las trampas de inyección
      strip_html: false
      normalise_unicode: true
      fix_encoding_artifacts: true
      collapse_whitespace: true
      remove_control_chars: true
  - type: exact_dedup
  - type: minhash_dedup
    minhash_threshold: 0.8
  - type: embedding_dedup
    embedding_model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    embedding_threshold: 0.88
    embedding_index_dir: {(ETAPA / 'indice').resolve()}
exporters:
  - type: alpaca
"""
    (ETAPA / "pipeline.yaml").write_text(yaml)
    salida = ETAPA / "salida"
    bin_ck = Path(sys.executable).parent / "curatorkit"
    import shutil
    shutil.rmtree(salida, ignore_errors=True)
    shutil.rmtree(ETAPA / "indice", ignore_errors=True)
    subprocess.run([str(bin_ck), "run", str(ETAPA / "pipeline.yaml"), "--output-dir", str(salida)], check=True)
    por_texto = {_clave(x["texto_es"]): x["id"] for x in xs}
    vivos = set()
    for l in open(salida / "sft_alpaca.jsonl", encoding="utf-8"):
        t = _clave(json.loads(l)["instruction"])
        if t in por_texto:
            vivos.add(por_texto[t])
    return vivos


def partir(xs: list[dict]) -> dict[str, list[dict]]:
    azar = random.Random(SEMILLA)
    por_cat = defaultdict(list)
    for x in xs:
        por_cat[x["categoria"]].append(x)
    partes = {k: [] for k in PARTES}
    for cat, ys in sorted(por_cat.items()):
        azar.shuffle(ys)
        n = len(ys)
        a = round(n * PARTES["train"])
        b = a + round(n * PARTES["validation"])
        partes["train"] += ys[:a]
        partes["validation"] += ys[a:b]
        partes["test"] += ys[b:]
    return partes


def respuesta(x: dict) -> str:
    j = x["juez_es"]
    return json.dumps({"veredicto": x["veredicto_objetivo"], **{b: bool(j[b]) for b in BANDERAS},
                       "explicacion": j.get("explicacion", "")}, ensure_ascii=False)


def filas(x: dict) -> list[dict]:
    out = []
    for idioma, texto in (("es", x["texto_es"]), ("en", x["texto_en"])):
        etiqueta = "Región declarada" if idioma == "es" else "Declared region"
        tema = "Tema declarado" if idioma == "es" else "Declared theme"
        out.append({
            "id": f"{x['id']}-{idioma}", "postit_id": x["id"], "idioma": idioma, "categoria": x["categoria"],
            "rasgo": x["rasgo"], "veredicto": x["veredicto_objetivo"],
            "messages": [
                {"role": "system", "content": SISTEMA[idioma]},
                {"role": "user", "content": f"{etiqueta}: {x['region']}. {tema}: {x['tema']}.\nPost-it:\n\"\"\"\n{texto}\n\"\"\""},
                {"role": "assistant", "content": respuesta(x)},
            ],
        })
    return out


def benchmark(x: dict) -> dict:
    claves = ["id", "categoria", "rasgo", "region", "tema", "texto_es", "texto_en", "expresiones_clave",
              "veredicto_objetivo", "banderas_objetivo", "sistema_es", "sistema_en"]
    b = {k: x[k] for k in claves}
    for capa in ("mod_es", "mod_en", "mini_es", "mini_en", "juez_es"):
        b[capa] = {k: v for k, v in x[capa].items() if k != "explicacion"}
    return b


def main() -> None:
    xs = cargar(sys.argv[1:])
    print(f"oro sin negativas: {len(xs)}")
    vivos = curatorkit(xs)
    xs = [x for x in xs if x["id"] in vivos]
    print(f"tras CuratorKIT: {len(xs)}")
    partes = partir(xs)
    base = Path("datos")
    (base / "guardian").mkdir(parents=True, exist_ok=True)
    (base / "benchmark").mkdir(parents=True, exist_ok=True)
    for nombre, ys in partes.items():
        with open(base / "guardian" / f"{nombre}.jsonl", "w", encoding="utf-8") as f:
            for x in ys:
                for fila in filas(x):
                    f.write(json.dumps(fila, ensure_ascii=False) + "\n")
        with open(base / "benchmark" / f"{nombre}.jsonl", "w", encoding="utf-8") as f:
            for x in ys:
                f.write(json.dumps(benchmark(x), ensure_ascii=False) + "\n")
        cats = defaultdict(int)
        for x in ys:
            cats[x["categoria"]] += 1
        print(f"{nombre:10} {len(ys):4} post-its ({2 * len(ys)} filas ES+EN) · {dict(sorted(cats.items()))}")


if __name__ == "__main__":
    main()
