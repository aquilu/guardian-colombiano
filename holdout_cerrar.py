"""Cierre del holdout v1: control de fuga, revisión humana y congelamiento.

1. Quita negativas del generador y casi-duplicados: similitud coseno ≥ 0,85 (paraphrase-multilingual-MiniLM-L12-v2)
   con cualquier texto existente (crudo/piloto, crudo/v1, datos/benchmark/*). Cada descarte queda registrado.
2. Revisión humana: TODOS los desacuerdos juez/objetivo y un 20 % al azar de los acuerdos → revision_holdout_v1.csv.
3. Congela datos/holdout/holdout_v1.jsonl (formato benchmark, con las líneas base) y un manifiesto con SHA-256.
   El holdout no se sube a Kaggle ni se usa para calibrar nada.
Uso: ../.venv-curator/bin/python holdout_cerrar.py verificado/holdout_v1.jsonl
"""
import csv
import hashlib
import json
import random
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

UMBRAL = 0.85
SEMILLA = 20261105
NEGATIVA = re.compile(r"^(no (puedo|ayudo)|si quieres|opci[oó]n segura|lo siento|i can.?t|sorry)", re.I)
EXISTENTES = ["crudo/piloto.jsonl", "crudo/v1.jsonl", "datos/benchmark/train.jsonl",
              "datos/benchmark/validation.jsonl", "datos/benchmark/test.jsonl"]


def main() -> None:
    xs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    descartes = []
    limpios = []
    for x in xs:
        if NEGATIVA.search(x["texto_es"]):
            descartes.append({"id": x["id"], "motivo": "negativa del generador", "texto": x["texto_es"][:120]})
        else:
            limpios.append(x)

    previos = sorted({json.loads(l)["texto_es"] for f in EXISTENTES for l in open(f, encoding="utf-8")})
    m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ep = m.encode(previos, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    eh = m.encode([x["texto_es"] for x in limpios], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    sim = eh @ ep.T
    maximo, cual = sim.max(1), sim.argmax(1)
    # casi-duplicados internos del propio holdout
    interna = eh @ eh.T
    np.fill_diagonal(interna, 0)
    quedan = []
    for i, x in enumerate(limpios):
        if maximo[i] >= UMBRAL:
            descartes.append({"id": x["id"], "motivo": f"casi-duplicado de datos existentes (coseno {maximo[i]:.3f})",
                              "texto": x["texto_es"][:120], "parecido_a": previos[cual[i]][:120]})
            continue
        if any(interna[i, j] >= UMBRAL for j in range(i)):
            descartes.append({"id": x["id"], "motivo": "casi-duplicado interno del holdout", "texto": x["texto_es"][:120]})
            continue
        x["similitud_max_con_existentes"] = round(float(maximo[i]), 3)
        quedan.append(x)

    # revisión humana: todos los desacuerdos + 20 % de los acuerdos
    azar = random.Random(SEMILLA)
    desacuerdo = [x for x in quedan if not x["oro"]]
    acuerdo = [x for x in quedan if x["oro"]]
    muestra = azar.sample(acuerdo, k=round(0.2 * len(acuerdo)))
    with open("revision_holdout_v1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "motivo_revision", "fuente", "categoria", "veredicto_objetivo", "veredicto_juez",
                    "texto_es", "texto_en", "explicacion_juez", "decision_comite", "nota"])
        for motivo, grupo in (("desacuerdo juez/objetivo", desacuerdo), ("auditoría aleatoria 20 %", muestra)):
            for x in grupo:
                w.writerow([x["id"], motivo, x["fuente"], x["categoria"], x["veredicto_objetivo"],
                            x["juez_es"]["veredicto"], x["texto_es"], x["texto_en"],
                            x["juez_es"].get("explicacion", ""), "", ""])

    salida = Path("datos/holdout")
    salida.mkdir(parents=True, exist_ok=True)
    claves = ["id", "fuente", "categoria", "rasgo", "perfil", "region", "tema", "texto_es", "texto_en",
              "expresiones_clave", "veredicto_objetivo", "banderas_objetivo", "sistema_es", "sistema_en",
              "similitud_max_con_existentes", "generador"]
    with open(salida / "holdout_v1.jsonl", "w", encoding="utf-8") as f:
        for x in quedan:
            fila = {k: x.get(k) for k in claves}
            fila["veredicto_juez"] = x["juez_es"]["veredicto"]
            fila["acuerdo_juez_objetivo"] = x["oro"]
            for capa in ("mod_es", "mod_en", "mini_es", "mini_en", "juez_es"):
                fila[capa] = {k: v for k, v in x[capa].items() if k != "explicacion"}
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    with open(salida / "holdout_v1_descartes.json", "w", encoding="utf-8") as f:
        json.dump(descartes, f, indent=2, ensure_ascii=False)
    sha = hashlib.sha256(open(salida / "holdout_v1.jsonl", "rb").read()).hexdigest()
    cats = {}
    for x in quedan:
        cats[x["categoria"]] = cats.get(x["categoria"], 0) + 1
    manifiesto = {
        "version": "holdout_v1", "fecha": date.today().isoformat(), "sha256": sha, "postits": len(quedan),
        "filas_es_en": 2 * len(quedan), "por_categoria": cats,
        "fuentes": {f: sum(x["fuente"] == f for x in quedan) for f in ("semilla_app", "generado_holdout")},
        "descartados": len(descartes), "umbral_casi_duplicado": UMBRAL,
        "acuerdo_juez_objetivo": f"{len(acuerdo)}/{len(quedan)}",
        "revision_humana": {"estado": "pendiente", "desacuerdos": len(desacuerdo), "auditoria_aleatoria": len(muestra),
                            "planilla": "revision_holdout_v1.csv"},
        "reglas": ["no se sube a Kaggle", "no se usa para entrenar ni calibrar (ρ, umbrales, prompts)",
                   "las etiquetas definitivas son las confirmadas por el comité"],
    }
    (salida / "holdout_v1_manifiesto.json").write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    print(json.dumps(manifiesto, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
