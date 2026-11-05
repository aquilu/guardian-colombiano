"""Auditoría de contaminación del holdout v2 (moderación y seguridad) frente a todo lo existente.

Compara contra: crudo/piloto y v1 (generación del entrenamiento), datos/benchmark/{train,validation,test},
datos/seguridad/{alineacion,eval}, y el holdout v1 (exploratorio). Para cada ítem del v2:
  1. duplicado exacto (texto normalizado) → descarte;
  2. casi-duplicado: coseno ≥ 0,85 (paraphrase-multilingual-MiniLM-L12-v2) con algún texto existente, en ES o en EN
     → descarte, con el texto más parecido;
  3. casi-duplicado interno del v2 (≥ 0,85) → se conserva el primero;
  4. familias de plantillas: componentes conexos con coseno ≥ 0,80 dentro del v2, y comienzos repetidos (4 primeras
     palabras) → se asigna «familia_id» para agrupar en la estadística (no se descarta).
Cada par ES/EN es una sola observación: la decisión se toma sobre el par.
Uso: ../.venv-curator/bin/python holdout_v2_auditar.py
"""
import collections
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

UMBRAL_DUP, UMBRAL_FAMILIA = 0.85, 0.80


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def leer(ruta, claves):
    p = Path(ruta)
    if not p.exists():
        return []
    out = []
    for l in open(p, encoding="utf-8"):
        x = json.loads(l)
        for k in claves:
            if x.get(k):
                out.append((ruta, k, x[k]))
    return out


def main() -> None:
    existentes = []
    for r in ("crudo/piloto.jsonl", "crudo/v1.jsonl", "datos/benchmark/train.jsonl", "datos/benchmark/validation.jsonl",
              "datos/benchmark/test.jsonl", "datos/holdout/holdout_v1.jsonl"):
        existentes += leer(r, ("texto_es", "texto_en"))
    existentes += leer("datos/seguridad/alineacion.jsonl", ("prompt",))
    existentes += leer("datos/seguridad/eval.jsonl", ("prompt",))
    textos_prev = [t for _, _, t in existentes]
    normalizados = {norm(t) for t in textos_prev}
    m = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ep = m.encode(textos_prev, normalize_embeddings=True, batch_size=128, show_progress_bar=False)

    informe = {"umbral_casi_duplicado": UMBRAL_DUP, "umbral_familia": UMBRAL_FAMILIA,
               "textos_existentes_comparados": len(textos_prev),
               "fuentes_comparadas": sorted({r for r, _, _ in existentes}), "conjuntos": {}}
    for conjunto, ces, cen in (("moderacion", "texto_es", "texto_en"), ("seguridad", "prompt_es", "prompt_en")):
        xs = [json.loads(l) for l in open(f"crudo/holdout_v2_{conjunto}.jsonl", encoding="utf-8")]
        e_es = m.encode([x[ces] for x in xs], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        e_en = m.encode([x[cen] for x in xs], normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        s_es, s_en = e_es @ ep.T, e_en @ ep.T
        descartes, quedan, idx = [], [], []
        for i, x in enumerate(xs):
            j_es, j_en = int(s_es[i].argmax()), int(s_en[i].argmax())
            mx = max(float(s_es[i, j_es]), float(s_en[i, j_en]))
            mas_parecido = textos_prev[j_es if s_es[i, j_es] >= s_en[i, j_en] else j_en]
            fuente = existentes[j_es if s_es[i, j_es] >= s_en[i, j_en] else j_en][0]
            if norm(x[ces]) in normalizados or norm(x[cen]) in normalizados:
                descartes.append({"id": x["id"], "motivo": "duplicado exacto de datos existentes", "texto": x[ces][:140]})
                continue
            if mx >= UMBRAL_DUP:
                descartes.append({"id": x["id"], "motivo": f"casi-duplicado (coseno {mx:.3f}) de {fuente}",
                                  "texto": x[ces][:140], "parecido_a": mas_parecido[:140]})
                continue
            interno = [k for k in idx if float(e_es[i] @ e_es[k]) >= UMBRAL_DUP]
            if interno:
                descartes.append({"id": x["id"], "motivo": f"casi-duplicado interno del v2 (de {xs[interno[0]]['id']})",
                                  "texto": x[ces][:140]})
                continue
            x["similitud_max_existentes"] = round(mx, 3)
            x["fuente_mas_parecida"] = fuente
            quedan.append(x)
            idx.append(i)
        # familias de plantillas (unión-búsqueda con coseno ≥ 0,80 entre los que quedan)
        padre = list(range(len(quedan)))

        def raiz(a):
            while padre[a] != a:
                padre[a] = padre[padre[a]]
                a = padre[a]
            return a
        E = e_es[idx]
        S = E @ E.T
        for a in range(len(quedan)):
            for b in range(a + 1, len(quedan)):
                if S[a, b] >= UMBRAL_FAMILIA:
                    padre[raiz(a)] = raiz(b)
        grupos = collections.defaultdict(list)
        for a in range(len(quedan)):
            grupos[raiz(a)].append(a)
        for n, (r, miembros) in enumerate(sorted(grupos.items(), key=lambda kv: -len(kv[1]))):
            for a in miembros:
                quedan[a]["familia_id"] = f"{conjunto[:3]}-f{n:03d}"
        comienzos = collections.Counter(" ".join(norm(x[ces]).split()[:4]) for x in quedan)
        with open(f"crudo/holdout_v2_{conjunto}_auditado.jsonl", "w", encoding="utf-8") as f:
            for x in quedan:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        sims = np.array([x["similitud_max_existentes"] for x in quedan])
        informe["conjuntos"][conjunto] = {
            "generados": len(xs), "conservados": len(quedan), "descartados": len(descartes), "descartes": descartes,
            "similitud_max_con_existentes": {"mediana": round(float(np.median(sims)), 3),
                                             "p95": round(float(np.percentile(sims, 95)), 3), "max": round(float(sims.max()), 3)},
            "familias": {"total": len(grupos), "con_mas_de_un_miembro": sum(len(g) > 1 for g in grupos.values()),
                         "tamano_max": max(len(g) for g in grupos.values())},
            "lotes_de_generacion": len({x["lote_id"] for x in quedan}),
            "comienzos_repetidos": {k: v for k, v in comienzos.most_common(8) if v > 1},
        }
    Path("informe_contaminacion_v2.json").write_text(json.dumps(informe, indent=2, ensure_ascii=False))
    print(json.dumps({c: {k: v for k, v in d.items() if k != "descartes"} for c, d in informe["conjuntos"].items()},
                     indent=2, ensure_ascii=False))
    for c, d in informe["conjuntos"].items():
        for x in d["descartes"]:
            print(c, "DESCARTE", x["motivo"], "|", x["texto"][:90], "| ~", x.get("parecido_a", "")[:90])


if __name__ == "__main__":
    main()
