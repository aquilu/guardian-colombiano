"""ROI: guardián cerrado de 12-cuentos frente a Tiny Aya afinado, sobre el MISMO holdout v2 congelado (sin llamadas nuevas).

Cadena cerrada = la de producción en 12-cuentos (ref_guardian.evaluar): reglas locales → omni-moderation →
gpt-5.4-mini (política completa), veredicto = el más severo; un «rechazado» temprano corta la cadena. Las respuestas de
omni-moderation y gpt-5.4-mini se guardaron al etiquetar el holdout (verificado/holdout_v2_moderacion_jueces.jsonl,
`lineas_base`); umbral de revisión de omni-moderation = 0,15, como en holdout_v2_juzgar.py.
Uso: python3 roi.py
"""
import json
from pathlib import Path

from analisis_final import H, bootstrap, item_mod

_ns: dict = {}
exec(Path("ref_guardian.py").read_text().replace("from .config import ajustes", "").replace("from . import ia", ""), _ns)
reglas = _ns["reglas_locales"]
ORDEN = {"aprobado": 0, "revision": 1, "rechazado": 2}
D = Path("resultados/final")


def cadena(x, base, idioma, capas):
    vs = []
    if "reglas" in capas:
        r = reglas(x[f"texto_{idioma}"])["veredicto"]
        if r == "rechazado":
            return r
        vs.append(r)
    if "omni" in capas:
        m = base[f"moderacion_{idioma}"]["veredicto"]
        if m == "rechazado":
            return m
        vs.append(m)
    if "mini" in capas:
        g = (base.get(f"mini_{idioma}") or {}).get("veredicto") or "revision"   # si la IA falla, revisión
        vs.append(g)
    return max(vs, key=ORDEN.get)


def main():
    base = {json.loads(l)["id"]: json.loads(l)["lineas_base"] for l in open("verificado/holdout_v2_moderacion_jueces.jsonl")}
    xs = [json.loads(l) for l in open("datos/holdout_v2/moderacion.jsonl")]
    sistemas = {}
    for nombre, capas in (("cadena_12cuentos", {"reglas", "omni", "mini"}), ("gpt-5.4-mini_solo", {"mini"}),
                          ("reglas+omni", {"reglas", "omni"})):
        sistemas[nombre] = [{"id": x["id"], "idioma": i, "pred": cadena(x, base[x["id"]], i, capas),
                             "json_ok": True, "esquema": True} for x in xs for i in ("es", "en")]
    for m in ("lisa", "plain", "safegrad", "base-oficial"):
        sistemas[m] = [json.loads(l) for l in open(D / f"mod-{m}.jsonl")]
    lotes = {i: h["lote_id"] for i, h in H.items()}
    res = {}
    for s, ys in sistemas.items():
        res[s] = {}
        for idioma in ("es", "en"):
            met = {}
            for y in ys:
                if y["idioma"] != idioma:
                    continue
                for k, v in item_mod(y).items():
                    if v is not None and k not in ("json_valido", "esquema_exacto"):
                        met.setdefault(k, {})[y["id"]] = float(v)
            res[s][idioma] = {}
            for k, v in met.items():
                e = {"valor": round(sum(v.values()) / len(v), 4), "n": len(v)}
                if s != "lisa":
                    w = {y["id"]: float(item_mod(y)[k]) for y in sistemas["lisa"] if y["idioma"] == idioma and item_mod(y)[k] is not None}
                    c = set(v) & set(w)
                    d, lo, hi = bootstrap({i: w[i] - v[i] for i in c}, lotes)
                    e["lisa_menos_este"] = {"dif": d, "ic95": [lo, hi], "excluye_0": lo > 0 or hi < 0}
                res[s][idioma][k] = e
    reg = [json.loads(l) for l in open("registro_openai.jsonl")]
    mini = [r for r in reg if r["op"].startswith("base-mini-") and r.get("ok")]
    tok = {"llamadas": len(mini), "entrada_media": round(sum(r["tokens_entrada"] for r in mini) / len(mini), 1),
           "salida_media": round(sum(r["tokens_salida"] for r in mini) / len(mini), 1)}
    rend = json.loads((Path("resultados/validacion-lisa.resumen.json")).read_text()).get("rendimiento") if Path("resultados/validacion-lisa.resumen.json").exists() else None
    out = {"metodo": __doc__.split("Uso:")[0].strip(), "calidad": res, "tokens_gpt-5.4-mini_por_postit": tok,
           "rendimiento_lisa_local": rend}
    (D / "roi.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for s in res:
        for i in ("es", "en"):
            print(f"{s:18} {i}: " + " · ".join(f"{k} {v['valor']:.1%}" for k, v in res[s][i].items()))
    print(tok, rend)


if __name__ == "__main__":
    main()
