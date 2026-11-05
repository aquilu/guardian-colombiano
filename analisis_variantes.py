"""Ejecución #22: variantes regionales de Tiny Aya (global, water, fire), original y +Lisa, sobre el holdout v2.

Mismas métricas, bootstrap pareado por lote y semilla que analisis_final.py (se importan de ahí). Los resultados de
global (mod-base-oficial, mod-lisa, seg-base, seg-lisa) son los de la evaluación única #20; no se recalculan.
Uso: python analisis_variantes.py  →  resultados/final/analisis_variantes.json
"""
import json

from analisis_final import D, consistencia, item_mod, item_seg, leer, resumir, tabla

MOD = {"base-oficial": "global original", "lisa": "global + Lisa", "base-water": "water original",
       "lisa-water": "water + Lisa", "base-fire": "fire original", "lisa-fire": "fire + Lisa"}
SEG = {"base": "global original", "lisa": "global + Lisa", "base-water": "water original",
       "lisa-water": "water + Lisa", "base-fire": "fire original", "lisa-fire": "fire + Lisa"}


def main():
    inf = {"metodo": __doc__.split("Uso:")[0].strip()}
    mod = {m: ys for m in MOD if (ys := leer(D / f"mod-{m}.jsonl"))}
    inf["moderacion"] = resumir(tabla(mod, item_mod), ["base-oficial", "lisa", "base-water", "base-fire"])
    for m, ys in mod.items():
        inf["moderacion"][m]["consistencia_es_en"] = consistencia(ys)
    seg = {m: ys for m in SEG if (ys := leer(D / f"seg-{m}.juzgado.jsonl"))}
    inf["seguridad"] = resumir(tabla(seg, item_seg), ["base", "lisa", "base-water", "base-fire"])
    inf["seguridad_cobertura_juez"] = {m: f"{sum(1 for y in ys if y.get('juicio'))}/{len(ys)}" for m, ys in seg.items()}
    (D / "analisis_variantes.json").write_text(json.dumps(inf, indent=2, ensure_ascii=False))
    for conj, nombres in (("moderacion", MOD), ("seguridad", SEG)):
        for m, por in inf[conj].items():
            for idioma in ("es", "en"):
                fila = " · ".join(f"{k} {v['valor']:.1%}" for k, v in por[idioma].items() if isinstance(v, dict))
                print(f"{conj:10} {nombres[m]:16} {idioma}: {fila}")


if __name__ == "__main__":
    main()
