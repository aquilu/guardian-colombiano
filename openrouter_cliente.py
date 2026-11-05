"""Cliente de OpenRouter para el holdout v2: modelo y proveedor fijos, privacidad, reintentos, tope de gasto y registro.

- Clave: solo la variable de entorno OPENROUTER_API_KEY (nunca se escribe en código, logs ni resultados).
- Proveedor fijo por modelo y sin respaldo automático (la cuantización no cambia entre llamadas), y
  data_collection="deny" (solo proveedores que no retienen ni entrenan con los datos).
- Cada llamada queda en registro_openrouter.jsonl: operación, modelo pedido y servido, proveedor, tokens y costo.
- Tope de gasto local (TOPE_USD, por defecto 3): al superarlo se detiene todo.
"""
import asyncio
import json
import os
import time
from pathlib import Path

from openai import AsyncOpenAI

BASE = "https://openrouter.ai/api/v1"
MODELOS = {   # rol: (modelo, proveedores, require_parameters)
    "generador": ("deepseek/deepseek-v4.1-flash-1017", ["DeepInfra"], True),
    # Claude por Google Vertex: en el endpoint de Anthropic, OpenRouter modera la entrada y devuelve 403 con los
    # post-its violentos, sexuales u odiosos (justo los que hay que juzgar). Con require_parameters=True no hay
    # endpoint (404). El cliente valida igual las claves requeridas de cada respuesta.
    "juez_claude": ("anthropic/claude-sonnet-5.5", ["Google"], False),
}
TOPE_USD = float(os.environ.get("TOPE_USD", "3"))   # tope local estricto; el límite de la clave se fija en el panel
REGISTRO = Path(__file__).parent / "registro_openrouter.jsonl"


class TopeAlcanzado(RuntimeError):
    pass


class OpenRouter:
    def __init__(self) -> None:
        clave = os.environ.get("OPENROUTER_API_KEY")
        if not clave:
            raise SystemExit("falta OPENROUTER_API_KEY en el entorno")
        self.cliente = AsyncOpenAI(base_url=BASE, api_key=clave, max_retries=0, timeout=180)
        self.gastado = sum(json.loads(l).get("costo_usd") or 0 for l in open(REGISTRO)) if REGISTRO.exists() else 0.0
        self._lock = asyncio.Lock()

    async def json(self, rol: str, operacion: str, sistema: str, usuario: str, esquema: dict, nombre: str,
                   max_tokens: int = 6000, temperatura: float = 0.7) -> dict | None:
        modelo, proveedores, requiere = MODELOS[rol]
        if self.gastado >= TOPE_USD:
            raise TopeAlcanzado(f"gasto acumulado {self.gastado:.4f} USD ≥ tope {TOPE_USD} USD")
        for intento in range(4):
            t0 = time.time()
            try:
                r = await self.cliente.chat.completions.create(
                    model=modelo, temperature=temperatura, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": sistema}, {"role": "user", "content": usuario}],
                    response_format={"type": "json_schema", "json_schema": {"name": nombre, "schema": esquema, "strict": True}},
                    extra_body={"provider": {"order": proveedores, "allow_fallbacks": False, "data_collection": "deny",
                                             "require_parameters": requiere},
                                "usage": {"include": True}})
                u = r.usage
                costo = getattr(u, "cost", None) if u is not None else None
                if costo is None and u is not None and isinstance(getattr(u, "model_extra", None), dict):
                    costo = u.model_extra.get("cost")
                ent = getattr(u, "prompt_tokens", None) if u is not None else None
                sal = getattr(u, "completion_tokens", None) if u is not None else None
                if costo is None and getattr(r, "id", None):   # sin usage en la respuesta: se pide el costo real
                    costo, ent, sal = await self._costo_generacion(r.id, ent, sal)
                texto = (r.choices[0].message.content if r.choices else "") or ""
                try:   # el costo se registra aunque la respuesta no sirva
                    datos = json.loads(texto)
                    faltan = [k for k in esquema.get("required", []) if k not in datos]
                    error = f"faltan claves {faltan}" if faltan else None
                except json.JSONDecodeError as e:
                    datos, error = None, f"JSON inválido: {e}"
                await self._registrar({"op": operacion, "rol": rol, "modelo_pedido": modelo, "modelo_servido": r.model,
                                       "proveedor": getattr(r, "provider", None) or (r.model_extra or {}).get("provider"),
                                       "tokens_entrada": ent, "tokens_salida": sal,
                                       "costo_usd": costo, "segundos": round(time.time() - t0, 2), "intento": intento + 1,
                                       "ok": error is None, **({"error": error} if error else {})})
                if error is None:
                    return datos
                if intento >= 1:   # salida vacía o cortada: suele ser un filtro del proveedor y se repite igual
                    return None
                await asyncio.sleep(2 ** intento)
                continue
            except TopeAlcanzado:
                raise
            except Exception as e:  # reintento con espera creciente; el error queda registrado sin la clave
                await self._registrar({"op": operacion, "rol": rol, "modelo_pedido": modelo, "ok": False,
                                       "error": f"{type(e).__name__}: {str(e)[:200]}", "intento": intento + 1})
                await asyncio.sleep(2 ** intento)
        return None

    async def _costo_generacion(self, gen_id: str, ent, sal):
        """Costo real de una generación por su id (/api/v1/generation); reintenta porque tarda unos segundos en estar."""
        import urllib.parse
        import urllib.request

        def pedir():
            url = f"{BASE}/generation?" + urllib.parse.urlencode({"id": gen_id})
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.cliente.api_key}"})
            return json.load(urllib.request.urlopen(req, timeout=30)).get("data") or {}
        for espera in (2, 4, 8):
            await asyncio.sleep(espera)
            try:
                d = await asyncio.to_thread(pedir)
                if d.get("total_cost") is not None:
                    return d["total_cost"], d.get("tokens_prompt", ent), d.get("tokens_completion", sal)
            except Exception:
                pass
        return None, ent, sal

    async def _registrar(self, fila: dict) -> None:
        async with self._lock:
            fila = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fila}
            if fila.get("costo_usd"):
                self.gastado += fila["costo_usd"]
            with open(REGISTRO, "a", encoding="utf-8") as f:
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
