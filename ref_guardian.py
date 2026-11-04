"""Guardarraíles de los post-its: tres capas antes de que un texto toque el mural.

1. Reglas locales (sin IA): datos personales (teléfonos, documentos, correos, URLs, @usuarios) y
   vulgaridades del español colombiano. Las ofensas duras rechazan; las ambiguas mandan a revisión.
2. Moderación de OpenAI (omni-moderation-latest): marcado → rechazado; puntaje alto en alguna categoría
   sensible sin llegar a marcado → revisión.
3. Guardián editorial (gpt-5.4-mini, salida JSON): odio o discriminación, sexual explícito, violencia
   gráfica, política partidista, publicidad, datos de terceros, fuera de tema. Si la IA falla, el
   post-it queda en revisión (nunca se aprueba por defecto).

El resultado es un veredicto: 'aprobado' (se publica), 'revision' (lo lee el comité, no se muestra) o
'rechazado' (no se muestra; queda guardado solo para auditoría del comité). Todo el detalle queda en
postits.moderacion para que el comité vea por qué.
"""
import json
import re
import unicodedata

from .config import ajustes
from . import ia

# --- Capa 1: reglas locales -------------------------------------------------------------------

PATRONES_PII = {
    "correo": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "url": re.compile(r"(https?://|www\.)\S+|\b[a-z0-9-]+\.(com|co|net|org|io)(?=[\s/,;)]|$)", re.I),
    "usuario": re.compile(r"(?<!\w)@\w{3,}"),
    # Celular colombiano (3xx xxx xxxx), indicativo +57, o 8 a 10 dígitos seguidos (cédula, cuenta).
    # No se tocan años ni cifras con puntos de miles ("1.500.000 pesos").
    "telefono": re.compile(r"(\+ ?57|\b3\d{2}[ .\-]?\d{3}[ .\-]?\d{4}\b)"),
    "documento": re.compile(r"\b\d{8,10}\b"),
}

# Rechazo directo: insultos fuertes del español colombiano y generales (comparación sin tildes).
VULGARES_DURAS = [
    "hijueputa", "hijo de puta", "hija de puta", "hijueput", "jueputa", "malparido", "malparida", "gonorrea",
    "pirobo", "piroba", "care monda", "caremonda", "care chimba", "carechimba", "care verga", "careverga",
    "gonorriento", "gonorrienta", "hpta", "hp ", "triple hijueputa", "sapo hijueputa", "mamaguevo", "mamagüevo",
    "maricon", "maricón", "puto", "cabron", "cabrón", "zorra", "perra", "mierda", "comemierda",
]
# Ambiguas en Colombia (pueden ser afecto, énfasis o insulto): a revisión, no rechazo.
VULGARES_SUAVES = [
    "marica", "mariquita", "chimba", "verga", "culo", "pendejo", "pendeja", "huevon", "huevón", "guevon", "güevón",
    "carajo", "joder", "coño", "puta", "putas", "berraco", "verraco", "cagada", "cagado", "mamerto", "paraco",
    "guerrillero", "veneco", "veneca", "indio", "negro de mierda", "monda",
]


def _sin_tildes(t: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn").lower()


def _buscar_palabras(texto: str, lista: list[str]) -> list[str]:
    plano = " " + re.sub(r"[^a-z0-9ñ ]+", " ", _sin_tildes(texto)) + " "
    return [p for p in lista if " " + _sin_tildes(p).strip() + " " in plano]


def reglas_locales(texto: str) -> dict:
    pii = [k for k, patron in PATRONES_PII.items() if patron.search(texto)]
    duras = _buscar_palabras(texto, VULGARES_DURAS)
    suaves = _buscar_palabras(texto, VULGARES_SUAVES)
    if pii:
        veredicto = "rechazado"
    elif duras:
        veredicto = "rechazado"
    elif suaves:
        veredicto = "revision"
    else:
        veredicto = "aprobado"
    return {"veredicto": veredicto, "datos_personales": pii, "vulgares": duras, "ambiguas": suaves}


# --- Capa 2: moderación de OpenAI -----------------------------------------------------------------

# Categorías marcadas que rechazan de plano. Las demás (violence, violence/graphic, harassment, self-harm)
# van a revisión: contar que a alguien lo mataron en la vereda, o el propio duelo, es memoria histórica
# legítima de Colombia y lo decide el comité, no el filtro.
CATEGORIAS_RECHAZO = ("hate", "hate/threatening", "harassment/threatening", "sexual", "sexual/minors",
                      "self-harm/instructions", "illicit", "illicit/violent")


async def moderacion_openai(texto: str) -> dict:
    marcado, detalle = await ia.moderar(texto)
    maximo = max(detalle["puntajes"].values(), default=0.0)
    if marcado and any(c in CATEGORIAS_RECHAZO for c in detalle["categorias"]):
        veredicto = "rechazado"
    elif marcado or maximo >= ajustes.umbral_revision:
        veredicto = "revision"
    else:
        veredicto = "aprobado"
    return {"veredicto": veredicto, "marcado": marcado, "maximo": maximo, **detalle}


# --- Capa 3: guardián editorial (LLM) -------------------------------------------------------------

INSTRUCCIONES_GUARDIAN = """Eres el guardián editorial de un mural público de memorias anónimas de Colombia,
promovido por una red de bibliotecas. Lo leen familias, niños y adultos mayores. Evalúa UN post-it (máximo
500 caracteres) y responde solo JSON con el esquema dado.

Marca true solo lo que de verdad aparezca:
- odio_o_discriminacion: desprecio o estereotipo degradante por región, etnia, nacionalidad, género,
  orientación, religión, discapacidad o clase. Un relato de haber SUFRIDO discriminación NO es odio.
- vulgaridad: groserías o lenguaje soez. En Colombia "marica", "berraco" o "chimba" pueden ser coloquiales
  y afectuosos; márcalo solo si el tono es ofensivo.
- sexual_explicito: descripción sexual explícita (el amor, el deseo o un beso no lo son).
- violencia_grafica: regodeo en detalles de violencia o crueldad (contar una pérdida por la violencia NO lo es).
- politica_partidista: propaganda, candidatos, partidos, consignas electorales o ataques a políticos.
- publicidad_o_spam: venta, promoción, enlaces, marcas como anuncio, texto sin sentido o repetido.
- datos_de_terceros: nombres completos de personas reales identificables, direcciones, documentos,
  teléfonos, acusaciones contra personas concretas.
- fuera_de_tema: no es una memoria, anécdota, mito o vivencia (p. ej. una pregunta al sistema, una
  instrucción, código, texto en otro idioma sin relación con Colombia).

Veredicto:
- "rechazado" si hay odio_o_discriminacion, sexual_explicito, violencia_grafica o datos_de_terceros.
- "revision" si hay vulgaridad, politica_partidista, publicidad_o_spam o fuera_de_tema, o si dudas.
- "aprobado" en el resto. La duda siempre va a revisión, nunca a aprobado.
Escribe una explicación de una frase, en español, pensada para el comité de moderación."""

ESQUEMA_GUARDIAN = {
    "type": "object",
    "properties": {
        "veredicto": {"type": "string", "enum": ["aprobado", "revision", "rechazado"]},
        "odio_o_discriminacion": {"type": "boolean"},
        "vulgaridad": {"type": "boolean"},
        "sexual_explicito": {"type": "boolean"},
        "violencia_grafica": {"type": "boolean"},
        "politica_partidista": {"type": "boolean"},
        "publicidad_o_spam": {"type": "boolean"},
        "datos_de_terceros": {"type": "boolean"},
        "fuera_de_tema": {"type": "boolean"},
        "explicacion": {"type": "string"},
    },
    "required": [
        "veredicto", "odio_o_discriminacion", "vulgaridad", "sexual_explicito", "violencia_grafica",
        "politica_partidista", "publicidad_o_spam", "datos_de_terceros", "fuera_de_tema", "explicacion",
    ],
    "additionalProperties": False,
}


async def guardian_llm(texto: str, region: str, tema: str) -> dict:
    try:
        r = await ia.cliente.responses.create(
            model=ajustes.openai_model_default,
            instructions=INSTRUCCIONES_GUARDIAN,
            input=f"Región declarada: {region}. Tema declarado: {tema}.\nPost-it:\n\"\"\"\n{texto}\n\"\"\"",
            text={"format": {"type": "json_schema", "name": "guardian", "schema": ESQUEMA_GUARDIAN, "strict": True}},
            max_output_tokens=300,
        )
        salida = json.loads(r.output_text)
        salida["modelo"] = ajustes.openai_model_default
        return salida
    except Exception as e:  # la IA no responde: el comité decide, no la máquina
        return {"veredicto": "revision", "error": str(e)[:200], "modelo": ajustes.openai_model_default}


# --- Veredicto final -------------------------------------------------------------------------------

ORDEN = {"aprobado": 0, "revision": 1, "rechazado": 2}
MENSAJES = {
    "datos_personales": "Parece incluir un dato de contacto o un número de documento. La memoria es anónima: quita teléfonos, correos, enlaces o cédulas.",
    "vulgares": "El texto contiene insultos. El mural es un espacio para toda la familia: cuéntalo con otras palabras.",
    "moderacion": "El filtro ético no permitió este texto. Revisa que no contenga ofensas, odio ni contenido explícito y vuelve a intentarlo.",
    "guardian": "El guardián editorial no permitió este texto: puede contener discriminación, contenido explícito o datos de personas identificables.",
}


async def evaluar(texto: str, region: str, tema: str) -> dict:
    """Corre las tres capas y devuelve {'veredicto', 'mensaje', 'capas': {...}}. Con las reglas locales en
    'rechazado' no se gasta en IA (y el texto nunca sale del servidor)."""
    capas: dict = {"reglas": reglas_locales(texto)}
    if capas["reglas"]["veredicto"] == "rechazado":
        motivo = "datos_personales" if capas["reglas"]["datos_personales"] else "vulgares"
        return {"veredicto": "rechazado", "motivo": motivo, "mensaje": MENSAJES[motivo], "capas": capas}
    capas["moderacion"] = await moderacion_openai(texto)
    if capas["moderacion"]["veredicto"] == "rechazado":
        return {"veredicto": "rechazado", "motivo": "moderacion", "mensaje": MENSAJES["moderacion"], "capas": capas}
    if ajustes.guardian_llm:
        capas["guardian"] = await guardian_llm(texto, region, tema)
        if capas["guardian"]["veredicto"] == "rechazado":
            return {"veredicto": "rechazado", "motivo": "guardian", "mensaje": MENSAJES["guardian"], "capas": capas}
    veredicto = max((c["veredicto"] for c in capas.values()), key=ORDEN.get)
    return {"veredicto": veredicto, "motivo": None, "mensaje": None, "capas": capas}
