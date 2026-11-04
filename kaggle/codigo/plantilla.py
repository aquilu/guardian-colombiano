"""Plantilla de chat compacta del guardián afinado.

Mismos tokens especiales de Tiny Aya (Cohere), sin el preámbulo por defecto de ~370 tokens que la plantilla oficial
mete en cada llamada. Se usa igual al entrenar (SFT, Lisa, SafeGrad) y al evaluar los modelos afinados; la línea base
zero-shot y la medición de seguridad general usan la plantilla oficial.
"""

PLANTILLA_COMPACTA = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "<|START_OF_TURN_TOKEN|>"
    "{% if m['role'] == 'system' %}<|SYSTEM_TOKEN|>{{ m['content'] }}"
    "{% elif m['role'] == 'user' %}<|USER_TOKEN|>{{ m['content'] }}"
    "{% else %}<|CHATBOT_TOKEN|><|START_RESPONSE|>{% generation %}{{ m['content'] }}<|END_RESPONSE|>{% endgeneration %}"
    "{% endif %}<|END_OF_TURN_TOKEN|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>{% endif %}"
)


def compacta(tok):
    """Pone la plantilla compacta en el tokenizer (en el sitio) y lo devuelve."""
    tok.chat_template = PLANTILLA_COMPACTA
    return tok
