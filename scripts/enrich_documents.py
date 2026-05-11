import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

INPUT_PATH = Path(os.getenv("ENRICH_INPUT_PATH", "data/documents.json"))
OUTPUT_PATH = Path(os.getenv("ENRICH_OUTPUT_PATH", "data/documents_enriched.json"))
MODEL = os.getenv("OPENAI_MODEL_ENRICH", "gpt-4.1-mini")


PRODUCT_TYPES = [
    "prestamos_general",
    "prestamo_personal",
    "prestamo_hipotecario_uva",
    "prestamo_prendario",
    "adelanto_sueldo",
    "prestamo_express",
    "cuotificacion",
    "tarjetas",
    "seguros",
    "cuentas",
    "transferencias",
    "inversiones",
    "pagos",
    "otro",
]


DOCUMENT_PURPOSES = [
    "requisitos",
    "solicitud",
    "condiciones",
    "general_info",
    "comparacion",
    "pagos",
    "cuotas",
    "cancelacion",
    "problema",
    "estado_seguimiento",
    "documentacion",
    "otro",
]


def shorten(text: str, max_chars: int = 3500) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def safe_json_loads(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def classify_document(client: OpenAI, doc: dict) -> dict:
    title = doc.get("title", "")
    content = shorten(doc.get("content", ""))

    system_prompt = f"""
Sos un clasificador de documentos para un sistema RAG bancario.

Tenés que clasificar el documento según su producto principal y propósito principal.

IMPORTANTE:
- No confundas rutas de navegación con el producto principal.
- No confundas sección general con producto específico.
- Si el documento habla de "Préstamos" en general y NO especifica claramente préstamo personal, hipotecario UVA, prendario, adelanto de sueldo, préstamo express o cuotificación, usá product_type = "prestamos_general".
- No asumas que un documento de préstamos generales es de préstamo personal.
- No asumas que préstamo prendario, préstamo hipotecario UVA, préstamo personal, adelanto de sueldo o cuotificación son lo mismo.
- Si el contenido dice "Préstamos > Nuevo préstamo Personal > Adelanto de sueldo", pero el documento habla de adelanto de sueldo, el product_type es "adelanto_sueldo".
- Elegí el producto principal según el tema real del documento, no según breadcrumbs, rutas de app o nombres de menú.
- Si el documento compara productos, usá product_type según el producto dominante o "prestamos_general" si no hay uno principal.
- Devolvé SOLO JSON válido.

product_type posibles:
{PRODUCT_TYPES}

document_purpose posibles:
{DOCUMENT_PURPOSES}

Formato obligatorio:
{{
  "product_type": "uno_de_los_valores_permitidos",
  "document_purpose": "uno_de_los_valores_permitidos",
  "topics": ["tema_1", "tema_2"],
  "confidence": 0.0,
  "reason": "motivo breve"
}}
""".strip()

    user_prompt = f"""
Título:
{title}

Contenido:
{content}
""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    parsed = safe_json_loads(raw)

    product_type = parsed.get("product_type", "otro")
    document_purpose = parsed.get("document_purpose", "otro")

    if product_type not in PRODUCT_TYPES:
        product_type = "otro"

    if document_purpose not in DOCUMENT_PURPOSES:
        document_purpose = "otro"

    topics = parsed.get("topics", [])
    if not isinstance(topics, list):
        topics = []

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return {
        "product_type": product_type,
        "document_purpose": document_purpose,
        "topics": topics[:5],
        "classification_confidence": confidence,
        "classification_reason": parsed.get("reason", ""),
    }


def main():
    client = OpenAI()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {INPUT_PATH}")

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    enriched_documents = []

    print(f"📄 Documentos a enriquecer: {len(documents)}")
    print(f"🤖 Modelo: {MODEL}")
    print(f"📥 Input: {INPUT_PATH}")
    print(f"📤 Output: {OUTPUT_PATH}")

    for index, doc in enumerate(documents, 1):
        print(f"\n[{index}/{len(documents)}] {doc.get('title', 'Sin título')}")

        try:
            metadata = classify_document(client, doc)
            enriched_doc = {**doc, **metadata}
            enriched_documents.append(enriched_doc)

            print(
                f"✅ product_type={metadata['product_type']} | "
                f"purpose={metadata['document_purpose']} | "
                f"confidence={metadata['classification_confidence']}"
            )

        except Exception as error:
            print(f"⚠️ Error clasificando documento: {error}")

            enriched_documents.append({
                **doc,
                "product_type": "otro",
                "document_purpose": "otro",
                "topics": [],
                "classification_confidence": 0.0,
                "classification_reason": f"Error: {str(error)}",
            })

        time.sleep(0.2)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(enriched_documents, file, ensure_ascii=False, indent=2)

    print("\n✅ Enrichment terminado.")
    print(f"Archivo generado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()