import json
import os
import re
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import uuid4

from flask import Flask, Response, jsonify, request
from twilio.twiml.messaging_response import MessagingResponse

from core.bot_runner import BotRuntime, prepare_runtime, run_bot_query
from core.privacy import mask_sensitive_text
from memory.local_memory import load_memory, mark_csat_sent, save_memory
from services.twilio_content import send_whatsapp_content_template
from services.twilio_media import build_media_payload, looks_like_pdf_media
from services.twilio_typing import send_whatsapp_typing_indicator
from utils.whatsapp_formatting import format_whatsapp_answer


app = Flask(__name__)
_runtime: BotRuntime | None = None
CSAT_FLOW_CONTENT_VARIABLES_JSON = os.getenv("CSAT_FLOW_CONTENT_VARIABLES_JSON")
CSAT_FLOW_TOKEN_PLACEHOLDER = "__FLOW_TOKEN__"


def get_runtime() -> BotRuntime:
    global _runtime

    if _runtime is None:
        _runtime = prepare_runtime(
            top_k=4,
            rebuild=False,
            include_graph=True,
            include_langfuse=True,
        )

    return _runtime


def sanitize_whatsapp_session_id(sender: str) -> str:
    digits = re.sub(r"\D", "", sender or "")
    if digits:
        return f"whatsapp-{digits}"

    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", (sender or "").strip())
    cleaned = cleaned.strip("-")
    return f"whatsapp-{cleaned}" if cleaned else "whatsapp-anonymous"


def twiml_message(body: str, status_code: int = 200) -> Response:
    response = MessagingResponse()
    response.message(body)
    return Response(str(response), mimetype="application/xml", status=status_code)


def _extract_media_from_request() -> dict[str, str]:
    num_media = (request.form.get("NumMedia") or "").strip()
    if not num_media or num_media == "0":
        return {}

    media_url = (request.form.get("MediaUrl0") or "").strip()
    content_type = (request.form.get("MediaContentType0") or "").strip()
    filename = (request.form.get("MediaFilename0") or "").strip()

    if not filename and media_url:
        filename = PurePosixPath(media_url.split("?", 1)[0]).name or ""

    if not filename:
        filename = "attachment.pdf" if "pdf" in content_type.lower() else "attachment"

    return build_media_payload(
        num_media=num_media,
        url=media_url,
        content_type=content_type,
        filename=filename,
    )


@app.get("/")
def root() -> Response:
    return jsonify({"status": "ok", "service": "gala-whatsapp-bot"})


@app.get("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


def handle_whatsapp_message() -> Response:
    body = (request.form.get("Body") or "").strip()
    sender = (request.form.get("From") or "").strip()
    recipient = (request.form.get("To") or "").strip()
    message_sid = (request.form.get("MessageSid") or "").strip()
    session_id = sanitize_whatsapp_session_id(sender)

    latitude = (request.form.get("Latitude") or "").strip()
    longitude = (request.form.get("Longitude") or "").strip()
    address = (request.form.get("Address") or "").strip()
    label = (request.form.get("Label") or "").strip()
    media = _extract_media_from_request()

    user_location = {}
    if latitude and longitude:
        user_location = {
            "latitude": latitude,
            "longitude": longitude,
            "address": address,
            "label": label,
        }

    try:
        if not body and user_location:
            body = "Ubicacion compartida por WhatsApp"
        elif not body and media:
            body = (
                "Analizar resumen de tarjeta adjunto"
                if looks_like_pdf_media(media)
                else "Archivo adjunto enviado"
            )

        if not body:
            return twiml_message("No recibí tu mensaje. Probá de nuevo, por favor.")

        print("\n[WHATSAPP] Mensaje recibido")
        print(f"  - from: {session_id}")
        print(f"  - body: {mask_sensitive_text(body)}")

        if user_location:
            print("  - location: recibida")
        if media:
            print("  - media: adjunto recibido")
            print(f"  - media_content_type: {media.get('content_type', '')}")

        send_whatsapp_typing_indicator(message_sid)

        result = run_bot_query(
            runtime=get_runtime(),
            question=body,
            session_id=session_id,
            langfuse_user_id=session_id,
            langfuse_tags=["gala", "langgraph", "rag", "whatsapp"],
            observation_name="gala-whatsapp-request",
            user_location=user_location,
            media=media,
        )

        formatted_answer = format_whatsapp_answer(
            result["final_answer"],
            topic=result.get("topic"),
            route=result.get("route"),
        )

        if result.get("send_csat"):
            _send_csat_flow_if_needed(
                session_id=session_id,
                sender=sender,
                recipient=recipient,
                template_sid=(result.get("csat_template_sid") or "").strip(),
            )

        return twiml_message(formatted_answer)

    except Exception as exc:
        print("\n[WHATSAPP] Error procesando mensaje")
        print(f"  - error: {str(exc)}")
        print(f"  - body: {mask_sensitive_text(body)}")
        return twiml_message(
            "Perdón, tuve un problema procesando tu consulta. Probá de nuevo en unos minutos.",
            status_code=200,
        )


@app.post("/whatsapp")
def whatsapp_webhook() -> Response:
    return handle_whatsapp_message()


@app.post("/webhook")
def webhook_alias() -> Response:
    return handle_whatsapp_message()


def _send_csat_flow_if_needed(
    *,
    session_id: str,
    sender: str,
    recipient: str,
    template_sid: str,
) -> None:
    try:
        memory = load_memory(session_id)
        if memory.get("csat_sent"):
            return

        sent = send_whatsapp_content_template(
            to_number=sender,
            from_number=recipient,
            content_sid=template_sid,
            content_variables=_build_csat_flow_content_variables(),
        )
    except Exception as exc:  # pragma: no cover - defensa extra
        print("\n[WHATSAPP] Error enviando CSAT")
        print(f"  - session_id: {session_id}")
        print(f"  - error: {str(exc)}")
        return

    if not sent:
        return

    try:
        updated_memory = mark_csat_sent(
            memory,
            template_sid=template_sid,
            sent_at=datetime.now(timezone.utc).isoformat(),
        )
        save_memory(session_id, updated_memory)
    except Exception as exc:  # pragma: no cover - defensa extra
        print("\n[WHATSAPP] Error guardando estado CSAT")
        print(f"  - session_id: {session_id}")
        print(f"  - error: {str(exc)}")


def _build_csat_flow_content_variables() -> dict[str, str]:
    if CSAT_FLOW_CONTENT_VARIABLES_JSON is not None:
        raw_content_variables = CSAT_FLOW_CONTENT_VARIABLES_JSON.strip()
        if not raw_content_variables:
            return {}

        try:
            parsed_content_variables = json.loads(raw_content_variables)
        except json.JSONDecodeError as exc:
            print("\n[WHATSAPP] Error parseando CSAT_FLOW_CONTENT_VARIABLES_JSON")
            print(f"  - error: {str(exc)}")
        else:
            if isinstance(parsed_content_variables, dict):
                built_content_variables: dict[str, str] = {}
                for key, value in parsed_content_variables.items():
                    if value is None:
                        continue

                    text_value = str(value).strip()
                    if not text_value:
                        continue

                    built_content_variables[str(key)] = (
                        uuid4().hex
                        if text_value == CSAT_FLOW_TOKEN_PLACEHOLDER
                        else text_value
                    )

                return built_content_variables

    return {}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
