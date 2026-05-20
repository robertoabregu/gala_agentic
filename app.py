import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from flask import Flask, Response, jsonify, request
from twilio.twiml.messaging_response import MessagingResponse

from agents.router import _has_benefits_context, _is_benefits_request, _normalize_text
from core.bot_runner import BotRuntime, prepare_runtime, run_bot_query
from core.privacy import mask_sensitive_text
from memory.local_memory import load_memory
from services.twilio_media import build_media_payload, looks_like_pdf_media
from services.twilio_messages import send_whatsapp_message
from services.twilio_typing import send_whatsapp_typing_indicator
from utils.whatsapp_formatting import format_whatsapp_answer


app = Flask(__name__)
_runtime: BotRuntime | None = None
_runtime_lock = threading.Lock()
_async_executor = ThreadPoolExecutor(
    max_workers=max(1, int(os.getenv("WHATSAPP_ASYNC_MAX_WORKERS", "4"))),
    thread_name_prefix="whatsapp-bg",
)
_inbound_registry: dict[str, dict[str, float | str]] = {}
_inbound_registry_lock = threading.Lock()


def get_runtime() -> BotRuntime:
    global _runtime

    if _runtime is None:
        with _runtime_lock:
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


def twiml_message(body: str = "", status_code: int = 200, media_url: str | None = None) -> Response:
    response = MessagingResponse()
    if body or media_url:
        message = response.message(body)
        if media_url:
            message.media(media_url)
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


def _ack_message() -> str:
    return (
        (os.getenv("TWILIO_ACK_MESSAGE", "") or "").strip()
        or "Estoy buscando opciones con beneficios Galicia, te escribo enseguida..."
    )


def _ack_media_url() -> str:
    return (os.getenv("TWILIO_ACK_MEDIA_URL", "") or "").strip()


def _status_callback_url() -> str:
    return (os.getenv("TWILIO_STATUS_CALLBACK_URL", "") or "").strip()


def _dedupe_ttl_seconds() -> int:
    raw_value = os.getenv("WHATSAPP_INBOUND_DEDUPE_TTL_SECONDS", "1800")
    try:
        return max(60, int(raw_value or "1800"))
    except (TypeError, ValueError):
        return 1800


def _has_location(memory: dict[str, object], user_location: dict[str, str]) -> bool:
    latitude = str((user_location or {}).get("latitude") or "").strip()
    longitude = str((user_location or {}).get("longitude") or "").strip()
    if latitude and longitude:
        return True

    persisted_location = memory.get("user_location") or {}
    if not isinstance(persisted_location, dict):
        return False

    persisted_latitude = str(persisted_location.get("latitude") or "").strip()
    persisted_longitude = str(persisted_location.get("longitude") or "").strip()
    return bool(persisted_latitude and persisted_longitude)


def _should_process_async(
    *,
    body: str,
    memory: dict[str, object],
    user_location: dict[str, str],
    media: dict[str, str],
) -> bool:
    if media and looks_like_pdf_media(media):
        return False

    normalized_body = _normalize_text(body)
    pending_route = str(memory.get("pending_route") or "").strip()
    last_topic = str(memory.get("last_topic") or "").strip()
    has_location = _has_location(memory, user_location)

    if not has_location:
        return False

    if pending_route == "benefits":
        return True

    if _is_benefits_request(normalized_body):
        return True

    if last_topic == "beneficios" and _has_benefits_context(normalized_body):
        return True

    return False


def _cleanup_inbound_registry() -> None:
    now = time.time()
    ttl_seconds = _dedupe_ttl_seconds()
    with _inbound_registry_lock:
        expired_keys = [
            message_sid
            for message_sid, entry in _inbound_registry.items()
            if float(entry.get("updated_at", 0)) + ttl_seconds <= now
        ]
        for message_sid in expired_keys:
            _inbound_registry.pop(message_sid, None)


def _reserve_inbound_message(message_sid: str) -> bool:
    if not message_sid:
        return True

    _cleanup_inbound_registry()
    now = time.time()
    with _inbound_registry_lock:
        entry = _inbound_registry.get(message_sid)
        if entry and str(entry.get("status") or "") in {"processing", "completed"}:
            return False

        _inbound_registry[message_sid] = {
            "status": "processing",
            "updated_at": now,
        }
    return True


def _complete_inbound_message(message_sid: str) -> None:
    if not message_sid:
        return

    with _inbound_registry_lock:
        if message_sid in _inbound_registry:
            _inbound_registry[message_sid] = {
                "status": "completed",
                "updated_at": time.time(),
            }


def _release_inbound_message(message_sid: str) -> None:
    if not message_sid:
        return

    with _inbound_registry_lock:
        _inbound_registry.pop(message_sid, None)


def _run_and_format_bot_reply(
    *,
    body: str,
    session_id: str,
    user_location: dict[str, str],
    media: dict[str, str],
) -> tuple[dict[str, object], str]:
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
    return result, formatted_answer


def _process_whatsapp_request_async(
    *,
    body: str,
    sender: str,
    session_id: str,
    message_sid: str,
    user_location: dict[str, str],
    media: dict[str, str],
) -> None:
    try:
        _result, formatted_answer = _run_and_format_bot_reply(
            body=body,
            session_id=session_id,
            user_location=dict(user_location),
            media=dict(media),
        )
        if formatted_answer:
            send_whatsapp_message(
                to_number=sender,
                body=formatted_answer,
                status_callback=_status_callback_url() or None,
            )
    except Exception as exc:
        print("\n[WHATSAPP] Error procesando mensaje async")
        print(f"  - error: {str(exc)}")
        print(f"  - body: {mask_sensitive_text(body)}")
        try:
            send_whatsapp_message(
                to_number=sender,
                body=(
                    "Perdón, tuve un problema procesando tu consulta. "
                    "Probá de nuevo en unos minutos."
                ),
                status_callback=_status_callback_url() or None,
            )
        except Exception:
            pass
    finally:
        _complete_inbound_message(message_sid)


def _submit_async_whatsapp_job(
    *,
    body: str,
    sender: str,
    session_id: str,
    message_sid: str,
    user_location: dict[str, str],
    media: dict[str, str],
) -> None:
    _async_executor.submit(
        _process_whatsapp_request_async,
        body=body,
        sender=sender,
        session_id=session_id,
        message_sid=message_sid,
        user_location=dict(user_location),
        media=dict(media),
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

        memory = load_memory(session_id)
        if _should_process_async(
            body=body,
            memory=memory,
            user_location=user_location,
            media=media,
        ):
            if not _reserve_inbound_message(message_sid):
                return twiml_message()

            try:
                _submit_async_whatsapp_job(
                    body=body,
                    sender=sender,
                    session_id=session_id,
                    message_sid=message_sid,
                    user_location=user_location,
                    media=media,
                )
            except Exception:
                _release_inbound_message(message_sid)
                raise

            return twiml_message(
                _ack_message(),
                media_url=_ack_media_url() or None,
            )

        send_whatsapp_typing_indicator(message_sid)
        _result, formatted_answer = _run_and_format_bot_reply(
            body=body,
            session_id=session_id,
            user_location=user_location,
            media=media,
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
