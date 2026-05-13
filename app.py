import re

from flask import Flask, Response, jsonify, request
from twilio.twiml.messaging_response import MessagingResponse

from core.bot_runner import BotRuntime, prepare_runtime, run_bot_query
from core.privacy import mask_sensitive_text
from services.twilio_typing import send_whatsapp_typing_indicator


app = Flask(__name__)
_runtime: BotRuntime | None = None


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

        if not body:
            return twiml_message("No recibí tu mensaje. Probá de nuevo, por favor.")

        print("\n[WHATSAPP] Mensaje recibido")
        print(f"  - from: {session_id}")
        print(f"  - body: {mask_sensitive_text(body)}")

        if user_location:
            print("  - location: recibida")

        send_whatsapp_typing_indicator(message_sid)

        result = run_bot_query(
            runtime=get_runtime(),
            question=body,
            session_id=session_id,
            langfuse_user_id=session_id,
            langfuse_tags=["gala", "langgraph", "rag", "whatsapp"],
            observation_name="gala-whatsapp-request",
            user_location=user_location,
        )

        return twiml_message(result["final_answer"])

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
