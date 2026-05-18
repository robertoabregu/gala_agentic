from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import requests


ALLOWED_PDF_CONTENT_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
    "text/pdf",
    "text/x-pdf",
}

REQUEST_TIMEOUT_SECONDS = 20
MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
CHUNK_SIZE_BYTES = 64 * 1024


class TwilioMediaError(RuntimeError):
    """Raised when the Twilio media cannot be downloaded safely."""


def looks_like_pdf_media(media: dict[str, Any] | None) -> bool:
    if not isinstance(media, dict):
        return False

    content_type = str(media.get("content_type") or "").strip().lower()
    filename = str(media.get("filename") or "").strip().lower()

    if content_type in ALLOWED_PDF_CONTENT_TYPES:
        return True

    if "pdf" in content_type:
        return True

    return filename.endswith(".pdf")


def build_media_payload(
    *,
    num_media: str,
    url: str,
    content_type: str,
    filename: str,
) -> dict[str, Any]:
    return {
        "num_media": num_media,
        "url": url,
        "content_type": content_type,
        "filename": filename,
    }


def download_twilio_pdf_to_tempfile(media: dict[str, Any]) -> Path:
    if not looks_like_pdf_media(media):
        raise TwilioMediaError("Adjuntá un PDF válido del resumen de tarjeta.")

    media_url = str(media.get("url") or "").strip()
    if not media_url:
        raise TwilioMediaError("No pude acceder al PDF adjunto.")

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    auth = (account_sid, auth_token) if account_sid and auth_token else None

    temp_path: Path | None = None

    try:
        with requests.get(
            media_url,
            auth=auth,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
        ) as response:
            if response.status_code >= 400:
                raise TwilioMediaError(
                    "No pude descargar el PDF adjunto desde WhatsApp."
                )

            response_content_type = str(
                response.headers.get("Content-Type") or media.get("content_type") or ""
            ).strip().lower()

            if response_content_type and "pdf" not in response_content_type and not looks_like_pdf_media(
                {
                    "content_type": response_content_type,
                    "filename": media.get("filename", ""),
                }
            ):
                raise TwilioMediaError("Adjuntá un PDF válido del resumen de tarjeta.")

            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    if int(content_length) > MAX_PDF_SIZE_BYTES:
                        raise TwilioMediaError(
                            "El PDF adjunto es demasiado grande para analizarlo por acá."
                        )
                except ValueError:
                    pass

            suffix = ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = Path(temp_file.name)
                bytes_written = 0

                for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                    if not chunk:
                        continue

                    bytes_written += len(chunk)
                    if bytes_written > MAX_PDF_SIZE_BYTES:
                        raise TwilioMediaError(
                            "El PDF adjunto es demasiado grande para analizarlo por acá."
                        )

                    temp_file.write(chunk)

        if temp_path is None or not temp_path.exists() or temp_path.stat().st_size == 0:
            raise TwilioMediaError("No pude descargar el PDF adjunto.")

        return temp_path

    except TwilioMediaError:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise TwilioMediaError(
            "No pude descargar el PDF adjunto en este momento."
        ) from exc


def cleanup_temp_file(path: str | Path | None) -> None:
    if not path:
        return

    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        return
