# Gala Agentic

Bot multiagente con LangGraph + RAG para responder consultas de ayuda de Banco Galicia y exponerlo como webhook para WhatsApp vía Twilio.

## Endpoints

- `GET /` devuelve estado básico del servicio.
- `GET /health` healthcheck para Render.
- `POST /webhook` webhook productivo recomendado para Twilio WhatsApp.
- `POST /whatsapp` alias del webhook.

## Variables de entorno

Copiar `.env.example` a `.env` para uso local y completar los valores reales. No subir `.env` al repositorio.

Variables mínimas para producción:

```bash
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
MIN_CHUNK_SIZE=500
MAX_CHUNK_SIZE=800
RAG_SCORE_THRESHOLD=0.5
```

Variables opcionales para observabilidad:

```bash
LANGFUSE_SECRET_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Uso local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py --rebuild
python main.py -q "Cómo abro una cuenta?"
python app.py
```

Para probar el webhook local:

```bash
curl -X POST http://localhost:5000/webhook \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "From=whatsapp:+5491111111111" \
  --data-urlencode "To=whatsapp:+5491164421355" \
  --data-urlencode "Body=Cómo abro una cuenta?"
```

## Deploy en Render

Este repo incluye `render.yaml`, pero también puede configurarse manualmente:

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app -k gthread --threads 4 --timeout 120 --workers 1`
- Health Check Path: `/health`

El vectorstore FAISS no se sube al repo porque es generado. Si no existe en Render, el primer arranque/primer request lo reconstruye desde `data/documents.json` usando `OPENAI_API_KEY`.

## Configuración en Twilio

En el sandbox o sender productivo de WhatsApp, configurar:

- When a message comes in: `https://TU-SERVICIO.onrender.com/webhook`
- Method: `POST`

La línea destino esperada es `whatsapp:+5491164421355`.

## Archivos ignorados

No se versionan archivos locales o sensibles:

- `.env`
- `data/memory/`
- `data/vectorstore/`
- `__pycache__/`
- `.vscode/`
- `ngrok.exe`
