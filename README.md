# Gala Agentic

Bot multiagente con LangGraph + RAG para responder consultas de ayuda de Banco Galicia y exponerlo como webhook para WhatsApp vía Twilio.

## Endpoints

- `GET /` devuelve estado básico del servicio.
- `GET /health` healthcheck para Render.
- `POST /webhook` webhook productivo recomendado para Twilio WhatsApp.
- `POST /whatsapp` alias del webhook.
- `POST /evaluate` endpoint JSON para correr datasets y experiments de Langfuse sin Twilio.

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
LANGFUSE_OBSERVABILITY_ENABLED=true
LANGFUSE_NODE_OBSERVABILITY_ENABLED=true
LANGFUSE_NODE_SCORE_ENABLED=true
LANGFUSE_QUALITY_EVAL_ENABLED=true
LANGFUSE_CATEGORICAL_SCORES_ENABLED=true
LANGFUSE_SESSION_METRICS_ENABLED=true
LANGFUSE_SESSION_CATEGORICAL_ENABLED=true
LLM_JUDGE_ENABLED=false
OPENAI_MODEL_JUDGE=
LLM_JUDGE_MAX_INPUT_CHARS=1000
LLM_JUDGE_MAX_CONTEXT_CHARS=2000
APP_ENV=dev
APP_VERSION=unknown
GALA_GRAPH_VERSION=gala_graph_v1
EVALUATION_ENDPOINT_ENABLED=true
EVALUATION_ENDPOINT_TOKEN=
EVALUATION_BACKEND_URL=http://localhost:5000
LANGFUSE_DATASET_NAME=
LANGFUSE_EXPERIMENT_NAME=
LANGFUSE_EXPERIMENT_DESCRIPTION=
EVALUATION_REQUEST_TIMEOUT_SECONDS=45
LANGFUSE_EXPERIMENT_MAX_CONCURRENCY=4
```

Si `LANGFUSE_OBSERVABILITY_ENABLED=false`, el bot sigue funcionando pero no intenta enviar la capa avanzada de metadata y scores a Langfuse.
Si `LANGFUSE_NODE_OBSERVABILITY_ENABLED=false`, el bot mantiene la observabilidad general del trace pero no abre spans por nodo.
Si `LANGFUSE_NODE_SCORE_ENABLED=false`, los spans por nodo pueden seguir existiendo pero sin scores por nodo.
Si `LANGFUSE_QUALITY_EVAL_ENABLED=false`, se desactiva la capa de evaluacion de calidad.
Si `LANGFUSE_CATEGORICAL_SCORES_ENABLED=false`, no se envian scores categóricos como `conversation_route`, `conversation_outcome` o `conversation_quality_status`.
Si `LANGFUSE_SESSION_METRICS_ENABLED=false`, no se envian scores numéricos agregados por sesion como `session_turn_count` o `session_fallback_count`.
Si `LANGFUSE_SESSION_CATEGORICAL_ENABLED=false`, no se envian scores categóricos de sesion como `session_status`, `session_primary_route` o `session_complexity`.
Si `LLM_JUDGE_ENABLED=false`, solo corren los evaluadores programaticos y no hay costo adicional de judge por LLM.

## Experiments

- `POST /evaluate` ejecuta el mismo grafo del bot sin pasar por Twilio, no devuelve TwiML y usa sesiones `eval-*`.
- `python experiments/run_langfuse_dataset.py --dataset "Gala Regression Cases" --experiment "local-regression-baseline-v1" --backend-url "http://localhost:5000"` corre un dataset completo contra el backend y registra el dataset run en Langfuse.
- La guia completa para proteger el endpoint, estructurar datasets y leer scores estÃ¡ en `docs/experiments.md`.
Variables opcionales para WhatsApp:

```bash
WHATSAPP_ASYNC_MAX_WORKERS=4
WHATSAPP_INBOUND_DEDUPE_TTL_SECONDS=1800
WHATSAPP_SESSION_TTL_SECONDS=900
```

`WHATSAPP_SESSION_TTL_SECONDS` controla la duracion de la sesion conversacional de WhatsApp por inactividad. El `user_id` se mantiene estable por numero normalizado, por ejemplo `whatsapp-5491125456750`, mientras que el `session_id` conversacional rota por charla, por ejemplo `whatsapp-5491125456750-20260525-190502-a8f3`. Ese `session_id` es el que usa la memoria local del bot y el `langfuse_session_id`, mientras que Langfuse sigue recibiendo el numero normalizado como `langfuse_user_id`.

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
- `data/session_store/`
- `data/vectorstore/`
- `__pycache__/`
- `.vscode/`
- `ngrok.exe`
