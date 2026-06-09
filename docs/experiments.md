# Experiments With Langfuse

## Objetivo

`POST /evaluate` y `experiments/run_langfuse_dataset.py` permiten ejecutar datasets de Langfuse contra el bot completo sin pasar por Twilio.

La idea de uso es:

1. Detectar conversaciones candidatas a dataset.
2. Cargar esos casos en Langfuse.
3. Modificar prompts, routing, RAG o tools.
4. Ejecutar un nuevo experiment contra la version actual del bot.
5. Comparar el dataset run nuevo con los anteriores en Langfuse.

No se compara texto exacto por default. Se evalua comportamiento esperado.

## Endpoint `/evaluate`

- Metodo: `POST`
- Content-Type: `application/json`
- Respuesta: JSON
- No devuelve TwiML.
- No envia mensajes por WhatsApp.
- No dispara templates ni notificaciones externas.
- Ejecuta el mismo grafo del bot via `run_bot_query()`.

Request esperado:

```json
{
  "question": "texto del usuario",
  "session_id": "eval-session-id-opcional",
  "channel": "langfuse_experiment",
  "metadata": {
    "dataset_name": "Gala Regression Cases",
    "dataset_item_id": "item-123",
    "case_type": "rag_false_fallback",
    "expected_route": "loans_rag",
    "expected_topic": "prestamos"
  }
}
```

Notas:

- `question` es requerido.
- Si `session_id` no viene, el backend genera uno con prefijo `eval-`.
- Si `session_id` viene sin prefijo `eval-`, se normaliza a `eval-*` para aislar sesiones reales de sesiones de evaluacion.
- `channel` default: `langfuse_experiment`.
- `metadata` es opcional y se adjunta al trace.

Respuesta base:

```json
{
  "answer": "respuesta final del bot",
  "route": "benefits",
  "topic": "beneficios",
  "used_rag": false,
  "used_tool": true,
  "fallback": false,
  "needs_clarification": false,
  "guardrail_blocked": false,
  "documents_count": 0,
  "needs_human_review": false,
  "dataset_candidate": false,
  "session_id": "eval-item-123",
  "trace_id": "trace-si-esta-disponible",
  "latency_ms": 842
}
```

Tambien puede incluir `app_version` y `graph_version` si estan disponibles en metadata del trace.

## Proteccion

Variables nuevas:

```bash
EVALUATION_ENDPOINT_ENABLED=true
EVALUATION_ENDPOINT_TOKEN=
```

Reglas:

- Si `EVALUATION_ENDPOINT_ENABLED=false`, `/evaluate` responde `404`.
- Si `EVALUATION_ENDPOINT_TOKEN` tiene valor, el request debe enviar `X-Eval-Token: <token>`.
- Si no hay token configurado, el endpoint solo queda habilitado en ambientes `local`, `dev`, `development` o `test`.
- En produccion, lo recomendable es configurar `EVALUATION_ENDPOINT_ENABLED=false` por default y activarlo solo de forma intencional.

Ejemplo local:

```bash
curl -X POST http://localhost:5000/evaluate \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Que beneficios tengo en gastronomia?\"}"
```

Ejemplo con token:

```bash
curl -X POST https://mi-backend-qa.render.com/evaluate \
  -H "Content-Type: application/json" \
  -H "X-Eval-Token: $EVALUATION_ENDPOINT_TOKEN" \
  -d "{\"question\":\"Quiero consultar mi situacion crediticia\"}"
```

## Dataset Items

El dataset puede mezclar casos de:

- `loans_rag`
- `benefits`
- `bcra_agent` o `bcra_credit_status`
- `branch_locator`
- `chitchat`
- `credit_card_statement`
- `fallback`

Formatos soportados:

Caso simple:

```json
{
  "question": "Puedo solicitar un prendario si ya tengo un auto?"
}
```

Caso con expectativas:

```json
{
  "question": "Quiero consultar mi situacion crediticia",
  "expected_route": "bcra_agent",
  "expected_topic": "situacion_crediticia",
  "case_type": "tool_followup"
}
```

Caso con rubrica flexible:

```json
{
  "question": "Puedo solicitar un prendario si ya tengo un auto?",
  "expected_behavior": "answer_partial_with_context",
  "must_include": ["prestamo prendario", "vehiculo", "garantia"],
  "must_not_include": [
    "No tengo informacion suficiente para resolver esa consulta bancaria desde aca"
  ]
}
```

Tambien se soporta `expected_output` como string o como objeto JSON. Si es string, no se hace comparacion exacta por default.

## Script

Archivo:

```bash
python experiments/run_langfuse_dataset.py
```

Variables usadas por el script:

```bash
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=https://cloud.langfuse.com
EVALUATION_BACKEND_URL=http://localhost:5000
EVALUATION_ENDPOINT_TOKEN=
LANGFUSE_DATASET_NAME=
LANGFUSE_EXPERIMENT_NAME=
LANGFUSE_EXPERIMENT_DESCRIPTION=
EVALUATION_REQUEST_TIMEOUT_SECONDS=45
LANGFUSE_EXPERIMENT_MAX_CONCURRENCY=4
```

Ejemplo local:

```bash
python experiments/run_langfuse_dataset.py \
  --dataset "Gala Regression Cases" \
  --experiment "local-regression-after-router-fix-v1" \
  --backend-url "http://localhost:5000"
```

Ejemplo remoto:

```bash
python experiments/run_langfuse_dataset.py \
  --dataset "Gala Regression Cases" \
  --experiment "qa-regression-2026-06-09" \
  --backend-url "https://mi-backend-qa.render.com" \
  --endpoint-token "$EVALUATION_ENDPOINT_TOKEN"
```

El script:

1. Lee el dataset desde Langfuse.
2. Construye un request `/evaluate` por item.
3. Usa sesiones `eval-*`.
4. Guarda el output real del bot.
5. Calcula scores por item.
6. Registra el dataset run en Langfuse usando el SDK instalado.

## Scores

Los evaluadores actuales calculan:

- `route_match`
- `topic_match`
- `fallback_avoided`
- `used_rag_expected`
- `used_tool_expected`
- `must_include_coverage`
- `forbidden_terms_avoided`
- `needs_human_review_after_run`
- `dataset_candidate_after_run`
- `answer_non_empty`
- `whatsapp_format_basic_ok`

Interpretacion rapida:

- `route_match`: compara la ruta esperada con la ruta real, tolerando aliases como `bcra_agent` y `bcra_credit_status`.
- `topic_match`: compara el tema esperado con el tema real usando normalizacion simple.
- `fallback_avoided`: idealmente `true` cuando el caso deberia responder sin fallback.
- `must_include_coverage`: porcentaje de terminos esperados encontrados en la respuesta, en rango `0.0` a `1.0`.
- `forbidden_terms_avoided`: `true` si no aparecieron frases prohibidas.
- `answer_non_empty`: `true` si el bot devolvio texto.
- `needs_human_review_after_run` y `dataset_candidate_after_run`: reflejan los flags del bot si la evaluacion de calidad estuvo disponible.

## Nombres De Experimento

Conviene usar nombres comparables entre iteraciones, por ejemplo:

- `local-regression-baseline-v1`
- `local-regression-after-router-fix-v1`
- `qa-regression-after-rag-prompt-fix-v1`
- `qa-regression-after-benefits-tool-fix-v1`

## Resultado En Langfuse

Cada item del dataset run conserva:

- input original
- expected output o rubrica
- output real del bot
- scores calculados
- metadata auxiliar del run, como `dataset_item_id`, `experiment_name`, `backend_url`, `route`, `topic`, `case_type`, `evaluation_timestamp` y `git_commit` si estuvo disponible

La vista del dataset run en Langfuse deberia mostrar:

- el nombre del experiment
- un item por caso ejecutado
- los scores por item
- el promedio de scores en el resumen del experimento
- el link del dataset run para comparar contra ejecuciones anteriores
