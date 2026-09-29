# SPS GenAI – Assignment 1

FastAPI service with two features:

- **Bigram text generation** (Module 3 class activity) – `POST /generate`
- **spaCy word embeddings** (Assignment 1) – `POST /embedding`, plus `POST /similarity`

## Run locally

```bash
uv sync
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs to try the endpoints.

### Example requests

```bash
curl -X POST http://127.0.0.1:8000/embedding \
  -H "Content-Type: application/json" -d '{"word": "apple"}'

curl -X POST http://127.0.0.1:8000/similarity \
  -H "Content-Type: application/json" -d '{"word1": "apple", "word2": "car"}'

curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" -d '{"start_word": "bigram", "length": 5}'
```

`/embedding` returns the 300-dimensional vector from spaCy's `en_core_web_lg` model:

```json
{"word": "apple", "dimension": 300, "embedding": [-0.63, -0.20, ...]}
```

## Run with Docker (optional)

```bash
docker build -t sps-genai .
docker run -p 8000:80 sps-genai
```
