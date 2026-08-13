# qbgen

Quiz bowl study tools over the [QBReader](https://qbreader.org) database: semantically
deduplicated clue generation, whole-set carding, bonus co-occurrence search, and Anki
(.apkg) export.

## Architecture

- `backend/` — stateless Flask API (`app.py`). Embeds clue sentences with
  all-MiniLM-L6-v2, splits with spaCy, clusters near-duplicates, builds Anki decks with
  genanki. No database; QBReader is queried live.
- `qbgen-landing/` — Next.js 14 static export (`output: 'export'`). Calls the API
  directly from the browser; the base URL comes from `NEXT_PUBLIC_API_BASE_URL`.
- `backend/build_corpus_deck.py` — offline CLI that mass-produces per-category corpus
  decks into `output/corpus-decks/`.

## Local development

Backend (http://localhost:8080):

    cd backend
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m spacy download en_core_web_sm
    .venv/bin/python app.py

Frontend (http://localhost:3000, auto-targets localhost:8080):

    cd qbgen-landing && pnpm install && pnpm dev

## Deployment

- Backend: Cloud Run (`us-central1`), built and deployed by a Cloud Build trigger on
  push. `service.yaml` is an exported snapshot of the live service, not an applied
  manifest. Keep `minScale: 0`, `maxScale: 2` (cost ceiling).
- Frontend: `pnpm build` produces `out/`, served from a static host with
  `NEXT_PUBLIC_API_BASE_URL` set to the Cloud Run URL at build time.
