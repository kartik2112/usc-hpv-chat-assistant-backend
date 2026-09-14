---
name: hpv-backend
description: Map of the HPV Chat Assistant Flask backend — variants (general vs post-partum), session lifecycle and on-disk layout, dashboard auth scopes, prompts, PHI guardrail, and how to test offline. Use before editing any file in usc-hpv-chat-assistant-backend/.
---

# HPV Chat Assistant — backend

Flask app (`flask_backend.py`, ~2.2k lines) + `rag_pipeline.py` (LangChain + Chroma Cloud) + `variants.py`.
Deployed on Render (`gunicorn … --workers 1`) and `sackend.isi.edu`. Dependencies come from `uv sync`.
**It must stay at one worker**: live sessions are an in-memory dict.

## Files
| File | Role |
|---|---|
| `variants.py` | **Single source of truth for variants**: `Variant(key, label, audience_instructions, summary_context)`, `VARIANTS`, `DEFAULT_VARIANT`, `parse_variant()` |
| `rag_sources.json` / `rag_sources.py` | Per-variant Chroma target + source URLs, and the validating loader (`load_rag_sources()`) |
| `rag_pipeline.py` | `BASE_SYSTEM_RULES` + `build_system_prompt(audience_instructions)`, `build_rag_pipeline(rag_sources)`, retrieval, `ask_rag_question_stream()` |
| `flask_backend.py` | Routes, sessions, PHI detection, summaries, dashboard |
| `conftest.py` | Offline fixtures (`fb`, `client`, `FakePipeline`) shared by the test files below |
| `test_variants.py`, `test_rag_sources.py` | Offline tests (stubbed RAG, PHI off) — `uv run pytest test_variants.py test_rag_sources.py -v` |
| `test_phi_guardrails.py` | Needs the spaCy models **and** a real RAG build (network) — run it separately |

## flask_backend.py anchors (grep these)
- `IS_RENDER`/`PHI_ENABLED` — Presidio/spaCy only off Render. `_run_phi_selftest()` **aborts startup** if it fails.
- `ALLOWED_ORIGINS` + `restrict_to_allowed_origins` — origin gate (requests without an Origin header pass).
- `_request_variant()` — the variant for patient calls: the live session's wins, otherwise the validated `variant` field.
- `_session_file_stem()`, `_migrate_legacy_session_files()`, `_variant_of()`.
- `SESSIONS_PASSWORD_HASH`, `_make_dashboard_token()`, `_dashboard_token_scope()`, `require_dashboard_token` (injects `variant=`).
- `generate_session_summary(messages, variant)`, `save_session_to_disk()`, `auto_expire_sessions()`.
- `_rag_pipelines` — one pipeline per variant, built from `load_rag_sources()`; `daily_task()` re-reads the file and keeps the old pipeline for any variant that fails to rebuild.
- Routes: `/api/chat` (SSE), `/api/audio-chat`, `/api/tts`, `/api/session/{start,activity,log,summary,end}`, `/api/sessions/auth`, `/api/sessions[/<file>]`, `/api/sessions/{favorite,delete,merge}`, `/api/rag/sources` (read-only sources viewer).

## Variants
- Keys are `general` and `postpartum`, and they must match `variants.js` in the frontend repo.
- `/api/session/start {variant}` binds the variant to the session. After that, `/log`, `/summary`, `/end`, `/chat` and expiry all read it from the session, so the client cannot switch it. An unknown key gets a 400; a missing key means `general`.
- Prompt = `BASE_SYSTEM_RULES` + `variant.audience_instructions` + the RAG context. General keeps the "assume age > 26" line; post-partum swaps it for post-partum framing. The summary prompt adds `variant.summary_context`.
- **Adding a variant:** create a `Variant` in `variants.py` and add it to `VARIANTS`. Folders are created at startup, and nothing else in the backend needs to change. Then add it to the frontend's `variants.js`.

## RAG sources
- `rag_sources.json` (path overridable with `RAG_SOURCES_FILE`) maps each variant to a Chroma target (`database`, `collection`, optional `api_key_env`/`tenant_env` for another Chroma account) and its web page / PDF URLs.
- Two variants sharing a collection is rejected at load: each build deletes chunks for URLs outside its own list.
- Most publishers (Wiley, JAMA, MDPI, cdc.gov, PMC, Springer, Elsevier) block the crawler. Use `europepmc.org/articles/PMC<id>?pdf=render`, the Europe PMC `fullTextXML` REST endpoint, or a PubMed abstract URL instead, and verify the character count before adding. `has_usable_content()` / `MIN_EXTRACTED_CHARS` (500) in rag_pipeline.py stop a short block page being indexed as if it were the source.
- `HPVRAGPipeline.describe_indexed_sources()` reads the collection's chunk metadata from Chroma (grouped by `source`) — that is what `/api/rag/sources` lists, so the viewer reflects Chroma, not the file. The file adds titles and the `not_indexed` / `not_in_config` flags; `index_report` (per URL: status/chunks/kind) and `built_at` describe the last crawl by this process.

## Storage layout
```
sessions/general/session_general_<uuid>_<UTCts>.{json,txt}
sessions/postpartum/session_postpartum_<uuid>_<UTCts>.{json,txt}
sessions/<variant>/session_<variant>_merged-<hex>_<UTCts>.*      # merges
```
The JSON has `"variant"`, and the TXT header has a `Variant :` line. Files from before variants existed are moved once into `sessions/general/` at startup (idempotent). PHI-flagged messages are removed before writing to disk and before the summary LLM call.

## Dashboard auth
- `SESSIONS_PASSWORD_HASH` (bcrypt) is the single dashboard password — it unlocks every variant and both dashboard pages. `SESSIONS_TOKEN_SECRET` signs tokens; if it's unset, a random secret is generated and tokens die on restart.
- Token: `<expiry>.<sorted,keys>.<hmac>` (2 h). Every dashboard route takes `?variant=`, gets 403 if the variant is outside the token's scope, and only touches `variant.sessions_dir`, using the basename of the filename. Merges can't cross variants.
- Auth is rate-limited per IP (5 failures → 15 min lockout, in memory).

## Session lifecycle
Start → a heartbeat every ≤2 min (`/activity`) → `/log` events and a message snapshot (`messages_seq` guards against stale writes) → `/summary` (cached by message fingerprint) → `/end` (saves the files). There's also a tab-close beacon to `/end`, and a backstop that expires a session after 5 min idle (`session_cleanup` runs every minute).

## Offline smoke run
`test_variants.py` shows the pattern: set `RENDER=1` and `OPENAI_API_KEY=x`, patch `rag_pipeline.build_rag_pipeline` with a fake pipeline class (see `conftest.FakePipeline`) **before** `import flask_backend`, then stub `generate_session_summary` (and `ask_rag_question_stream` / `retrieve_context_docs` if you need `/api/chat`). Add `http://localhost:<port>` to `ALLOWED_ORIGINS` to drive it from a local frontend.
