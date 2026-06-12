How to set it up on Render / server:

# 1. Generate a bcrypt hash for your chosen password (run locally):
python -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASSWORD', bcrypt.gensalt(rounds=12)).decode())"

# 2. Generate a signing secret for tokens:
python -c "import secrets; print(secrets.token_hex(32))"

Then add two environment variables in Render's dashboard / server:

Key |	Value
`SESSIONS_PASSWORD_HASH` |	`$2b$12$...` (output from step 1)
`SESSIONS_TOKEN_SECRET` |	the hex string from step 2

---

## Session inactivity & close detection

A conversation is closed and written to disk (so it appears in `sessions.html`)
as soon as it ends, rather than waiting out a long timeout. Three mechanisms
cooperate, designed to keep the server's view accurate **without** flooding the
backend with heartbeat requests:

1. **Throttled activity heartbeat (frontend).** Any real user interaction —
   click, keypress, scroll, mousemove, touch — counts as activity and touches
   the session's `last_activity` via `POST /api/session/activity`, but at most
   **once every 2 minutes** (`HEARTBEAT_THROTTLE_MS` in `index.html`). A user
   who scrolls continuously still generates ≤ 1 request per window; an idle user
   generates **zero**. Silence therefore reliably means "inactive".

2. **End-on-close beacon (frontend).** When the tab is actually unloaded
   (closed or navigated away), `pagehide` fires `navigator.sendBeacon` to
   `POST /api/session/end` with the final transcript. The beacon is delivered
   even as the page goes away, so an intentionally- or accidentally-closed
   conversation is persisted almost immediately. The `e.persisted` check skips
   back/forward (bfcache) navigations, which may be restored.

3. **Server-side timeout backstop.** `SESSION_TIMEOUT_MINUTES = 5` and the
   `session_cleanup` job (every **1 min**) expire and save any session whose
   `last_activity` is older than the cutoff. This covers the rare case where the
   close beacon never arrives (browser crash, network loss, OS killing the tab).
   A hidden-tab `visibilitychange` snapshot beacons the latest messages to
   `/api/session/log` first, so the saved transcript is still complete.

Net effect: clean tab-close → dashboard within seconds; abandoned tab →
dashboard within ~5–6 minutes; near-zero requests from idle users.