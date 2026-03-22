How to set it up on Render / server:

# 1. Generate a bcrypt hash for your chosen password (run locally):
python -c "import bcrypt; print(bcrypt.hashpw(b'YOUR_PASSWORD', bcrypt.gensalt(rounds=12)).decode())"

# 2. Generate a signing secret for tokens:
python -c "import secrets; print(secrets.token_hex(32))"

Then add two environment variables in Render's dashboard / server:

Key |	Value
`SESSIONS_PASSWORD_HASH` |	`$2b$12$...` (output from step 1)
`SESSIONS_TOKEN_SECRET` |	the hex string from step 2