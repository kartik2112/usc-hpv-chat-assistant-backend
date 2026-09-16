"""
test_variants.py
================
General vs post-partum variant behaviour: prompt selection, per-variant session
folders/filenames, the session↔variant binding, and dashboard token scoping.

Runs offline (see conftest.py for the fixtures):

    uv run pytest test_variants.py -v

Run it on its own: test_phi_guardrails.py imports flask_backend with PHI
enabled and a real RAG build, which this file deliberately avoids.
"""

import glob
import json
import os

import pytest

from conftest import DASHBOARD_PW, LEGACY_FILE, auth_header as _auth, token_for as _token


def _run_conversation(client, variant):
    """Start → log one exchange → end. Returns the session_id."""
    resp = client.post('/api/session/start', json={'variant': variant})
    assert resp.status_code == 200
    sid = resp.get_json()['session_id']
    messages = [{'role': 'user', 'content': 'Can I breastfeed after the HPV vaccine?'},
                {'role': 'assistant', 'content': 'Yes.'}]
    client.post('/api/session/log', json={'session_id': sid, 'messages': messages, 'messages_seq': 1})
    assert client.post('/api/session/end', json={'session_id': sid}).status_code == 200
    return sid


# ── Registry + prompts ────────────────────────────────────────────────────────

def test_parse_variant():
    from variants import parse_variant, GENERAL, POSTPARTUM
    assert parse_variant(None) is GENERAL
    assert parse_variant('') is GENERAL
    assert parse_variant(' PostPartum ') is POSTPARTUM
    assert parse_variant('../general') is None
    assert parse_variant('unknown') is None


def test_system_prompt_per_variant():
    from rag_pipeline import build_system_prompt
    from variants import GENERAL, POSTPARTUM
    general = build_system_prompt(GENERAL.audience_instructions)
    postpartum = build_system_prompt(POSTPARTUM.audience_instructions)
    assert 'ABOVE 26' in general and 'given birth' not in general
    assert 'given birth' in postpartum and 'ABOVE 26' not in postpartum
    assert general.endswith('Use the following context in your response:')
    assert 'Use the following context' not in build_system_prompt('', with_context_header=False)


# ── Storage layout ────────────────────────────────────────────────────────────

def test_legacy_files_migrated_to_general(fb):
    assert not os.path.exists(os.path.join('sessions', LEGACY_FILE))
    assert os.path.exists(os.path.join('sessions', 'general', 'session_general_11111111-legacy_20250101_000000.json'))


@pytest.mark.parametrize('variant,label', [('general', 'General HPV'), ('postpartum', 'Post-partum HPV')])
def test_saved_in_variant_folder_with_tag(client, variant, label):
    sid = _run_conversation(client, variant)
    [json_path] = glob.glob(os.path.join('sessions', variant, f'session_{variant}_{sid}_*.json'))
    data = json.load(open(json_path))
    assert data['variant'] == variant
    assert data['summary']['patient_questions'] == f'• asked ({variant})'
    txt = open(json_path[:-5] + '.txt', encoding='utf-8').read()
    assert f'Variant    : {label} ({variant})' in txt
    other = 'general' if variant == 'postpartum' else 'postpartum'
    assert not glob.glob(os.path.join('sessions', other, f'*{sid}*'))


def test_unknown_variant_rejected(client):
    assert client.post('/api/session/start', json={'variant': 'nope'}).status_code == 400


def test_session_variant_cannot_be_switched(fb, client):
    sid = client.post('/api/session/start', json={'variant': 'postpartum'}).get_json()['session_id']
    assert fb._request_variant({'session_id': sid, 'variant': 'general'}).key == 'postpartum'
    assert fb._request_variant({'variant': 'general'}).key == 'general'
    assert fb._request_variant({'variant': 'bogus'}) is None


# ── Dashboard scoping ─────────────────────────────────────────────────────────

def test_one_password_opens_both_views_separately(client):
    """A single dashboard password sees every variant, but each view lists only
    its own conversations."""
    pp_sid = _run_conversation(client, 'postpartum')
    resp = client.post('/api/sessions/auth', json={'password': DASHBOARD_PW})
    assert resp.get_json()['variants'] == ['general', 'postpartum']
    token = resp.get_json()['token']
    general = client.get('/api/sessions?variant=general', headers=_auth(token)).get_json()
    postpartum = client.get('/api/sessions?variant=postpartum', headers=_auth(token)).get_json()
    assert pp_sid in {s['session_id'] for s in postpartum['sessions']}
    assert pp_sid not in {s['session_id'] for s in general['sessions']}


def test_token_scope_is_enforced(fb, client):
    """Scope still gates every dashboard route, so a narrower token (should one
    ever be issued) cannot reach another variant."""
    narrow = _auth(fb._make_dashboard_token({'postpartum'}))
    assert client.get('/api/sessions?variant=postpartum', headers=narrow).status_code == 200
    assert client.get('/api/sessions?variant=general', headers=narrow).status_code == 403
    assert client.post('/api/sessions/delete?variant=general', headers=narrow,
                       json={'filenames': ['x.json']}).status_code == 403


def test_tampered_or_legacy_tokens_rejected(fb, client):
    # Editing the scope in a token breaks its signature.
    expiry, _scope, sig = fb._make_dashboard_token({'postpartum'}).split('.')
    widened = f'{expiry}.general,postpartum.{sig}'
    assert client.get('/api/sessions?variant=general', headers=_auth(widened)).status_code == 401
    # A token in the pre-variant '<expiry>.<sig>' format is no longer accepted.
    assert client.get('/api/sessions', headers=_auth(f'{expiry}.{sig}')).status_code == 401


def test_dashboard_cannot_reach_other_variant_files(client):
    sid = _run_conversation(client, 'general')
    [path] = glob.glob(os.path.join('sessions', 'general', f'*{sid}*.json'))
    fname = os.path.basename(path)
    token = _token(client, DASHBOARD_PW)
    # Traversal is stripped to the basename, which does not exist in postpartum/.
    resp = client.get(f'/api/sessions/..%2Fgeneral%2F{fname}?variant=postpartum', headers=_auth(token))
    assert resp.status_code == 404
    resp = client.post('/api/sessions/merge?variant=postpartum', headers=_auth(token),
                       json={'filenames': [fname, fname + 'x.json']})
    assert resp.status_code in (400, 404)
    assert os.path.exists(path)


def test_event_offsets_survive_browser_timestamps(fb):
    """Event offsets are computed against created_at, which must be UTC-aware.

    index.html sends `new Date().toISOString()`, i.e. a 'Z'-suffixed string that
    fromisoformat() parses as aware. When created_at was naive (datetime.utcnow())
    the subtraction raised TypeError and every offset in the transcript fell back
    to '?'. Both sides are aware now, so the offsets are real.
    """
    from datetime import datetime, timedelta, timezone

    created = datetime.now(timezone.utc) - timedelta(seconds=90)
    session_id = 'tz-offsets'
    fb.sessions[session_id] = {
        'variant': 'general', 'created_at': created,
        'last_activity': datetime.now(timezone.utc),
        'messages': [], 'survey_responses': [], 'last_messages_seq': -1,
        'events': [
            {'type': 'user_message',
             'timestamp': (created + timedelta(seconds=30)).replace(tzinfo=None).isoformat() + 'Z'},
            {'type': 'bot_message',
             'timestamp': (created + timedelta(seconds=75)).replace(tzinfo=None).isoformat() + 'Z'},
        ],
    }
    fb.save_session_to_disk(session_id, fb.sessions[session_id],
                            {'patient_questions': 'q', 'action_items': 'a'})

    [txt] = glob.glob(os.path.join('sessions', 'general', f'*{session_id}*.txt'))
    event_log = open(txt).read().split('EVENT LOG')[1]
    assert '+30s' in event_log and '+1m15s' in event_log
    assert '?' not in event_log

    # Stored timestamps stay UTC, now carrying an explicit offset. to_pst()
    # already accepted both forms, so the dashboard reads the same instant.
    [js] = glob.glob(os.path.join('sessions', 'general', f'*{session_id}*.json'))
    ended_at = json.load(open(js))['ended_at']
    assert ended_at.endswith('+00:00')
    assert fb.to_pst(ended_at) == fb.to_pst(ended_at.replace('+00:00', ''))
