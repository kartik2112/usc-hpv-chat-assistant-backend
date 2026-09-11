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

from conftest import LEGACY_FILE, MASTER_PW, POSTPARTUM_PW, auth_header as _auth, token_for as _token


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

def test_master_password_sees_each_variant_separately(client):
    pp_sid = _run_conversation(client, 'postpartum')
    token = _token(client, MASTER_PW)
    general = client.get('/api/sessions?variant=general', headers=_auth(token)).get_json()
    postpartum = client.get('/api/sessions?variant=postpartum', headers=_auth(token)).get_json()
    assert pp_sid in {s['session_id'] for s in postpartum['sessions']}
    assert pp_sid not in {s['session_id'] for s in general['sessions']}


def test_postpartum_password_is_scoped(client):
    resp = client.post('/api/sessions/auth', json={'password': POSTPARTUM_PW})
    assert resp.get_json()['variants'] == ['postpartum']
    token = resp.get_json()['token']
    assert client.get('/api/sessions?variant=postpartum', headers=_auth(token)).status_code == 200
    assert client.get('/api/sessions?variant=general', headers=_auth(token)).status_code == 403
    assert client.post('/api/sessions/delete?variant=general', headers=_auth(token),
                       json={'filenames': ['x.json']}).status_code == 403


def test_tampered_or_legacy_tokens_rejected(client):
    expiry, _scope, sig = _token(client, POSTPARTUM_PW).split('.')
    widened = f'{expiry}.general,postpartum.{sig}'
    assert client.get('/api/sessions?variant=general', headers=_auth(widened)).status_code == 401
    assert client.get('/api/sessions', headers=_auth(f'{expiry}.{sig}')).status_code == 401


def test_dashboard_cannot_reach_other_variant_files(client):
    sid = _run_conversation(client, 'general')
    [path] = glob.glob(os.path.join('sessions', 'general', f'*{sid}*.json'))
    fname = os.path.basename(path)
    token = _token(client, MASTER_PW)
    # Traversal is stripped to the basename, which does not exist in postpartum/.
    resp = client.get(f'/api/sessions/..%2Fgeneral%2F{fname}?variant=postpartum', headers=_auth(token))
    assert resp.status_code == 404
    resp = client.post('/api/sessions/merge?variant=postpartum', headers=_auth(token),
                       json={'filenames': [fname, fname + 'x.json']})
    assert resp.status_code in (400, 404)
    assert os.path.exists(path)
