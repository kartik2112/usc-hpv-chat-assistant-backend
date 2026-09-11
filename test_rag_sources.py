"""
test_rag_sources.py
===================
Per-variant RAG sources: validation of rag_sources.json, one Chroma collection
per variant, retrieval from the right collection, and the /api/rag/sources
viewer endpoint (including its variant scoping).

Runs offline (see conftest.py):

    uv run pytest test_rag_sources.py -v
"""

import json

import pytest

import rag_sources
from conftest import MASTER_PW, POSTPARTUM_PW, auth_header as _auth, token_for as _token

VALID = {
    "_comment": "ignored",
    "general": {
        "chroma": {"database": "Demo", "collection": "general_col"},
        "sources": [{"url": "https://example.org/a", "title": "A"},
                    "https://example.org/a.pdf"],
    },
    "postpartum": {
        "chroma": {"database": "Demo", "collection": "pp_col"},
        "sources": [{"url": "https://example.org/b"}],
    },
}


def _write(tmp_path, config, name='rag_sources.json'):
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return str(path)


# ── Config file ───────────────────────────────────────────────────────────────

def test_loads_targets_and_sources(tmp_path):
    config = rag_sources.load_rag_sources(_write(tmp_path, VALID))
    assert set(config) == {'general', 'postpartum'}
    assert config['general'].chroma.collection == 'general_col'
    assert config['postpartum'].chroma.collection == 'pp_col'
    # Bare strings and {url, title} objects both work; defaults fill the rest.
    assert config['general'].urls == ['https://example.org/a', 'https://example.org/a.pdf']
    assert config['general'].sources[0].title == 'A'
    assert config['general'].chroma.api_key_env == 'CHROMA_API_KEY'


def test_shipped_file_is_valid_and_separates_collections():
    config = rag_sources.load_rag_sources()      # the repo's rag_sources.json
    collections = {key: cfg.chroma.collection for key, cfg in config.items()}
    assert len(set(collections.values())) == len(collections)
    assert collections['general'] == 'hpv_facts_rag'   # unchanged: no re-embedding
    assert all(cfg.sources for cfg in config.values())


@pytest.mark.parametrize('mutate, message', [
    (lambda c: c.pop('postpartum'), 'missing variants'),
    (lambda c: c.update(other={'chroma': {'database': 'd', 'collection': 'c'}, 'sources': []}), 'unknown variants'),
    # Sharing a collection would let one variant delete the other's chunks.
    (lambda c: c['postpartum']['chroma'].update(collection='general_col'), 'same Chroma collection'),
    (lambda c: c['general']['sources'].append('ftp://example.org/x'), 'http(s) URL'),
    (lambda c: c['general']['sources'].append('https://example.org/a'), 'duplicate URL'),
    (lambda c: c['general']['chroma'].pop('collection'), 'required'),
])
def test_rejects_bad_config(tmp_path, mutate, message):
    config = json.loads(json.dumps(VALID))
    mutate(config)
    with pytest.raises(ValueError) as excinfo:
        rag_sources.load_rag_sources(_write(tmp_path, config))
    assert message in str(excinfo.value)


# ── One pipeline per variant ──────────────────────────────────────────────────

def test_each_variant_has_its_own_collection(fb):
    assert (fb._rag_pipelines['general'].rag_sources.chroma.collection
            != fb._rag_pipelines['postpartum'].rag_sources.chroma.collection)


def test_chat_retrieves_from_the_sessions_variant(fb, client, monkeypatch):
    """A post-partum session's chat must hit the post-partum collection even if
    the request body claims another variant."""
    used = {}
    monkeypatch.setattr(fb, 'retrieve_context_docs',
                        lambda pipeline, messages: used.setdefault('collection', pipeline.rag_sources.chroma.collection) and [])
    monkeypatch.setattr(fb, 'ask_rag_question_stream',
                        lambda *a, **kw: iter([kw['audience_instructions'][:20]]))
    sid = client.post('/api/session/start', json={'variant': 'postpartum'}).get_json()['session_id']
    resp = client.post('/api/chat', json={'session_id': sid, 'variant': 'general',
                                          'messages': [{'role': 'user', 'content': 'hi'}]})
    assert resp.status_code == 200
    resp.get_data()   # drain the stream so the generator runs
    assert used['collection'] == fb._rag_pipelines['postpartum'].rag_sources.chroma.collection


# ── Viewer endpoint ───────────────────────────────────────────────────────────

def test_sources_endpoint_lists_indexed_sources(fb, client):
    body = client.get('/api/rag/sources?variant=postpartum',
                      headers=_auth(_token(client, MASTER_PW))).get_json()
    assert body['variant'] == 'postpartum'
    assert body['chroma']['collection'] == fb._rag_pipelines['postpartum'].rag_sources.chroma.collection
    assert body['config_error'] is None and body['chroma_pending'] is None
    assert body['sources'] and all(s['status'] == 'unchanged' and s['chunks'] == 3 for s in body['sources'])
    assert {s['kind'] for s in body['sources']} <= {'pdf', 'web'}
    # Only database/collection — no credentials and no env var names.
    assert set(body['chroma']) == {'database', 'collection'}
    assert not {'api_key_env', 'tenant_env', 'CHROMA_API_KEY', 'CHROMA_TENANT'} & set(json.dumps(body).split('"'))


def test_sources_endpoint_shows_pending_edits(fb, client, tmp_path, monkeypatch):
    """Edits to the file show up before the nightly refresh applies them."""
    edited = json.loads(json.dumps(VALID))
    live = fb._rag_pipelines['general'].rag_sources
    edited['general']['chroma']['collection'] = live.chroma.collection
    edited['general']['sources'] = [live.urls[0], 'https://example.org/new']
    monkeypatch.setattr(rag_sources, 'RAG_SOURCES_FILE', _write(tmp_path, edited))

    body = client.get('/api/rag/sources?variant=general',
                      headers=_auth(_token(client, MASTER_PW))).get_json()
    rows = {s['url']: s['status'] for s in body['sources']}
    assert rows['https://example.org/new'] == 'pending'
    assert rows[live.urls[0]] == 'unchanged'
    assert rows[live.urls[1]] == 'pending_removal'


def test_sources_endpoint_reports_a_broken_config(fb, client, tmp_path, monkeypatch):
    bad = tmp_path / 'broken.json'
    bad.write_text('{ not json')
    monkeypatch.setattr(rag_sources, 'RAG_SOURCES_FILE', str(bad))
    body = client.get('/api/rag/sources?variant=general',
                      headers=_auth(_token(client, MASTER_PW))).get_json()
    assert body['config_error']                      # surfaced to the provider
    assert body['sources']                           # live index still described


def test_sources_endpoint_respects_token_scope(client):
    pp_token = _token(client, POSTPARTUM_PW)
    assert client.get('/api/rag/sources?variant=postpartum', headers=_auth(pp_token)).status_code == 200
    assert client.get('/api/rag/sources?variant=general', headers=_auth(pp_token)).status_code == 403
    assert client.get('/api/rag/sources?variant=general').status_code == 401
