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
from rag_pipeline import MIN_EXTRACTED_CHARS, crawl_candidates, has_usable_content
from conftest import DASHBOARD_PW, auth_header as _auth, token_for as _token

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
    # The post-partum reading list is curated separately and is the longer one;
    # general keeps its original nine sources.
    assert len(config['general'].sources) == 9
    assert len(config['postpartum'].sources) > len(config['general'].sources)
    assert all(s.title for s in config['postpartum'].sources)


class _Doc:
    def __init__(self, text): self.page_content = text


def test_block_pages_are_not_treated_as_content():
    """Bot-protected publishers answer 200 with a short "Access Denied" page."""
    assert not has_usable_content([_Doc('Access Denied Reference #18.921c1602 https://errors.edgesuite.net/')])
    assert not has_usable_content([_Doc('Please enable JavaScript to proceed.')])
    assert not has_usable_content([])
    assert has_usable_content([_Doc('a' * MIN_EXTRACTED_CHARS)])
    assert has_usable_content([_Doc('a' * 300), _Doc('b' * 300)])   # summed across pages


def test_europepmc_article_links_try_the_rest_api_first():
    """europepmc.org answers the crawler with a bot challenge (HTTP 403); the
    same articles are served unchallenged as JATS XML by the REST API."""
    rest = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6818701/fullTextXML'
    for url in ('https://europepmc.org/articles/PMC6818701?pdf=render',
                'https://europepmc.org/articles/PMC6818701',
                'https://www.europepmc.org/article/pmc/PMC6818701'):
        # The configured URL stays as a fallback: only open-access articles have
        # full text at the REST endpoint, which 404s for the rest.
        assert crawl_candidates(url) == [rest, url]


def test_other_sources_are_crawled_as_configured():
    for url in ('https://pubmed.ncbi.nlm.nih.gov/29477308/',
                'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8391101/fullTextXML',
                'https://www.cancer.org/some.pdf'):
        assert crawl_candidates(url) == [url]


def test_no_configured_source_is_a_known_blocked_publisher():
    """Sources fronted by a bot challenge are configured as their open-access
    equivalent instead (see the 'note' on each), so the nightly refresh can
    actually fetch them."""
    blocked = ('bmj.com', 'tandfonline.com', 'nejm.org', 'jamanetwork.com', 'wiley.com')
    for key, variant in rag_sources.load_rag_sources().items():
        for url in variant.urls:
            assert not any(host in url for host in blocked), f'{key}: {url}'


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

def test_sources_endpoint_lists_what_chroma_holds(fb, client):
    body = client.get('/api/rag/sources?variant=postpartum',
                      headers=_auth(_token(client, DASHBOARD_PW))).get_json()
    pipeline = fb._rag_pipelines['postpartum']
    assert body['variant'] == 'postpartum'
    assert body['chroma']['collection'] == pipeline.rag_sources.chroma.collection
    assert body['live_error'] is None and body['config_error'] is None
    assert body['fetched_at'] and body['built_at']
    # Rows mirror the collection's contents.
    assert {s['url'] for s in body['sources']} == set(pipeline.indexed)
    assert all(s['status'] == 'indexed' and s['chunks'] == 3 for s in body['sources'])
    assert {s['kind'] for s in body['sources']} <= {'pdf', 'web'}
    # Only database/collection — no credentials and no env var names.
    assert set(body['chroma']) == {'database', 'collection'}
    assert not {'api_key_env', 'tenant_env', 'CHROMA_API_KEY', 'CHROMA_TENANT'} & set(json.dumps(body).split('"'))


def test_rows_come_from_chroma_not_the_file(fb, client, tmp_path, monkeypatch):
    """A URL only in Chroma is listed; one only in the file is flagged, not counted."""
    pipeline = fb._rag_pipelines['general']
    monkeypatch.setitem(pipeline.indexed, 'https://example.org/only-in-chroma',
                        {'chunks': 5, 'kind': 'web', 'fulltext_hash': 'z'})
    edited = json.loads(json.dumps(VALID))
    edited['general']['chroma']['collection'] = pipeline.rag_sources.chroma.collection
    edited['general']['sources'] = [pipeline.rag_sources.urls[0], 'https://example.org/only-in-file']
    monkeypatch.setattr(rag_sources, 'RAG_SOURCES_FILE', _write(tmp_path, edited))

    body = client.get('/api/rag/sources?variant=general',
                      headers=_auth(_token(client, DASHBOARD_PW))).get_json()
    rows = {s['url']: s for s in body['sources']}
    assert rows['https://example.org/only-in-chroma']['status'] == 'not_in_config'
    assert rows['https://example.org/only-in-chroma']['chunks'] == 5
    assert rows[pipeline.rag_sources.urls[0]]['status'] == 'indexed'
    assert rows['https://example.org/only-in-file']['status'] == 'not_indexed'
    assert rows['https://example.org/only-in-file']['chunks'] == 0


def test_sources_endpoint_survives_a_broken_config(fb, client, tmp_path, monkeypatch):
    """Chroma is still described in full when the sources file can't be read."""
    bad = tmp_path / 'broken.json'
    bad.write_text('{ not json')
    monkeypatch.setattr(rag_sources, 'RAG_SOURCES_FILE', str(bad))
    body = client.get('/api/rag/sources?variant=general',
                      headers=_auth(_token(client, DASHBOARD_PW))).get_json()
    assert body['config_error']                                    # surfaced to the provider
    assert {s['url'] for s in body['sources']} == set(fb._rag_pipelines['general'].indexed)
    assert all(s['status'] == 'indexed' for s in body['sources'])   # config unknown → no false flags


def test_sources_endpoint_reports_chroma_failure(fb, client, monkeypatch):
    pipeline = fb._rag_pipelines['general']
    monkeypatch.setattr(pipeline, 'raise_on_read', RuntimeError('chroma unreachable'))
    body = client.get('/api/rag/sources?variant=general',
                      headers=_auth(_token(client, DASHBOARD_PW))).get_json()
    assert 'chroma unreachable' in body['live_error']
    # Configured URLs are still listed, as not indexed.
    assert body['sources'] and all(s['status'] == 'not_indexed' for s in body['sources'])


def test_sources_endpoint_respects_token_scope(fb, client):
    """The one dashboard password reaches both variants' sources; a narrower
    token (should one ever be issued) is still confined to its variant."""
    full = _auth(_token(client, DASHBOARD_PW))
    assert client.get('/api/rag/sources?variant=general', headers=full).status_code == 200
    assert client.get('/api/rag/sources?variant=postpartum', headers=full).status_code == 200

    narrow = _auth(fb._make_dashboard_token({'postpartum'}))
    assert client.get('/api/rag/sources?variant=postpartum', headers=narrow).status_code == 200
    assert client.get('/api/rag/sources?variant=general', headers=narrow).status_code == 403
    assert client.get('/api/rag/sources?variant=general').status_code == 401
