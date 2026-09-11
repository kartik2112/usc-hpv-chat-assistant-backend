"""
Shared pytest fixtures: import flask_backend offline.

The RAG build (crawling + Chroma) and PHI detection (RENDER=1 → no spaCy) are
stubbed, so the whole app can be exercised with no network access and no keys.
Used by test_variants.py and test_rag_sources.py.

Note: test_phi_guardrails.py imports flask_backend for real (PHI on, real RAG
build) — run that file on its own.
"""

import importlib
import json
import os
import sys

import bcrypt
import pytest

DASHBOARD_PW = 'dashboard-pw'
LEGACY_FILE = 'session_11111111-legacy_20250101_000000.json'


def _hash(pw):
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=4)).decode()


class FakePipeline:
    """Stand-in for HPVRAGPipeline with a fake Chroma collection.

    `indexed` is what describe_indexed_sources() reports — tests mutate it to
    simulate Chroma holding more or less than the sources file lists.
    """

    def __init__(self, rag_sources, **kwargs):
        self.rag_sources = rag_sources
        self.built_at = '2026-09-11T00:00:00+00:00'
        self.removed_urls = []
        self.index_report = {
            s.url: {'status': 'unchanged', 'chunks': 3,
                    'kind': 'pdf' if s.url.endswith('.pdf') else 'web'}
            for s in rag_sources.sources
        }
        self.indexed = {
            s.url: {'chunks': 3, 'kind': 'pdf' if s.url.endswith('.pdf') else 'web',
                    'fulltext_hash': 'abc'}
            for s in rag_sources.sources
        }
        self.raise_on_read = None       # set to an Exception to simulate Chroma being down

    def describe_indexed_sources(self):
        if self.raise_on_read:
            raise self.raise_on_read
        return dict(self.indexed)


@pytest.fixture(scope='session')
def fb(tmp_path_factory):
    """flask_backend imported inside a temp working dir with offline stubs."""
    workdir = tmp_path_factory.mktemp('backend')
    os.makedirs(workdir / 'sessions')
    (workdir / 'sessions' / LEGACY_FILE).write_text(json.dumps({'session_id': 'legacy'}))

    old_cwd, old_env = os.getcwd(), dict(os.environ)
    os.chdir(workdir)
    os.environ.update({
        'RENDER': '1',                     # PHI_ENABLED=False → no spaCy
        'OPENAI_API_KEY': 'test-key',
        'SESSIONS_TOKEN_SECRET': 'test-secret',
        'SESSIONS_PASSWORD_HASH': _hash(DASHBOARD_PW),
    })
    import rag_pipeline
    real_build = rag_pipeline.build_rag_pipeline
    rag_pipeline.build_rag_pipeline = FakePipeline   # no crawling / Chroma
    sys.modules.pop('flask_backend', None)
    module = importlib.import_module('flask_backend')
    module.generate_session_summary = lambda messages, variant=None: {
        'patient_questions': f'• asked ({variant.key})', 'action_items': ''}
    yield module
    module.scheduler.shutdown(wait=False)
    rag_pipeline.build_rag_pipeline = real_build
    sys.modules.pop('flask_backend', None)
    os.chdir(old_cwd)
    os.environ.clear()
    os.environ.update(old_env)


@pytest.fixture
def client(fb):
    return fb.app.test_client()


def token_for(client, password):
    resp = client.post('/api/sessions/auth', json={'password': password})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()['token']


def auth_header(token):
    return {'Authorization': f'Bearer {token}'}
