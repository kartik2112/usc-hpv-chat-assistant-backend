# Loader for rag_sources.json — which Chroma collection and which web pages /
# PDFs back each variant's RAG index.
#
# The file lives on the server (path: $RAG_SOURCES_FILE, default rag_sources.json
# next to this module) so sources can be curated without a code change. It is
# read at startup and again by the nightly RAG refresh. Shape:
#
#   {
#     "<variant key>": {
#       "chroma":  {"database": "...", "collection": "...",
#                   "api_key_env": "CHROMA_API_KEY",  # optional: env var names,
#                   "tenant_env":  "CHROMA_TENANT"},  # to use another Chroma account
#       "sources": [{"url": "https://...", "title": "optional"}, ...]
#     },
#     "_anything": "keys starting with _ are ignored (comments)"
#   }
#
# Every variant in variants.py must have an entry, and no two variants may share
# a collection: each pipeline deletes chunks of URLs that are not in its own
# list, so a shared collection would wipe the other variant's sources.

import json
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from variants import VARIANTS

RAG_SOURCES_FILE = os.getenv('RAG_SOURCES_FILE',
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rag_sources.json'))


@dataclass(frozen=True)
class ChromaTarget:
    database: str
    collection: str
    api_key_env: str = 'CHROMA_API_KEY'   # names of env vars, never the secrets themselves
    tenant_env: str = 'CHROMA_TENANT'


@dataclass(frozen=True)
class Source:
    url: str
    title: str = ''


@dataclass(frozen=True)
class RagSources:
    chroma: ChromaTarget
    sources: tuple  # of Source

    @property
    def urls(self):
        return [s.url for s in self.sources]


def _parse_variant(key, raw):
    if not isinstance(raw, dict) or not isinstance(raw.get('chroma'), dict):
        raise ValueError(f"'{key}': expected an object with 'chroma' and 'sources'")
    chroma = raw['chroma']
    if not chroma.get('database') or not chroma.get('collection'):
        raise ValueError(f"'{key}.chroma': 'database' and 'collection' are required")
    target = ChromaTarget(**{f: str(chroma[f]) for f in ChromaTarget.__dataclass_fields__ if f in chroma})

    sources, seen = [], set()
    for item in raw.get('sources') or []:
        item = {'url': item} if isinstance(item, str) else item
        url = str(item.get('url', '')).strip() if isinstance(item, dict) else ''
        if urlparse(url).scheme not in ('http', 'https'):
            raise ValueError(f"'{key}.sources': not an http(s) URL: {item!r}")
        if url in seen:
            raise ValueError(f"'{key}.sources': duplicate URL {url}")
        seen.add(url)
        sources.append(Source(url=url, title=str(item.get('title', '')).strip()))
    return RagSources(chroma=target, sources=tuple(sources))


def load_rag_sources(path=None):
    """Read and validate the sources file. Returns {variant key: RagSources}.

    Raises ValueError (or OSError) with a readable message on any problem, so a
    bad edit is caught before it touches a vector store.
    """
    with open(path or RAG_SOURCES_FILE, encoding='utf-8') as f:
        raw = json.load(f)
    entries = {k: v for k, v in raw.items() if not k.startswith('_')}

    missing, unknown = VARIANTS.keys() - entries.keys(), entries.keys() - VARIANTS.keys()
    if missing or unknown:
        raise ValueError(f"rag_sources: missing variants {sorted(missing)}, unknown variants {sorted(unknown)}")

    config = {key: _parse_variant(key, value) for key, value in entries.items()}
    # Compare resolved tenants: two env var names may hold the same account.
    targets = [(os.getenv(c.chroma.tenant_env, c.chroma.tenant_env), c.chroma.database, c.chroma.collection)
               for c in config.values()]
    if len(set(targets)) != len(targets):
        raise ValueError("rag_sources: two variants point at the same Chroma collection")
    return config
