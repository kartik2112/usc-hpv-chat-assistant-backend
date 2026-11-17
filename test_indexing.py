"""What a RAG build embeds. Offline: uv run pytest test_indexing.py -v"""

from collections import defaultdict

import pytest

import rag_pipeline as rp
from rag_sources import ChromaTarget, RagSources, Source
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CLOUD_READ_LIMIT = 300


class FakeCloudStore:
    """Chroma stand-in with Chroma Cloud's 300-record read cap."""

    def __init__(self):
        self.records = {}            # id -> metadata
        self.embedded = 0            # chunks embedded

    def get(self, include=None, limit=None, offset=None, where=None):
        rows = [(i, m) for i, m in self.records.items()
                if not where or all(m.get(k) == v for k, v in where.items())]
        rows = rows[offset or 0:]
        rows = rows[:min(limit or CLOUD_READ_LIMIT, CLOUD_READ_LIMIT)]   # silently capped
        return {"ids": [i for i, _ in rows], "metadatas": [dict(m) for _, m in rows]}

    def add_documents(self, documents, ids):
        assert len(documents) <= 300 and len(documents) == len(ids)
        self.embedded += len(documents)
        for doc, i in zip(documents, ids):      # upsert
            self.records[i] = dict(doc.metadata)
        return list(ids)

    def delete(self, ids):
        assert len(ids) <= 300
        for i in ids:
            self.records.pop(i, None)


def make_pipeline(store, pages):
    sources = RagSources(chroma=ChromaTarget(database="d", collection="c"),
                         sources=tuple(Source(url=u) for u in pages))
    p = rp.HPVRAGPipeline.__new__(rp.HPVRAGPipeline)
    p.rag_sources, p.vector_store = sources, store
    p.urls = sources.urls
    p.content_selectors = {u: "" for u in pages}
    p.index_report, p.removed_urls = {}, []
    p.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    p._crawl_url = lambda url: [Document(page_content=pages[url], metadata={"source": url})]
    return p


def build(store, pages):
    p = make_pipeline(store, pages)
    p._setup_rag_agent()
    return p


def chunks_per_source(store):
    counts = defaultdict(int)
    for m in store.records.values():
        counts[m["source"]] += 1
    return dict(counts)


# Well past the 300-record read cap.
PAGES = {f"https://example.org/{n}": " ".join(f"word{n}-{i}" for i in range(12000)) for n in range(4)}


def test_unchanged_rebuild_embeds_nothing_even_past_the_read_limit():
    store = FakeCloudStore()
    build(store, PAGES)
    first = store.embedded
    assert len(store.records) > CLOUD_READ_LIMIT
    p = build(store, PAGES)
    assert store.embedded == first
    assert all(r["status"] == "unchanged" for r in p.index_report.values())
    assert len(store.records) == first


def test_only_the_edited_source_is_re_embedded_and_old_chunks_go():
    store = FakeCloudStore()
    build(store, PAGES)
    before = chunks_per_source(store)
    edited = dict(PAGES, **{"https://example.org/2": "fresh text " * 3000})
    embedded_before = store.embedded
    p = build(store, edited)
    after = chunks_per_source(store)
    assert p.index_report["https://example.org/2"]["status"] == "updated"
    assert store.embedded - embedded_before == after["https://example.org/2"]
    assert {u: n for u, n in after.items() if u != "https://example.org/2"} == \
           {u: n for u, n in before.items() if u != "https://example.org/2"}


def test_duplicates_and_other_model_chunks_are_replaced_once():
    store = FakeCloudStore()
    url = "https://example.org/0"
    for copy in range(3):
        for i in range(200):
            store.records[f"old-{copy}-{i}"] = {"source": url, "fulltext_hash": "h"}
    pages = {url: PAGES[url]}
    build(store, pages)
    assert not any(i.startswith("old-") for i in store.records)
    clean = dict(store.records)
    embedded = store.embedded
    build(store, pages)
    assert store.records == clean and store.embedded == embedded


def test_sources_dropped_from_the_list_are_deleted_past_the_read_limit():
    store = FakeCloudStore()
    build(store, PAGES)
    remaining = {u: t for u, t in PAGES.items() if not u.endswith("/3")}
    p = build(store, remaining)
    assert set(chunks_per_source(store)) == set(remaining)
    assert p.removed_urls == ["https://example.org/3"]


def test_describe_counts_every_chunk_not_just_the_first_page():
    store = FakeCloudStore()
    p = build(store, PAGES)
    assert {u: r["chunks"] for u, r in p.describe_indexed_sources().items()} == chunks_per_source(store)


def test_chunks_record_the_embedding_model():
    store = FakeCloudStore()
    build(store, {"https://example.org/0": "text " * 400})
    assert {m["embedding_model"] for m in store.records.values()} == {rp.EMBEDDING_MODEL}
    assert rp.EMBEDDING_MODEL == "text-embedding-3-small"


def test_extra_copies_are_deleted_without_re_embedding():
    store = FakeCloudStore()
    pages = {"https://example.org/0": "text " * 2000}
    build(store, pages)
    clean, embedded = dict(store.records), store.embedded
    store.records["dup"] = {"source": "https://example.org/0", "fulltext_hash": "old"}
    build(store, pages)
    assert store.records == clean and store.embedded == embedded


def test_partially_written_source_is_completed():
    store = FakeCloudStore()
    pages = {"https://example.org/0": "text " * 2000}
    build(store, pages)
    store.records.pop(next(iter(store.records)))
    p = build(store, pages)
    assert p.index_report["https://example.org/0"]["status"] == "updated"
    assert len(store.records) == p.index_report["https://example.org/0"]["chunks"]


# ── Text extraction ───────────────────────────────────────────────────────────

def test_pdf_bytes_read_as_a_web_page_are_rejected():
    garbage = "%PDF-1.7\n" + "".join(chr(c % 32) for c in range(20000))
    assert rp.looks_like_binary(garbage)
    assert not rp.looks_like_binary("An ordinary page.\n" * 500)
    assert not rp.looks_like_binary("Vacunación contra el VPH — información para pacientes. " * 300)


def test_content_selector_keeps_only_the_article():
    html = """<html><body><nav><ul><li>Home</li><li>About</li></ul></nav>
      <main><h1>HPV vaccine</h1><p>Two doses are recommended.</p>
        <ul><li><p>Nested item</p></li></ul></main>
      <footer><p>Page last updated: May 3, 2026</p></footer></body></html>"""
    assert rp.extract_page_text(html, "main") == "HPV vaccine\nTwo doses are recommended.\nNested item"
    whole = rp.extract_page_text(html)
    assert "Home" in whole and "last updated" not in whole
    with pytest.raises(ValueError):
        rp.extract_page_text(html, "article.missing")


@pytest.mark.parametrize("line", [
    "Last reviewed: March 2, 2026", "Page last updated November 1, 2026",
    "Last updated on 2026-11-01", "Updated on: Jan 5, 2026", "Content reviewed: 01/02/2026",
    "Date accessed 11/17/2026",
])
def test_volatile_lines_do_not_change_the_hash(line):
    assert rp.strip_volatile_lines(f"HPV is common.\n{line}\nMost infections clear.") == \
           "HPV is common.\nMost infections clear."


def test_ordinary_sentences_are_kept():
    text = "Updated guidelines recommend self-collection.\nThe last reviewed study was small."
    assert rp.strip_volatile_lines(text) == text


def test_pdf_links_try_the_pdf_loader_first():
    p = make_pipeline(FakeCloudStore(), {})
    del p._crawl_url
    calls = []
    p._load_as_pdf = lambda u: calls.append("pdf") or [Document(page_content="x", metadata={"page": 0})]
    p._load_as_webpage = lambda u, s="": calls.append("web") or None
    p.content_selectors = {}
    p._crawl_url("https://www.cancer.org/booklet.pdf")
    assert calls == ["pdf"]
