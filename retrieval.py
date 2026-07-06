"""
Phase 2 retrieval functions -- these get wrapped as MCP tools in Phase 3.

Both functions operate on a single Chroma collection at a time (i.e. one
indexed repo). Which collection is used is deliberately explicit here
(collection_name param) rather than hardcoded, since the MCP server will
need to pick a repo/collection per session (or expose a switch_codebase
tool later -- see PROGRESS.md optional stretch goal).
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer

from index_repo import CHROMA_PATH, EMBEDDING_MODEL, collection_name_for

_model = None  # lazy-loaded, shared across calls in a process

# Groq's free tier caps at 12,000 tokens PER MINUTE for llama-3.3-70b-versatile.
# Measured real payloads at limit=15/cap=1200 came in around ~2,100 tokens --
# well under budget -- so this cap can be raised substantially without risking
# another 413. Raised from 1200 -> 2500 after diagnosing a real bug: a
# 2,218-char cert_verify() function was truncated mid-body, hiding the exact
# lines (conn.cert_reqs access after the OSError branch) that would have told
# the model conn=None breaks things. The model didn't reason incorrectly --
# it was never shown the information needed to reason correctly.
MAX_CODE_CHARS = 2500


def _truncate_code(code: str) -> str:
    if len(code) <= MAX_CODE_CHARS:
        return code
    kept_lines = code[:MAX_CODE_CHARS].rstrip()
    total_lines = code.count("\n") + 1
    return (
        f"{kept_lines}\n"
        f"... [TRUNCATED -- {total_lines} lines total, only part of this "
        f"function is shown above. Do NOT write assertions about behavior "
        f"in the missing part -- call read_file first to see the rest.]"
    )


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection(collection_name: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(collection_name)


def list_indexed_repos() -> list[dict]:
    """List every collection currently in the Chroma DB, with repo_path
    metadata and chunk count -- useful for a 'which repos are indexed?'
    check before calling search_code/find_untested."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    results = []
    for coll in client.list_collections():
        c = client.get_collection(coll.name)
        results.append({
            "collection_name": coll.name,
            "repo_path": coll.metadata.get("repo_path") if coll.metadata else None,
            # older collections indexed before this field existed won't
            # have it -- None is handled by the UI as "unknown, can't
            # check staleness" rather than crashing
            "indexed_at": coll.metadata.get("indexed_at") if coll.metadata else None,
            "chunk_count": c.count(),
        })
    return results


def search_code(
    query: str,
    collection_name: str,
    k: int = 5,
    path_prefix: str | None = None,
) -> list[dict]:
    """Semantic search: embed the query, find the k most similar chunks.

    path_prefix (optional) scopes results to files whose path contains
    this substring, e.g. path_prefix='frontend' to search only frontend
    code. This was a REAL gap found via testing: without it, an agent
    trying to search a specific subdirectory (e.g. "find the frontend
    project component") had no way to scope the search at all, and would
    burn many tool calls trying different keyword phrasings instead,
    without ever actually restricting WHICH files got searched --
    consistently surfacing irrelevant top-k matches from unrelated code.

    Implementation note: when path_prefix is set, we fetch a much larger
    candidate pool from Chroma BEFORE filtering by path, then trim to k
    AFTER filtering -- same ordering fix as find_untested's path_prefix
    bug. Filtering an already-small top-k pool by path would frequently
    return zero matches even when the target directory has good matches,
    for the same reason as before."""
    collection = _get_collection(collection_name)
    model = _get_model()

    query_embedding = model.encode([query])[0].tolist()
    query_k = 200 if path_prefix else k
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=query_k,
    )

    out = []
    for i in range(len(results["ids"][0])):
        if len(out) >= k:
            break
        meta = results["metadatas"][0][i]
        if path_prefix:
            normalized_path = meta["file_path"].replace("\\", "/")
            normalized_prefix = path_prefix.replace("\\", "/")
            if normalized_prefix not in normalized_path:
                continue
        out.append({
            "qualified_name": meta["qualified_name"],
            "file_path": meta["file_path"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "kind": meta["kind"],
            "language": meta["language"],
            "has_test": meta["has_test"],
            "code": _truncate_code(meta["code"]),
            "distance": results["distances"][0][i],  # lower = more similar
        })
    return out


def get_repo_path_for_collection(collection_name: str) -> str:
    """Resolve a collection name back to its filesystem repo path (stored as
    collection metadata at index time) -- needed by read_file, which reads
    from disk rather than from Chroma."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(collection_name)
    repo_path = collection.metadata.get("repo_path") if collection.metadata else None
    if not repo_path:
        raise ValueError(f"No repo_path metadata found for collection '{collection_name}'")
    return repo_path


def read_file(
    repo_path: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Raw file read, for full surrounding context beyond a single chunk.
    Line numbers are 1-indexed and inclusive, matching Chunk.start_line/end_line."""
    full_path = os.path.join(repo_path, file_path)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"No such file: {file_path} (under {repo_path})")

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if start_line is None and end_line is None:
        return "".join(lines)

    start = (start_line or 1) - 1  # convert to 0-indexed
    end = end_line if end_line is not None else len(lines)
    start = max(0, start)
    end = min(len(lines), end)

    return "".join(lines[start:end])


def find_untested(
    collection_name: str,
    path_prefix: str | None = None,
    language: str | None = None,
    limit: int = 15,
) -> list[dict]:
    """Metadata-only filter -- no embedding needed. Returns chunks where
    has_test == False, optionally scoped to a subdirectory and/or language.

    path_prefix matches as a SUBSTRING anywhere in the file path, not a
    strict prefix -- e.g. path_prefix='requests/adapters.py' correctly
    matches a real path of 'src/requests/adapters.py'.

    IMPORTANT ordering fix: path_prefix filtering happens in Python AFTER
    fetching from Chroma, so `limit` must NOT be applied at the Chroma
    query stage when path_prefix is set -- otherwise we'd fetch an
    arbitrary `limit`-sized sample from the WHOLE repo first, and only
    then filter by path, which can return zero results even when the
    target file genuinely has untested chunks (they just didn't happen to
    be in that arbitrary first sample). Fixed by fetching a large
    candidate pool whenever path_prefix is set, filtering by path first,
    THEN applying `limit` to the filtered results.

    Default limit lowered to 15 (was 100) -- returning many full-code
    chunks in one tool result is what caused real Groq free-tier TPM
    errors in testing (16K+ tokens in a single request against a 12K/min
    budget). Ask again with a path_prefix to drill into a specific
    subdirectory rather than raising this limit."""
    collection = _get_collection(collection_name)

    where_clauses = [{"has_test": False}]
    if language:
        where_clauses.append({"language": language})

    where = where_clauses[0] if len(where_clauses) == 1 else {"$and": where_clauses}

    # if scoping by path, fetch a large candidate pool (not the final
    # user-facing limit) so path filtering happens over the FULL matching
    # set, not an arbitrary pre-limited sample
    query_limit = 5000 if path_prefix else limit
    results = collection.get(where=where, limit=query_limit)

    out = []
    for i in range(len(results["ids"])):
        if len(out) >= limit:
            break
        meta = results["metadatas"][i]
        if path_prefix:
            normalized_path = meta["file_path"].replace("\\", "/")
            normalized_prefix = path_prefix.replace("\\", "/")
            if normalized_prefix not in normalized_path:
                continue
        out.append({
            "qualified_name": meta["qualified_name"],
            "file_path": meta["file_path"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "kind": meta["kind"],
            "language": meta["language"],
            "code": _truncate_code(meta["code"]),
        })
    return out


if __name__ == "__main__":
    import sys

    print("=== Indexed repos ===")
    for repo in list_indexed_repos():
        print(f"  {repo['collection_name']:35s} {repo['chunk_count']:4d} chunks  {repo['repo_path']}")
    print()

    if len(sys.argv) < 2:
        print("Usage: python retrieval.py <collection_name> [search query]")
        sys.exit(0)

    coll_name = sys.argv[1]

    print(f"=== find_untested (first 10) on '{coll_name}' ===")
    for r in find_untested(coll_name, limit=10):
        print(f"  {r['kind']:8s} {r['qualified_name']:40s} {r['file_path']}:{r['start_line']}")
    print()

    if len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"=== search_code('{query}') on '{coll_name}' ===")
        for r in search_code(query, coll_name, k=5):
            print(f"  [{r['distance']:.3f}] {r['qualified_name']:40s} {r['file_path']}:{r['start_line']} (tested={r['has_test']})")