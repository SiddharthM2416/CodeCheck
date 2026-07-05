"""
Standalone indexing script. Usage:

    python index_repo.py <path_to_repo>

Chunks the repo (Python + Java for now), applies the test-linkage heuristic,
embeds each chunk with sentence-transformers, and stores everything in a
local Chroma collection. The collection name is derived from the repo path
(see PROGRESS.md / Section 7 pattern), so multiple codebases can be indexed
into the same Chroma instance and the MCP server can switch which collection
it queries without needing to wipe/rebuild anything.

Re-running this script on the same repo path will overwrite that repo's
collection from scratch (simplest correct behavior -- avoids stale chunks
lingering after code changes).
"""

import os
import sys
import hashlib
import chromadb
from sentence_transformers import SentenceTransformer

from walker import chunk_repo, find_source_and_test_files
from test_linkage import apply_test_linkage
from chunkers.base import make_chunk_id

CHROMA_PATH = os.environ.get("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def collection_name_for(repo_path: str) -> str:
    abs_path = os.path.abspath(repo_path)
    digest = hashlib.md5(abs_path.encode()).hexdigest()[:8]
    repo_name = os.path.basename(abs_path.rstrip("/\\"))
    # keep it human-readable AND collision-safe
    safe_name = "".join(c if c.isalnum() else "_" for c in repo_name)
    return f"code_{safe_name}_{digest}"


def build_embedding_text(chunk) -> str:
    """What actually gets embedded. Including the qualified name and kind
    helps semantic search match queries like 'authentication logic' even
    when the code itself doesn't use that exact word."""
    return f"{chunk.kind} {chunk.qualified_name} ({chunk.language})\n{chunk.code}"


def index_repo(repo_path: str) -> None:
    repo_path = os.path.abspath(repo_path)
    print(f"Indexing: {repo_path}")

    print("  chunking...")
    source_files, test_files = find_source_and_test_files(repo_path)
    chunks = chunk_repo(repo_path)
    print(f"  {len(chunks)} chunks extracted from {len(source_files)} source files")

    print("  applying test-linkage heuristic...")
    chunks = apply_test_linkage(chunks, repo_path, test_files)
    tested = sum(1 for c in chunks if c.has_test)
    print(f"  {tested} tested / {len(chunks) - tested} untested")

    print(f"  loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("  embedding chunks...")
    texts = [build_embedding_text(c) for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

    print(f"  connecting to Chroma at {CHROMA_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    coll_name = collection_name_for(repo_path)

    # fresh rebuild -- delete if it already exists
    try:
        client.delete_collection(coll_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=coll_name,
        metadata={"repo_path": repo_path},
    )

    print(f"  storing {len(chunks)} chunks in collection '{coll_name}'...")
    ids = [make_chunk_id(c) for c in chunks]
    # dedupe ids (e.g. @typing.overload stubs can collide) by appending index
    seen = {}
    for i, cid in enumerate(ids):
        if cid in seen:
            seen[cid] += 1
            ids[i] = f"{cid}#{seen[cid]}"
        else:
            seen[cid] = 0

    metadatas = []
    for c in chunks:
        metadatas.append({
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "name": c.name,
            "qualified_name": c.qualified_name,
            "kind": c.kind,
            "language": c.language,
            "has_test": bool(c.has_test) if c.has_test is not None else False,
            "code": c.code,
        })

    # Chroma has a max batch size on some backends -- chunk the insert
    BATCH = 500
    for i in range(0, len(chunks), BATCH):
        collection.add(
            ids=ids[i:i + BATCH],
            embeddings=[e.tolist() for e in embeddings[i:i + BATCH]],
            metadatas=metadatas[i:i + BATCH],
            documents=texts[i:i + BATCH],
        )

    print(f"Done. Collection '{coll_name}' has {collection.count()} chunks.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python index_repo.py <path_to_repo>")
        sys.exit(1)
    index_repo(sys.argv[1])