"""
Walks a repo, separates source files from test files, and dispatches each
source file to the right language chunker.

Scope note: JS/TS support is intentionally deferred. Adding it later means
writing chunkers/js_chunker.py and adding '.js'/'.jsx' to LANGUAGE_BY_EXT --
nothing else in this file, or anything downstream, needs to change.
"""

import os
from chunkers.base import Chunk
from chunkers.python_chunker import extract_python_chunks
from chunkers.java_chunker import extract_java_chunks
from chunkers.js_chunker import extract_javascript_chunks

SKIP_DIRS = {
    ".git", "venv", ".venv", "__pycache__", "node_modules",
    "target", "build", "dist", ".pytest_cache", "egg-info",
    "coverage", ".nyc_output",  # generated test-coverage reports (e.g.
    # Istanbul/nyc's lcov-report includes vendored, minified third-party
    # JS like prettify.js -- not the user's own code, and indexing it
    # produces meaningless "untested function" noise (single-letter
    # function names from minification).
}

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".java": "java",
    ".js": "javascript",
    # ".jsx": "javascript",  # still deferred -- React/JSX component chunking
    # needs different unit rules (whole-component, not sub-function) per
    # PROGRESS.md's earlier scope discussion. Plain .js works today.
}

EXTRACTORS = {
    "python": extract_python_chunks,
    "java": extract_java_chunks,
    "javascript": extract_javascript_chunks,
}


def _is_test_file(rel_path: str, language: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    filename = parts[-1]

    if language == "python":
        return "tests" in parts or "test" in parts or filename.startswith("test_") or filename.endswith("_test.py")
    if language == "java":
        # standard Maven/Gradle layout: src/test/java/...
        return "test" in parts and filename.endswith(".java") or filename.endswith("Test.java") or filename.endswith("Tests.java")
    if language == "javascript":
        # Mocha/Jest convention: *.test.js, or anything under a top-level
        # test/ or __tests__/ directory (e.g. validator.js's
        # test/testFunctions.js, which doesn't match *.test.js but is
        # clearly test-support code, not library source)
        return "test" in parts or "__tests__" in parts or filename.endswith(".test.js") or filename.endswith(".spec.js")
    return False


def find_source_and_test_files(repo_root: str) -> tuple[list[str], list[str]]:
    """Returns (source_files, test_files), both as paths relative to repo_root."""
    source_files, test_files = [], []

    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for fname in files:
            ext = os.path.splitext(fname)[1]
            language = LANGUAGE_BY_EXT.get(ext)
            if language is None:
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, repo_root)

            if _is_test_file(rel_path, language):
                test_files.append(rel_path)
            else:
                source_files.append(rel_path)

    return source_files, test_files


def chunk_repo(repo_root: str) -> list[Chunk]:
    """Chunk every source file in the repo (test files are NOT chunked here --
    they're consumed separately by the test-linkage step)."""
    source_files, _ = find_source_and_test_files(repo_root)
    all_chunks: list[Chunk] = []

    for rel_path in source_files:
        ext = os.path.splitext(rel_path)[1]
        language = LANGUAGE_BY_EXT[ext]
        extractor = EXTRACTORS[language]
        try:
            chunks = extractor(rel_path, repo_root)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [WARN] failed to parse {rel_path}: {e}")

    return all_chunks


if __name__ == "__main__":
    import sys
    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."

    source_files, test_files = find_source_and_test_files(repo_root)
    print(f"Repo: {repo_root}")
    print(f"Source files: {len(source_files)}")
    print(f"Test files:   {len(test_files)}")
    print()

    chunks = chunk_repo(repo_root)
    print(f"Total chunks extracted: {len(chunks)}")

    by_kind: dict[str, int] = {}
    for c in chunks:
        by_kind[c.kind] = by_kind.get(c.kind, 0) + 1
    print(f"By kind: {by_kind}")