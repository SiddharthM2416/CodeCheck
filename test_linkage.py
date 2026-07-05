"""
Sets `has_test` on each Chunk by checking whether its name is referenced
anywhere inside the repo's test files.

Heuristic (documented limitation -- see PROGRESS.md / README): this is
name-based matching, not true call-graph analysis. It will have false
positives (name coincidentally appears in a comment or unrelated call) and
false negatives (function is tested only indirectly, e.g. called internally
by another function that IS directly tested). Keeping it simple and
name-based is a deliberate weekend-project scope decision.

Matching rule (same for both languages right now):
  A chunk is "tested" if its `name` appears followed by `(` anywhere in the
  concatenated text of all test files -- i.e. it looks like the test file
  calls it directly. Python additionally checks for `test_<name>` as a
  test function name, which is a stronger, more deliberate signal.
"""

import os
import re
from chunkers.base import Chunk


def _read_test_corpus(repo_root: str, test_files: list[str]) -> str:
    """Concatenate all test file contents into one searchable string."""
    parts = []
    for rel_path in test_files:
        full_path = os.path.join(repo_root, rel_path)
        try:
            with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                parts.append(f.read())
        except OSError as e:
            print(f"  [WARN] could not read test file {rel_path}: {e}")
    return "\n".join(parts)


def _is_tested_python(name: str, test_corpus: str) -> bool:
    if name in ("<anonymous>", "__init__", "__repr__", "__str__"):
        # dunder/anonymous names are too generic to name-match meaningfully
        pass  # still attempt the match below, just noting the caveat

    # strong signal: an explicit test_<name> function exists
    if re.search(rf"\bdef\s+test_{re.escape(name)}\b", test_corpus):
        return True
    # weaker signal: the name is called somewhere in the test files
    if re.search(rf"\b{re.escape(name)}\s*\(", test_corpus):
        return True
    return False


def _is_tested_java(name: str, test_corpus: str) -> bool:
    if name == "<anonymous>":
        return False
    # JUnit tests typically call the method/constructor directly, e.g.
    # `obj.getString("key")` or `new JSONObject(...)`
    if re.search(rf"\b{re.escape(name)}\s*\(", test_corpus):
        return True
    return False


def _is_tested_javascript(name: str, test_corpus: str) -> bool:
    if name is None:
        return False
    # validator.js's real convention (confirmed by inspection): tests
    # dispatch by STRING NAME through a shared test-runner helper, e.g.
    # `validator: 'isEmail'` -- not a direct function call. This pattern
    # held even in the per-function test files (test/validators/isIP.test.js
    # still uses `validator: 'isIP'`, not a direct isIP(...) call). Check
    # for the quoted name first since that's the dominant real pattern here;
    # fall back to a direct call match for other JS repos/conventions.
    if re.search(rf"""['"]{re.escape(name)}['"]""", test_corpus):
        return True
    if re.search(rf"\b{re.escape(name)}\s*\(", test_corpus):
        return True
    return False


_MATCHERS = {
    "python": _is_tested_python,
    "java": _is_tested_java,
    "javascript": _is_tested_javascript,
}


def apply_test_linkage(chunks: list[Chunk], repo_root: str, test_files: list[str]) -> list[Chunk]:
    """Mutates and returns the same chunk list with has_test set."""
    test_corpus_cache: dict[str, str] = {}

    # test files can be in different languages too (mixed repo); group by
    # nothing for now since both our matchers work on the same corpus, but
    # keep the structure ready in case JS test files (very different syntax)
    # get added later and need their own corpus.
    full_corpus = _read_test_corpus(repo_root, test_files)

    for chunk in chunks:
        matcher = _MATCHERS.get(chunk.language)
        if matcher is None:
            chunk.has_test = None
            continue
        # use the qualified method name's leaf (e.g. "path_url" not
        # "RequestEncodingMixin.path_url") since that's what a test would
        # actually reference
        chunk.has_test = matcher(chunk.name, full_corpus)

    return chunks


if __name__ == "__main__":
    import sys
    from walker import chunk_repo, find_source_and_test_files

    repo_root = sys.argv[1] if len(sys.argv) > 1 else "."

    source_files, test_files = find_source_and_test_files(repo_root)
    chunks = chunk_repo(repo_root)
    chunks = apply_test_linkage(chunks, repo_root, test_files)

    tested = sum(1 for c in chunks if c.has_test)
    untested = sum(1 for c in chunks if c.has_test is False)
    print(f"Total chunks: {len(chunks)}")
    print(f"Tested:       {tested}")
    print(f"Untested:     {untested}")
    print()
    print("Sample untested chunks:")
    shown = 0
    for c in chunks:
        if c.has_test is False and shown < 15:
            print(f"  {c.kind:8s} {c.qualified_name:40s} {c.file_path}:{c.start_line}")
            shown += 1