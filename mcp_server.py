"""
MCP server exposing three tools to any MCP client (Claude via Agent SDK,
the MCP inspector, etc.):

  - search_code(query, collection_name, k=5)
  - find_untested(collection_name, path_prefix=None, language=None, limit=50)
  - read_file(collection_name, file_path, start_line=None, end_line=None)

Run standalone for testing with the MCP inspector:
    npx @modelcontextprotocol/inspector python mcp_server.py

Or run directly (stdio transport, for wiring into an agent client):
    python mcp_server.py
"""

from mcp.server.fastmcp import FastMCP

from retrieval import (
    search_code as _search_code,
    find_untested as _find_untested,
    read_file as _read_file,
    list_indexed_repos as _list_indexed_repos,
    get_repo_path_for_collection,
)

mcp = FastMCP("testgap-agent")


@mcp.tool()
def list_repos() -> list[dict]:
    """List every codebase currently indexed and available to query. Returns
    each repo's collection_name (needed by the other tools), its filesystem
    path, and how many code chunks were indexed. Call this first if you
    don't already know which collection_name to use."""
    return _list_indexed_repos()


@mcp.tool()
def search_code(query: str, collection_name: str, k: int = 5) -> list[dict]:
    """Semantic search over an indexed codebase. Finds code chunks
    (functions/methods/classes) whose meaning matches the query, even if
    the exact words don't appear in the code. Use this for questions like
    'where is authentication handled?' or 'find the retry logic'.

    Returns a list of chunks, each with: qualified_name, file_path,
    start_line, end_line, kind, language, has_test, code, distance
    (lower distance = more semantically similar)."""
    return _search_code(query, collection_name, k=k)


@mcp.tool()
def find_untested(
    collection_name: str,
    path_prefix: str | None = None,
    language: str | None = None,
    limit: int = 15,
) -> list[dict]:
    """Find functions/methods/classes that have no corresponding test,
    based on a name-matching heuristic against the repo's test files. This
    is the primary tool for the test-gap detection use case. Optionally
    scope to a subdirectory (path_prefix, e.g. 'src/requests/') or a
    specific language ('python' or 'java'). Default limit is intentionally
    small (15) to keep response size reasonable for smaller-context/
    free-tier models -- call again with a narrower path_prefix rather than
    raising the limit if you need to see more.

    Returns a list of untested chunks, each with: qualified_name,
    file_path, start_line, end_line, kind, language, code (code is
    truncated for very large chunks -- use read_file for the full body).

    Known limitation: this is name-based matching (does the function name
    get called somewhere in a test file), not true call-graph analysis. It
    can have false positives for very generic/short names, and won't catch
    indirect testing (function A is only exercised because test covers
    function B, which calls A internally)."""
    return _find_untested(collection_name, path_prefix=path_prefix, language=language, limit=limit)


@mcp.tool()
def read_file(
    collection_name: str,
    file_path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Read raw source code directly from disk, for full surrounding
    context beyond a single chunk returned by search_code or find_untested
    (e.g. to see imports, a class's other methods, or nearby comments).
    file_path should be relative to the repo root (as returned by the
    other tools' file_path field). If start_line/end_line are omitted,
    reads the whole file. Line numbers are 1-indexed and inclusive."""
    repo_path = get_repo_path_for_collection(collection_name)
    return _read_file(repo_path, file_path, start_line=start_line, end_line=end_line)


if __name__ == "__main__":
    mcp.run()