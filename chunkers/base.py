"""
Shared data model for all language chunkers.

Every language extractor (python_chunker, js_chunker, java_chunker) must
return a list of Chunk objects in this exact shape. Nothing downstream
(embedding, Chroma storage, MCP tools, agent loop) should ever need to
know which language a chunk came from.
"""

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Chunk:
    file_path: str          # relative path from repo root, e.g. "src/requests/models.py"
    start_line: int         # 1-indexed, inclusive
    end_line: int           # 1-indexed, inclusive
    name: str                # function/method/component name, e.g. "calculate_refund"
    qualified_name: str      # includes class/component context, e.g. "Response.raise_for_status"
    kind: str                 # "function" | "method" | "class" | "component" | "hook"
    language: str             # "python" | "javascript" | "java"
    code: str                  # the raw source text of this chunk
    has_test: Optional[bool] = None   # set later by the test-linkage step, not by the chunker

    def to_dict(self) -> dict:
        return asdict(self)


def make_chunk_id(chunk: Chunk) -> str:
    """Stable ID for storing/updating this chunk in Chroma."""
    return f"{chunk.language}:{chunk.file_path}:{chunk.start_line}:{chunk.name}"