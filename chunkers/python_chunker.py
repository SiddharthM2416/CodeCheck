"""
Extracts function/method/class chunks from a Python source file using tree-sitter.

Unit rules:
- Top-level `def` -> one chunk, kind="function"
- Top-level `class` -> each method inside becomes its own chunk, kind="method",
  qualified_name = "ClassName.method_name"
- A class with no methods (rare, e.g. a plain data container) is chunked as
  a whole, kind="class" -- documented limitation: such a class won't get
  method-level test granularity.
"""

from tree_sitter_languages import get_parser
from .base import Chunk

_parser = get_parser("python")


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_name(node, source: bytes) -> str:
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source)
    return "<anonymous>"


def extract_python_chunks(file_path: str, repo_root: str) -> list[Chunk]:
    full_path = f"{repo_root}/{file_path}"
    with open(full_path, "rb") as f:
        source = f.read()

    tree = _parser.parse(source)
    chunks: list[Chunk] = []

    def _unwrap_decorated(node):
        """decorated_definition wraps a function_definition/class_definition
        as a child -- return that inner node, plus the outer node (which
        has the correct start_line including the decorator lines)."""
        if node.type == "decorated_definition":
            inner = next(
                (c for c in node.children if c.type in ("function_definition", "class_definition")),
                None,
            )
            return inner, node  # inner node for name/type, outer node for span
        return node, node

    def _contains_function(body_node) -> bool:
        for c in body_node.children:
            inner, _ = _unwrap_decorated(c)
            if inner is not None and inner.type == "function_definition":
                return True
        return False

    def walk(node, class_context: str | None = None):
        for raw_child in node.children:
            child, span_node = _unwrap_decorated(raw_child)
            if child is None:
                continue

            if child.type == "function_definition":
                name = _get_name(child, source)
                qualified = f"{class_context}.{name}" if class_context else name
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=span_node.start_point[0] + 1,
                    end_line=span_node.end_point[0] + 1,
                    name=name,
                    qualified_name=qualified,
                    kind="method" if class_context else "function",
                    language="python",
                    code=_node_text(span_node, source),
                ))
                # don't recurse into nested functions/closures -- keep chunks
                # at the top function/method level, matching how you'd test them

            elif child.type == "class_definition":
                class_name = _get_name(child, source)
                body = next((c for c in child.children if c.type == "block"), None)
                has_methods = body is not None and _contains_function(body)
                if has_methods:
                    walk(body, class_context=class_name)
                else:
                    chunks.append(Chunk(
                        file_path=file_path,
                        start_line=span_node.start_point[0] + 1,
                        end_line=span_node.end_point[0] + 1,
                        name=class_name,
                        qualified_name=class_name,
                        kind="class",
                        language="python",
                        code=_node_text(span_node, source),
                    ))
            else:
                # keep descending (module -> block -> etc.)
                walk(raw_child, class_context)

    walk(tree.root_node)
    return chunks