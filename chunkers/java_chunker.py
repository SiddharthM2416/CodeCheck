"""
Extracts method/constructor/class chunks from a Java source file using tree-sitter.

Unit rules:
- Each `method_declaration` inside a class -> one chunk, kind="method",
  qualified_name = "ClassName.methodName"
- Each `constructor_declaration` -> one chunk, kind="method" (constructors are
  just as testable/risky as methods -- e.g. JSONObject's constructors parse
  input and are prime test-gap candidates)
- Nested/inner classes (e.g. JSONObject.Null) are walked recursively, with
  qualified names like "JSONObject.Null.methodName"
- A class with no methods/constructors is chunked as a whole, kind="class"
  (documented limitation, same as Python)
"""

from tree_sitter_languages import get_parser
from .base import Chunk

_parser = get_parser("java")


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_name(node, source: bytes) -> str:
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source)
    return "<anonymous>"


def extract_java_chunks(file_path: str, repo_root: str) -> list[Chunk]:
    full_path = f"{repo_root}/{file_path}"
    with open(full_path, "rb") as f:
        source = f.read()

    tree = _parser.parse(source)
    chunks: list[Chunk] = []

    def has_member(body_node, types: tuple[str, ...]) -> bool:
        return any(c.type in types for c in body_node.children)

    def walk(node, class_context: str | None = None):
        for child in node.children:
            if child.type in ("method_declaration", "constructor_declaration"):
                name = _get_name(child, source)
                qualified = f"{class_context}.{name}" if class_context else name
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    name=name,
                    qualified_name=qualified,
                    kind="method",
                    language="java",
                    code=_node_text(child, source),
                ))

            elif child.type in ("class_declaration", "interface_declaration"):
                class_name = _get_name(child, source)
                qualified_class = f"{class_context}.{class_name}" if class_context else class_name
                body = next((c for c in child.children if c.type in ("class_body", "interface_body")), None)
                has_methods = body is not None and has_member(
                    body, ("method_declaration", "constructor_declaration", "class_declaration")
                )
                if has_methods:
                    walk(body, class_context=qualified_class)
                else:
                    chunks.append(Chunk(
                        file_path=file_path,
                        start_line=child.start_point[0] + 1,
                        end_line=child.end_point[0] + 1,
                        name=class_name,
                        qualified_name=qualified_class,
                        kind="class",
                        language="java",
                        code=_node_text(child, source),
                    ))
            else:
                # keep descending (program -> class_body -> etc.)
                walk(child, class_context)

    walk(tree.root_node)
    return chunks