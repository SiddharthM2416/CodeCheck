"""
Extracts function/method/class chunks from a JavaScript source file using
tree-sitter.

Unit rules (designed against validator.js's real structure, but written to
be reasonably general for other plain-JS/utility-style repos too):
- `export default function name(...) {}` -- by far the dominant pattern in
  validator.js (101/103 lib files). Chunked with the export_statement's
  full span (includes "export default", small but useful context), kind="function".
- Plain top-level `function name(...) {}` (no export) -- also chunked,
  kind="function".
- `const name = (...) => {}` / `export const name = (...) => {}` -- arrow
  functions assigned to a top-level const, kind="function".
- `class Name { ... }` with methods -- each method_definition becomes its
  own chunk, kind="method", qualified as "Name.methodName" (for reusability
  against other JS repos that do use classes; validator.js itself has none).

Known limitation, documented here rather than discovered later: files that
only export plain data objects (e.g. validator.js's alpha.js, which exports
a lookup table, not a function) correctly produce zero chunks -- there's no
testable "function" there, which is the right behavior, not a bug.
"""

from tree_sitter_languages import get_parser
from .base import Chunk

_parser = get_parser("javascript")


def _node_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _get_name(node, source: bytes) -> str | None:
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source)
    return None


def extract_javascript_chunks(file_path: str, repo_root: str) -> list[Chunk]:
    full_path = f"{repo_root}/{file_path}"
    with open(full_path, "rb") as f:
        source = f.read()

    tree = _parser.parse(source)
    chunks: list[Chunk] = []

    def handle_function_declaration(node, span_node, class_context: str | None):
        name = _get_name(node, source)
        if name is None:
            return
        qualified = f"{class_context}.{name}" if class_context else name
        chunks.append(Chunk(
            file_path=file_path,
            start_line=span_node.start_point[0] + 1,
            end_line=span_node.end_point[0] + 1,
            name=name,
            qualified_name=qualified,
            kind="method" if class_context else "function",
            language="javascript",
            code=_node_text(span_node, source),
        ))

    def handle_lexical_declaration(node, span_node):
        # const name = (...) => {} -- only chunk if the value is a function
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = next((c for c in child.children if c.type == "identifier"), None)
            value_node = next(
                (c for c in child.children if c.type in ("arrow_function", "function")), None
            )
            if name_node is not None and value_node is not None:
                name = _node_text(name_node, source)
                chunks.append(Chunk(
                    file_path=file_path,
                    start_line=span_node.start_point[0] + 1,
                    end_line=span_node.end_point[0] + 1,
                    name=name,
                    qualified_name=name,
                    kind="function",
                    language="javascript",
                    code=_node_text(span_node, source),
                ))

    def handle_class(node, class_context: str | None):
        class_name = _get_name(node, source)
        body = next((c for c in node.children if c.type == "class_body"), None)
        if body is None:
            return
        for member in body.children:
            if member.type == "method_definition":
                handle_function_declaration(member, member, class_name)

    def walk(node, class_context: str | None = None):
        for child in node.children:
            if child.type == "export_statement":
                # export_statement's children include the actual declaration
                # (function_declaration / class_declaration / lexical_declaration)
                inner = next(
                    (c for c in child.children
                     if c.type in ("function_declaration", "class_declaration", "lexical_declaration")),
                    None,
                )
                if inner is None:
                    continue
                if inner.type == "function_declaration":
                    handle_function_declaration(inner, child, class_context)  # span = export_statement
                elif inner.type == "class_declaration":
                    handle_class(inner, class_context)
                elif inner.type == "lexical_declaration":
                    handle_lexical_declaration(inner, child)

            elif child.type == "function_declaration":
                handle_function_declaration(child, child, class_context)

            elif child.type == "class_declaration":
                handle_class(child, class_context)

            elif child.type == "lexical_declaration":
                handle_lexical_declaration(child, child)

            else:
                walk(child, class_context)

    walk(tree.root_node)
    return chunks