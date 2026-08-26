"""Unity shader extraction (.shader, .compute, .cginc, .hlslinc, .hlsl, .hlsli).

Every Unity shader extension routes here, not only the ShaderLab ones. A .hlsl
in URP's ShaderLibrary has no ShaderLab wrapper, but it is written in the same
SRP macros as a .shader, and shader.py's extractor does not know them - so it
still needs the neutralization pass below. The ShaderLab-specific layers (the
Shader node, Pass nodes) simply find nothing in a bare include and no-op.

A Unity .shader file is a ShaderLab document with HLSL/Cg embedded in
HLSLPROGRAM..ENDHLSL (or CGPROGRAM..ENDCG) blocks, so tree-sitter-hlsl cannot
parse it directly. Rather than slice the program blocks out - which would put
every node's line number in block-local coordinates - this module BLANKS the
ShaderLab chrome in place, preserving newlines. Line numbers therefore stay
absolute and shader.py's extractor needs no offset threading.

Two more Unity-isms are handled before the parse:

  * SRP macros. CBUFFER_START(name)/CBUFFER_END is rewritten to a real
    "cbuffer name { ... }", which is the flat form shader.py's
    recover_known_forms already reflects. A line holding a lone ALL_CAPS macro
    (UNITY_VERTEX_INPUT_INSTANCE_ID) has no terminating ';' and makes the
    grammar swallow the enclosing struct, so one is appended.
  * Entry points. Unity declares them with "#pragma vertex/fragment/kernel",
    not the [shader("...")] attribute or numthreads that shader.py looks for.

Every rewrite above preserves the newline count, so node line numbers stay
faithful to the file on disk.
"""
from __future__ import annotations

import re
from pathlib import Path

from graphify.extractors.base import _make_id
from graphify.extractors.shader import _extract

_PROGRAM_RE = re.compile(
    r"\b(?:HLSLPROGRAM|CGPROGRAM|HLSLINCLUDE|CGINCLUDE)\b(.*?)\b(?:ENDHLSL|ENDCG)\b",
    re.DOTALL,
)
# Anchor on [ \t]* rather than \s*: \s would consume the preceding newline, so
# match.start() would land on the blank line ABOVE the keyword and every line
# number derived from it would be one too low. This also anchors GrabPass and
# UsePass out of _PASS_RE, which is what we want - neither declares shader code.
_SHADER_NAME_RE = re.compile(r'(?m)^[ \t]*Shader\s+"([^"]*)"')
_PASS_RE = re.compile(r"(?m)^[ \t]*Pass\b")
_PASS_NAME_RE = re.compile(r'(?m)^[ \t]*Name\s+"([^"]*)"')

_CBUFFER_START_RE = re.compile(r"CBUFFER_START\s*\(([^)]*)\)")
_CBUFFER_END_RE = re.compile(r"\bCBUFFER_END\b")
# The trailing lookahead, not "[ \t]*$": on a CRLF file the \r sits between the
# last space and the newline, so a plain $ never matches and the rule silently
# does nothing. Unity checkouts are CRLF by default, which is exactly where this
# rule has to work. The lookahead also leaves the \r in place for the rewrite.
_BARE_MACRO_RE = re.compile(r"(?m)^([ \t]*[A-Z_][A-Z0-9_]+)[ \t]*(?=\r?$)")

# ALL_CAPS_WITH_UNDERSCORES callees are Unity/SRP macros (SAMPLE_TEXTURE2D,
# ZERO_INITIALIZE, UNITY_ACCESS_INSTANCED_PROP), not functions the graph should
# chase across files.
_MACRO_CALL_RE = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")

# "#pragma <stage> <entry>". Unity's stage names, mapped onto shader.py's
# _STAGES vocabulary so a pragma-declared entry point is indistinguishable
# downstream from an attribute-declared one.
_PRAGMA_STAGES = {
    "vertex": "vertex",
    "fragment": "fragment",
    "geometry": "geometry",
    "hull": "hull",
    "domain": "domain",
    "kernel": "compute",
    "surface": "fragment",
}
_PRAGMA_ENTRY_RE = re.compile(
    r"(?m)^[ \t]*#[ \t]*pragma\s+(" + "|".join(_PRAGMA_STAGES) + r")\s+(\w+)"
)


def _blank(text: str) -> str:
    """Replace every character except newlines with a space."""
    return re.sub(r"[^\r\n]", " ", text)


def _mask_shaderlab(text: str, is_shaderlab: bool) -> str:
    """Blank everything outside HLSL/Cg program blocks, preserving line count.

    ``is_shaderlab`` must be True for a .shader file and False for a bare
    include/compute unit. It decides what "no program blocks" means: in a
    .shader it means a fixed-function shader with no HLSL at all (blank the
    whole document - Unity's Legacy/Mobile and editor-gizmo shaders are full of
    `Material { Diffuse [_Color] }` and `SetTexture [_MainTex] { }`, which the
    HLSL grammar reads as one long error), while in a .cginc/.compute it means
    the whole file is HLSL and must be handed over untouched.
    """
    spans = [match.span(1) for match in _PROGRAM_RE.finditer(text)]
    if not spans:
        return _blank(text) if is_shaderlab else text
    out: list[str] = []
    cursor = 0
    for start, end in spans:
        out.append(_blank(text[cursor:start]))
        out.append(text[start:end])
        cursor = end
    out.append(_blank(text[cursor:]))
    return "".join(out)


def _neutralize_macros(text: str) -> str:
    """Rewrite Unity SRP macros into forms tree-sitter-hlsl can parse."""
    text = _CBUFFER_START_RE.sub(lambda m: "cbuffer " + m.group(1) + " {", text)
    text = _CBUFFER_END_RE.sub("}", text)
    return _BARE_MACRO_RE.sub(r"\1;", text)


def _pass_spans(text: str) -> list[tuple[int, int, "str | None"]]:
    """Return (first_line, last_line, name) for each ShaderLab Pass block.

    Brace matching skips // and /* */ comments and string literals, so a brace
    inside a comment or a Name string does not close the block early.
    """
    spans: list[tuple[int, int, "str | None"]] = []
    n = len(text)
    for match in _PASS_RE.finditer(text):
        i = match.end()
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n or text[i] != "{":
            continue  # a Pass keyword with no block declares nothing
        depth = 0
        j = i
        line_comment = block_comment = in_string = False
        while j < n:
            char = text[j]
            nxt = text[j + 1] if j + 1 < n else ""
            if line_comment:
                if char == "\n":
                    line_comment = False
            elif block_comment:
                if char == "*" and nxt == "/":
                    block_comment = False
                    j += 1
            elif in_string:
                if char == '"':
                    in_string = False
            elif char == "/" and nxt == "/":
                line_comment = True
                j += 1
            elif char == "/" and nxt == "*":
                block_comment = True
                j += 1
            elif char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    j += 1
                    break
            j += 1
        name_match = _PASS_NAME_RE.search(text[match.start():j])
        spans.append((
            text.count("\n", 0, match.start()) + 1,
            text.count("\n", 0, j) + 1,
            name_match.group(1) if name_match else None,
        ))
    return spans


def _mark_pragma_entries(extractor, text: str) -> None:
    """Promote functions named by a #pragma stage directive to entry points.

    Runs from shader.py's hook - after declarations are collected, before stage
    I/O - so these functions get the same stage_input/stage_output edges an
    attribute-declared entry point gets.
    """
    by_name: dict[str, list[dict]] = {}
    for fact in extractor.functions:
        by_name.setdefault(fact["name"], []).append(fact)
    for match in _PRAGMA_ENTRY_RE.finditer(text):
        stage = _PRAGMA_STAGES[match.group(1)]
        for fact in by_name.get(match.group(2), []):
            fact["entry"] = True
            # shader.py sets "unknown" when it sees an HLSL semantic but no
            # attribute naming the stage. The pragma is that missing name, so it
            # wins over "unknown" as well as over None.
            if fact["stage"] in (None, "unknown"):
                fact["stage"] = stage
            shader = extractor.node_by_id[fact["nid"]]["metadata"]["shader"]
            shader["kind"] = "entry_point"
            shader["entry_point"] = True
            shader["stage"] = fact["stage"]


def _line_of(item: dict) -> int:
    location = item.get("source_location") or ""
    return int(location[1:]) if location[1:].isdigit() else 0


def _add_shaderlab_structure(extractor, text: str) -> None:
    """Add the Shader and Pass nodes, and reparent contents into their Pass."""
    name_match = _SHADER_NAME_RE.search(text)
    if name_match:
        line = text.count("\n", 0, name_match.start()) + 1
        nid = extractor.add_node(
            _make_id(extractor.stem, "shader", name_match.group(1)),
            name_match.group(1), line, "shader_program", node_type="shader",
        )
        extractor.add_edge(extractor.file_nid, nid, "defines", line)

    spans = _pass_spans(text)
    if not spans:
        return
    pass_nids: list[tuple[int, int, str]] = []
    for index, (first, last, name) in enumerate(spans):
        label = name or f"Pass {index}"
        nid = extractor.add_node(
            _make_id(extractor.stem, "pass", label), f"Pass:{label}", first,
            "shader_pass", node_type="pass", pass_name=name,
        )
        extractor.add_edge(extractor.file_nid, nid, "contains", first)
        pass_nids.append((first, last, nid))

    # A declaration inside a Pass belongs to that Pass, not loosely to the file.
    own = {nid for _, _, nid in pass_nids}
    for edge in extractor.edges:
        if edge["source"] != extractor.file_nid or edge["relation"] != "contains":
            continue
        if edge["target"] in own:
            continue
        line = _line_of(edge)
        for first, last, nid in pass_nids:
            if first <= line <= last:
                edge["source"] = nid
                break


def extract_shaderlab(path: Path) -> dict:
    """Reflect a Unity .shader, .compute, or HLSL include.

    Delegates the HLSL body to shader.py's extractor and layers the ShaderLab
    structure (Shader name, Pass blocks) on top. On a bare include there is no
    such structure to add, and the value is the SRP macro neutralization alone.
    """
    try:
        text = path.read_bytes().decode("utf-8", errors="replace")
    except OSError as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}

    is_shaderlab = path.suffix.lower() == ".shader"
    source = _neutralize_macros(_mask_shaderlab(text, is_shaderlab)).encode("utf-8")

    captured: list = []

    def hook(extractor) -> None:
        captured.append(extractor)
        _mark_pragma_entries(extractor, text)

    result = _extract(path, "hlsl", source=source, hook=hook)
    if not captured:
        return result  # parse or import failed; result carries the error
    extractor = captured[0]

    _add_shaderlab_structure(extractor, text)

    # Drop macro-shaped unresolved calls so cross-file resolution does not hunt
    # for a function named SAMPLE_TEXTURE2D.
    result["raw_calls"] = [
        call for call in result.get("raw_calls", [])
        if not _MACRO_CALL_RE.fullmatch(call.get("callee", ""))
    ]
    result["nodes"] = extractor.nodes
    result["edges"] = extractor.edges
    return result
