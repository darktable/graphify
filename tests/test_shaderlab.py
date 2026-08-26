from pathlib import Path

from graphify.detect import CODE_EXTENSIONS
from graphify.extract import _get_extractor
from graphify.extractors.shader import _extract
from graphify.extractors.shaderlab import extract_shaderlab

FIXTURES = Path(__file__).parent / "fixtures"
SHADER = FIXTURES / "sample.shader"
COMPUTE = FIXTURES / "sample.compute"


def _node(result: dict, label: str) -> dict:
    return next(node for node in result["nodes"] if node["label"] == label)


def _meta(result: dict, label: str) -> dict:
    return _node(result, label)["metadata"]["shader"]


def _line(result: dict, label: str) -> int:
    return int(_node(result, label)["source_location"][1:])


def _edges(result: dict, relation: str) -> list[dict]:
    return [edge for edge in result["edges"] if edge["relation"] == relation]


def _parent(result: dict, label: str) -> str:
    target = _node(result, label)["id"]
    by_id = {node["id"]: node["label"] for node in result["nodes"]}
    return next(by_id[edge["source"]] for edge in _edges(result, "contains")
                if edge["target"] == target)


def test_shaderlab_dispatch_is_case_insensitive() -> None:
    # .hlsl/.hlsli are here too: a bare URP include carries no ShaderLab
    # wrapper but is written in the same SRP macros, so it needs this
    # extractor's neutralization pass. See LANGUAGE_EXTRACTORS in extract.py.
    for ext in (".shader", ".compute", ".cginc", ".hlslinc", ".hlsl", ".hlsli"):
        assert _get_extractor(Path("x" + ext)) is extract_shaderlab
        assert _get_extractor(Path("x" + ext.upper())) is extract_shaderlab
        assert ext in CODE_EXTENSIONS


def test_shaderlab_parses_without_errors() -> None:
    result = extract_shaderlab(SHADER)
    assert "error" not in result
    assert "parse_errors" not in result


def test_raw_hlsl_parse_of_shaderlab_fails_without_preprocessing() -> None:
    """The negative control for the test above.

    ShaderLab chrome plus SRP macros must actually break the bare grammar,
    otherwise test_shaderlab_parses_without_errors proves nothing about the
    masking and macro neutralization.
    """
    raw = _extract(SHADER, "hlsl")
    assert raw["parse_errors"]["count"] > 0


def test_shader_and_pass_nodes() -> None:
    result = extract_shaderlab(SHADER)
    assert _meta(result, "Custom/URPLit")["kind"] == "shader_program"
    assert [edge["relation"] for edge in result["edges"]
            if edge["target"] == _node(result, "Custom/URPLit")["id"]] == ["defines"]
    assert _meta(result, "Pass:ForwardLit")["kind"] == "shader_pass"
    assert _meta(result, "Pass:ShadowCaster")["kind"] == "shader_pass"


def test_line_numbers_are_absolute_in_the_file() -> None:
    """Masking (not slicing) the ShaderLab chrome keeps lines file-absolute."""
    lines = SHADER.read_text(encoding="utf-8").splitlines()
    result = extract_shaderlab(SHADER)
    assert "Varyings LitVertex" in lines[_line(result, "LitVertex()") - 1]
    assert "float3 ApplyTint" in lines[_line(result, "ApplyTint()") - 1]
    assert "CBUFFER_START" in lines[_line(result, "UnityPerMaterial") - 1]
    assert "Pass" in lines[_line(result, "Pass:ForwardLit") - 1]


def test_pragma_declares_entry_points_and_stages() -> None:
    result = extract_shaderlab(SHADER)
    stages = {label: _meta(result, label)["stage"]
              for label in ("LitVertex()", "LitFragment()",
                            "ShadowVertex()", "ShadowFragment()")}
    assert stages == {"LitVertex()": "vertex", "LitFragment()": "fragment",
                      "ShadowVertex()": "vertex", "ShadowFragment()": "fragment"}
    assert all(_meta(result, label)["kind"] == "entry_point" for label in stages)
    # ApplyTint has no pragma and no semantic, so it stays a plain function.
    assert _meta(result, "ApplyTint()")["kind"] == "function"


def test_pragma_entry_points_get_stage_io() -> None:
    """Marking entries must happen before stage I/O, not after."""
    result = extract_shaderlab(SHADER)
    entry = _node(result, "LitVertex()")["id"]
    io = [edge for edge in result["edges"]
          if edge["source"] == entry and edge["relation"].startswith("stage_")]
    assert io


def test_declarations_are_reparented_into_their_pass() -> None:
    result = extract_shaderlab(SHADER)
    assert _parent(result, "LitFragment()") == "Pass:ForwardLit"
    assert _parent(result, "Attributes") == "Pass:ForwardLit"
    assert _parent(result, "ShadowVertex()") == "Pass:ShadowCaster"
    # HLSLINCLUDE sits outside every Pass, so it stays on the file.
    assert _parent(result, "ApplyTint()") == "sample.shader"


def test_cbuffer_start_macro_is_reflected_as_a_uniform_buffer() -> None:
    result = extract_shaderlab(SHADER)
    assert _meta(result, "UnityPerMaterial")["resource_kind"] == "uniform_buffer"
    assert _parent(result, "_BaseColor") == "UnityPerMaterial"


def test_unity_macros_are_not_reported_as_unresolved_calls() -> None:
    result = extract_shaderlab(SHADER)
    callees = {call["callee"] for call in result["raw_calls"]}
    assert not callees & {"SAMPLE_TEXTURE2D", "UNITY_SETUP_INSTANCE_ID",
                          "TRANSFORM_TEX", "TEXTURE2D", "SAMPLER"}
    # A real SRP function still is one.
    assert "TransformObjectToHClip" in callees


def test_local_calls_resolve_across_a_program_block_boundary() -> None:
    result = extract_shaderlab(SHADER)
    calls = {(edge["source"], edge["target"]) for edge in _edges(result, "calls")}
    assert (_node(result, "LitFragment()")["id"], _node(result, "ApplyTint()")["id"]) in calls


def test_compute_kernel_pragma_and_resources() -> None:
    result = extract_shaderlab(COMPUTE)
    assert "parse_errors" not in result
    assert _meta(result, "Accumulate()")["stage"] == "compute"
    assert _meta(result, "Accumulate()")["thread_group_size"] == [64, 1, 1]
    assert _meta(result, "_Output")["resource_kind"] == "storage_resource"
    assert _meta(result, "_Input")["resource_kind"] == "buffer"
    assert _meta(result, "Params")["resource_kind"] == "uniform_buffer"


def test_bare_include_file_without_shaderlab_wrapper(tmp_path: Path) -> None:
    include = tmp_path / "Common.cginc"
    include.write_text(
        "CBUFFER_START(UnityPerFrame)\n"
        "    float4 _Time;\n"
        "CBUFFER_END\n"
        "\n"
        "float3 Desaturate(float3 c)\n"
        "{\n"
        "    return dot(c, float3(0.3, 0.6, 0.1));\n"
        "}\n",
        encoding="utf-8",
    )
    result = extract_shaderlab(include)
    assert "parse_errors" not in result
    assert _line(result, "Desaturate()") == 5
    assert _meta(result, "UnityPerFrame")["resource_kind"] == "uniform_buffer"


def test_grabpass_without_a_block_is_not_a_pass(tmp_path: Path) -> None:
    shader = tmp_path / "Grab.shader"
    shader.write_text(
        'Shader "X/Grab"\n'
        "{\n"
        "    SubShader\n"
        "    {\n"
        '        GrabPass { "_GrabTexture" }\n'
        "        UsePass \"X/Other/FORWARD\"\n"
        "        Pass\n"
        "        {\n"
        '            Name "Only"\n'
        "            HLSLPROGRAM\n"
        "            #pragma vertex vert\n"
        "            float4 vert(float4 p : POSITION) : SV_POSITION { return p; }\n"
        "            ENDHLSL\n"
        "        }\n"
        "    }\n"
        "}\n",
        encoding="utf-8",
    )
    result = extract_shaderlab(shader)
    passes = [node["label"] for node in result["nodes"]
              if node.get("metadata", {}).get("shader", {}).get("kind") == "shader_pass"]
    assert passes == ["Pass:Only"]


def test_macro_neutralization_survives_crlf_line_endings(tmp_path: Path) -> None:
    """Unity checks out CRLF by default, so the rules must not need a bare LF.

    A '$' anchor sits after the '\r', so a rule written as "[ \t]*$" silently
    stops firing on every real Unity file while still passing on LF fixtures.
    """
    body = (
        'Shader "X/CRLF"\n'
        "{\n"
        "    SubShader\n"
        "    {\n"
        "        Pass\n"
        "        {\n"
        "            HLSLPROGRAM\n"
        "            #pragma vertex vert\n"
        "            CBUFFER_START(UnityPerMaterial)\n"
        "                float4 _Color;\n"
        "            CBUFFER_END\n"
        "            struct Attributes\n"
        "            {\n"
        "                float4 positionOS : POSITION;\n"
        "                UNITY_VERTEX_INPUT_INSTANCE_ID\n"
        "            };\n"
        "            float4 vert(Attributes i) : SV_POSITION { return i.positionOS; }\n"
        "            ENDHLSL\n"
        "        }\n"
        "    }\n"
        "}\n"
    )
    lf = tmp_path / "lf.shader"
    lf.write_bytes(body.encode())
    crlf = tmp_path / "crlf.shader"
    crlf.write_bytes(body.replace("\n", "\r\n").encode())

    lf_result, crlf_result = extract_shaderlab(lf), extract_shaderlab(crlf)
    assert "parse_errors" not in lf_result
    assert "parse_errors" not in crlf_result
    for result in (lf_result, crlf_result):
        assert _meta(result, "UnityPerMaterial")["resource_kind"] == "uniform_buffer"
        assert _meta(result, "vert()")["stage"] == "vertex"
        assert _node(result, "Attributes")


def test_fixed_function_shaderlab_has_no_hlsl_to_parse(tmp_path: Path) -> None:
    """A Legacy/Mobile shader is pure ShaderLab; feeding it to the HLSL grammar
    produces one long error, so the whole document must be blanked."""
    shader = tmp_path / "Legacy.shader"
    shader.write_text(
        'Shader "Legacy Shaders/Diffuse Fast" {\n'
        "Properties {\n"
        '    _Color ("Main Color", Color) = (1,1,1,1)\n'
        '    _MainTex ("Base (RGB)", 2D) = "white" {}\n'
        "}\n"
        "SubShader {\n"
        "    Pass {\n"
        "        Material { Diffuse [_Color] }\n"
        "        Lighting On\n"
        "        SetTexture [_MainTex] { combine texture * primary }\n"
        "    }\n"
        "}\n"
        'Fallback "Legacy Shaders/VertexLit"\n'
        "}\n",
        encoding="utf-8",
    )
    result = extract_shaderlab(shader)
    assert "parse_errors" not in result
    assert _node(result, "Legacy Shaders/Diffuse Fast")


def test_bare_hlsl_include_gets_srp_macro_neutralization(tmp_path: Path) -> None:
    """URP's .hlsl library is written in SRP macros despite having no ShaderLab
    wrapper, which is why .hlsl routes through this extractor rather than
    straight to shader.py's extract_hlsl."""
    include = tmp_path / "LitInput.hlsl"
    include.write_text(
        "CBUFFER_START(UnityPerMaterial)\n"
        "    float4 _BaseMap_ST;\n"
        "    half4 _BaseColor;\n"
        "CBUFFER_END\n"
        "\n"
        "struct SurfaceDescription\n"
        "{\n"
        "    half3 albedo;\n"
        "    UNITY_VERTEX_INPUT_INSTANCE_ID\n"
        "};\n"
        "\n"
        "half3 SampleAlbedo(float2 uv)\n"
        "{\n"
        "    return _BaseColor.rgb;\n"
        "}\n",
        encoding="utf-8",
    )

    routed = extract_shaderlab(include)
    assert "parse_errors" not in routed
    assert _meta(routed, "UnityPerMaterial")["resource_kind"] == "uniform_buffer"
    assert _node(routed, "SurfaceDescription")
    assert _line(routed, "SampleAlbedo()") == 12

    # The negative control: without the pass, the same file does not parse.
    from graphify.extractors.shader import extract_hlsl
    assert extract_hlsl(include)["parse_errors"]["count"] > 0
