"""Windows console-window suppression for external-binary spawns.

A hook-triggered rebuild runs with no console window of its own, so every
console binary it spawns (git, cpp, gh, ...) gets one allocated - a window
that flashes on screen. CREATE_NO_WINDOW must be passed at each spawn; it is
a creation flag, not an inherited property.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

from graphify.proc import no_window_kwargs

_PKG = Path(__file__).resolve().parent.parent / "graphify"

# Spawns that are not launching an external console binary, or that already
# handle the flag themselves.
_EXEMPT = {
    # hooks.py builds shell text, and its embedded launcher passes the
    # Windows creation flags inline rather than through this helper.
    "hooks.py",
}


def _spawn_calls(path: Path):
    """Yield (lineno, call) for every subprocess.run/Popen/call in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (isinstance(fn, ast.Attribute)
                and fn.attr in {"run", "Popen", "call", "check_call", "check_output"}
                and isinstance(fn.value, ast.Name) and fn.value.id in {"subprocess", "_sp"}):
            yield node.lineno, node


def _passes_no_window(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg is None:  # **something
            src = ast.dump(kw.value)
            if "no_window_kwargs" in src:
                return True
        elif kw.arg == "creationflags":
            return True
    return False


def test_every_external_spawn_suppresses_the_console_window():
    missing = []
    for path in sorted(_PKG.rglob("*.py")):
        if path.name in _EXEMPT:
            continue
        for lineno, call in _spawn_calls(path):
            if not _passes_no_window(call):
                missing.append(f"{path.relative_to(_PKG.parent)}:{lineno}")
    assert not missing, (
        "these subprocess spawns would flash a console window on Windows; "
        "pass **no_window_kwargs() (graphify/proc.py):\n  " + "\n  ".join(missing)
    )


def test_no_window_kwargs_matches_platform():
    if sys.platform == "win32":
        assert no_window_kwargs() == {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        assert no_window_kwargs() == {}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows creation flags")
def test_flag_is_accepted_by_a_real_spawn():
    """The flag must be usable as passed - a wrong value raises on spawn."""
    res = subprocess.run([sys.executable, "-c", "print('ok')"],
                         capture_output=True, text=True, **no_window_kwargs())
    assert res.stdout.strip() == "ok"
