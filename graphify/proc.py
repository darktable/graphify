"""Subprocess spawn kwargs shared across graphify.

Windows only: every external console binary graphify shells out to (git, gh,
cpp, claude.cmd, the Google Workspace CLI) is a console-subsystem program. When
the parent has no console window of its own - which is the normal case for a
hook-triggered rebuild, launched detached from a GUI git client or an agent
shell - the console host allocates one per spawn, and with Windows Terminal as
the default host that is a real window that appears and vanishes on screen.
A long rebuild calls git once per scan and cpp once per capital-F Fortran file,
so the flashing is per-spawn, not once.

CREATE_NO_WINDOW suppresses it. It has to be passed at each spawn: it is a
creation flag, not an inherited process property, so suppressing it on the
rebuild process does not cover the binaries that process goes on to run.

No-op off Windows.
"""
from __future__ import annotations

import subprocess
import sys


def no_window_kwargs() -> dict:
    """subprocess kwargs that keep a spawned console binary invisible on Windows."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW}
    return {}
