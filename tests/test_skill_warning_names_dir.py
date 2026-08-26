"""Skill-staleness warnings must name the directory they are about.

The check runs once per install location, so a user with several agent
platforms installed used to see the same pathless sentence repeated with no
way to tell which dir needed repairing (#user-report).
"""
from pathlib import Path

import pytest

from graphify.__main__ import (
    _check_skill_version,
    _skill_label,
    _skill_repair_command,
    __version__,
)


def _make_skill(tmp_path: Path, stamp: str, *, body: str = "hello") -> Path:
    d = tmp_path / "skills" / "graphify"
    d.mkdir(parents=True)
    (d / ".graphify_version").write_text(stamp, encoding="utf-8")
    dst = d / "SKILL.md"
    dst.write_text(body, encoding="utf-8")
    return dst


def test_stale_warning_names_the_directory(tmp_path, capsys):
    dst = _make_skill(tmp_path, "0.0.1")
    _check_skill_version(dst)
    err = capsys.readouterr().err
    assert "0.0.1" in err
    assert str(dst.parent) in err or _skill_label(dst) in err, err


def test_newer_skill_warning_names_the_directory(tmp_path, capsys):
    """The downgrade-guard branch must identify its dir too."""
    dst = _make_skill(tmp_path, "999.0.0")
    _check_skill_version(dst)
    err = capsys.readouterr().err
    assert "would downgrade" in err
    assert _skill_label(dst) in err, err


def test_missing_sidecar_warning_names_the_directory(tmp_path, capsys):
    dst = _make_skill(tmp_path, __version__, body="see references/foo.md")
    _check_skill_version(dst)
    err = capsys.readouterr().err
    assert "references/" in err
    assert _skill_label(dst) in err, err


def test_missing_skill_md_warning_names_the_directory(tmp_path, capsys):
    dst = _make_skill(tmp_path, __version__)
    dst.unlink()
    _check_skill_version(dst)
    err = capsys.readouterr().err
    assert "SKILL.md is missing" in err
    assert _skill_label(dst) in err, err


def test_current_version_is_silent(tmp_path, capsys):
    """Guards the tests above: they only prove anything if a matching stamp
    produces no output at all."""
    dst = _make_skill(tmp_path, __version__)
    _check_skill_version(dst)
    assert capsys.readouterr().err == ""


def test_label_shortens_paths_under_home():
    inside = Path.home() / ".claude" / "skills" / "graphify" / "SKILL.md"
    assert _skill_label(inside) == "~/.claude/skills/graphify"


def test_label_leaves_paths_outside_home_absolute(tmp_path):
    outside = tmp_path / "skills" / "graphify" / "SKILL.md"
    label = _skill_label(outside)
    assert label == str(outside.parent)
    assert not label.startswith("~")


def test_suggested_command_is_actually_runnable():
    """Every command the warning suggests must parse.

    `--platform` takes exactly ONE value, so a dir shared by several platforms
    cannot be named with one - suggesting `--platform a|b` there would hand the
    user a command that exits 1 with "unknown platform".
    """
    from graphify.install import _PLATFORM_CONFIG as _INSTALL_PLATFORMS
    from graphify.__main__ import _PLATFORM_CONFIG, _platform_skill_destination

    for name in _PLATFORM_CONFIG:
        cmd = _skill_repair_command(_platform_skill_destination(name))
        parts = cmd.split()
        assert parts[:2] == ["graphify", "install"], cmd
        if len(parts) > 2:
            assert parts[2] == "--platform" and len(parts) == 4, cmd
            assert parts[3] in _INSTALL_PLATFORMS, cmd


def test_shared_dir_falls_back_to_plain_install():
    """claude and windows share ~/.claude/skills and ship different bodies, so
    the warning must not pick one of them arbitrarily."""
    from graphify.__main__ import _platform_skill_destination

    shared = _platform_skill_destination("claude")
    if _platform_skill_destination("windows") == shared:
        assert _skill_repair_command(shared) == "graphify install"


def test_sole_platform_gets_a_targeted_command():
    """Guards the test above: it only means something if a dir with exactly one
    platform DOES get the --platform form."""
    from graphify.__main__ import _PLATFORM_CONFIG, _platform_skill_destination

    counts = {}
    for name in _PLATFORM_CONFIG:
        counts.setdefault(_platform_skill_destination(name), []).append(name)
    solos = [(d, n[0]) for d, n in counts.items() if len(n) == 1]
    assert solos, "expected at least one platform with its own dir"
    for dst, name in solos:
        assert _skill_repair_command(dst) == f"graphify install --platform {name}"
