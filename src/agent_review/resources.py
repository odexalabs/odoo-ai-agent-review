"""Bundled data: the default configuration, the example run profiles, the safety profiles, the scenarios
and the fixture scripts.

They live at the top of the source tree (`config/`, `profiles/`, `scenarios/`, `fixtures/`) so a reader of
the repository finds them there, and the wheel carries them inside the package as `agent_review/_bundled/`
(`pyproject.toml`, `force-include`). ONE function resolves them, in this order:

    1. `agent_review/_bundled/`   an installed wheel
    2. the source checkout       an editable install or `pytest` from the repository

A checkout never has `_bundled/`, and an installed wheel never has the checkout, so the two cannot disagree.
If neither holds the files, loading fails loudly rather than guessing another location.

Explicit paths a user passes (`--profile my.yaml`, a scenario file) are always used as given; only bare names
are looked up here."""
from __future__ import annotations

from pathlib import Path

BUNDLED_DIRS = ("config", "profiles", "scenarios", "fixtures")
_PACKAGE = Path(__file__).resolve().parent


class BundledDataMissing(FileNotFoundError):
    pass


def bundled_root() -> Path:
    """The directory holding config/, profiles/, scenarios/ and fixtures/."""
    installed = _PACKAGE / "_bundled"
    if (installed / "config").is_dir():
        return installed
    checkout = _PACKAGE.parents[1]          # <checkout>/src/agent_review -> <checkout>
    if (checkout / "config" / "classification.yaml").is_file() and (checkout / "src" / "agent_review").is_dir():
        return checkout
    raise BundledDataMissing(
        f"bundled data not found: neither {installed} (installed wheel) nor a source checkout at {checkout} "
        "holds config/classification.yaml. Reinstall the package.")


def bundled(*parts: str) -> Path:
    """A path inside the bundled data, e.g. bundled("config", "pricing.yaml")."""
    if not parts or parts[0] not in BUNDLED_DIRS:
        raise ValueError(f"bundled data lives under {BUNDLED_DIRS}, got {parts!r}")
    return bundled_root().joinpath(*parts)


def bundled_names(kind: str, suffix: str = ".yaml") -> list[str]:
    """Bare names of the bundled files under one directory, e.g. bundled_names('scenarios')."""
    d = bundled(*kind.split("/"))
    return sorted(p.stem for p in d.glob(f"*{suffix}")) if d.is_dir() else []
