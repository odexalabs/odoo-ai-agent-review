"""The publication boundary, checked over the DISTRIBUTABLE tree (code, comments, docs, bundled data, tests), not over
rendered output: a sentence in a comment ships exactly like a sentence in the README.

Each pattern is assembled from fragments, so this file never contains the strings it forbids. A match names the
file and line. The scan is a guard against known leak shapes; it is not a licence review and cannot prove that
nothing else is wrong."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from agent_review.resources import bundled, bundled_names, bundled_root

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ["src", "config", "profiles", "scenarios", "fixtures", "examples", "tests", ".github",
          "README.md", "CONTRIBUTING.md", "SECURITY.md", "pyproject.toml", ".gitignore"]
PRIVATE_EVIDENCE_NAMES = {"run.json", "raw_diff.json", "odoo.log", "odoo.conf", "standin.jsonl", "driver20.json"}

PATTERNS = {
    "a developer's absolute path": re.compile("/" + "Users/|/" + "home/[a-z]+/"),
    "an Enterprise build id": re.compile(r"\b\d{2}\.0\+e[.-]" + r"\d{8}\b"),
    "an internal section number": re.compile(chr(0xA7)),
    "an internal decision id": re.compile(r"\bdecision" + r" \d{2,3}\b"),
    "an Enterprise line citation": re.compile(r"\bai(?:_[a-z_]+)?/[\w/]+\.(?:py|xml|csv|js):" + r"\d+"),
    "a provider key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9]{32,}"),
}


def _public_files():
    for entry in PUBLIC:
        p = ROOT / entry
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from (f for f in sorted(p.rglob("*")) if f.is_file() and "__pycache__" not in f.parts)


def test_the_scan_sees_the_tree():
    files = [f.relative_to(ROOT).as_posix() for f in _public_files()]
    assert "src/agent_review/cli.py" in files and "README.md" in files and len(files) > 40     # positive control


@pytest.mark.parametrize("name", sorted(PATTERNS))
def test_no_public_file_carries(name):
    rx, hits = PATTERNS[name], []
    for f in _public_files():
        try:
            text = f.read_text()
        except UnicodeDecodeError:
            continue
        hits += [f"{f.relative_to(ROOT)}:{i}" for i, line in enumerate(text.splitlines(), 1) if rx.search(line)]
    assert not hits, f"{name}: {hits}"


def test_each_pattern_bites():
    """Negative controls: every pattern matches a sample of what it exists to stop."""
    samples = {"a developer's absolute path": "/" + "Users/someone/odoo",
               "an Enterprise build id": "19.0+e." + "20260101",
               "an internal section number": "see " + chr(0xA7) + "4", "an internal decision id": "decision" + " 111",
               "an Enterprise line citation": "ai/models/ai_tool.py:" + "12",
               "a provider key": "sk-" + "a1B2" * 9}
    assert set(samples) == set(PATTERNS)
    for name, sample in samples.items():
        assert PATTERNS[name].search(sample), name


def test_no_private_evidence_file_is_in_the_tree():
    assert not [f for f in _public_files() if f.name in PRIVATE_EVIDENCE_NAMES]


def test_bundled_data_resolves_and_loads():
    from agent_review.core.profile import load_run_profile, load_safety_profile
    from agent_review.core.scenario import load_scenario
    assert (bundled_root() / "config" / "classification.yaml").is_file()
    assert {"lead-from-prose", "reassign-opportunities", "noop"} <= set(bundled_names("scenarios"))
    for name in bundled_names("scenarios"):
        load_scenario(bundled("scenarios", f"{name}.yaml"))
    assert {"odoo19-example", "odoo20-standin-example", "noop"} <= set(bundled_names("profiles/run"))
    for name in bundled_names("profiles/run"):
        load_run_profile(name)
    for name in bundled_names("profiles/safety"):
        load_safety_profile(name)
    assert (bundled("fixtures", "odoo20") / "load_fixture.py").is_file()


def test_an_example_profile_is_refused_until_it_is_edited():
    """The examples carry placeholder paths; a run against one refuses before creating anything, and says why."""
    from agent_review.core.profile import ProfileError, load_run_profile
    from agent_review.drivers.native_ai_20.driver import NativeAi20Driver
    prof = load_run_profile("odoo20-standin-example")
    with pytest.raises(ProfileError, match="not an Odoo source tree"):
        NativeAi20Driver().preflight(prof, "dbname=agent_review_tpl20")


def test_the_committed_sample_report_is_what_the_code_renders():
    """The published sample is regenerated from the committed record, so it cannot drift from the report code."""
    import subprocess
    import sys
    script = ROOT / "examples" / "reassign-opportunities" / "make_sample_report.py"
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"}).stdout
    assert out == (script.parent / "sample-report.txt").read_text(), "regenerate sample-report.txt (see the example README)"
    assert out.startswith("SYNTHETIC EXAMPLE.") and "Not a run against Odoo" in out


def test_the_release_gate_note_is_cleared_before_publication():
    """The README opens with a release-gate note while the release candidate may not be published. It is removed
    by the owner, as a whole, only when every item in it is cleared; this reports it as a skip until then."""
    readme = (ROOT / "README.md").read_text()
    begin, end = "<!-- RELEASE-GATE:" + "BEGIN", "<!-- RELEASE-GATE:" + "END -->"
    if begin in readme:
        assert end in readme, "a partial release-gate note: remove the whole block, or none of it"
        pytest.skip("RELEASE GATE present in README.md: publication is blocked until the owner clears it")
    assert "NOT FOR " + "PUBLICATION" not in readme and "VERIFIED-" + "RESULTS" not in readme
