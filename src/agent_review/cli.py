"""agent-review CLI.

    agent-review run <scenario> --profile <run-profile> [--repeat N] [--probe] [--go] [--runs-dir DIR]
    agent-review review --profile <run-profile> --prompt "..." [--safety basic_write_agent] [--go]
    agent-review inspect --profile <run-profile>
    agent-review scenarios | profiles
    agent-review init-profile <example> [--output FILE]
    agent-review fixture-script <19|20>

Selection is explicit: the run profile names the driver (native_ai = Odoo 19, native_ai_20 = Odoo 20) and, for
Odoo 20, `transport: standin`. The target's Odoo version, the template's schema and the transport are checked
before anything is created or any spend is planned; a mismatch refuses, it never falls back.

No provider request happens before the credential gate: the key source, its last 4 characters and the
estimated planned spend are printed, and a paid run needs an explicit --go. The spend guard sums the ESTIMATED
cost recorded under one runs directory; it is an estimate-based guard, not a hard budget."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .core.cost import Pricing
from .core.credentials import Credential, CredentialError, load_credential, spend_plan_line
from .core.lifecycle import RunRecord, TeardownError, run_once
from .core.profile import ProfileError, load_run_profile, load_safety_profile
from .core.report import run_summary, suite_summary, write_suite
from .core.scenario import Scenario, ScenarioError, level1_scenario, load_scenario
from .drivers.base import Driver
from .drivers.native_ai import capabilities as caps_mod
from .drivers.native_ai.driver import NativeAiDriver
from .drivers.native_ai_20.capabilities import discover20
from .drivers.native_ai_20.driver import NativeAi20Driver
from .drivers.noop import NoopDriver
from .invariants.generic import load_invariants
from .resources import BundledDataMissing, bundled, bundled_names

DRIVERS: dict[str, type[Driver]] = {"native_ai": NativeAiDriver, "native_ai_20": NativeAi20Driver, "noop": NoopDriver}


def default_runs_dir() -> tuple[str, str]:
    """Where run directories go unless --runs-dir says otherwise: AGENT_REVIEW_RUNS, else a per-user data
    directory. Never inside the installed package."""
    env = os.environ.get("AGENT_REVIEW_RUNS")
    if env:
        return env, "AGENT_REVIEW_RUNS"
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    return os.path.join(base, "agent-review", "runs"), "default (per-user data directory)"


RUNS_ROOT, RUNS_SOURCE = default_runs_dir()


def _driver(name: str, probe: bool = False) -> Driver:
    if name not in DRIVERS:
        sys.exit(f"unknown driver {name!r}; known: {sorted(DRIVERS)}")
    d = DRIVERS[name]()
    d.probe = probe
    return d


def _find_scenario(name: str) -> Scenario:
    p = Path(name)
    if not p.exists():
        p = bundled("scenarios", f"{name}.yaml")
    if not p.exists():
        sys.exit(f"scenario not found: {name} (a path, or a bundled name: `agent-review scenarios`)")
    return load_scenario(p)


def _run_files(runs_root: str) -> list[str]:
    return glob.glob(os.path.join(runs_root, "*", "*", "run.json"))


def spent_so_far(runs_root: str) -> float:
    """Estimated spend recorded by every run under `runs_root`, ever. The profile's `spend_cap_usd` is therefore
    a cap on this directory's recorded estimates, not on one suite and not on the provider account; runs recorded
    in another directory are not counted. A run whose token usage was incomplete counts its lower bound
    (`at_least_usd`); a run with no cost measurement counts nothing, and `unmeasured_runs` says how many there are,
    so the figure is never read as the whole spend."""
    total = 0.0
    for f in _run_files(runs_root):
        try:
            with open(f) as fh:
                cost = json.load(fh).get("cost") or {}
            total += float(cost.get("usd") or cost.get("at_least_usd") or 0.0)
        except (OSError, ValueError):
            pass
    return total


def unmeasured_runs(runs_root: str) -> int:
    """Runs under `runs_root` whose recorded cost is not a complete measurement — unknown, or only a lower
    bound — leaving out probes and capability refusals, which call no provider."""
    n = 0
    for f in _run_files(runs_root):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        if r.get("probe") or r.get("status") == "refused_capability":
            continue
        n += (r.get("cost") or {}).get("usd") is None
    return n


def _gate(profile, repeat: int, go: bool, probe: bool = False) -> Credential | None:
    """The paid-call gate. Returns the credential, or exits before any run."""
    drv = DRIVERS.get(profile.driver)
    if probe or drv is None or not drv().is_paid():
        return None
    try:
        cred = load_credential(profile.provider.get("key_file"))
    except CredentialError as e:
        sys.exit(str(e))
    planning = profile.planning or {}
    est = planning.get("estimated_usd_per_run")
    src = planning.get("estimated_usd_source", "no estimate configured in the run profile (planning.estimated_usd_per_run)")
    cap = float(planning.get("spend_cap_usd", 0) or 0)
    spent = spent_so_far(RUNS_ROOT)
    recorded = len(_run_files(RUNS_ROOT))
    unmeasured = unmeasured_runs(RUNS_ROOT)
    print(spend_plan_line(cred, repeat, float(est) if est is not None else None, src, profile.provider_name, profile.model))
    print(f"SPEND GUARD (estimate-based, not a hard budget): ${spent:.4f} estimated spend recorded in {recorded} run(s) "
          f"under {RUNS_ROOT}"
          + (f" (a LOWER BOUND: {unmeasured} run(s) have no complete cost measurement)" if unmeasured else "")
          + (f" | cap ${cap:.2f}" if cap else " | NO spend cap configured"))
    print("  only runs recorded in that directory are counted"
          + ("; none are recorded there yet, so $0 is not a statement that nothing was spent elsewhere" if not recorded else ""))
    if cap and spent >= cap:
        sys.exit(f"STOP: spend cap reached ({spent:.4f} >= {cap:.2f}); no paid call made")
    if cap and est is not None and spent + repeat * float(est) > cap:
        sys.exit(f"STOP: planned spend {repeat} x ${float(est):.4f} + ${spent:.4f} spent would exceed the cap ${cap:.2f}; no paid call made")
    if not go:
        sys.exit("STOP: no paid call made. Re-run with --go once this plan has been read and approved.")
    return cred


def _template_dsn(profile, fixture: str) -> str:
    fx = profile.fixture
    hp = " ".join(p for p in [f"host={fx['host']}" if fx.get("host") else "", f"port={fx['port']}" if fx.get("port") else ""] if p)
    return f"dbname={profile.template_for(fixture)} {hp}".strip()


def _preflight(sc: Scenario, profile, probe: bool) -> None:
    """Driver/scenario/target compatibility, BEFORE the spend gate: a paid plan is never offered for a run that
    would refuse. The lifecycle checks again before it clones."""
    drv = _driver(profile.driver, probe)
    drv.check_scenario(sc)
    for note in drv.preflight(profile, _template_dsn(profile, sc.fixture)):
        print(f"preflight: {note}")


def _suite_dir(sc: Scenario, profile) -> str:
    """A directory no other invocation can share: second-resolution stamp, then the pid on collision."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = os.path.join(RUNS_ROOT, f"{sc.name}--{profile.name}--{stamp}")
    for candidate in (base, f"{base}-{os.getpid()}"):
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    sys.exit(f"suite directory collision: {base} and {base}-{os.getpid()} both exist")


def _suite(sc: Scenario, profile, repeat: int, cred: Credential | None, safety: list[str] | None, keep: bool,
           probe: bool = False) -> list[RunRecord]:
    suite_dir = _suite_dir(sc, profile)
    pricing = Pricing()
    invariants = load_invariants(profile.invariants)
    records: list[RunRecord] = []
    try:
        for i in range(1, repeat + 1):
            try:
                rec = run_once(sc, profile, _driver(profile.driver, probe), i, suite_dir, cred, invariants=invariants,
                               safety_names=safety, pricing=pricing, keep=keep)
            except TeardownError as e:
                # the record is on disk and joins the summary; the suite stops before cloning again
                records.append(e.record)
                print(run_summary(e.record))
                print()
                raise
            records.append(rec)
            print(run_summary(rec))
            print()
            if rec.status == "refused_capability":
                break   # the same target refuses every repeat identically
    finally:
        # whatever happened — Ctrl-C, a scenario error, a preflight refusal — the runs that did
        # complete are summarised, and an incomplete suite says so
        text, data = suite_summary(records, sc.name, profile.name, sc.level)
        if len(records) < repeat and not any(r.status == "refused_capability" for r in records):
            text = f"** INCOMPLETE SUITE: {len(records)} of {repeat} planned runs recorded **\n\n" + text
            data["incomplete"] = True
        write_suite(suite_dir, records, text, data)
        print(text)
        print(f"\nartifacts: {suite_dir}\n"
              f"  redacted report — review before sharing outside your team (business-field values remain): "
              f"summary.txt, summary.json\n"
              f"  PRIVATE evidence (unredacted, never share): NNN/run.json, NNN/raw_diff.json and every other file under NNN/")
    return records


def cmd_run(a: argparse.Namespace) -> None:
    sc = _find_scenario(a.scenario)
    profile = load_run_profile(a.profile)
    repeat = a.repeat or profile.repeat
    _preflight(sc, profile, a.probe)
    cred = _gate(profile, repeat, a.go, a.probe)
    _suite(sc, profile, repeat, cred, a.safety or None, a.keep, a.probe)


def cmd_review(a: argparse.Namespace) -> None:
    """Level 1: agent + one representative task prompt + a safety profile. Zero assertion authoring."""
    profile = load_run_profile(a.profile)
    safety = a.safety or ["basic_write_agent"]
    for s in safety:
        load_safety_profile(s)
    sc = level1_scenario(a.name or "level1-review", a.prompt, a.continuation, safety, a.fixture)
    repeat = a.repeat or profile.repeat
    _preflight(sc, profile, a.probe)
    cred = _gate(profile, repeat, a.go, a.probe)
    _suite(sc, profile, repeat, cred, safety, a.keep, a.probe)


def cmd_inspect(a: argparse.Namespace) -> None:
    profile = load_run_profile(a.profile)
    dsn = _template_dsn(profile, a.fixture)
    drv = _driver(profile.driver)
    for note in drv.preflight(profile, dsn):
        print(f"preflight: {note}")
    agent_cfg = profile.agent(a.agent)
    if profile.driver == "native_ai_20":
        caps = discover20(dsn, profile.target.get("odoo_root"))
    elif profile.driver == "native_ai":
        caps = caps_mod.discover(dsn, profile.target.get("odoo_root"))
    else:
        sys.exit(f"driver {profile.driver} has nothing to inspect")
    # the agent's name, which is what the tool inventory is keyed on: a role configured by xml_id is looked up, or
    # every tool would be shown as reachable
    agent = agent_cfg.get("name") or _agent_name(dsn, agent_cfg.get("xml_id"))
    if agent is None:
        sys.exit(f"agent role {a.agent!r}: no agent named by `name` or `xml_id` in {profile.template_for(a.fixture)}")
    print(f"profile {profile.name} · fixture {profile.template_for(a.fixture)} · driver {profile.driver} · "
          f"transport {profile.transport} · agent role {a.agent!r} = {agent!r} · model {profile.model} (run profile)")
    print(caps_mod.render(caps, agent))
    if a.json:
        from .core.contracts import to_dict
        print(json.dumps(to_dict(caps), indent=1))


def _agent_name(dsn: str, xml_id: str | None) -> str | None:
    if not xml_id:
        return None
    import psycopg
    mod, _, nm = xml_id.partition(".")
    with psycopg.connect(dsn) as c:
        row = c.execute("select p.name from ir_model_data d join ai_agent a on a.id = d.res_id "
                        "join res_partner p on p.id = a.partner_id where d.model = 'ai.agent' and d.module = %s "
                        "and d.name = %s", (mod, nm)).fetchone()
    return row[0] if row else None


def cmd_scenarios(a: argparse.Namespace) -> None:
    for name in bundled_names("scenarios"):
        p = bundled("scenarios", f"{name}.yaml")
        try:
            sc = load_scenario(p)
            odoo = ",".join(sc.required_odoo) or "any"
            print(f"{sc.name:<34} L{sc.level} {sc.kind or '-':<7} odoo {odoo:<6} requires {sc.required_capabilities} "
                  f"agent={sc.required_agent}  {sc.title or ''}")
        except ScenarioError as e:
            print(f"{p.name:<34} INVALID: {e}")


def cmd_profiles(a: argparse.Namespace) -> None:
    print("run profiles (examples: copy one with `agent-review init-profile <name>` and edit it):")
    for name in bundled_names("profiles/run"):
        raw = yaml.safe_load(bundled("profiles", "run", f"{name}.yaml").read_text()) or {}
        print(f"  {name:<24} driver {raw.get('driver', 'native_ai'):<13} transport {raw.get('transport', '-'):<8} "
              f"{(raw.get('description') or '').strip()}")
    print("safety profiles:")
    for name in bundled_names("profiles/safety"):
        sp = load_safety_profile(name)
        print(f"  {name:<24} {len(sp.rules)} rule(s)  {sp.description.strip()[:90]}")


def cmd_init_profile(a: argparse.Namespace) -> None:
    src = bundled("profiles", "run", f"{a.example}.yaml")
    if not src.is_file():
        sys.exit(f"no bundled example profile {a.example!r}; see `agent-review profiles`")
    out = Path(a.output or f"{a.example}.yaml")
    if out.exists():
        sys.exit(f"{out} exists; not overwritten")
    shutil.copyfile(src, out)
    driver = (yaml.safe_load(src.read_text()) or {}).get("driver")
    after = f"agent-review run noop --profile {out}" if driver == "noop" else f"agent-review inspect --profile {out}"
    print(f"wrote {out}: set every value marked EDIT, then run `{after}`")


def cmd_fixture_script(a: argparse.Namespace) -> None:
    d = bundled("fixtures", f"odoo{a.version}")
    if not d.is_dir():
        sys.exit(f"no bundled fixture for Odoo {a.version}; available: 19, 20")
    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix != ".pyc":
            print(f)
    # the hint goes to stderr, so stdout is only the paths: `odoo shell … < "$(agent-review fixture-script 20)"`
    print(f"see {bundled('fixtures', 'README.md')} for the preparation steps", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    global RUNS_ROOT, RUNS_SOURCE
    ap = argparse.ArgumentParser(prog="agent-review",
                                 description="Compare an Odoo AI agent's response with observed database changes.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    probe_help = ("exercise the plumbing without any provider request (Odoo 19: post the prompt only; Odoo 20: the local "
                  "stand-in answers every round with a canned reply). Never a measurement")
    runs_help = f"where run directories are written (default: $AGENT_REVIEW_RUNS, else {default_runs_dir()[0]})"
    r = sub.add_parser("run", help="run a scenario against a run profile")
    r.add_argument("scenario"); r.add_argument("--profile", required=True); r.add_argument("--repeat", type=int)
    r.add_argument("--go", action="store_true", help="explicit approval of the printed spend plan"); r.add_argument("--safety", action="append")
    r.add_argument("--keep", action="store_true", help="keep the run database (debugging only)")
    r.add_argument("--probe", action="store_true", help=probe_help)
    r.add_argument("--runs-dir", help=runs_help)
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("review", help="Level 1 safety review: one prompt, a safety profile, no assertions")
    v.add_argument("--profile", required=True); v.add_argument("--prompt", required=True)
    v.add_argument("--continuation", help="a follow-up message; on Odoo 20 also confirmation:<confirm_once|auto_confirm|decline>")
    v.add_argument("--safety", action="append"); v.add_argument("--repeat", type=int); v.add_argument("--go", action="store_true")
    v.add_argument("--name"); v.add_argument("--fixture", default="default"); v.add_argument("--keep", action="store_true")
    v.add_argument("--probe", action="store_true", help=probe_help)
    v.add_argument("--runs-dir", help=runs_help)
    v.set_defaults(fn=cmd_review)
    i = sub.add_parser("inspect", help="check the target and discover what its agent exposes")
    i.add_argument("--profile", required=True); i.add_argument("--fixture", default="default"); i.add_argument("--agent", default="default")
    i.add_argument("--json", action="store_true")
    i.set_defaults(fn=cmd_inspect)
    s = sub.add_parser("scenarios", help="list bundled scenarios"); s.set_defaults(fn=cmd_scenarios)
    pr = sub.add_parser("profiles", help="list bundled example run profiles and safety profiles"); pr.set_defaults(fn=cmd_profiles)
    ip = sub.add_parser("init-profile", help="copy a bundled example run profile for editing")
    ip.add_argument("example"); ip.add_argument("--output")
    ip.set_defaults(fn=cmd_init_profile)
    fs = sub.add_parser("fixture-script", help="print the bundled synthetic fixture script(s) for an Odoo version")
    fs.add_argument("version", choices=["19", "20"])
    fs.set_defaults(fn=cmd_fixture_script)
    a = ap.parse_args(argv)
    if getattr(a, "runs_dir", None):
        RUNS_ROOT, RUNS_SOURCE = a.runs_dir, "--runs-dir"
    # SIGTERM would skip every `finally` and leave a clone behind; raise it as an interrupt instead so
    # the run tears down like Ctrl-C does.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    # a teardown that failed while another error was propagating rides on that error as a note: print it,
    # or the leaked database would go unmentioned
    try:
        a.fn(a)
    except (ScenarioError, ProfileError, TeardownError, BundledDataMissing) as e:
        sys.exit("\n".join([f"error: {e}", *getattr(e, "__notes__", [])]))
    except KeyboardInterrupt as e:
        sys.exit("\n".join([("interrupted: the suite summary is marked incomplete; each run's `teardown` records whether "
                             "its database was dropped"), *getattr(e, "__notes__", [])]))


if __name__ == "__main__":
    main()
