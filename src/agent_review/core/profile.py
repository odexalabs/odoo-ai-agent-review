"""Run profiles (WHERE and WITH WHAT a scenario executes) and safety profiles (what must not
happen). Model identity lives here, never in a scenario.

A bare name (`--profile odoo19-example`) is looked up among the bundled profiles; a path is used as given."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..resources import bundled
from .scenario import _rules

TRANSPORTS = {
    # driver -> the transports it supports. Selection is explicit and never falls back to another.
    "native_ai": ("direct",),        # Odoo 19: the Odoo process calls the provider with the named key
    "native_ai_20": ("standin",),     # Odoo 20: a per-run local stand-in answers Odoo's AI endpoint
    "noop": ("none",),
}


class ProfileError(ValueError):
    pass


@dataclass
class RunProfile:
    name: str
    driver: str
    target: dict[str, Any]
    fixture: dict[str, Any]
    agents: dict[str, dict[str, Any]]          # role -> agent config; "default" required
    provider: dict[str, Any]
    planning: dict[str, Any] = field(default_factory=dict)
    egress: dict[str, Any] = field(default_factory=dict)
    mode: str = "customer"                     # customer | benchmark
    repeat: int = 1
    invariants: list[str] = field(default_factory=list)   # generic names or package.module:attr adapters
    source_path: str | None = None
    transport: str | None = None               # how the agent's model requests travel; see TRANSPORTS
    # Exact messages THIS target shows its users, per verdict (the scenario `report:` shape). The tool ships none of
    # Odoo's own wording: it varies by version and language, so the operator supplies it for their deployment.
    report_rules: list = field(default_factory=list)
    standin: dict[str, Any] = field(default_factory=dict)  # Odoo 20 stand-in settings (turn timeout)

    @property
    def model(self) -> str:
        return str(self.provider.get("model", ""))

    @property
    def provider_name(self) -> str:
        return str(self.provider.get("name", ""))

    def agent(self, role: str = "default") -> dict[str, Any]:
        if role not in self.agents:
            raise ProfileError(f"profile {self.name}: no agent for role {role!r} (have {sorted(self.agents)})")
        return self.agents[role]

    def template_for(self, fixture_name: str) -> str:
        t = (self.fixture.get("templates") or {}).get(fixture_name)
        if not t:
            raise ProfileError(f"profile {self.name}: fixture {fixture_name!r} is not mapped in fixture.templates")
        return t


def load_run_profile(name_or_path: str) -> RunProfile:
    p = Path(name_or_path)
    if not p.exists():
        p = bundled("profiles", "run", f"{name_or_path}.yaml")
    if not p.exists():
        raise ProfileError(f"run profile not found: {name_or_path} (a path, or a bundled name: `agent-review profiles`)")
    with open(p) as fh:
        raw = yaml.safe_load(fh) or {}
    agents = raw.get("agents") or ({"default": raw["agent"]} if raw.get("agent") else {})
    if "default" not in agents:
        raise ProfileError(f"{p}: profile needs an agent (agent: {{...}} or agents: {{default: {{...}}}})")
    if raw.get("mode", "customer") not in ("customer", "benchmark"):
        raise ProfileError(f"{p}: mode must be customer | benchmark")
    driver = raw.get("driver", "native_ai")
    transport = raw.get("transport")
    if driver in TRANSPORTS:
        allowed = TRANSPORTS[driver]
        if transport is None and len(allowed) == 1 and driver != "native_ai_20":
            transport = allowed[0]        # the driver has exactly one way to reach a model; nothing to choose
        if transport not in allowed:
            hosted = " Odoo's hosted AI service (IAP) is not supported by this tool." if driver == "native_ai_20" else ""
            raise ProfileError(f"{p}: driver {driver} needs `transport: {' | '.join(allowed)}` (got {transport!r}).{hosted}")
    return RunProfile(
        name=raw.get("name") or p.stem, driver=driver, target=raw.get("target", {}) or {},
        fixture=raw.get("fixture", {}) or {}, agents=agents, provider=raw.get("provider", {}) or {},
        planning=raw.get("planning", {}) or {}, egress=raw.get("egress", {}) or {}, mode=raw.get("mode", "customer"),
        repeat=int(raw.get("repeat", 1)), invariants=list(raw.get("invariants", []) or []), source_path=str(p),
        transport=transport, standin=raw.get("standin", {}) or {},
        report_rules=[dataclasses.replace(r, source="profile") for r in _rules(raw.get("report_rules"), 2, str(p))],
    )


@dataclass
class SafetyRule:
    id: str
    rule: str
    rationale: str
    observable_via: str = "db_diff"        # db_diff | tool_trace
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyProfile:
    name: str
    description: str
    rules: list[SafetyRule]
    extends: list[str] = field(default_factory=list)


def load_safety_profile(name_or_path: str, _seen: set[str] | None = None) -> SafetyProfile:
    seen = _seen or set()
    p = Path(name_or_path)
    if not p.exists():
        p = bundled("profiles", "safety", f"{name_or_path}.yaml")
    if not p.exists():
        raise ProfileError(f"safety profile not found: {name_or_path}")
    with open(p) as fh:
        raw = yaml.safe_load(fh) or {}
    name = raw.get("name") or p.stem
    if name in seen:
        raise ProfileError(f"safety profile cycle at {name}")
    seen.add(name)
    rules: list[SafetyRule] = []
    for parent in raw.get("extends", []) or []:
        rules += load_safety_profile(parent, seen).rules
    for r in raw.get("rules", []) or []:
        if not r.get("rationale"):
            raise ProfileError(f"{p}: rule {r.get('id')!r} has no rationale; a rule with no rationale does not ship")
        params = {k: v for k, v in r.items() if k not in ("id", "rule", "rationale", "observable_via")}
        rules.append(SafetyRule(r["id"], r["rule"], r["rationale"], r.get("observable_via", "db_diff"), params))
    return SafetyProfile(name, raw.get("description", ""), rules, list(raw.get("extends", []) or []))
