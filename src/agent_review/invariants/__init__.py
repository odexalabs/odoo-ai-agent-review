"""Invariant checking. The INTERFACE, the identity-based differential, and two generic
invariants with provenance (`generic.py`). Other packages' checks plug in through an adapter on
their side (`package.module:attr` in the run profile); this package imports nothing of theirs.

Every invariant OWNS its `compare(before, after)` and returns stable violation IDENTITIES plus an
optional severity measure. Classification is by identity, never by count:

    NEW           identity appears only after
    RESOLVED      identity existed only before
    PRE_EXISTING  same identity, equivalent state
    WORSENED      same identity, severity increased by that invariant's own measure

Two baseline modes: benchmark (baseline MUST be clean — refuse otherwise) and customer (record
pre-existing failures, report differentially). Every invariant carries provenance; one without
provenance does not ship."""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Violation:
    identity: str                 # stable across runs on the same fixture
    severity: float | None = None
    detail: str = ""


@dataclass
class InvariantResult:
    invariant: str
    provenance: str
    scope: str                    # scoped | global
    violations: list[Violation]
    runtime_s: float | None = None
    skipped: str | None = None    # set when the check could not run (e.g. the model is not installed): never "clean"


class Invariant(abc.ABC):
    name: str
    provenance: str               # Odoo SQL constraint | Python constraint | observed source logic | published repair script
    scope: str = "global"

    @abc.abstractmethod
    def check(self, observer_dsn: str, changed_ids: dict[str, set[Any]] | None) -> InvariantResult: ...

    def compare(self, before: InvariantResult, after: InvariantResult) -> dict[str, list[Violation]]:
        b = {v.identity: v for v in before.violations}
        a = {v.identity: v for v in after.violations}
        out: dict[str, list[Violation]] = {"new": [], "resolved": [], "pre_existing": [], "worsened": []}
        for ident, v in a.items():
            if ident not in b:
                out["new"].append(v)
            elif v.severity is not None and b[ident].severity is not None and v.severity > b[ident].severity:
                out["worsened"].append(v)
            else:
                out["pre_existing"].append(v)
        for ident, v in b.items():
            if ident not in a:
                out["resolved"].append(v)
        return out


@dataclass
class InvariantReport:
    configured: list[str] = field(default_factory=list)
    baseline_clean: bool | None = None
    by_invariant: dict[str, dict[str, list[dict]]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)      # invariant -> why it could not be evaluated
    note: str = "no invariants configured; pass generic names or package.module:attr adapters in the run profile"

    @property
    def new_count(self) -> int:
        return sum(len(v.get("new", [])) for v in self.by_invariant.values())

    @property
    def worsened_count(self) -> int:
        return sum(len(v.get("worsened", [])) for v in self.by_invariant.values())


def run_invariants(invariants: list[Invariant], observer_dsn: str, mode: str) -> tuple[list[InvariantResult], bool]:
    results = [inv.check(observer_dsn, None) for inv in invariants]
    clean = all(not r.violations for r in results)
    if mode == "benchmark" and not clean:
        raise RuntimeError("benchmark mode requires a clean invariant baseline; refusing to run: "
                           + "; ".join(f"{r.invariant}: {len(r.violations)}" for r in results if r.violations))
    return results, clean


def differential(invariants: list[Invariant], before: list[InvariantResult], after: list[InvariantResult]) -> InvariantReport:
    rep = InvariantReport(configured=[i.name for i in invariants])
    rep.baseline_clean = all(not r.violations for r in before)
    for inv, b, a in zip(invariants, before, after):
        if b.skipped or a.skipped:
            rep.skipped[inv.name] = a.skipped or b.skipped or ""
            continue
        cmp = inv.compare(b, a)
        rep.by_invariant[inv.name] = {k: [{"identity": v.identity, "severity": v.severity, "detail": v.detail} for v in vs]
                                      for k, vs in cmp.items()}
    if invariants:
        rep.note = f"{len(invariants)} invariant(s) checked before and after; classified by violation identity"
    return rep
