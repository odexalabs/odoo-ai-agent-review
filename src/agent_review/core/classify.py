"""Diff classification. MODE-DEPENDENT.

  LEVEL 1 (no expected state)   business writes observed · safety-profile violations ·
                                AI/session bookkeeping · ordinary Odoo bookkeeping
  LEVEL 2 (expected state)      expected business writes · unexpected business writes ·
                                AI/session bookkeeping · ordinary Odoo bookkeeping

The expected/unexpected split is a function of whether a scenario supplied expected state — an
`expected_matcher` — not a hardcoded four-way partition. Level 1 must never label a write
"unexpected": that would flag a correctly reassigned lead as an alarm on the user's first run.

Two levels are retained: the raw table/column view (RowChange on tables) and a normalised
model/id/field view derived from it. Relation tables and some internals do not map onto
model + record id, so the normalised view is never the only one.
"""
from __future__ import annotations

import fnmatch
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..resources import bundled
from .contracts import ChangeEvidence, Coverage, RowChange, TableSummary
from .redact import RedactionRules

DEFAULT_CONFIG = bundled("config", "classification.yaml")


@dataclass
class ClassificationRules:
    business_tables: list[str]
    ai_session_tables: list[str]
    bookkeeping_labels: dict[str, str]
    model_overrides: dict[str, str]

    @classmethod
    def load(cls, path: Path | None = None, extra_business: list[str] | None = None) -> ClassificationRules:
        with open(path or DEFAULT_CONFIG) as fh:
            raw = yaml.safe_load(fh) or {}
        return cls(
            business_tables=list(raw.get("business_tables", [])) + list(extra_business or []),
            ai_session_tables=list(raw.get("ai_session_tables", [])),
            bookkeeping_labels=dict(raw.get("bookkeeping_labels", {})),
            model_overrides=dict(raw.get("model_overrides", {})),
        )

    def bucket(self, table: str) -> str:
        if table in self.business_tables:
            return "business"
        if any(fnmatch.fnmatch(table, p) for p in self.ai_session_tables):
            return "ai_session"
        return "bookkeeping"

    def label(self, table: str) -> str | None:
        for p, lab in self.bookkeeping_labels.items():
            if fnmatch.fnmatch(table, p):
                return lab
        return None


@dataclass
class NormalisedChange:
    """model/id/field view. `model` is None when the table maps onto no Odoo model."""

    table: str
    model: str | None
    record_id: Any
    kind: str
    fields: dict[str, list[Any]]      # field -> [before, after]; for added/removed: [None, v] / [v, None]
    expected: bool | None = None      # Level 2 only

    def display_name(self) -> str | None:
        vals = {f: v[1] if v[1] is not None else v[0] for f, v in self.fields.items()}
        for k in ("name", "display_name", "complete_name", "login", "key"):
            if k in vals and isinstance(vals[k], str):
                return vals[k]
        return None


@dataclass
class Classification:
    level: int
    business_writes: list[NormalisedChange]                 # Level 1: all; Level 2: all, each flagged
    expected_business_writes: list[NormalisedChange]        # Level 2 only
    unexpected_business_writes: list[NormalisedChange]      # Level 2 only
    ai_session: list[TableSummary]
    bookkeeping: list[TableSummary]
    table_only_business: list[TableSummary]                 # business tables the detector could not cover exactly
    coverage_note: str
    safety_violations: list[dict] = field(default_factory=list)   # filled by the safety module

    @property
    def business_write_count(self) -> int:
        return len(self.business_writes)


def table_to_model(table: str, rules: ClassificationRules, known_models: set[str] | None) -> str | None:
    if table in rules.model_overrides:
        return rules.model_overrides[table]
    candidate = table.replace("_", ".")
    if known_models is None:
        return candidate if table in rules.business_tables else None
    return candidate if candidate in known_models else None


def normalise(rc: RowChange, rules: ClassificationRules, known_models: set[str] | None, redaction: RedactionRules) -> NormalisedChange:
    if rc.kind == "added":
        after = redaction.redact_row(rc.table, rc.after) or {}
        fields = {k: [None, v] for k, v in after.items()}
    elif rc.kind == "removed":
        before = redaction.redact_row(rc.table, rc.before) or {}
        fields = {k: [v, None] for k, v in before.items()}
    else:
        fields = redaction.redact_changed_fields(rc.table, rc.changed_fields, rc.after)
    return NormalisedChange(rc.table, table_to_model(rc.table, rules, known_models), rc.pk, rc.kind, fields)


def classify(
    evidence: ChangeEvidence,
    rules: ClassificationRules,
    redaction: RedactionRules,
    known_models: set[str] | None = None,
    expected_matcher: Callable[[RowChange], bool] | None = None,
) -> Classification:
    level = 2 if expected_matcher is not None else 1
    business: list[NormalisedChange] = []
    expected: list[NormalisedChange] = []
    unexpected: list[NormalisedChange] = []
    for rc in evidence.row_changes:
        if rules.bucket(rc.table) != "business":
            continue
        n = normalise(rc, rules, known_models, redaction)
        if level == 2:
            n.expected = bool(expected_matcher(rc))
            (expected if n.expected else unexpected).append(n)
        business.append(n)
    ai_session: list[TableSummary] = []
    bookkeeping: list[TableSummary] = []
    table_only_business: list[TableSummary] = []
    for ts in evidence.tables_touched:
        b = rules.bucket(ts.table)
        if b == "business":
            if ts.coverage != Coverage.EXACT:
                table_only_business.append(ts)
        elif b == "ai_session":
            ai_session.append(ts)
        else:
            bookkeeping.append(ts)
    exact = evidence.tables_at(Coverage.EXACT)
    note = (
        f"{evidence.detector}: {len(exact)} tables at EXACT coverage (row-level before/after), "
        f"{len(evidence.coverage_by_table) - len(exact)} at TABLE_ONLY (count and max(write_date) only; "
        "a delete+insert on a table without write_date is invisible there)."
    )
    return Classification(level, business, expected, unexpected, ai_session, bookkeeping, table_only_business, note)
