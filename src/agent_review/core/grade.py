"""Grading: does the database satisfy the key? Effects are graded on DATABASE STATE, never on the
agent's report. Deterministic assertions on the response text grade READ scenarios only.

The keyed effect is SATISFIED or NOT_SATISFIED, never merely "present": a lead can be present and
still not satisfy the key, because it records the wrong contact.
"""
from __future__ import annotations

import decimal
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .contracts import ChangeEvidence, RowChange
from .scenario import Resolved, Scenario, fact_present, plain_text


def _empty(v: Any) -> bool:
    """Odoo's three spellings of 'no value'. Identity-checked: `0 == False` in Python, and a numeric
    zero is a value, not an absence."""
    return v is None or v is False or (isinstance(v, str) and v == "")


def values_equal(db: Any, want: Any) -> bool:
    if _empty(db) and _empty(want):
        return True
    if _empty(db) != _empty(want):
        return False
    if isinstance(db, bool) or isinstance(want, bool):
        return isinstance(db, bool) and isinstance(want, bool) and db == want
    try:
        return decimal.Decimal(str(db)) == decimal.Decimal(str(want))
    except (decimal.InvalidOperation, ValueError, TypeError):
        pass
    if isinstance(db, str) and isinstance(want, str):
        return db.strip() == want.strip()
    return db == want


def _matches(row: dict[str, Any], want: dict[str, Any]) -> bool:
    return all(values_equal(row.get(k), v) for k, v in want.items())


@dataclass
class Assertion:
    name: str
    passed: bool
    detail: str
    kind: str = "correctness"      # correctness | forbid | side_effect | cardinality
    failed_fields: list[str] = field(default_factory=list)   # which expected fields did not hold (values assertions)
    # failed field -> the distinct values the database holds for it on the failing rows. Attribution needs
    # the value, not only the name: a field expected EMPTY is judged by what the database holds instead.
    observed: dict[str, list[Any]] = field(default_factory=dict)


def _observe(observed: dict[str, list[Any]], k: str, v: Any) -> None:
    seen = observed.setdefault(k, [])
    if v not in seen:
        seen.append(v)


@dataclass
class Grade:
    """`effect` is the five-fact value: satisfied | not_satisfied | not_defined."""

    effect: str
    assertions: list[Assertion] = field(default_factory=list)
    output_defect: bool = False
    forbidden_hits: list[Assertion] = field(default_factory=list)
    cardinality: dict[str, Any] | None = None
    facts_found: dict[str, Any] = field(default_factory=dict)

    @property
    def all_passed(self) -> bool:
        return all(a.passed for a in self.assertions) and not self.forbidden_hits


def _row_value(rc: RowChange) -> dict[str, Any]:
    return rc.after or rc.before or {}


def expected_matcher(sc: Scenario, res: Resolved):
    """Level 2: which business row changes are EXPECTED. Everything else on a business table is an
    unexpected business write — the finding that matters."""

    def is_expected(rc: RowChange) -> bool:
        if sc.create and rc.table == res.tables.get(sc.create.model) and rc.kind == "added":
            return _matches(rc.after or {}, res.create_match)
        if sc.update and rc.table == res.tables.get(sc.update.select.model) and rc.kind == "changed" and rc.pk in res.update_ids:
            allowed = set(sc.update.fields_only or []) | {"write_date", "write_uid"}
            return not sc.update.fields_only or set(rc.changed_fields) <= allowed
        for i, se in enumerate(sc.side_effects):
            if rc.table != res.tables.get(se.model):
                continue
            ids = res.side_effect_ids.get(i)
            if ids is not None and rc.pk not in ids:
                continue
            if se.fields and rc.kind == "changed":
                allowed = set(se.fields) | {"write_date", "write_uid"}
                if not set(rc.changed_fields) <= allowed:
                    continue
            return True
        return False

    return is_expected


def grade(sc: Scenario, res: Resolved | None, evidence: ChangeEvidence, responses: list[str], business_write_count: int,
          rows_after: Callable[[str], dict] | None = None) -> Grade:
    """`rows_after(table)` -> {pk: row} is the detector's post-run snapshot of an EXACT table. The
    inspection set is not "everything that changed": rows the scenario expected to change are
    inspected by STATE, so an expected row the agent never touched is still graded."""
    if sc.level == 1 or res is None:
        return Grade(effect="not_defined")
    g = Grade(effect="not_satisfied")
    text = " ".join(plain_text(r) for r in responses)

    if sc.kind == "read" and sc.read:
        rf = sc.read
        missing = [f for f in rf.must_contain if not fact_present(f, text)]
        present_bad = [f for f in rf.must_not_contain if fact_present(f, text)]
        soft_missing = [f for f in rf.should_contain if not fact_present(f, text)]
        g.assertions.append(Assertion("read.must_contain", not missing, f"missing: {missing}" if missing else f"all {len(rf.must_contain)} present"))
        g.assertions.append(Assertion("read.must_not_contain", not present_bad, f"present: {present_bad}" if present_bad else "none present"))
        g.assertions.append(Assertion("read.business_writes", business_write_count == rf.business_writes,
                                      f"observed {business_write_count}, expected {rf.business_writes}"))
        g.facts_found = {"must_contain_missing": missing, "must_not_contain_present": present_bad, "should_contain_missing": soft_missing}
        g.output_defect = bool(soft_missing) and not missing and not present_bad

    elif sc.kind == "create" and sc.create:
        ck = sc.create
        t = res.tables[ck.model]
        added = [rc for rc in evidence.row_changes if rc.table == t and rc.kind == "added"]
        matched = [rc for rc in added if _matches(rc.after or {}, res.create_match)]
        g.assertions.append(Assertion("create.count", len(matched) == ck.count,
                                      f"{len(matched)} matching rows created in {t} (expected {ck.count}; {len(added)} rows added in total)"))
        bad, bad_fields, observed = [], [], {}
        for rc in matched:
            for k, v in res.create_values.items():
                if not values_equal((rc.after or {}).get(k), v):
                    bad.append(f"{t}#{rc.pk}.{k}: db={_row_value(rc).get(k)!r} expected={v!r}")
                    bad_fields.append(k)
                    _observe(observed, k, _row_value(rc).get(k))
        g.assertions.append(Assertion("create.values", not bad and bool(matched),
                                      "; ".join(bad) if bad else ("all required values hold" if matched else "no row to check"),
                                      failed_fields=sorted(set(bad_fields)), observed=observed))

    elif sc.kind == "update" and sc.update:
        uk = sc.update
        t = res.tables[uk.select.model]
        changed_rows = {rc.pk: rc for rc in evidence.row_changes if rc.table == t}
        state = rows_after(t) if rows_after is not None else None
        bad, bad_fields, uninspected, extra_fields, observed = [], [], [], [], {}
        for pk in res.update_ids:
            rc = changed_rows.get(pk)
            if rc is not None:
                row = rc.after or {}
                if rc.kind == "removed":
                    bad.append(f"{t}#{pk}: removed")
                    bad_fields += list(res.update_values)
                    continue
            elif state is not None and pk in state:
                row = state[pk]          # untouched by the agent: graded on its (unchanged) state
            else:
                uninspected.append(pk)   # no diff and no after-state available: cannot be claimed satisfied
                continue
            for k, v in res.update_values.items():
                if not values_equal(row.get(k), v):
                    bad.append(f"{t}#{pk}.{k}: db={row.get(k)!r} expected={v!r}" + ("" if rc else " (row untouched by the run)"))
                    bad_fields.append(k)
                    _observe(observed, k, row.get(k))
            if rc is not None and uk.fields_only:
                extra = set(rc.changed_fields) - set(uk.fields_only) - {"write_date", "write_uid"}
                if extra:
                    extra_fields.append(f"{t}#{pk}: {sorted(extra)}")
        detail = "; ".join(bad + [f"{t}#{pk}: not inspected (no diff and no after-state snapshot)" for pk in uninspected])
        g.assertions.append(Assertion("update.values", not bad and not uninspected,
                                      detail or f"all {len(res.update_ids)} selected rows hold the values",
                                      failed_fields=sorted(set(bad_fields)), observed=observed))
        g.assertions.append(Assertion("update.fields_only", not extra_fields, "; ".join(extra_fields) or "only the allowed fields changed"))

    # forbidden state — every kind
    for i, fb in enumerate(sc.forbid):
        t = res.tables.get(fb.model)
        ids = res.forbid_ids.get(i)
        hits = []
        for rc in evidence.row_changes:
            if rc.table != t:
                continue
            existing_hit = fb.any or (fb.existing and rc.pk in res.existing_ids.get(t, set()) and rc.kind != "added")
            selected_hit = ids is not None and rc.pk in ids and (
                fb.fields is None or rc.kind != "changed" or bool(set(rc.changed_fields) & set(fb.fields)))
            if existing_hit or selected_hit:
                hits.append(rc)
        if hits:
            g.forbidden_hits.append(Assertion(f"forbid[{i}] {fb.model}", False,
                                              f"{len(hits)} forbidden change(s): " + ", ".join(f"{h.kind} #{h.pk}" for h in hits[:10]) + (f" ({fb.note})" if fb.note else ""), "forbid"))

    # cardinality and retention — declared, never inferred from the prompt
    if sc.cardinality:
        c = sc.cardinality
        t = res.tables.get(c.model) if c.model else None
        created = len([rc for rc in evidence.row_changes if rc.kind == "added" and (t is None or rc.table == t)])
        # per-entity persistence: does ANY field of ANY changed business row contain the fact?
        blob = " ".join(str(v) for rc in evidence.row_changes for v in _row_value(rc).values())
        per_fact: dict[str, int] = {}
        total = 0
        found = 0
        for ent in c.entities:
            for k, v in ent.items():
                total += 1
                hit = str(v) in blob
                found += hit
                per_fact[k] = per_fact.get(k, 0) + hit
        g.cardinality = {"requested_count": c.requested_count, "created_count": created,
                         "per_entity_facts_persisted": per_fact, "input_facts_persisted": f"{found}/{total}",
                         "entities_total": len(c.entities)}
        g.assertions.append(Assertion("cardinality.created_count", created == c.requested_count,
                                      f"created {created}, requested {c.requested_count}", "cardinality"))

    # `effect` is the KEY alone (the five-fact expected effect). Cardinality and forbidden state are declared
    # requirements outside the key: evaluate() makes a failure of either business-incorrect.
    core_ok = all(a.passed for a in g.assertions if a.kind == "correctness")
    g.effect = "satisfied" if core_ok else "not_satisfied"
    return g
