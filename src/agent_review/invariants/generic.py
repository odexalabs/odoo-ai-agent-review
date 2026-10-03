"""Generic invariants shipped with agent-review. Each is a binary predicate with a provenance
citation into the Odoo 19 tree, stable violation identities, and no threshold.

Other packages can expose their own checks TO this interface through an adapter on their side (see
`load_invariants`); this package imports nothing of theirs.
"""
from __future__ import annotations

import importlib
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row

from . import Invariant, InvariantResult, Violation


class SqlInvariant(Invariant):
    """One SELECT returning (identity, severity, detail) rows; each row is a violation."""

    sql: str = ""
    scope = "global"

    def check(self, observer_dsn: str, changed_ids: dict[str, set[Any]] | None) -> InvariantResult:
        t0 = time.perf_counter()
        try:
            with psycopg.connect(observer_dsn, row_factory=dict_row) as c:
                rows = c.execute(self.sql).fetchall()
        except psycopg.errors.UndefinedTable as e:
            # the model is not installed in this fixture: NOT EVALUABLE, which is not the same as clean
            return InvariantResult(self.name, self.provenance, self.scope, [], round(time.perf_counter() - t0, 4),
                                   skipped=f"not evaluable: {str(e).splitlines()[0]}")
        viol = [Violation(str(r["identity"]), float(r["severity"]) if r.get("severity") is not None else None, str(r.get("detail") or ""))
                for r in rows]
        return InvariantResult(self.name, self.provenance, self.scope, viol, round(time.perf_counter() - t0, 4))


class PostedMovesBalanced(SqlInvariant):
    name = "posted_moves_balanced"
    provenance = ("Python constraint: account.move._check_balanced (Community `account` module), which asserts that "
                  "a posted move is balanced: debit = credit")
    sql = """
        select m.id::text as identity, abs(sum(l.debit) - sum(l.credit)) as severity,
               m.name || ': debit ' || sum(l.debit) || ' credit ' || sum(l.credit) as detail
        from account_move m join account_move_line l on l.move_id = m.id
        where m.state = 'posted'
        group by m.id, m.name
        having abs(sum(l.debit) - sum(l.credit)) > 0.005
    """


class InvoiceResidualConsistent(SqlInvariant):
    name = "invoice_residual_consistent"
    provenance = ("Observed source logic: account.move._compute_amount (Community `account` module) sets "
                  "amount_residual_signed to the sum of amount_residual over the move's payment_term lines")
    sql = """
        select m.id::text as identity,
               abs(m.amount_residual_signed - coalesce(s.residual, 0)) as severity,
               m.name || ': stored ' || m.amount_residual_signed || ' lines ' || coalesce(s.residual, 0) as detail
        from account_move m
        left join (select move_id, sum(amount_residual) as residual from account_move_line
                   where display_type = 'payment_term' group by move_id) s on s.move_id = m.id
        where m.state = 'posted' and m.move_type <> 'entry'
          and abs(m.amount_residual_signed - coalesce(s.residual, 0)) > 0.005
    """


GENERIC: dict[str, type[Invariant]] = {
    PostedMovesBalanced.name: PostedMovesBalanced,
    InvoiceResidualConsistent.name: InvoiceResidualConsistent,
}


def load_invariants(names: list[str]) -> list[Invariant]:
    """`name` = a generic invariant, or `package.module:attr` = an Invariant subclass or instance
    provided by an external adapter (the dependency points from the adapter to this interface,
    never the other way)."""
    out: list[Invariant] = []
    for n in names:
        if n in GENERIC:
            out.append(GENERIC[n]())
            continue
        if ":" not in n:
            raise ValueError(f"unknown invariant {n!r}; generic: {sorted(GENERIC)}; external: 'package.module:attr'")
        mod, _, attr = n.partition(":")
        obj = getattr(importlib.import_module(mod), attr)
        inst = obj() if isinstance(obj, type) else obj
        if not isinstance(inst, Invariant):
            raise TypeError(f"{n} is not an agent_review Invariant")
        if not getattr(inst, "provenance", None):
            raise ValueError(f"{n} carries no provenance; an invariant without provenance does not ship")
        out.append(inst)
    return out
