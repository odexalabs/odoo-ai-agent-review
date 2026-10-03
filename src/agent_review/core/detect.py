"""ChangeDetector — the second seam. Discovers what changed, at a stated coverage.

TableDiffDetector (implemented): before/after snapshot. EXACT for the tables it snapshots in
full — the business set plus anything the scenario or the invariants name — and TABLE_ONLY
elsewhere (per-table count and max(write_date)). Its blind spot is real and stated in the
evidence: a relation table with no write_date where one row is deleted and another inserted
leaves the count unchanged and is invisible. Cost is proportional to DATABASE SIZE.

WalChangeDetector (interface only): record the LSN before the run, decode the WAL written
during it. Cost proportional to WHAT CHANGED. Gives deletes.

Rejected, recorded so nobody re-proposes it: locating changed rows by `xmin` (32-bit, wraps,
seq-scans anyway) and treating `pg_stat_user_tables` as proof (cumulative stats lag).
"""
from __future__ import annotations

import abc
import datetime as dt
import decimal
import hashlib
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from .contracts import ChangeEvidence, Coverage, EnvironmentHandle, RowChange, TableSummary


def normalise_value(v: Any) -> Any:
    """Make a PostgreSQL value JSON-able and comparable. bytea is hashed: comparing memoryview reprs once
    reported a false `res_company.logo_web` change."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, decimal.Decimal):
        return str(int(v)) if v == v.to_integral() else str(v)   # never .normalize(): 1200.00 -> '1.2E+3'
    if isinstance(v, (bytes, memoryview, bytearray)):
        return "sha1:" + hashlib.sha1(bytes(v)).hexdigest()
    if isinstance(v, (dt.datetime, dt.date, dt.time)):
        return v.isoformat()
    if isinstance(v, (list, tuple)):
        return [normalise_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): normalise_value(x) for k, x in v.items()}
    return str(v)


class ChangeDetector(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def before_run(self, env: EnvironmentHandle, exact_tables: set[str]) -> None: ...

    @abc.abstractmethod
    def after_run(self, env: EnvironmentHandle) -> None: ...

    @abc.abstractmethod
    def collect_changes(self) -> ChangeEvidence: ...


class TableDiffDetector(ChangeDetector):
    name = "TableDiffDetector"

    def __init__(self, schema: str = "public"):
        self.schema = schema
        self._before: dict | None = None
        self._after: dict | None = None
        self._exact: set[str] = set()

    # -- snapshot
    def _snapshot(self, env: EnvironmentHandle) -> dict:
        snap: dict[str, Any] = {"tables": {}, "rows": {}, "pk": {}}
        with psycopg.connect(env.observer_dsn, row_factory=dict_row) as c:
            tables = [
                r["table_name"]
                for r in c.execute(
                    "select table_name from information_schema.tables "
                    "where table_schema = %s and table_type = 'BASE TABLE' order by 1",
                    (self.schema,),
                )
            ]
            has_wd = {
                r["table_name"]
                for r in c.execute(
                    "select table_name from information_schema.columns "
                    "where table_schema = %s and column_name = 'write_date'",
                    (self.schema,),
                )
            }
            pks = {}
            for r in c.execute(
                """select tc.table_name, kcu.column_name
                   from information_schema.table_constraints tc
                   join information_schema.key_column_usage kcu
                     on kcu.constraint_name = tc.constraint_name and kcu.table_schema = tc.table_schema
                   where tc.table_schema = %s and tc.constraint_type = 'PRIMARY KEY'
                   order by kcu.ordinal_position""",
                (self.schema,),
            ):
                pks.setdefault(r["table_name"], []).append(r["column_name"])
            snap["pk"] = pks
            for t in tables:
                if t in has_wd:
                    row = c.execute(
                        sql.SQL("select count(*) as n, max(write_date)::text as wd from {}").format(sql.Identifier(t))
                    ).fetchone()
                else:
                    row = c.execute(sql.SQL("select count(*) as n, null as wd from {}").format(sql.Identifier(t))).fetchone()
                snap["tables"][t] = (row["n"], row["wd"])
                if t in self._exact:
                    pk = pks.get(t) or ["id"]
                    rows = c.execute(sql.SQL("select * from {}").format(sql.Identifier(t))).fetchall()
                    keyed = {}
                    for r in rows:
                        key = tuple(r.get(k) for k in pk)
                        key = key[0] if len(key) == 1 else "|".join(str(k) for k in key)
                        keyed[key] = {k: normalise_value(v) for k, v in r.items()}
                    snap["rows"][t] = keyed
        return snap

    def before_run(self, env: EnvironmentHandle, exact_tables: set[str]) -> None:
        self._exact = set(exact_tables)
        self._before = self._snapshot(env)
        # only tables that exist can be exact
        self._exact &= set(self._before["tables"])

    def after_run(self, env: EnvironmentHandle) -> None:
        self._after = self._snapshot(env)

    def collect_changes(self) -> ChangeEvidence:
        assert self._before is not None and self._after is not None, "before_run/after_run not called"
        a, b = self._before, self._after
        touched: list[TableSummary] = []
        coverage: dict[str, Coverage] = {}
        rows: list[RowChange] = []
        for t, (n1, wd1) in b["tables"].items():
            n0, wd0 = a["tables"].get(t, (None, None))
            cov = Coverage.EXACT if t in self._exact else Coverage.TABLE_ONLY
            coverage[t] = cov
            if (n0, wd0) != (n1, wd1) or (t in self._exact and a["rows"].get(t) != b["rows"].get(t)):
                touched.append(TableSummary(t, n0, n1, wd0, wd1, cov))
        for t in self._exact:
            A, B = a["rows"].get(t, {}), b["rows"].get(t, {})
            for k in B:
                if k not in A:
                    rows.append(RowChange(t, k, "added", None, B[k]))
                elif A[k] != B[k]:
                    changed = {f: [A[k].get(f), B[k].get(f)] for f in B[k] if A[k].get(f) != B[k].get(f)}
                    rows.append(RowChange(t, k, "changed", A[k], B[k], changed))
            for k in A:
                if k not in B:
                    rows.append(RowChange(t, k, "removed", A[k], None))
        return ChangeEvidence(tables_touched=touched, row_changes=rows, coverage_by_table=coverage, detector=self.name)

    def rows_before(self, table: str) -> dict:
        assert self._before is not None
        return self._before["rows"].get(table, {})

    def rows_after(self, table: str) -> dict:
        assert self._after is not None
        return self._after["rows"].get(table, {})


class WalChangeDetector(ChangeDetector):
    """INTERFACE ONLY. Coverage: EXACT for the WAL interval.

    Old values: a primary key identifies an updated or deleted row but does not carry its old
    column values. Since the fixture is immutable, the normal route is to read the old row BY
    PRIMARY KEY FROM THE FIXTURE. `REPLICA IDENTITY FULL` is a fallback for the cases that route
    cannot serve — never the default, and never global.
    """

    name = "WalChangeDetector"

    def before_run(self, env: EnvironmentHandle, exact_tables: set[str]) -> None:  # pragma: no cover
        raise NotImplementedError("WalChangeDetector is declared, not implemented, in v1")

    def after_run(self, env: EnvironmentHandle) -> None:  # pragma: no cover
        raise NotImplementedError("WalChangeDetector is declared, not implemented, in v1")

    def collect_changes(self) -> ChangeEvidence:  # pragma: no cover
        raise NotImplementedError("WalChangeDetector is declared, not implemented, in v1")
