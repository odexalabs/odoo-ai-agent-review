"""PostgreSQL statistics: SNAPSHOT AND DELTA, NEVER RESET. A global pg_stat_statements_reset()
would wipe monitoring data belonging to every other database and user on the server.

The delta is scoped by the run database's oid AND the execution role, so the harness's own
grading and snapshot SQL — run as a separate observer role — is excluded: a no-op run reads
0 statements / 0 calls only because of this.

pg_stat_statements reports execution time, not CPU."""
from __future__ import annotations

import psycopg
from psycopg.rows import dict_row


def snapshot(observer_dsn: str, run_db: str, execution_role: str) -> dict | None:
    try:
        with psycopg.connect(observer_dsn, row_factory=dict_row) as c:
            me = c.execute("select current_user as u").fetchone()["u"]
            if me == execution_role:
                raise RuntimeError(f"observer role {me!r} must differ from the execution role {execution_role!r}; "
                                   "the delta would include the harness's own SQL")
            oid = c.execute("select oid from pg_database where datname = %s", (run_db,)).fetchone()["oid"]
            rows = c.execute(
                "select queryid, calls, total_exec_time, rows, left(query, 200) as query from pg_stat_statements "
                "where dbid = %s and userid = (select oid from pg_roles where rolname = %s)",
                (oid, execution_role),
            ).fetchall()
            return {str(r["queryid"]): (r["calls"], r["total_exec_time"], r["rows"], r["query"]) for r in rows}
    except psycopg.errors.UndefinedTable:
        return None   # extension not installed in this database: report "unavailable", never fabricate
    except psycopg.errors.ObjectNotInPrerequisiteState:
        # created in the database but not in shared_preload_libraries: reading it raises. Unavailable, not a
        # harness error (raising would fail every run)
        return None


def delta(before: dict | None, after: dict | None, top: int = 12) -> dict:
    if before is None or after is None:
        return {"available": False, "reason": "pg_stat_statements not available in the run database"}
    out = []
    for k, (calls, ms, rows, text) in after.items():
        c0, m0, r0, _ = before.get(k, (0, 0.0, 0, ""))
        if calls - c0:
            out.append({"queryid": k, "calls": calls - c0, "exec_ms": round(ms - m0, 3), "rows": rows - r0, "query": text})
    out.sort(key=lambda d: -d["exec_ms"])
    return {"available": True, "statements": len(out), "calls": sum(d["calls"] for d in out),
            "exec_ms": round(sum(d["exec_ms"] for d in out), 3), "top": out[:top],
            "note": "execution time, not CPU; scoped to the run database oid and the execution role; never reset"}
