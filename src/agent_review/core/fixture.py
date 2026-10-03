"""FixtureBackend — the seam between 'a disposable environment per run' and the mechanism.

    customer staging -> dump/snapshot -> prepared fixture -> immutable template -> run-001 ...

PostgresTemplateBackend is implemented: `CREATE DATABASE ... TEMPLATE`, proven on the ~60 MB
synthetic fixtures. SnapshotBackend is an interface only — see its docstring for what the
implementer must not get wrong.
"""
from __future__ import annotations

import abc
import os
import re
import shutil
import time
from dataclasses import dataclass

import psycopg
from psycopg import sql

from .contracts import EnvironmentHandle

FX_PREFIX = "odexalabs_fx_"          # the tool's namespace for disposable databases
RUN_PREFIX = FX_PREFIX + "run_"       # what this backend creates and the ONLY thing it will ever drop.
                                      # Templates also live under FX_PREFIX (odexalabs_fx_tpl), so the
                                      # bare prefix is not a sufficient guard.
_RUN_NAME_RE = re.compile(r"^[a-z0-9_]{1,40}$")


class FixtureBackend(abc.ABC):
    @abc.abstractmethod
    def prepare(self) -> None:
        """Build the immutable fixture, once. Idempotent."""

    @abc.abstractmethod
    def clone_for_run(self, run_name: str) -> EnvironmentHandle:
        """A disposable environment for one run."""

    @abc.abstractmethod
    def destroy_run(self, env: EnvironmentHandle) -> None:
        """Tear it down. Must be safe to call twice and from a `finally`."""

    @abc.abstractmethod
    def describe(self) -> dict:
        """What it is, for the run conditions."""

    def template_dsn(self) -> str:
        """A read-only connection string for the immutable fixture, used by driver preflight checks. A backend
        that cannot offer one returns "", and a driver that needs to inspect the template then refuses."""
        return ""


class PreflightRefused(RuntimeError):
    pass


@dataclass
class PostgresTemplateBackend(FixtureBackend):
    """One disposable DATABASE per run, cloned from an immutable template database.

    template          the prepared fixture database. Never written after preparation.
    execution_role    the role the agent's Odoo connects as; owns every clone. The observer
                      (this harness) connects as a different role so pg_stat_statements can be
                      scoped to the agent's work alone.
    admin_dsn         how the harness connects to create/drop databases.
    """

    template: str
    execution_role: str
    admin_dsn: str = "dbname=postgres"
    host: str | None = None
    port: int | None = None
    execution_password: str | None = None    # only when the cluster does not trust local connections
    disk_headroom_bytes: int = 2 * 1024**3

    def _admin(self) -> psycopg.Connection:
        return psycopg.connect(self.admin_dsn, autocommit=True)

    def prepare(self) -> None:
        # The tool does not build fixtures; it consumes a template the operator prepared (fixtures/README.md).
        # prepare() only verifies that it exists.
        with self._admin() as c:
            row = c.execute("select 1 from pg_database where datname = %s", (self.template,)).fetchone()
            if not row:
                raise PreflightRefused(f"template database {self.template!r} does not exist")

    def template_dsn(self) -> str:
        """How the harness reads the template for its read-only preflight checks (never for writes)."""
        hp = " ".join(p for p in [f"host={self.host}" if self.host else "", f"port={self.port}" if self.port else ""] if p)
        return f"dbname={self.template} {hp}".strip()

    def template_size(self) -> int:
        with self._admin() as c:
            return c.execute("select pg_database_size(%s)", (self.template,)).fetchone()[0]

    def data_directory(self) -> str | None:
        try:
            with self._admin() as c:
                return c.execute("show data_directory").fetchone()[0]
        except psycopg.Error:
            return None

    def preflight(self, run_db: str) -> dict:
        """Refuse rather than crash. CREATE DATABASE ... TEMPLATE needs about 3x the source's size free (PostgreSQL
        16 writes WAL for the whole copy), measured at the data directory, plus a hard floor for WAL headroom."""
        size = self.template_size()
        pgdata = self.data_directory()
        measured_at = pgdata if pgdata and os.path.isdir(pgdata) else None
        report = {"template_bytes": size, "measured_at": measured_at}
        if measured_at:
            free = shutil.disk_usage(measured_at).free
            report["free_bytes"] = free
            if free < 3 * size + self.disk_headroom_bytes:
                raise PreflightRefused(
                    f"disk: {free} bytes free at {measured_at}, need 3 x {size} + {self.disk_headroom_bytes}"
                )
        else:
            report["free_bytes"] = None   # could not measure; a guard that cannot measure must not block
        with self._admin() as c:
            # A backend that is still connecting or exiting is listed briefly with no user; PostgreSQL's own
            # CREATE DATABASE ... TEMPLATE waits a few seconds for such backends and ends autovacuum workers on
            # the source itself. So wait the same way, and refuse only for a session that stays (a
            # one-shot look refused on a transient backend during the test suite).
            open_on_tpl: list = []
            for _ in range(20):
                open_on_tpl = c.execute(
                    "select usename, application_name, backend_type from pg_stat_activity where datname = %s "
                    "and backend_type <> 'autovacuum worker'", (self.template,)).fetchall()
                if not open_on_tpl:
                    break
                time.sleep(0.25)
            if open_on_tpl:
                raise PreflightRefused(f"template {self.template} is held open by {open_on_tpl}")
            if c.execute("select 1 from pg_database where datname = %s", (run_db,)).fetchone():
                raise PreflightRefused(f"run database {run_db} already exists")
        return report

    def clone_for_run(self, run_name: str) -> EnvironmentHandle:
        if not _RUN_NAME_RE.match(run_name or ""):
            raise PreflightRefused(f"invalid run name {run_name!r}: must match {_RUN_NAME_RE.pattern}")
        run_db = f"{RUN_PREFIX}{run_name}"
        if len(run_db) > 63:
            raise PreflightRefused(f"database name too long: {run_db}")
        if run_db == self.template:
            raise PreflightRefused(f"run database name equals the template {self.template!r}")
        pre = self.preflight(run_db)
        with self._admin() as c:
            c.execute(
                sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE {}").format(
                    sql.Identifier(run_db), sql.Identifier(self.execution_role), sql.Identifier(self.template)
                )
            )
        hp = " ".join(p for p in [f"host={self.host}" if self.host else "", f"port={self.port}" if self.port else ""] if p)
        pw = f" password={self.execution_password}" if self.execution_password else ""
        return EnvironmentHandle(
            name=run_db,
            dsn=f"dbname={run_db} user={self.execution_role}{pw} {hp}".strip(),
            execution_role=self.execution_role,
            observer_dsn=f"dbname={run_db} {hp}".strip(),
            extras={"preflight": pre, "template": self.template},
        )

    def destroy_run(self, env: EnvironmentHandle) -> None:
        """Drops ONLY a database this backend could have created: RUN_PREFIX plus a non-empty name,
        never the configured template, never anything else in the cluster. Idempotent."""
        name = env.name or ""
        if not name.startswith(RUN_PREFIX) or len(name) <= len(RUN_PREFIX):
            raise RuntimeError(f"refusing to drop {name!r}: not a {RUN_PREFIX}* database")
        if name == self.template:
            raise RuntimeError(f"refusing to drop {name!r}: it is the configured template")
        if not _RUN_NAME_RE.match(name[len(RUN_PREFIX):]):
            raise RuntimeError(f"refusing to drop {name!r}: not a name this backend creates")
        with self._admin() as c:
            c.execute(
                "select pg_terminate_backend(pid) from pg_stat_activity where datname = %s and pid <> pg_backend_pid()",
                (env.name,),
            )
            c.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(env.name)))

    def describe(self) -> dict:
        return {
            "backend": "PostgresTemplateBackend",
            "template": self.template,
            "execution_role": self.execution_role,
            "mechanism": "CREATE DATABASE ... TEMPLATE; cost proportional to fixture size",
        }


class SnapshotBackend(FixtureBackend):
    """INTERFACE ONLY. Copy-on-write clone of the PostgreSQL data directory AND the Odoo filestore,
    for fixtures too large to copy per run.

    Notes for whoever implements this, so the interface does not have to change:
      - a data-directory clone needs its OWN postmaster, port and socket — a cluster per run, not
        a database per run. destroy_run() stops a server rather than dropping a database.
      - shut the fixture's PostgreSQL down CLEANLY before snapshotting; a clone of a running
        directory is crash-consistent and recovers on startup.
      - set `wal_level = logical` in the fixture BEFORE the snapshot so every clone inherits it.
        Create replication slots on the clone at run start, never in the image.
      - a large fixture is an ENVIRONMENT snapshot: database AND filestore. A database-only
        snapshot leaves filesystem effects behind and the isolation claim is false.
      - needs ZFS, btrfs, LVM thin provisioning or cloud block snapshots — a hard infrastructure
        dependency, which is why v1 does not ship it.
    """

    def prepare(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError("SnapshotBackend is declared, not implemented, in v1")

    def clone_for_run(self, run_name: str) -> EnvironmentHandle:  # pragma: no cover
        raise NotImplementedError("SnapshotBackend is declared, not implemented, in v1")

    def destroy_run(self, env: EnvironmentHandle) -> None:  # pragma: no cover
        raise NotImplementedError("SnapshotBackend is declared, not implemented, in v1")

    def describe(self) -> dict:  # pragma: no cover
        return {"backend": "SnapshotBackend", "implemented": False}
