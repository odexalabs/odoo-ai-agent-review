"""The no-op driver. Starts no process, makes no provider call, writes nothing. It checks the lifecycle
end to end — a no-op task produces an empty business diff on a real copy of the template — without
any spend.

With `simulate_sql` in the agent config it executes that SQL as the execution role so the
diff, classification and grading paths can be exercised against a real clone (tests only)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import psycopg

from ...core.contracts import (
    DriverRunResult,
    EnvironmentHandle,
    ModelMetadata,
    RequestStatus,
    TokenUsage,
    Turn,
)
from ...core.credentials import Credential
from ...core.profile import RunProfile
from ..base import Capabilities, Driver


class NoopDriver(Driver):
    name = "noop"

    def __init__(self) -> None:
        self.env: EnvironmentHandle | None = None
        self.turns: list[Turn] = []
        self.cfg: dict[str, Any] = {}

    def prepare(self, env: EnvironmentHandle, profile: RunProfile, agent_role: str, credential: Credential | None,
                run_dir: str, session_kind: str | None) -> None:
        self.env = env
        self.cfg = profile.agent(agent_role)

    def capabilities(self) -> Capabilities:
        return Capabilities(None, None, [], "no", False, False, True, ["noop driver: no tools, no provider"])

    def session_context(self) -> dict[str, Any]:
        return {"session_id": f"noop:{self.env.name if self.env else '-'}", "kind": "none", "agent": "noop"}

    def execute(self, first_turn: str, continuation: str | None, should_continue: Callable[[], bool]) -> None:
        assert self.env
        reply = self.cfg.get("reply")
        sql = self.cfg.get("simulate_sql")
        if sql:
            with psycopg.connect(self.env.dsn, autocommit=True) as c:
                c.execute(sql)
        self.turns.append(Turn(first_turn, RequestStatus.RETURNED, reply, None, 0.0))
        if continuation and should_continue():
            self.turns.append(Turn(continuation, RequestStatus.RETURNED, reply, None, 0.0))

    def collect(self) -> DriverRunResult:
        return DriverRunResult(
            session_id=self.session_context()["session_id"], turns=list(self.turns),
            model_metadata=ModelMetadata("none", "none", False, None, "n/a"),
            tool_trace=[] if self.cfg.get("trace_observable", True) else None,
            token_usage=TokenUsage(), driver_notes=["noop driver"],
            provider_calls=0, usage_basis="the noop driver has no provider client and starts no process",
        )

    def close(self) -> None:
        pass

    def is_paid(self) -> bool:
        return False

    def transport_description(self) -> str:
        return "none: the noop driver starts no process and calls no provider"
