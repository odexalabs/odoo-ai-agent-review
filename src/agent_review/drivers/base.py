"""Driver — the third seam. Owns the substrate's own lifecycle, because "a fresh session" means
something different for a native Odoo agent, an MCP client, a script and a human.

    prepare -> execute -> (wait_for_completion inside execute) -> collect -> close

The driver reports only what it can observe technically. It is never asked whether the business
result was correct. The core supplies `should_continue()` for the optional second turn; the
driver calls it and never decides itself."""
from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..core.contracts import DriverRunResult, EnvironmentHandle, TokenUsage
from ..core.credentials import Credential
from ..core.profile import RunProfile
from ..core.scenario import Scenario, ScenarioError


@dataclass
class ToolInfo:
    key: str                     # normalised capability key, e.g. search, read_group, create_lead
    name: str
    xml_id: str | None
    model: str | None
    kind: str                    # read | write | unknown
    attached_agents: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Capabilities:
    odoo_version: str | None
    odoo_build: str | None
    tools: list[ToolInfo]
    trajectory_observable: str          # yes (logs) | yes (wire) | no
    token_usage: bool
    provider_cost: bool
    db_access: bool
    notes: list[str] = field(default_factory=list)
    # agent name -> the agents it may start sub-agent sessions with (Odoo 20). Their tools are reachable through the
    # delegation and their calls are part of the trace, but they are not the agent's own capabilities.
    delegation: dict[str, list[str]] = field(default_factory=dict)

    def reachable_by_delegation(self, agent_name: str, levels: int = 3) -> set[str]:
        """The agents a session of `agent_name` can reach by delegating, directly or through its sub-agents (Odoo 20
        lets a session delegate while fewer than four agent levels sit above it: three levels under the root)."""
        seen: set[str] = set()
        frontier = [agent_name]
        for _ in range(levels):
            frontier = [c for a in frontier for c in self.delegation.get(a, []) if c not in seen and c != agent_name]
            seen.update(frontier)
        return seen

    def keys_for_agent(self, agent_name: str | None) -> set[str]:
        if agent_name is None:
            return {t.key for t in self.tools}
        return {t.key for t in self.tools if agent_name in t.attached_agents}

    def write_tool_names(self) -> set[str]:
        return {t.name for t in self.tools if t.kind == "write"}

    def missing_for(self, required: list[str], agent_name: str | None) -> list[str]:
        have = self.keys_for_agent(agent_name)
        return [k for k in required if k not in have]


class Driver(abc.ABC):
    name: str = "abstract"
    probe: bool = False        # set by the CLI: exercise the plumbing, make no provider call

    @abc.abstractmethod
    def prepare(self, env: EnvironmentHandle, profile: RunProfile, agent_role: str, credential: Credential | None,
                run_dir: str, session_kind: str | None) -> None: ...

    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    @abc.abstractmethod
    def session_context(self) -> dict[str, Any]:
        """session_id, kind (internal|public), uid, company_id, agent name, model as configured."""

    @abc.abstractmethod
    def execute(self, first_turn: str, continuation: str | None, should_continue: Callable[[], bool]) -> None: ...

    @abc.abstractmethod
    def collect(self) -> DriverRunResult: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    def quiesce(self) -> list[str]:
        """Stop everything this driver started that could still write to the run's copy, before the final evidence
        is taken. Called once the last turn has returned, whether it completed or timed out: a timed-out turn can
        still have work under way. Returns notes for the record; raises EvidenceIncomplete when it cannot establish
        that nothing will write any more. The default is for drivers that start nothing."""
        return []

    def is_paid(self) -> bool:
        return not self.probe

    def cron_threads_zero(self) -> bool:
        return True

    def smtp_fallback_disabled(self) -> bool:
        """True when the substrate process cannot fall back to a config-level SMTP host. A driver
        that starts no application process has nothing that could send."""
        return True

    # ---- selection and compatibility, checked BEFORE any environment exists
    # A structured continuation (Odoo 20's question/confirmation protocol) is handed over as the Continuation
    # object only to a driver that declares it can send it; every other driver receives the plain text.
    accepts_structured_continuation: bool = False

    def preflight(self, profile: RunProfile, template_dsn: str) -> list[str]:
        """Validate that this driver can run this profile against this template: the Odoo version of the source
        tree, the template's schema, the transport. Raise ProfileError to refuse, naming what does not match;
        never fall back to another driver or transport. Returns notes for the run conditions."""
        return []

    def check_scenario(self, sc: Scenario) -> None:
        """Refuse a scenario this driver cannot execute as written (ScenarioError), before any environment."""

    def refuse_other_versions(self, sc: Scenario, major: str) -> None:
        if sc.required_odoo and major not in sc.required_odoo:
            raise ScenarioError(f"{sc.name} is written for Odoo {', '.join(sc.required_odoo)}; the {self.name} driver runs "
                                f"Odoo {major}")

    def refuse_structured_continuation(self, sc: Scenario) -> None:
        if sc.continuation is not None and sc.continuation.structured and not self.accepts_structured_continuation:
            raise ScenarioError(f"{sc.name}: its continuation uses Odoo 20's question/confirmation protocol, which the "
                                f"{self.name} driver cannot send; use a plain `text` continuation or the native_ai_20 driver")

    def transport_description(self) -> str:
        """How the agent's model requests travel, in words, for the report."""
        return "none"

    def scope_note(self) -> str | None:
        """What a run on this driver does and does not evaluate, when that is narrower than it looks."""
        return None

    def usage_after_failure(self) -> tuple[TokenUsage | None, int | None, str]:
        """(usage, provider_calls, basis) for a run that never reached collect(): what the driver can still vouch
        for about provider spend. Default: nothing — unknown, never zero."""
        return None, None, "the run did not complete, and this driver keeps no record of provider requests outside collect()"
