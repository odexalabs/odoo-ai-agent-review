"""The seam contracts. Everything above a seam sees only what is defined here.

Three seams:

  FixtureBackend   builds an immutable fixture and hands out disposable environments
  ChangeDetector   discovers what changed, at a stated COVERAGE
  Driver           runs the substrate's own session and reports only what it can observe

The driver reports technical facts. The core DERIVES transaction outcome, expected-effect
status and everything else — a driver is never asked whether the business result was correct,
and nothing in the core may require a field this file marks optional.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Coverage(str, Enum):
    """How much of a table's change a detector can vouch for. Part of the ChangeEvidence
    contract because v1 cannot prove exact row changes everywhere."""

    EXACT = "exact"            # every changed row is known, with before/after values
    SCOPED = "scoped"          # exact within a declared scope (e.g. a WAL interval)
    TABLE_ONLY = "table_only"  # per-table count and max(write_date) only; deletes+inserts invisible


class RequestStatus(str, Enum):
    """Technical outcome of one turn's request. Says nothing about the business result."""

    RETURNED = "returned"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class Turn:
    """One entry of the structured conversation. The driver always knows what it submitted and
    what came back."""

    user_input: str
    request_status: RequestStatus
    assistant_response: str | None = None
    error: str | None = None
    wall_s: float | None = None
    message_id: int | None = None
    # Substrate state, reported by the driver where the substrate records it.
    # None = the substrate cannot report it; the core then falls back to what it did before.
    #   pending_interaction   "none" | "question" | "confirmation" | "client_result" | "external_result"
    #   user_request_returned True when the user's own request came back normally (whatever the
    #                         agent loop did afterwards); False when it raised and the user saw an error
    pending_interaction: str | None = None
    user_request_returned: bool | None = None
    # What the user was shown, when the SUBSTRATE itself establishes it rather than the reply's wording — e.g. on
    # Odoo 20 the stand-in answered the turn's last round with a failure, so the message the instance then posted
    # is its failure notice. None = not established this way; the report is classified from the text as before.
    substrate_report: str | None = None
    substrate_report_basis: str | None = None


@dataclass
class ToolCall:
    """One observed tool invocation. Only present when the substrate exposes it."""

    name: str
    arguments: dict[str, Any] | None   # parsed where possible
    arguments_raw: str                  # verbatim, as logged
    error: str | None = None
    turn_index: int | None = None
    session_depth: int = 0              # 0: the conversation's own session; 1+: a sub-agent session it delegated to


@dataclass
class ModelMetadata:
    provider: str
    identifier: str                 # what was sent to the provider
    pinnable: bool                  # does the provider expose an immutable build?
    served_identifier: str | None = None   # the immutable build, where observed
    reproducibility: str = "limited"       # "pinned" | "limited"
    # The defaults keep stored records loadable. `configured_identifier` is what the run profile
    # names; `identifier` is what was (or would be) sent; `served_identifier` stays None unless a provider
    # response was observed naming it. `selected_by` says who chose the model (never Odoo's hosted service here).
    configured_identifier: str | None = None
    selected_by: str | None = None


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    llm_round_trips: int = 0
    # None = a complete measurement. Otherwise why these totals are a LOWER BOUND (some response's usage was not
    # observed): a cost built from them is "at least", never the cost.
    partial: str | None = None

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.llm_round_trips += other.llm_round_trips
        self.partial = self.partial or other.partial


@dataclass
class DriverRunResult:
    """The one contract every driver returns. Optional fields are the point: MCP may expose
    excellent trajectories while native Odoo exposes none, and the core works either way."""

    session_id: str
    turns: list[Turn]
    model_metadata: ModelMetadata
    tool_trace: list[ToolCall] | None = None      # None = unobservable on this substrate
    token_usage: TokenUsage | None = None
    provider_cost_usd: float | None = None        # provider-REPORTED only; never estimated here
    driver_notes: list[str] = field(default_factory=list)   # e.g. runtime log lines of interest
    known_response_rules: list[ResponseRule] = field(default_factory=list)
    # How many provider requests the run made, where the driver can vouch for it: 0 only on POSITIVE evidence
    # that none was made (e.g. the Odoo 20 stand-in held no key and counted no provider attempt); None = unknown.
    provider_calls: int | None = None
    usage_basis: str | None = None               # where the token counts (or the zero) come from

    @property
    def request_status(self) -> RequestStatus:
        return self.turns[-1].request_status if self.turns else RequestStatus.ERROR

    @property
    def final_response(self) -> str | None:
        for t in reversed(self.turns):
            if t.assistant_response:
                return t.assistant_response
        return None

    @property
    def error(self) -> str | None:
        return self.turns[-1].error if self.turns else "no turns executed"

    @property
    def turn_count(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class ResponseRule:
    """A deterministic rule that classifies a user-facing report. Exact string or regex.
    Supplied by the driver (known substrate responses) or by the scenario. Never a model."""

    verdict: str            # success | failure | neutral | clarification | refused
    pattern: str
    regex: bool = False
    source: str = "scenario"   # scenario | substrate:<driver>


@dataclass
class RowChange:
    table: str
    pk: Any
    kind: str                                   # added | removed | changed
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    changed_fields: dict[str, list[Any]] = field(default_factory=dict)   # field -> [before, after]


@dataclass
class TableSummary:
    table: str
    count_before: int | None
    count_after: int | None
    max_write_date_before: str | None
    max_write_date_after: str | None
    coverage: Coverage


@dataclass
class ChangeEvidence:
    """Output of a ChangeDetector. `coverage` is part of the contract."""

    tables_touched: list[TableSummary]
    row_changes: list[RowChange]
    coverage_by_table: dict[str, Coverage]
    detector: str

    def tables_at(self, cov: Coverage) -> list[str]:
        return [t for t, c in self.coverage_by_table.items() if c == cov]


@dataclass
class EnvironmentHandle:
    """A disposable environment for ONE run. For PostgresTemplateBackend it is a database; for a
    SnapshotBackend it may be a cluster plus a filestore. Nothing above the seam assumes
    'database' beyond what `dsn` exposes."""

    name: str
    dsn: str                    # connection string for the execution role's database
    execution_role: str         # the role the agent's Odoo process connects as
    observer_dsn: str           # what the harness itself uses (must be a different role)
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConditions:
    """Everything a reader needs to know to interpret one run. Recorded with every run."""

    odoo_version: str | None = None
    odoo_build: str | None = None
    fixture: str | None = None
    fixture_backend: str | None = None
    benchmark_date: str | None = None
    egress_isolation: str = "unavailable"      # verified | partial | unavailable
    egress_allowed_endpoints: list[str] = field(default_factory=list)
    mail_blocked_verified: bool | None = None
    key_source: str | None = None
    key_last4: str | None = None
    host: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    driver: str | None = None                  # native_ai (Odoo 19) | native_ai_20 | noop
    transport: str | None = None               # how the agent's model requests travel, in words
    scope: str | None = None                   # what a run on this driver does and does not evaluate


def to_dict(obj: Any) -> Any:
    """dataclass -> JSON-able, enums to their values."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


class EvidenceIncomplete(RuntimeError):
    """The final evidence could not be taken with nothing left that could still write to the run's copy (the agent's
    process would not stop, or a connection of the execution role stayed open). Such a run is not graded: its diff
    could miss a write that landed after the snapshot."""
