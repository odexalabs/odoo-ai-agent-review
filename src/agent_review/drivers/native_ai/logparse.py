"""Parse the run's own Odoo log for the trajectory native Odoo exposes: tool calls with
arguments at INFO (`AI: Call action <name> with arguments: <python repr>`), tool errors, the
runtime's early-termination line, and the `[AI Summary]` token line per response. Odoo logs in
UTC; the boundary marker is compared as text on the timestamp prefix.

WHAT THE LOG CAN VOUCH FOR. Odoo 19 writes one `[AI Summary]` line when each AI response generation
ends (in a `finally`, so an errored generation is summarised too, provided it made a provider call),
carrying its own count of the tools it ran; every tool run is logged as a `Call action` line. So:

  tool trace   observed (`tool_calls_or_none`) only when the log was read to the end and every AI response
               the driver requested is closed by exactly ONE summary line inside its own turn's window (at
               or after its turn marker, at or before the next; second resolution, both ends inclusive),
               whose tool count equals the calls logged since the previous summary — with no call left
               after the last. Summaries are matched turn by turn, never counted globally: two summaries
               in the first turn must not stand in for a second turn that never finished. Anything less is
               NO trace (None, unobservable) — never `[]`, which says "no tool was called". A readable log
               with no summary (truncated, or the `ai` logger configured away) shows nothing about the AI
               operation: in an earlier version, an unopenable log, a lone HTTP line, and then summaries counted
               globally each produced `[]`.
  token usage  complete only when the log was read to the end and every requested response has its
               summary, matched as above. A missing summary or a read that stopped gives a LOWER BOUND
               (`TokenUsage.partial`); summaries that cannot be matched to the requested turns (excess,
               or outside their window) could over- as well as under-count, so the usage is unknown
               (None), as it is with no summary at all. Never a partial total presented as the total."""
from __future__ import annotations

import ast
import dataclasses
import re
from dataclasses import dataclass, field

from ...core.contracts import TokenUsage, ToolCall

CALL_RE = re.compile(r"AI: Call action (.+?) with arguments: (.*)$")
ERR_RE = re.compile(r"An error occurred while executing (.+)$")   # name and message split below: names contain ': '

SUMMARY_RE = re.compile(
    r"\[AI Summary\] Total: ([\d.]+)s \| API calls: (\d+) \(([\d.]+)s\) \| Tools: (\d+) \(([\d.]+)s\) \| "
    r"Tokens: (\d+) \(in: (\d+), out: (\d+), cached: (\d+)\) \| Batches: (\d+)"
)
EARLY_RE = re.compile(r"AI: action terminate early.*$")
LIMIT_RE = re.compile(r"Tool call limit reached|forbidden action|max_successive_calls", re.IGNORECASE)
TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


@dataclass
class ParsedLog:
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=TokenUsage)
    summaries: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)          # early termination, limits, ERROR lines
    error_lines: list[str] = field(default_factory=list)
    # why the tool trace cannot be vouched for; None only once `parse` has seen the AI operation complete
    trace_unavailable: str | None = "no log was parsed"
    # why the token totals are only a lower bound; None = complete (whenever any summary was parsed)
    usage_incomplete: str | None = None
    # why no total can be given at all: the summaries could not be matched to the requested responses
    usage_unknown: str | None = None

    def usage_or_none(self) -> TokenUsage | None:
        """None when no `[AI Summary]` line was parsed (log unreadable, format changed, no provider call):
        unknown, never zero. Totals from a log read part-way, or missing a requested response's summary,
        come back marked `partial` — a lower bound, not the usage."""
        if not self.summaries or self.usage_unknown:
            return None
        return dataclasses.replace(self.usage, partial=self.usage_incomplete) if self.usage_incomplete else self.usage

    def tool_calls_or_none(self) -> list[ToolCall] | None:
        """The tool trace, or None when the log does not show the AI operation complete. `[]` is reserved
        for a fully observed operation that ran no tool."""
        return None if self.trace_unavailable else self.tool_calls


def _parse_args(raw: str) -> dict | None:
    try:
        v = ast.literal_eval(raw.strip())
        return v if isinstance(v, dict) else None
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def parse(log_path: str, start_marker: str, turn_markers: list[str] | None = None, responses: int | None = None) -> ParsedLog:
    """`start_marker` and `turn_markers` are 'YYYY-MM-DD HH:MM:SS' UTC strings; a call is assigned
    to the last turn whose marker precedes its timestamp. `responses` is how many AI responses the
    driver requested (`/ai/generate_response` calls); None = unknown, and at least one is required."""
    out = ParsedLog()
    markers = sorted(turn_markers or [])
    try:
        fh = open(log_path, errors="replace")  # noqa: SIM115 — closed by the `with` below; open() failure is reported, not raised
    except OSError as e:
        out.notes.append(f"log unreadable: {e}")
        out.trace_unavailable = f"the run's log could not be opened ({type(e).__name__})"
        out.notes.append(f"tool trace UNAVAILABLE: {out.trace_unavailable}")
        return out
    in_window = 0          # timestamped lines at or after the start marker: does this log cover the run at all?
    read_failed = None
    try:
        with fh:
            for line in fh:
                ts = line[:19]
                if ts < start_marker:
                    continue
                if TS_RE.fullmatch(ts):
                    in_window += 1
                _line(line, ts, out, markers)
    except OSError as e:
        read_failed = f"the run's log could not be read to the end ({type(e).__name__})"
    sums = out.summaries
    expected = responses if responses is not None else (len(markers) or None)
    match, match_why = _match_summaries(sums, markers, expected)
    if sums:
        if match == "ambiguous":
            out.usage_unknown = match_why
        elif read_failed or match == "missing":
            out.usage_incomplete = read_failed or match_why
    if read_failed:
        why = read_failed
    elif not in_window:
        why = "the run's log has no line inside the run window"
    elif responses == 0:
        why = "no AI response was requested, so there is no tool trace to observe"
    elif match != "complete":
        why = match_why
    else:
        why = _tool_count_problem(sums, len(out.tool_calls))
    out.trace_unavailable = why
    if why:
        out.notes.append(f"tool trace UNAVAILABLE: {why}")
    if out.usage_unknown:
        out.notes.append(f"token usage UNKNOWN: {out.usage_unknown}")
    elif out.usage_incomplete:
        out.notes.append(f"token usage PARTIAL (a lower bound): {out.usage_incomplete}")
    return out


def _match_summaries(sums: list[dict], markers: list[str], expected: int | None) -> tuple[str, str | None]:
    """Match `[AI Summary]` lines to the requested responses, turn by turn: "complete" (exactly one per turn,
    each inside its own turn's window), "missing" (fewer — and every one placed in a DISTINCT requested turn, so
    none can be a duplicate: a lower bound) or "ambiguous" (more than requested, or any summary that cannot be
    placed in a turn of its own: it may double-count, so nothing can be vouched for). Placement is checked
    BEFORE a shortfall is called "missing" (previously two duplicates in the first of three turns were summed
    as a lower bound)."""
    n, want = len(sums), (1 if expected is None else expected)
    if n > want:
        return "ambiguous", f"{n} [AI Summary] lines for {want} requested AI response(s): the extra cannot be matched"
    if markers:
        if len(markers) != want:
            return "ambiguous", f"{len(markers)} turn marker(s) for {want} requested AI response(s): summaries cannot be matched to turns"
        unplaced = _place(sums, markers)
        if unplaced:
            return "ambiguous", unplaced
    elif n > 1:
        return "ambiguous", f"{n} [AI Summary] lines and no turn markers: they cannot be matched to distinct turns"
    if n < want:
        return "missing", (f"{n} of {want} AI responses have an [AI Summary] line: the log does not show the AI "
                           "operation complete (truncated, or the ai logger not at INFO)")
    return "complete", None


def _place(sums: list[dict], markers: list[str]) -> str | None:
    """Place each summary, in log order, in a distinct turn whose window holds it (at or after that turn's marker, at
    or before the next one: second resolution, both ends inclusive). A turn whose window passes without one is a
    missing response. Returns why a summary cannot be placed, or None when every one has a turn of its own."""
    k = 0
    for s in sums:
        while True:
            if k >= len(markers):
                return f"the [AI Summary] line at {s['ts']} has no requested turn left to close"
            lo, hi = markers[k], (markers[k + 1] if k + 1 < len(markers) else None)
            if s["ts"] < lo:
                return (f"the [AI Summary] line at {s['ts']} is before the first turn" if k == 0 else
                        f"the [AI Summary] line at {s['ts']} falls in turn {k}'s window, which another summary already closed")
            k += 1
            if hi is None or s["ts"] <= hi:
                break          # placed in turn k (1-based); otherwise that turn is missing and the next is tried
    return None


def _tool_count_problem(sums: list[dict], n_calls: int) -> str | None:
    """Each summary's tool count must equal the calls logged since the previous summary, with none after the last."""
    before = 0
    for k, s in enumerate(sums):
        if s["calls_before"] - before != s["tools"]:
            return (f"response {k + 1}: the log's own [AI Summary] reports {s['tools']} tool call(s) and "
                    f"{s['calls_before'] - before} were parsed")
        before = s["calls_before"]
    if n_calls > before:
        return f"{n_calls - before} tool call(s) logged after the last [AI Summary]: a response did not finish in this log"
    return None


def _line(line: str, ts: str, out: ParsedLog, markers: list[str]) -> None:
    """One log line inside the run window."""
    turn = sum(1 for m in markers if m <= ts) - 1 if markers else None
    m = CALL_RE.search(line)
    if m:
        raw = m.group(2).strip()
        out.tool_calls.append(ToolCall(m.group(1), _parse_args(raw), raw[:4000], None, turn if turn is not None and turn >= 0 else None))
        return
    m = ERR_RE.search(line)
    if m:
        rest = m.group(1)
        # tool names themselves contain ': ' ("AI: Search"), so match the longest known
        # call name as a prefix of the message rather than splitting on the first colon
        names = sorted({c.name for c in out.tool_calls if c.error is None}, key=len, reverse=True)
        hit = next((n for n in names if rest.startswith(n + ": ")), None)
        if hit:
            err = rest[len(hit) + 2:].strip()
            for call in reversed(out.tool_calls):
                if call.name == hit and call.error is None:
                    call.error = err[:1000]
                    break
        else:
            out.error_lines.append(line.strip()[:500])
        return
    m = SUMMARY_RE.search(line)
    if m:
        s = dict(zip(["total_s", "api_calls", "api_s", "tools", "tool_s", "tokens", "in", "out", "cached", "batches"],
                     [float(m.group(1)), int(m.group(2)), float(m.group(3)), int(m.group(4)), float(m.group(5)),
                      int(m.group(6)), int(m.group(7)), int(m.group(8)), int(m.group(9)), int(m.group(10))]))
        s["ts"], s["calls_before"] = ts, len(out.tool_calls)
        out.summaries.append(s)
        out.usage.add(TokenUsage(s["in"], s["out"], s["cached"], s["api_calls"]))
        return
    if EARLY_RE.search(line) or LIMIT_RE.search(line):
        out.notes.append(line.strip()[-300:])
    elif " ERROR " in line or " CRITICAL " in line:
        out.error_lines.append(line.strip()[:500])
