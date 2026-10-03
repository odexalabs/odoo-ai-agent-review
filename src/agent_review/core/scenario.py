"""The scenario format. SEMANTICS ONLY: what correct means. Where and with what it executes
is the run profile (profile.py). The model is never in a scenario file.

Expected state is written as SELECTORS and resolved to ids against the disposable environment
BEFORE the agent starts; grading is against that snapshot. Raw record ids are stable only inside
one template clone and cannot express a CREATE expectation at all.

Three key types, because they grade differently:
    read     facts-in-answer with zero business writes
    create   model + count + required values on the created rows
    update   existing ids (resolved) + required values, and which fields may change

Multi-turn, minimally: turn 1 is the task prompt; an optional fixed continuation is sent when
  Level 2 — the keyed mutation has not occurred (`when: key_not_satisfied`, the default)
  Level 1 — only if the scenario says `when: always`; there is no key to consult.
Never inferred by asking a model to read the reply.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
import yaml
from psycopg import sql

from .contracts import ResponseRule

OPS = {"=": "=", "!=": "<>", "<": "<", "<=": "<=", ">": ">", ">=": ">=", "in": "IN", "not in": "NOT IN",
       "ilike": "ILIKE", "like": "LIKE", "is null": "IS NULL", "is not null": "IS NOT NULL"}


class ScenarioError(ValueError):
    pass


@dataclass
class Selector:
    model: str
    where: list[list[Any]] = field(default_factory=list)   # [[field, op, value], ...]; value may be {ref: {...}}

    @classmethod
    def parse(cls, raw: dict, ctx: str) -> Selector:
        if not isinstance(raw, dict) or "model" not in raw:
            raise ScenarioError(f"{ctx}: a selector needs `model` (and optionally `where`)")
        where = raw.get("where", []) or []
        for w in where:
            if not (isinstance(w, list) and len(w) in (2, 3) and w[1] in OPS):
                raise ScenarioError(f"{ctx}: bad where clause {w!r}; use [field, op, value] with op in {sorted(OPS)}")
        return cls(raw["model"], where)


@dataclass
class ReadFacts:
    must_contain: list[Any] = field(default_factory=list)
    should_contain: list[Any] = field(default_factory=list)
    must_not_contain: list[Any] = field(default_factory=list)
    business_writes: int = 0


@dataclass
class CreateKey:
    model: str
    count: int
    match: dict[str, Any] = field(default_factory=dict)
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class UpdateKey:
    select: Selector
    values: dict[str, Any] = field(default_factory=dict)
    fields_only: list[str] | None = None


@dataclass
class Forbid:
    model: str
    any: bool = False
    existing: bool = False
    where: list[list[Any]] | None = None
    fields: list[str] | None = None
    note: str | None = None


@dataclass
class SideEffect:
    """A write the scenario declares EXPECTED-BY-DESIGN: a change the substrate makes on purpose beside the
    requested one. Counted as expected, reported separately."""

    model: str
    where: list[list[Any]] | None = None
    fields: list[str] | None = None
    note: str | None = None


@dataclass
class Cardinality:
    requested_count: int
    entities: list[dict[str, str]] = field(default_factory=list)
    model: str | None = None


CONFIRMATION_VALUES = ("confirm_once", "auto_confirm", "decline")


@dataclass
class Continuation:
    """The optional second turn. Either a plain follow-up message (`text`), or — Odoo 20 only — the substrate's
    own interaction protocol, each part sent only when Odoo records the session as waiting for it:

      on_question       a text reply, posted only if the session waits for an answer to a question
      on_confirmation   a STRUCTURED confirmation (confirm_once | auto_confirm | decline), sent only if Odoo
                        requests one; never typed as a chat message, which Odoo would read as a new request

    For the structured form `text` holds its canonical encoding (`confirmation:<value>` or
    `on_question:<text>;on_confirmation:<value>`), the form earlier scenario files were written in, so
    a driver that takes a string receives exactly what it always did. A driver that cannot send the protocol
    refuses such a scenario before any environment is created."""

    text: str
    when: str = "key_not_satisfied"    # key_not_satisfied | always
    on_question: str | None = None
    on_confirmation: str | None = None

    @property
    def structured(self) -> bool:
        return self.on_question is not None or self.on_confirmation is not None


def parse_continuation(raw: Any, ctx: str) -> Continuation:
    """A continuation from YAML: a string, `{text, when}`, or `{on_question, on_confirmation, when}`. A text that
    starts with `confirmation:` or `on_question:` is read as the structured protocol's canonical encoding."""
    c = raw if isinstance(raw, dict) else {"text": raw}
    when = str(c.get("when", "key_not_satisfied"))
    text, question, confirm = c.get("text"), c.get("on_question"), c.get("on_confirmation")
    if text is not None and (question is not None or confirm is not None):
        raise ScenarioError(f"{ctx}: continuation takes `text` OR `on_question`/`on_confirmation`, not both")
    if text is not None:
        text = str(text)
        if text.startswith("confirmation:"):
            confirm, text = text.split(":", 1)[1].strip(), None
        elif text.startswith(("on_question:", "on_confirmation:")):
            spec: dict[str, str] = {}
            for part in text.split(";"):
                if ":" not in part:
                    raise ScenarioError(f"{ctx}: cannot read the continuation protocol string {text!r}")
                k, v = part.split(":", 1)
                spec[k.strip()] = v.strip()
            if set(spec) - {"on_question", "on_confirmation"}:
                raise ScenarioError(f"{ctx}: unknown continuation keys {sorted(set(spec) - {'on_question', 'on_confirmation'})}")
            question, confirm, text = spec.get("on_question") or None, spec.get("on_confirmation") or None, None
    if question is None and confirm is None:
        if not text or not text.strip():
            raise ScenarioError(f"{ctx}: continuation needs `text`, or `on_question` / `on_confirmation`")
        return Continuation(text, when)
    if confirm is not None and confirm not in CONFIRMATION_VALUES:
        raise ScenarioError(f"{ctx}: on_confirmation must be one of {CONFIRMATION_VALUES}, got {confirm!r}")
    if question is not None:
        question = str(question)
        if not question.strip() or ";" in question:
            raise ScenarioError(f"{ctx}: on_question must be a non-empty message without ';'")
    canonical = (f"confirmation:{confirm}" if question is None
                 else f"on_question:{question}" + (f";on_confirmation:{confirm}" if confirm else ""))
    return Continuation(canonical, when, question, confirm)


@dataclass
class Scenario:
    name: str
    turns: list[str]
    title: str | None = None
    fixture: str = "default"
    benchmark_date: str | None = None
    required_capabilities: list[str] = field(default_factory=list)
    required_session: str | None = None       # internal | public
    required_agent: str = "default"           # which agent ROLE of the run profile this scenario needs
    required_odoo: list[str] = field(default_factory=list)   # Odoo major versions it is written for; [] = any
    continuation: Continuation | None = None
    kind: str | None = None                   # read | create | update | None (Level 1)
    read: ReadFacts | None = None
    create: CreateKey | None = None
    update: UpdateKey | None = None
    forbid: list[Forbid] = field(default_factory=list)
    side_effects: list[SideEffect] = field(default_factory=list)
    cardinality: Cardinality | None = None
    response_rules: list[ResponseRule] = field(default_factory=list)
    safety_profiles: list[str] = field(default_factory=list)
    extra_business_tables: list[str] = field(default_factory=list)
    source_path: str | None = None

    @property
    def level(self) -> int:
        return 2 if self.kind else 1

    @property
    def prompt(self) -> str:
        return self.turns[0]

    def validate(self) -> None:
        if not self.turns or not self.turns[0].strip():
            raise ScenarioError(f"{self.name}: at least one turn (the task prompt) is required")
        if self.level == 1 and self.continuation and self.continuation.when != "always":
            raise ScenarioError(
                f"{self.name}: Level 1 has no key, so a continuation must say `when: always` explicitly"
            )
        if self.continuation and self.continuation.when not in ("always", "key_not_satisfied"):
            raise ScenarioError(f"{self.name}: continuation.when must be always | key_not_satisfied")
        if self.kind == "read" and self.read is None:
            raise ScenarioError(f"{self.name}: expect.kind read needs `facts`")
        if self.kind == "create" and self.create is None:
            raise ScenarioError(f"{self.name}: expect.kind create needs model and count")
        if self.kind == "update" and self.update is None:
            raise ScenarioError(f"{self.name}: expect.kind update needs select")
        if self.cardinality and self.level == 1:
            raise ScenarioError(f"{self.name}: cardinality is a Level 2 assertion")

    def models_named(self) -> set[str]:
        out: set[str] = set()
        if self.create:
            out.add(self.create.model)
        if self.update:
            out.add(self.update.select.model)
        for f in self.forbid:
            out.add(f.model)
        for s in self.side_effects:
            out.add(s.model)
        if self.cardinality and self.cardinality.model:
            out.add(self.cardinality.model)
        return out


def _rules(raw: dict | None, level: int, ctx: str) -> list[ResponseRule]:
    out = []
    for verdict in ("success", "failure", "neutral", "clarification", "refused"):
        for p in (raw or {}).get(verdict, []) or []:
            if isinstance(p, str) and p.startswith("regex:"):
                out.append(ResponseRule(verdict, p[6:], True, "scenario"))
            elif isinstance(p, str):
                out.append(ResponseRule(verdict, p, False, "scenario"))
            else:
                raise ScenarioError(f"{ctx}: report rule must be a string or 'regex:<pattern>'")
    return out


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    ctx = str(path)
    name = raw.get("name") or path.stem
    for banned in ("model_identifier", "provider", "llm_model", "profile"):
        if banned in raw:
            raise ScenarioError(f"{ctx}: `{banned}` belongs in a run profile, never in a scenario")
    turns = raw.get("turns") or ([raw["prompt"]] if raw.get("prompt") else [])
    if isinstance(turns, str):
        turns = [turns]
    cont = parse_continuation(raw["continuation"], ctx) if raw.get("continuation") else None
    req = raw.get("requires", {}) or {}
    exp = raw.get("expect") or {}
    kind = exp.get("kind")
    sc = Scenario(
        name=name, turns=[str(t) for t in turns], title=raw.get("title"), fixture=raw.get("fixture", "default"),
        benchmark_date=str(raw["benchmark_date"]) if raw.get("benchmark_date") else None,
        required_capabilities=list(req.get("capabilities", []) or []), required_session=req.get("session"),
        required_agent=str(req.get("agent", "default")),
        required_odoo=[str(v) for v in (req.get("odoo") if isinstance(req.get("odoo"), list) else [req["odoo"]])]
        if req.get("odoo") is not None else [],
        continuation=cont, kind=kind, source_path=str(path),
        safety_profiles=list(raw.get("safety_profiles", []) or []),
        extra_business_tables=list(raw.get("extra_business_tables", []) or []),
    )
    if kind == "read":
        f = exp.get("facts", {}) or {}
        for fact in f.get("must_not_contain", []) or []:
            if isinstance(fact, dict) and "amount" in fact:
                raise ScenarioError(
                    f"{ctx}: `{{amount: {fact['amount']}}}` in must_not_contain. A correct answer may name an amount it "
                    "excludes. If this value can only appear when the wrong set was aggregated, "
                    f"write `{{wrong_aggregate: {fact['amount']}}}`; if a correct answer could legitimately mention it, "
                    "it cannot be a must_not fact at all."
                )
        sc.read = ReadFacts(list(f.get("must_contain", []) or []), list(f.get("should_contain", []) or []),
                            list(f.get("must_not_contain", []) or []), int(exp.get("business_writes", 0)))
    elif kind == "create":
        sc.create = CreateKey(exp["model"], int(exp.get("count", 1)), dict(exp.get("match", {}) or {}), dict(exp.get("values", {}) or {}))
    elif kind == "update":
        sc.update = UpdateKey(Selector.parse(exp["select"], ctx), dict(exp.get("values", {}) or {}), exp.get("fields_only"))
    elif kind is not None:
        raise ScenarioError(f"{ctx}: expect.kind must be read | create | update")
    for fr in raw.get("forbid", []) or []:
        sc.forbid.append(Forbid(fr["model"], bool(fr.get("any")), bool(fr.get("existing")), fr.get("where"), fr.get("fields"), fr.get("note")))
    for se in raw.get("expected_side_effects", []) or []:
        sc.side_effects.append(SideEffect(se["model"], se.get("where"), se.get("fields"), se.get("note")))
    if raw.get("cardinality"):
        c = raw["cardinality"]
        sc.cardinality = Cardinality(int(c["requested_count"]), list(c.get("entities", []) or []), c.get("model"))
    sc.response_rules = _rules(raw.get("report"), sc.level, ctx)
    sc.validate()
    return sc


def level1_scenario(name: str, prompt: str, continuation: str | None, safety_profiles: list[str],
                    fixture: str = "default", required_capabilities: list[str] | None = None) -> Scenario:
    """The onboarding path: agent + one representative task prompt + a profile. No assertions."""
    sc = Scenario(name=name, turns=[prompt], fixture=fixture, safety_profiles=safety_profiles,
                  required_capabilities=required_capabilities or [],
                  continuation=parse_continuation({"text": continuation, "when": "always"}, "--continuation") if continuation else None)
    sc.validate()
    return sc


# ---------------------------------------------------------------------------------------------
# Resolution: selectors -> ids, refs -> values, against the run environment before the agent starts


@dataclass
class Resolved:
    """Snapshot of the scenario's expectations resolved against the fixture."""

    tables: dict[str, str] = field(default_factory=dict)              # model -> table
    update_ids: list[Any] = field(default_factory=list)
    update_values: dict[str, Any] = field(default_factory=dict)
    create_values: dict[str, Any] = field(default_factory=dict)
    create_match: dict[str, Any] = field(default_factory=dict)
    forbid_ids: dict[int, list[Any] | None] = field(default_factory=dict)      # forbid index -> ids (None = any)
    side_effect_ids: dict[int, list[Any] | None] = field(default_factory=dict)
    existing_ids: dict[str, set[Any]] = field(default_factory=dict)            # table -> ids before the run
    notes: list[str] = field(default_factory=list)


class Resolver:
    def __init__(self, observer_dsn: str):
        self.dsn = observer_dsn
        self._models: dict[str, str] | None = None

    def known_models(self) -> dict[str, str]:
        """model -> table, from ir_model. Odoo's default `_table` rule; the classification config
        holds overrides for the few that differ."""
        if self._models is None:
            with psycopg.connect(self.dsn) as c:
                rows = c.execute("select model from ir_model where transient = false").fetchall()
            self._models = {m: m.replace(".", "_") for (m,) in rows}
        return self._models

    def table(self, model: str) -> str:
        t = self.known_models().get(model)
        if not t:
            raise ScenarioError(f"model {model!r} is not installed in the fixture")
        return t

    def _value(self, v: Any) -> Any:
        if isinstance(v, dict) and "ref" in v:
            sel = Selector.parse(v["ref"], "ref")
            ids = self.resolve_ids(sel)
            if len(ids) != 1:
                raise ScenarioError(f"ref {v['ref']!r} resolved to {len(ids)} rows; a ref must name exactly one")
            return ids[0]
        return v

    @staticmethod
    def _as_text(v: Any) -> str:
        """The text PostgreSQL renders for the value, so `[active, =, true]` compares against 'true'
        (not Python's 'True') and numbers keep their YAML spelling."""
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    def _where_sql(self, where: list[list[Any]]) -> tuple[sql.Composed, list[Any]]:
        parts, params = [], []
        for clause in where:
            f, op = clause[0], clause[1]
            value = self._value(clause[2]) if len(clause) == 3 else None
            if op in ("is null", "is not null"):
                parts.append(sql.SQL("{} {}").format(sql.Identifier(f), sql.SQL(OPS[op])))
            elif op in ("in", "not in"):
                vals = [self._value(x) for x in clause[2]]
                if any(x is None for x in vals):
                    raise ScenarioError(f"selector {clause!r}: null cannot be a member of an `in` list; use `is null`")
                # compared as text so a selector can name ids, names or codes without a type map
                parts.append(sql.SQL("{}::text {} (SELECT unnest(%s::text[]))").format(sql.Identifier(f), sql.SQL(OPS[op])))
                params.append([self._as_text(x) for x in vals])
            elif value is None:
                if op not in ("=", "!="):
                    raise ScenarioError(f"selector {clause!r}: null only supports = and != (or `is null`)")
                parts.append(sql.SQL("{} {}").format(sql.Identifier(f), sql.SQL("IS NULL" if op == "=" else "IS NOT NULL")))
            else:
                parts.append(sql.SQL("{}::text {} %s::text").format(sql.Identifier(f), sql.SQL(OPS[op])))
                params.append(self._as_text(value))
        if not parts:
            return sql.SQL("true"), []
        return sql.SQL(" AND ").join(parts), params

    def resolve_ids(self, sel: Selector) -> list[Any]:
        t = self.table(sel.model)
        cond, params = self._where_sql(sel.where)
        with psycopg.connect(self.dsn) as c:
            rows = c.execute(sql.SQL("select id from {} where {} order by id").format(sql.Identifier(t), cond), params).fetchall()
        return [r[0] for r in rows]

    def resolve(self, sc: Scenario) -> Resolved:
        r = Resolved()
        required = {sc.create.model} if sc.create else ({sc.update.select.model} if sc.update else set())
        for m in sc.models_named():
            if m in self.known_models():
                r.tables[m] = self.table(m)
            elif m in required:
                raise ScenarioError(f"{sc.name}: expected model {m!r} is not installed in the fixture")
            else:
                # a forbid / side effect on a model the fixture does not have is vacuously satisfied
                r.notes.append(f"model {m!r} is not installed in the fixture; its forbid/side-effect rules cannot fire")
        if sc.update:
            r.update_ids = self.resolve_ids(sc.update.select)
            if not r.update_ids:
                raise ScenarioError(f"{sc.name}: update selector matched no rows in the fixture")
            r.update_values = {k: self._value(v) for k, v in sc.update.values.items()}
        if sc.create:
            r.create_values = {k: self._value(v) for k, v in sc.create.values.items()}
            r.create_match = {k: self._value(v) for k, v in sc.create.match.items()}
        for i, f in enumerate(sc.forbid):
            r.forbid_ids[i] = self.resolve_ids(Selector(f.model, f.where)) if f.where and f.model in r.tables else None
        for i, s in enumerate(sc.side_effects):
            r.side_effect_ids[i] = self.resolve_ids(Selector(s.model, s.where)) if s.where and s.model in r.tables else None
        with psycopg.connect(self.dsn) as c:
            for m, t in r.tables.items():
                r.existing_ids[t] = {row[0] for row in c.execute(sql.SQL("select id from {}").format(sql.Identifier(t)))}
        return r


# ---------------------------------------------------------------------------------------------
# Deterministic text normalisation for READ facts. Never a model.

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def plain_text(html_or_text: str | None) -> str:
    import html as _html

    if not html_or_text:
        return ""
    return _WS.sub(" ", _html.unescape(_TAG.sub(" ", html_or_text))).strip()


def amount_renderings(x: float | str) -> list[str]:
    """The finite set of deterministic renderings an amount may take in a reply. A forward-looking
    constraint from the spec: `14,260.00`, `14260` and `14.260,00` can all be the same amount."""
    from decimal import Decimal

    d = Decimal(str(x))
    neg = d < 0
    d = abs(d)
    whole, _, frac = f"{d:.2f}".partition(".")
    grouped_comma = f"{int(whole):,}"
    grouped_dot = grouped_comma.replace(",", ".")
    grouped_space = grouped_comma.replace(",", " ")
    out = {
        f"{grouped_comma}.{frac}", f"{whole}.{frac}", f"{grouped_dot},{frac}", f"{grouped_space},{frac}",
        f"{grouped_space}.{frac}", f"{whole},{frac}",
    }
    if frac == "00":
        out |= {grouped_comma, whole, grouped_dot, grouped_space, f"{whole}.0", f"{grouped_comma}.0"}
    elif frac.endswith("0"):
        out |= {f"{grouped_comma}.{frac[0]}", f"{whole}.{frac[0]}", f"{grouped_dot},{frac[0]}"}
    return sorted((("-" + s) if neg else s) for s in out)


def fact_present(fact: Any, text: str) -> bool:
    if isinstance(fact, dict):
        if "wrong_aggregate" in fact:
            # an aggregate (total, count) that only the wrong inclusion set produces; the one
            # deterministic way to assert "not reported as included" on prose
            return fact_present({"amount": fact["wrong_aggregate"]}, text)
        if "amount" in fact:
            # boundary-aware: 1200 must not match inside 12000, 1200.5 or 31200
            return any(re.search(r"(?<![\d.,])" + re.escape(r) + r"(?!\d|[.,]\d)", text) for r in amount_renderings(fact["amount"]))
        if "regex" in fact:
            return re.search(fact["regex"], text) is not None
        if "any" in fact:
            return any(fact_present(f, text) for f in fact["any"])
        raise ScenarioError(f"unknown fact form {fact!r}")
    if isinstance(fact, str) and fact.startswith("regex:"):   # same prefix the report rules accept
        return re.search(fact[6:], text) is not None
    return str(fact) in text
