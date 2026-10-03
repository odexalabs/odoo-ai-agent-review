"""Cost is a first-class output beside correctness. Two labels, never merged:
  reported cost    provider-returned, measured
  estimated cost   tokens x published price, with pricing source, date and separate
                   input / output / cached rates recorded
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..resources import bundled
from .contracts import TokenUsage

DEFAULT_PRICING = bundled("config", "pricing.yaml")


@dataclass
class Price:
    key: str
    input: float
    cached_input: float
    output: float
    source: str
    read_on: str


class Pricing:
    def __init__(self, path: Path | None = None):
        with open(path or DEFAULT_PRICING) as fh:
            raw = yaml.safe_load(fh) or {}
        self.models = {k: Price(k, float(v["input"]), float(v["cached_input"]), float(v["output"]), v["source"], str(v["read_on"]))
                       for k, v in (raw.get("models") or {}).items()}

    def get(self, provider: str, model: str) -> Price | None:
        return self.models.get(f"{provider}/{model}")

    def estimate(self, provider: str, model: str, usage: TokenUsage | None) -> dict:
        p = self.get(provider, model)
        if usage is None or p is None:
            return {"label": "estimated", "usd": None,
                    "reason": "no token usage reported" if usage is None else f"no published price for {provider}/{model}"}
        uncached = max(usage.input_tokens - usage.cached_input_tokens, 0)
        usd = (uncached * p.input + usage.cached_input_tokens * p.cached_input + usage.output_tokens * p.output) / 1e6
        out = {"label": "estimated", "usd": round(usd, 6), "pricing_source": p.source, "pricing_read_on": p.read_on,
               "rates_per_1m": {"input": p.input, "cached_input": p.cached_input, "output": p.output},
               "tokens": {"input": usage.input_tokens, "cached_input": usage.cached_input_tokens, "output": usage.output_tokens}}
        if usage.partial:
            # a lower bound: some response's usage was not observed. Never presented as the cost.
            out.update({"usd": None, "at_least_usd": round(usd, 6), "tokens_partial": usage.partial,
                        "reason": f"token usage incomplete ({usage.partial}); at least ${usd:.4f}"})
        return out
