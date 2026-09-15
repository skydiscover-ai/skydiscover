"""Router policy interface. The harness owns the clock, the fleet, billing, and all
correctness/SLO accounting; the router owns only placement policy."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Request:
    """One trace record."""

    req_id: int
    t_ms: int
    session_id: str
    cls: str
    model_requested: str
    equiv_class: list
    prompt_tokens: int
    prefix_id: str
    prefix_tokens: int
    expected_output_tokens: int
    max_tokens: int
    stream: bool
    slo: dict
    downgrade_ok: bool
    retry_safe: bool
    temperature: float
    # ground-truth per-model quality in [0,1] for models in equiv_class, and the floor the
    # deployment tolerates. The harness scores with these; whether a *router* may read them
    # (oracle) or must estimate from features is a per-experiment choice; see task.md.
    quality: dict = None  # type: ignore[assignment]
    quality_floor: float = 0.0
    # observable per-request features a deployed router may legitimately use
    # (e.g. a noisy difficulty signal extracted from the prompt)
    features: dict = None  # type: ignore[assignment]

    @classmethod
    def from_json(cls, d):
        return cls(
            req_id=d["req_id"],
            t_ms=d["t_ms"],
            session_id=d["session_id"],
            cls=d["class"],
            model_requested=d["model_requested"],
            equiv_class=d["equiv_class"],
            prompt_tokens=d["prompt_tokens"],
            prefix_id=d["prefix_id"],
            prefix_tokens=d["prefix_tokens"],
            expected_output_tokens=d["expected_output_tokens"],
            max_tokens=d["max_tokens"],
            stream=d["stream"],
            slo=d["slo"],
            downgrade_ok=d["downgrade_ok"],
            retry_safe=d["retry_safe"],
            temperature=d["temperature"],
            quality=d.get("quality", {}),
            quality_floor=d.get("quality_floor", 0.0),
            features=d.get("features", {}),
        )


@dataclass
class Action:
    """DISPATCH(provider, model) sends now; DEFER(until_ms) re-invokes decide() at that
    time; SHED rejects explicitly (legal only under declared overload)."""

    kind: str  # "dispatch" | "defer" | "shed"
    provider: Optional[str] = None
    model: Optional[str] = None
    until_ms: Optional[int] = None


class Router:
    """Subclass and implement decide(). The fleet_view is read-only truth about the fleet:
    {name: {"models": {model: {"in","out","cached"}}, "inflight": int, "rpm_left": int,
    "tpm_left": int, "error_rate": float, "has_prefix": callable(prefix_id)->bool}}.
    Mutating it is a correctness violation."""

    def decide(self, req: Request, now_ms: int, fleet_view: dict) -> Action:
        raise NotImplementedError

    def on_error(self, req: Request, now_ms: int, kind: str, retry_after_ms: int) -> None:
        """Informational: a dispatch failed (429/503). decide() is re-invoked afterwards."""

    def on_complete(self, req: Request, now_ms: int) -> Optional[Action]:
        """Informational hook after a request completes. Returning an Action re-enters the
        harness (a correct router returns None; the test's double-bill mutant abuses this)."""
        return None
