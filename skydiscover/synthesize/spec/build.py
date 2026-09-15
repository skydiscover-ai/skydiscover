"""Build cards/requirements.json, the specification card the synthesis loop reads: the interface,
the guarantees, the required properties, and the operating point.

The system's own fields (interface, guarantees, design space) come from spec.json; the required
properties from answers.json; harness fields (hardware, benchmark) from a base card when the
caller supplies one. The other two cards, cards/workload.json and cards/environment.json, are
the spec-builder's measurements and are not touched here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional

_DEFAULT_BASE: Dict[str, Any] = {
    "title": "Requirements Card",
    "optimization_objective": "Maximize the spec's stated objective.",
    "design_space": {"note": "Open -- any architecture that satisfies the spec is acceptable."},
}


def build_requirements_card(
    spec: Dict[str, Any],
    *,
    design_principles: Optional[Dict[str, Any]] = None,
    answers: Optional[Dict[str, Any]] = None,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Overlay spec.json and answers.json onto a base requirements card."""
    card = copy.deepcopy(base if base is not None else _DEFAULT_BASE)

    for key in ("title", "source", "purpose", "operating_point"):
        if spec.get(key):
            card[key] = spec[key]

    # The interface is whatever the spec declares, passed through whole under `api`.
    iface = spec.get("interface") or {}
    if iface:
        card["api"] = {**(card.get("api") or {}), **iface}

    # Guarantees from axes ({name: {property, guarantee, evidence}}) or a contract ({name: statement}).
    # Declared but unreadable is an error, not an empty card.
    axes = spec.get("axes") or {}
    contract = spec.get("contract") or {}
    spec_axes: Dict[str, Any] = {}
    if isinstance(axes, dict):
        for name, a in axes.items():
            if isinstance(a, dict):
                spec_axes[name] = {
                    "property": a.get("property"),
                    "guarantee": a.get("guarantee"),
                    "evidence": a.get("evidence"),
                }
            elif a:
                spec_axes[name] = {"guarantee": a}
    if isinstance(contract, dict):
        for name, stmt in contract.items():
            if name not in spec_axes and stmt:
                spec_axes[name] = {"guarantee": stmt}
    if spec_axes:
        card["spec_axes"] = spec_axes
    elif axes or contract:
        raise ValueError(
            "spec declares axes/contract but none mapped to guarantees -- the requirements card "
            "would ship with no guarantees. Check the spec shape passed to build_requirements_card."
        )

    if spec.get("design_space_notes"):
        ds = dict(card.get("design_space") or {})
        ds["from_spec"] = spec["design_space_notes"]
        card["design_space"] = ds

    if design_principles:
        card["design_principles"] = _principles_list(design_principles)

    # The required properties: answers.json's decisions[], as spec.decisions exports it.
    # Rows that are present but unreadable are an error, not an empty card.
    if answers:
        decisions = answers.get("decisions")
        chosen = [
            {
                "id": d.get("id") or d.get("axis"),
                "axis": d.get("axis"),
                "requirement": d.get("q") or d.get("property"),
                "value": d.get("chosen"),
                "by": d.get("answered_by"),
                "rationale": d.get("rationale", ""),
            }
            for d in (decisions if isinstance(decisions, list) else [])
            if isinstance(d, dict) and d.get("chosen")
        ]
        if chosen:
            card["required_properties"] = chosen
        elif decisions:
            raise ValueError(
                "answers.json has decision rows but none carries `chosen`; the requirements card "
                "would ship with no requirements. Write answers.json with spec.decisions."
            )

    # The operating point: kept from the base when supplied, else an empty placeholder so the field
    # is visible. The evaluator writes at least one test that runs at this load.
    if "operating_point" not in card:
        card["operating_point"] = {
            "_todo": "FILL IN for THIS system: the load the headline score is measured at "
            "(cards/workload.json's scored_configuration; a bare number means at least that, "
            "or write {min|max|eq: N}).",
            "params": {},
        }

    return card


def _principles_list(design_principles: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten design_principles.json ({principle_N_slug: {name, summary, ...}}) into a list."""
    out: List[Dict[str, Any]] = []
    for key, p in design_principles.items():
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "name": p.get("name", key),
                "summary": p.get("summary"),
                "tradeoff": p.get("tradeoff"),
                "evidence": p.get("evidence"),
            }
        )
    return out


def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m skydiscover.synthesize.spec.build",
        description="Compile spec.json and answers.json into cards/requirements.json.",
    )
    ap.add_argument("spec", help="specification/spec.json")
    ap.add_argument("--answers", help="specification/answers.json (from spec.decisions)")
    ap.add_argument("-d", "--out-dir", required=True, help="the cards/ directory to write into")
    args = ap.parse_args(argv)

    card = build_requirements_card(_load(args.spec), answers=_load(args.answers))
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "requirements.json")
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(card, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    from .paths import run_cli

    raise SystemExit(run_cli(main, "build"))
