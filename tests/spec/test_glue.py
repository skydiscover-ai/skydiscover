"""Tests for the spec-synthesis glue (requirements / cards).

These modules are pure-stdlib. We load them directly from their files so the test does not
trigger ``skydiscover/__init__`` (which pulls heavy optional deps); the same tests therefore
pass under a bare ``python3`` and under the full project env.

The guiding rule under test: questions elicit **properties** (what the system must do), never
**designs** (how to build it). ``lint`` enforces that mechanically.
"""

import importlib.util
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PKG = _ROOT / "skydiscover" / "synthesize" / "spec"
# Grounded spec fixtures: a spec.json per reference system, plus axes and design principles for faster.
_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "specs"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_specsyn_{name.replace('/', '_')}", _PKG / f"{name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


requirements = _load("requirements")
cards = _load("build")


def _fx(system, artifact):
    with open(_FIXTURES / system / f"{artifact}.json", encoding="utf-8") as f:
        return json.load(f)


# A hand-authored property questionnaire (what discovery produces).
# Options are property VALUES -- no system names, no mechanisms.
_GOOD_REQS = {
    "domain": "key-value store",
    "questions": [
        {
            "id": "working-set",
            "axis": "memory",
            "q": "Is your working set expected to fit in RAM, or be larger than memory?",
            "why": "Decides whether the store is memory-resident or must tier/persist to disk.",
            "options": [
                {"value": "fits in RAM"},
                {"value": "larger than memory"},
                {"value": "unsure / mixed"},
            ],
            "default": "fits in RAM",
            "multi": False,
        },
        {
            "id": "consistency",
            "axis": "consistency",
            "q": "What consistency must operations provide?",
            "why": "Sets the correctness contract the implementation must uphold.",
            "options": [
                {"value": "per-operation atomicity only"},
                {"value": "multi-key transactions"},
                {"value": "cross-node strong consistency"},
            ],
            "default": "per-operation atomicity only",
            "multi": False,
        },
        {
            "id": "durability",
            "axis": "durability",
            "q": "What may a crash lose?",
            "why": "Sets the recovery contract and acceptable loss window.",
            "options": [
                {"value": "may lose everything (cache/benchmark)"},
                {"value": "bounded loss window"},
                {"value": "no acknowledged write may be lost"},
            ],
            "default": "bounded loss window",
            "multi": False,
        },
    ],
}

# A questionnaire that leaks DESIGNS into the question -- lint must reject this.
_LEAKY_REQS = {
    "domain": "key-value store",
    "questions": [
        {
            "id": "execution",
            "axis": "concurrency",
            "q": "Should the keyspace be latch-free (FASTER) or single-threaded (Redis)?",
            "why": "Pick the execution model.",
            "options": [{"value": "latch-free + epoch"}, {"value": "single-thread"}],
        }
    ],
}


# requirements: scaffold


def test_gather_value_space_is_grounded_in_pulled_specs():
    # The consistency value-space must come from what the real pulled systems guarantee.
    specs = {s: _fx(s, "spec") for s in ("redis", "rocksdb", "faster")}
    vs = requirements.gather_value_space("consistency", specs)
    froms = {v["from"] for v in vs}
    assert froms == {"redis", "rocksdb", "faster"}
    blob = json.dumps(vs).lower()
    # the real spectrum surfaces (eventual + snapshot + monotonic), not generic guesses
    assert "eventual" in blob and "snapshot" in blob and "monoton" in blob


def test_seed_from_specs_attaches_value_space():
    specs = {s: _fx(s, "spec") for s in ("redis", "rocksdb", "faster")}
    doc = requirements.seed_from_specs(_fx("faster", "axes"), specs)
    cons = next(q for q in doc["questions"] if q["axis"] == "consistency")
    assert cons["value_space"], "each axis slot should carry its real value-space for authoring"


def test_evidence_field_is_not_linted():
    # Provenance (system names) belongs in an internal 'evidence' field; lint must ignore it.
    doc = {
        "questions": [
            {
                "id": "c",
                "q": "What must a read observe?",
                "why": "the consistency model",
                "options": [{"value": "strong"}, {"value": "eventual"}],
                "evidence": [{"from": "redis", "guarantee": "eventual when replicated"}],
            }
        ]
    }
    assert requirements.lint(doc, ["redis"]) == []


def test_seed_from_axes_one_slot_per_axis():
    axes = _fx("faster", "axes")
    doc = requirements.seed_from_axes(axes)
    assert len(doc["questions"]) == len(axes["axes"])
    ids = [q["id"] for q in doc["questions"]]
    assert ids == [a["id"] for a in axes["axes"]]
    # scaffold leaves options empty for the skill to fill with property values
    assert all(q["options"] == [] for q in doc["questions"])


# requirements: validate


def test_validate_accepts_good_property_questions():
    assert requirements.validate(_GOOD_REQS) == []


def test_validate_rejects_missing_options_and_design_refs():
    bad = {
        "questions": [
            {"id": "x", "q": "?", "why": "because", "options": [{"value": "a"}]},  # <2 options
            {
                "id": "y",
                "q": "?",
                "why": "because",
                "options": [{"value": "a", "ref": "Redis"}, {"value": "b"}],  # design ref
            },
        ]
    }
    problems = requirements.validate(bad)
    assert any(">=2 value options" in p for p in problems)
    assert any("not designs" in p for p in problems)


# requirements: lint (the key rule)


def test_lint_passes_property_questions():
    assert requirements.lint(_GOOD_REQS, ["FASTER", "Redis", "latch-free", "epoch"]) == []


def test_lint_flags_the_words_it_is_given_and_nothing_else():
    assert requirements.lint(_LEAKY_REQS) == []  # no words, no list of designs built in
    terms = {f["term"] for f in requirements.lint(_LEAKY_REQS, ["FASTER", "Redis", "latch-free"])}
    assert terms == {"faster", "redis", "latch-free"}


def test_lint_matches_whole_words_only():
    doc = {
        "questions": [
            {"id": "a", "q": "How fast must a read be?", "why": "w", "options": [{"value": "yes"}]}
        ]
    }
    assert requirements.lint(doc, ["FASTER"]) == []  # "fast" is not "faster"
    doc["questions"][0]["q"] = "Should reclamation use epochs?"
    assert [f["term"] for f in requirements.lint(doc, ["epoch"])] == ["epoch"]  # plurals count


def test_terms_come_from_the_run(tmp_path):
    """The forbidden words are the reference systems' names (folder and spec.json source) and the
    mechanisms axes.json lists; nothing is built in."""
    refs = tmp_path / "references"
    (refs / "faster").mkdir(parents=True)
    (refs / "faster" / "spec.json").write_text(
        json.dumps({"source": "microsoft/FASTER @ 7f71289 (cc/src)", "axes": {}})
    )
    (refs / "rocksdb").mkdir()
    (refs / "axes.json").write_text(
        json.dumps({"axes": [], "mechanisms": ["latch-free index", "HybridLog", "epoch"]})
    )
    terms = requirements.terms_from(tmp_path)
    assert [t.lower() for t in terms] == [
        "latch-free index",
        "hybridlog",
        "epoch",
        "faster",
        "rocksdb",
    ]
    assert requirements.terms_from(tmp_path / "nowhere") == []


# cards


def test_cards_carry_required_properties_not_designs():
    spec = _fx("faster", "spec")
    # the shape spec.decisions exports to answers.json
    r = {
        "decisions": [
            {
                "id": "working-set",
                "axis": "memory",
                "chosen": "larger than memory",
                "answered_by": "human",
            },
            {
                "id": "durability",
                "axis": "durability",
                "chosen": "bounded loss window",
                "answered_by": "human",
            },
        ]
    }
    sc = cards.build_requirements_card(
        spec,
        design_principles=_fx("faster", "design_principles"),
        answers=r,
        base=_requirements_card_base(),
    )
    # the property profile is attached as requirements, not design decisions
    assert "required_properties" in sc and "prior_decisions" not in sc
    rp = {x["axis"]: x["value"] for x in sc["required_properties"]}
    assert rp["memory"] == "larger than memory"
    assert rp["durability"] == "bounded loss window"
    # environment scaffold preserved; spec axes overlaid; the interface passed through whole
    assert sc["hardware"]["machine"] == _requirements_card_base()["hardware"]["machine"]
    assert "spec_axes" in sc
    assert sc["api"]["operations"] == spec["interface"]["operations"]


def test_cards_json_serializable_all_systems():
    for system in ("redis", "rocksdb", "faster"):
        out = cards.build_requirements_card(_fx(system, "spec"), base=_requirements_card_base())
        json.dumps(out)


# helpers


def _requirements_card_base():
    p = _ROOT / "skydiscover/synthesize/examples/single-machine-kvstore/spec/requirements.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)
