"""Tests for the findings store's fail-closed read/write contract (``spec/findings.py``).

The rule under test: an ABSENT store is empty, a PRESENT but unparseable store is a hard error. The
distinction is the whole game -- every mutator is a read-modify-write on top of ``all()``, so if a
corrupt file read as ``[]`` the prod-ready checks would see no defects AND the next write would
overwrite the file, destroying the user's answers in it.
"""

import json
from dataclasses import asdict

import pytest

from skydiscover.synthesize.spec.findings import (
    Finding,
    Findings,
    FindingsUnreadable,
    FindingsWriteRefused,
)


def _defect(fid="d1"):
    return Finding(
        id=fid,
        title="unflushed write",
        kind="spec",
        detail="acknowledged writes must survive a crash",
        severity="defect",
        status="confirmed",
        answered_by="human",
        asked=True,
    )


def _seeded(tmp_path, n=2):
    p = tmp_path / "decision_log.json"
    store = Findings(p)
    for i in range(n):
        store.add(_defect(f"d{i}"))
    return p, store


def _junk_row(fid):
    """A row hand-written past the store, with a free-form kind ``all()`` cannot reconstruct."""
    return {"id": fid, "title": "hand-written", "kind": "build", "detail": "bypassed the store"}


def _write_raw(p, rows):
    p.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


# reads


def test_an_absent_store_is_empty(tmp_path):
    store = Findings(tmp_path / "never-written.json")
    assert store.all() == []
    assert store.open_defects() == [] and store.for_spec() == []


def test_a_truncated_store_raises_instead_of_reading_as_empty(tmp_path):
    """A write cut short by a crash must raise on every read path, never read as ``[]``:
    open_defects() blocks the release while it is non-empty, so a corrupt log that read as empty
    would answer 'no open defects'."""
    p, _ = _seeded(tmp_path)
    p.write_text(p.read_text()[:40], encoding="utf-8")
    for read in (lambda: Findings(p).all(), Findings(p).open_defects, Findings(p).for_spec):
        with pytest.raises(FindingsUnreadable):
            read()


def test_a_store_that_is_not_a_list_raises(tmp_path):
    p = tmp_path / "decision_log.json"
    p.write_text(json.dumps({"findings": []}), encoding="utf-8")
    with pytest.raises(FindingsUnreadable):
        Findings(p).all()


def test_an_empty_file_raises(tmp_path):
    p = tmp_path / "decision_log.json"
    p.write_text("", encoding="utf-8")
    with pytest.raises(FindingsUnreadable):
        Findings(p).all()


# writes


def test_a_corrupt_store_is_not_overwritten_by_the_next_add(tmp_path):
    """The destructive half: add() re-reads, appends and saves. If the read had swallowed the
    corruption, this add would have replaced two of the user's answers with one row."""
    p, _ = _seeded(tmp_path)
    corrupt = p.read_text()[:40]
    p.write_text(corrupt, encoding="utf-8")
    with pytest.raises(FindingsUnreadable):
        Findings(p).add(_defect("d9"))
    assert p.read_text() == corrupt  # the bytes that were there are still there


def test_a_save_that_would_drop_rows_is_refused(tmp_path):
    p, store = _seeded(tmp_path, n=3)
    with pytest.raises(FindingsWriteRefused):
        store._save([_defect("d0")])
    assert len(json.loads(p.read_text())) == 3


def test_an_explicit_delete_may_shrink_the_store(tmp_path):
    p, store = _seeded(tmp_path, n=3)
    store._save([_defect("d0")], allow_shrink=True)
    assert [r["id"] for r in json.loads(p.read_text())] == ["d0"]


def test_a_write_over_an_unparseable_file_sets_its_bytes_aside(tmp_path):
    p, store = _seeded(tmp_path, n=2)
    corrupt = p.read_text()[:40]
    p.write_text(corrupt, encoding="utf-8")  # corrupted by another writer, after our read
    store._save([_defect("d0"), _defect("d1"), _defect("d2")])
    kept = list(tmp_path.glob("decision_log.json.corrupt.*"))
    assert len(kept) == 1 and kept[0].read_text() == corrupt
    assert len(json.loads(p.read_text())) == 3


def test_junk_rows_on_disk_do_not_wedge_later_writes(tmp_path):
    """A row hand-written past the store (a free-form kind that ``all()`` drops on read) must not
    make later writes look like a row-dropping shrink: the overwrite guard counts recoverable rows,
    so the store drops the junk and sets the pre-write bytes aside."""
    p, _ = _seeded(tmp_path, n=3)  # 3 valid, human-ruled
    _write_raw(p, json.loads(p.read_text()) + [_junk_row("j0"), _junk_row("j1")])

    store = Findings(p)
    assert len(store.all()) == 3  # junk is dropped on read

    assert store.set_status("d0", "waived", note="fixed", who="human") is True

    saved = {r["id"] for r in json.loads(p.read_text())}
    assert saved == {"d0", "d1", "d2"}  # every valid row kept, both junk rows gone

    sidecars = list(tmp_path.glob("decision_log.json.malformed.*"))
    assert len(sidecars) == 1  # the pre-write bytes set aside once
    assert {r["id"] for r in json.loads(sidecars[0].read_text())} == {"d0", "d1", "d2", "j0", "j1"}


def test_upsert_over_junk_rows_succeeds(tmp_path):
    p, _ = _seeded(tmp_path, n=2)
    _write_raw(p, json.loads(p.read_text()) + [_junk_row("j0")])
    store = Findings(p)
    replacement = _defect("d0")
    replacement.note = "re-answered"
    assert (
        store.upsert(replacement) is True
    )  # replacing a valid row must not be refused as a shrink
    assert {r["id"] for r in json.loads(p.read_text())} == {"d0", "d1"}


def test_a_valid_row_loss_still_refuses_even_with_junk_present(tmp_path):
    """Protection preserved: dropping a VALID row still raises, whether or not junk is also present."""
    p, store = _seeded(tmp_path, n=3)
    _write_raw(p, json.loads(p.read_text()) + [_junk_row("j0")])
    with pytest.raises(FindingsWriteRefused):
        store._save([_defect("d0")])  # 1 valid < 3 recoverable -> refused
    assert len(json.loads(p.read_text())) == 4  # decision_log.json untouched by the refused write


def test_the_ordinary_grow_and_update_paths_still_work(tmp_path):
    p, store = _seeded(tmp_path, n=1)
    assert store.add(_defect("d1")) is True
    assert store.set_status("d1", "waived", note="fixed", who="human") is True
    rows = {f.id: f for f in Findings(p).all()}
    assert set(rows) == {"d0", "d1"} and rows["d1"].status == "waived"
