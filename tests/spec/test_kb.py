"""Read-only smoke test for the kept tests. The write path (record/lookup/sync) is covered end-to-end
in test_run_finish.py against an isolated temp knowledge base."""

from skydiscover.synthesize.spec import kept_tests


def test_lookup_unknown_domain_is_all_gaps(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYDISCOVER_HOME", str(tmp_path))
    r = kept_tests.lookup("nonesuch", [{"id": "x", "text": "y"}])
    assert r["candidates"] == [] and len(r["gaps"]) == 1
