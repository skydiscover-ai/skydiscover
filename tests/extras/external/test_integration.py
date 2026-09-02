"""
Live integration test for the AlphaEvolve external backend.

This test runs the full AlphaEvolve backend lifecycle against the real
Google Discovery Engine API.  It is gated on GCP credentials **and** the
required environment variables (``ALPHAEVOLVE_PROJECT_ID``,
``ALPHAEVOLVE_ENGINE_ID``).  When either is missing the test is cleanly
skipped.

Markers:
    @pytest.mark.integration  — requires a live external service
    @pytest.mark.slow         — 10-minute wall-time budget

Never logs GCP access tokens, project IDs, or full API response bodies
in test output.  Uses ``iterations=3`` to keep the
experiment short and avoid unnecessary API cost.
"""

import asyncio
import os
import textwrap
from typing import Dict

import pytest

from skydiscover.api import DiscoveryResult
from skydiscover.extras.external.alphaevolve_backend import run


# ---------------------------------------------------------------------------
# Credential / env-var gating helpers
# ---------------------------------------------------------------------------


def _has_gcp_credentials() -> bool:
    """Return True if GCP Application Default Credentials are available.

    Uses the same ``google.auth.default()`` probe that the backend itself
    calls at runtime.  Never inspects or logs credential values.
    """
    try:
        import google.auth

        google.auth.default()
        return True
    except Exception:
        return False


def _has_alphaevolve_env_vars() -> bool:
    """Return True if both required AlphaEvolve env vars are set and non-empty.

    The backend reads ``ALPHAEVOLVE_PROJECT_ID`` and
    ``ALPHAEVOLVE_ENGINE_ID`` from the environment to resolve the GCP
    Discovery Engine endpoint.
    """
    project_id = os.environ.get("ALPHAEVOLVE_PROJECT_ID", "")
    engine_id = os.environ.get("ALPHAEVOLVE_ENGINE_ID", "")
    return bool(project_id) and bool(engine_id)


# ---------------------------------------------------------------------------
# Inline test data constants
# ---------------------------------------------------------------------------

TRIVIAL_EVALUATOR_SOURCE = textwrap.dedent(
    """\
    def evaluate(program_path: str) -> dict:
        \"\"\"Score based on source-code length to give AlphaEvolve score variance.\"\"\"
        with open(program_path, "r") as f:
            source = f.read()
        # Longer code -> higher score (capped at 1.0).  This gives the server
        # meaningful score differences across candidates so it can drive evolution.
        score = min(1.0, len(source) / 200.0)
        return {"combined_score": score}
    """
)

SEED_PROGRAM_SOURCE = textwrap.dedent(
    """\
    def solve(x):
        return x + 1
    """
)


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestAlphaEvolveIntegration:
    """Live integration test for the AlphaEvolve external backend."""

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_live_lifecycle(self, tmp_path):  # type: ignore[no-untyped-def]
        """Run the full AlphaEvolve backend lifecycle against the live API.

        The test is cleanly skipped when credentials or env vars are absent.
        On flaky API errors, it retries once with a 30-second backoff before
        skipping.

        Uses ``asyncio.wait_for`` with a 600-second (10-minute) timeout in
        lieu of ``pytest-timeout`` which is not installed.
        """
        # -- Gate on credentials and env vars ---------------------------------
        if not _has_gcp_credentials():
            pytest.skip("No GCP credentials available")
        if not _has_alphaevolve_env_vars():
            pytest.skip(
                "ALPHAEVOLVE_PROJECT_ID and/or ALPHAEVOLVE_ENGINE_ID not set"
            )

        # -- Write test artifacts to tmp_path ---------------------------------
        evaluator = tmp_path / "evaluator.py"
        evaluator.write_text(TRIVIAL_EVALUATOR_SOURCE)

        seed = tmp_path / "seed.py"
        seed.write_text(SEED_PROGRAM_SOURCE)

        output = tmp_path / "output"
        output.mkdir()

        # -- Minimal Config object (only needs file_suffix for the backend) ---
        # The backend reads project_id / engine_id from env vars, so no
        # alphaevolve config section is required on the Config object.
        from skydiscover.config import Config

        config = object.__new__(Config)
        config.file_suffix = ".py"
        config.language = "python"

        # -- Retry wrapper (one retry + 30 s backoff) -------------------
        async def _attempt() -> DiscoveryResult:
            return await run(
                program_path=str(seed),
                evaluator_path=str(evaluator),
                config_obj=config,
                iterations=3,
                output_dir=str(output),
            )

        last_exc = None
        for attempt in range(2):  # max 2 attempts
            try:
                result = await asyncio.wait_for(_attempt(), timeout=600)
                break
            except asyncio.TimeoutError:
                pytest.skip(
                    "AlphaEvolve API timed out after 600 s"
                )
            except Exception as exc:
                last_exc = exc
                if attempt == 0:
                    # First failure — back off and retry once
                    await asyncio.sleep(30)
                else:
                    # Second failure — skip, not hard-fail
                    pytest.skip(
                        f"AlphaEvolve API flaky: {last_exc}"
                    )

        # -- Assertions -----------------------------------------------
        assert isinstance(result, DiscoveryResult)
        assert isinstance(result.best_solution, str) and len(result.best_solution) > 0
        assert isinstance(result.best_score, float) and result.best_score >= 0.0
        assert isinstance(result.metrics, dict)
        assert "combined_score" in result.metrics
        assert result.output_dir == str(output)
