"""Unit tests for AlphaEvolve config resolution."""

import os
from unittest.mock import MagicMock, patch

import pytest

from skydiscover.extras.external.alphaevolve_backend import (
    _expand_env_vars_in_value,
    _get_alphaevolve_config,
    _resolve_credentials_file,
)

# Patch target for load_defaults (lazy import inside _get_alphaevolve_config)
_LOAD_DEFAULTS = "skydiscover.extras.external.defaults.load_defaults"


class TestGetAlphaevolveConfig:
    """Tests for _get_alphaevolve_config."""

    @patch(_LOAD_DEFAULTS, return_value={"alphaevolve": {"location": "global"}})
    @patch.dict(
        os.environ,
        {
            "ALPHAEVOLVE_PROJECT_ID": "env-project",
            "ALPHAEVOLVE_ENGINE_ID": "env-engine",
        },
        clear=False,
    )
    def test_env_var_overrides(self, mock_defaults: MagicMock) -> None:
        config_obj = MagicMock(spec=[])

        result = _get_alphaevolve_config(config_obj)

        assert result["project_id"] == "env-project"
        assert result["engine_id"] == "env-engine"

    @patch(_LOAD_DEFAULTS, return_value={"alphaevolve": {}})
    @patch.dict(os.environ, {}, clear=True)
    def test_missing_required_fields_raises(
        self, mock_defaults: MagicMock
    ) -> None:
        config_obj = MagicMock(spec=[])

        with pytest.raises(ValueError, match="project_id"):
            _get_alphaevolve_config(config_obj)

    @patch(
        _LOAD_DEFAULTS,
        return_value={
            "alphaevolve": {
                "project_id": "p",
                "engine_id": "e",
                "location": "us-central1",
            }
        },
    )
    @patch.dict(os.environ, {}, clear=True)
    def test_defaults_fallback(self, mock_defaults: MagicMock) -> None:
        config_obj = MagicMock(spec=[])

        result = _get_alphaevolve_config(config_obj)

        assert result["location"] == "us-central1"
        assert result["project_id"] == "p"

    @patch(
        _LOAD_DEFAULTS,
        return_value={
            "alphaevolve": {"project_id": "default-p", "engine_id": "default-e"}
        },
    )
    @patch.dict(os.environ, {}, clear=True)
    def test_config_obj_merge(self, mock_defaults: MagicMock) -> None:
        config_obj = MagicMock()
        config_obj.alphaevolve = {
            "project_id": "from-obj",
            "engine_id": "from-obj",
        }

        result = _get_alphaevolve_config(config_obj)

        assert result["project_id"] == "from-obj"
        assert result["engine_id"] == "from-obj"

    @patch(
        _LOAD_DEFAULTS,
        return_value={
            "alphaevolve": {"project_id": "default-p", "engine_id": "default-e"}
        },
    )
    @patch.dict(
        os.environ,
        {"ALPHAEVOLVE_PROJECT_ID": "from-env"},
        clear=True,
    )
    def test_env_beats_config_section(self, mock_defaults: MagicMock) -> None:
        config_obj = MagicMock()
        config_obj.alphaevolve = {
            "project_id": "from-obj",
            "engine_id": "from-obj",
        }

        result = _get_alphaevolve_config(config_obj)

        # Env is the ad-hoc override, the config file supplies the rest.
        assert result["project_id"] == "from-env"
        assert result["engine_id"] == "from-obj"

    @patch(
        _LOAD_DEFAULTS,
        return_value={"alphaevolve": {"location": "global"}},
    )
    @patch.dict(os.environ, {}, clear=True)
    def test_section_from_yaml_config_is_used(
        self, mock_defaults: MagicMock
    ) -> None:
        """A user's `alphaevolve:` YAML section must survive Config parsing."""
        from skydiscover.config import Config

        config_obj = Config.from_dict(
            {"alphaevolve": {"project_id": "yaml-p", "engine_id": "yaml-e"}}
        )

        result = _get_alphaevolve_config(config_obj)

        assert result["project_id"] == "yaml-p"
        assert result["engine_id"] == "yaml-e"
        assert result["location"] == "global"

    @patch(
        _LOAD_DEFAULTS,
        return_value={
            "alphaevolve": {"project_id": "p", "engine_id": "e"}
        },
    )
    @patch.dict(
        os.environ,
        {
            "ALPHAEVOLVE_NUM_EVALUATORS": "4",
            "ALPHAEVOLVE_NUM_SAMPLERS": "2",
        },
        clear=False,
    )
    def test_int_keys_converted(self, mock_defaults: MagicMock) -> None:
        config_obj = MagicMock(spec=[])

        result = _get_alphaevolve_config(config_obj)

        assert result["num_evaluators"] == 4
        assert isinstance(result["num_evaluators"], int)
        assert result["num_samplers"] == 2
        assert isinstance(result["num_samplers"], int)


class TestExpandEnvVarsInValue:
    """Tests for _expand_env_vars_in_value standalone helper."""

    def test_single_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_DIR", "/opt/creds")
        assert _expand_env_vars_in_value("${MY_DIR}/sa.json") == "/opt/creds/sa.json"

    def test_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BASE", "/home")
        monkeypatch.setenv("USER", "alice")
        assert _expand_env_vars_in_value("${BASE}/${USER}/key.json") == "/home/alice/key.json"

    def test_unmatched_var_left_as_is(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        assert _expand_env_vars_in_value("${NONEXISTENT_VAR_XYZ}/f.json") == "${NONEXISTENT_VAR_XYZ}/f.json"

    def test_no_vars_passthrough(self) -> None:
        assert _expand_env_vars_in_value("/plain/path.json") == "/plain/path.json"

    def test_empty_string(self) -> None:
        assert _expand_env_vars_in_value("") == ""


class TestCredentialsFile:
    """Tests for _resolve_credentials_file and credentials_file config."""

    def test_credentials_file_sets_env_var(
        self, tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        sa_file = tmp_path / "sa.json"
        sa_file.write_text("{}")
        ae_config = {"credentials_file": str(sa_file)}

        _resolve_credentials_file(ae_config)

        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(sa_file)

    def test_credentials_file_expands_env_vars(
        self, tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setenv("TEST_CRED_DIR", str(tmp_path))
        sa_file = tmp_path / "sa.json"
        sa_file.write_text("{}")
        ae_config = {"credentials_file": "${TEST_CRED_DIR}/sa.json"}

        _resolve_credentials_file(ae_config)

        assert os.environ["GOOGLE_APPLICATION_CREDENTIALS"] == str(sa_file)

    def test_credentials_file_missing_raises(self) -> None:
        ae_config = {"credentials_file": "/nonexistent/path.json"}

        with pytest.raises(ValueError, match="credentials_file not found"):
            _resolve_credentials_file(ae_config)

    def test_credentials_file_unset_no_clobber(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "GOOGLE_APPLICATION_CREDENTIALS", "/original/path.json"
        )
        ae_config: dict = {}

        _resolve_credentials_file(ae_config)

        assert (
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
            == "/original/path.json"
        )

    def test_credentials_file_empty_string_no_clobber(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        ae_config = {"credentials_file": ""}

        _resolve_credentials_file(ae_config)

        assert "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ

    @patch(
        _LOAD_DEFAULTS,
        return_value={
            "alphaevolve": {"project_id": "p", "engine_id": "e"}
        },
    )
    @patch.dict(
        os.environ,
        {"ALPHAEVOLVE_CREDENTIALS_FILE": "/some/path.json"},
        clear=False,
    )
    def test_env_var_override_credentials_file(
        self, mock_defaults: MagicMock
    ) -> None:
        config_obj = MagicMock(spec=[])

        result = _get_alphaevolve_config(config_obj)

        assert result["credentials_file"] == "/some/path.json"
