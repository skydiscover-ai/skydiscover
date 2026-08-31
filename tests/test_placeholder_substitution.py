"""Tests for placeholder substitution helpers in the default context builder."""

import logging

from skydiscover.context_builder.default.builder import (
    _KNOWN_TEMPLATE_VARS,
    _EnvFormatDict,
    _substitute_placeholders,
    _warn_unsubstituted_placeholders,
)


class TestSubstitutePlaceholders:
    def test_resolves_env_var(self, monkeypatch):
        monkeypatch.setenv("PROBLEM_STATEMENT", "Optimize the loss function")
        result = _substitute_placeholders("Task: {problem_statement}")
        assert result == "Task: Optimize the loss function"

    def test_leaves_unknown_var_literal(self, monkeypatch):
        monkeypatch.delenv("UNKNOWN_VAR", raising=False)
        result = _substitute_placeholders("Value is {unknown_var}")
        assert result == "Value is {unknown_var}"

    def test_no_braces_returns_unchanged(self):
        text = "No placeholders here."
        assert _substitute_placeholders(text) is text

    def test_empty_string_returns_unchanged(self):
        assert _substitute_placeholders("") == ""

    def test_none_returns_none(self):
        assert _substitute_placeholders(None) is None

    def test_mixed_resolved_and_unresolved(self, monkeypatch):
        monkeypatch.setenv("FOO", "bar")
        monkeypatch.delenv("BAZ", raising=False)
        result = _substitute_placeholders("{foo} and {baz}")
        assert result == "bar and {baz}"


class TestWarnUnsubstitutedPlaceholders:
    def test_warns_for_unknown_placeholder(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unsubstituted_placeholders("Hello {problem_statement}")
        assert "Unsubstituted placeholder" in caplog.text
        assert "{problem_statement}" in caplog.text

    def test_no_warn_for_known_template_vars(self, caplog):
        msg = " ".join(f"{{{v}}}" for v in _KNOWN_TEMPLATE_VARS)
        with caplog.at_level(logging.WARNING):
            _warn_unsubstituted_placeholders(msg)
        assert caplog.text == ""

    def test_no_warn_when_all_resolved(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unsubstituted_placeholders("No placeholders here")
        assert caplog.text == ""

    def test_no_warn_for_empty_input(self, caplog):
        with caplog.at_level(logging.WARNING):
            _warn_unsubstituted_placeholders("")
            _warn_unsubstituted_placeholders(None)
        assert caplog.text == ""


class TestEnvFormatDict:
    def test_returns_env_value_uppercased(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "my_value")
        d = _EnvFormatDict()
        assert d["my_key"] == "my_value"

    def test_returns_literal_when_env_not_set(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        d = _EnvFormatDict()
        assert d["missing_key"] == "{missing_key}"
