"""Tests for explicit switch_interval in EvoX config."""

from skydiscover.config import Config, SearchConfig


class TestSwitchIntervalConfig:
    def test_switch_interval_default_none(self):
        config = SearchConfig()
        assert config.switch_interval is None

    def test_switch_interval_from_yaml_dict(self):
        config = Config.from_dict(
            {
                "search": {
                    "type": "evox",
                    "switch_interval": 5,
                },
            }
        )
        assert config.search.switch_interval == 5

    def test_switch_interval_omitted_stays_none(self):
        config = Config.from_dict(
            {
                "search": {
                    "type": "evox",
                },
            }
        )
        assert config.search.switch_interval is None
