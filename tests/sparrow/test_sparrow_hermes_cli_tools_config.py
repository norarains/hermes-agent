"""Sparrow-specific tools-config tests kept out of upstream test files."""


def test_enabled_platforms_include_onebot_napcat_when_enabled(monkeypatch):
    from hermes_cli.tools_config import _get_enabled_platforms

    monkeypatch.delenv("NAPCAT_ENABLED", raising=False)
    assert "onebot_napcat" not in _get_enabled_platforms()

    monkeypatch.setenv("NAPCAT_ENABLED", "true")
    assert "onebot_napcat" in _get_enabled_platforms()
