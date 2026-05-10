"""Sparrow-specific gateway config tests kept out of upstream test files."""

import os
from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides


def test_onebot_napcat_connected_when_enabled_env_is_truthy(monkeypatch):
    monkeypatch.setenv("NAPCAT_ENABLED", "true")
    config = GatewayConfig(
        platforms={Platform.ONEBOT_NAPCAT: PlatformConfig(enabled=True)},
    )

    assert Platform.ONEBOT_NAPCAT in config.get_connected_platforms()


def test_onebot_napcat_not_connected_without_enabled_env(monkeypatch):
    monkeypatch.delenv("NAPCAT_ENABLED", raising=False)
    config = GatewayConfig(
        platforms={Platform.ONEBOT_NAPCAT: PlatformConfig(enabled=True)},
    )

    assert Platform.ONEBOT_NAPCAT not in config.get_connected_platforms()


def test_napcat_env_override_creates_platform_and_home_channel():
    config = GatewayConfig()
    env = {
        "NAPCAT_ENABLED": "true",
        "ONEBOT_NAPCAT_HOME_CHANNEL": "group_123",
        "ONEBOT_NAPCAT_HOME_CHANNEL_NAME": "QQ Home",
    }

    with patch.dict(os.environ, env, clear=True):
        _apply_env_overrides(config)

    platform_config = config.platforms[Platform.ONEBOT_NAPCAT]
    assert platform_config.enabled is True
    assert platform_config.home_channel is not None
    assert platform_config.home_channel.chat_id == "group_123"
    assert platform_config.home_channel.name == "QQ Home"
