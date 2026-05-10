"""Tests for the Edge TTS ``pitch`` config option.

The pitch knob is what makes a Chinese voice like ``zh-CN-XiaoyiNeural``
sound lively (``+5Hz``) instead of newscaster-flat.  Without it,
operators who set the recipe in config.yaml would silently get the
neutral default — no SSML pitch shift, no obvious failure, just a
flatter voice than expected.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ("HERMES_SESSION_PLATFORM",):
        monkeypatch.delenv(key, raising=False)


def _run(tts_config, tmp_path):
    """Invoke ``_generate_edge_tts`` against a mock Communicate so we
    can introspect the kwargs that would be sent to the SSML server."""
    mock_comm = MagicMock()
    mock_comm.save = AsyncMock()
    mock_edge = MagicMock()
    mock_edge.Communicate = MagicMock(return_value=mock_comm)

    with patch("tools.tts_tool._import_edge_tts", return_value=mock_edge):
        from tools.tts_tool import _generate_edge_tts
        asyncio.run(_generate_edge_tts(
            "你好", str(tmp_path / "out.mp3"), tts_config,
        ))
    return mock_edge.Communicate


class TestEdgeTtsPitch:
    def test_default_no_pitch_kwarg(self, tmp_path):
        """Without ``tts.edge.pitch`` configured, no pitch kwarg is
        passed to Communicate — Edge keeps its native ``+0Hz``."""
        comm_cls = _run({}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert "pitch" not in kwargs

    def test_positive_pitch_passed_through(self, tmp_path):
        """The headline use case: ``+5Hz`` for the Chinese Xiaoyi
        voice arrives at Communicate verbatim."""
        comm_cls = _run({"edge": {"pitch": "+5Hz"}}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["pitch"] == "+5Hz"

    def test_negative_pitch_passed_through(self, tmp_path):
        """Negative pitch (``-3Hz``) for a deeper voice is also
        valid Edge syntax and must reach Communicate as-is."""
        comm_cls = _run({"edge": {"pitch": "-3Hz"}}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["pitch"] == "-3Hz"

    def test_invalid_pitch_format_silently_skipped(self, tmp_path):
        """A typo like ``5Hz`` (missing sign) or ``+5%`` (wrong unit)
        must NOT be sent to Communicate — Edge would reject it with
        a 500 / SSML error mid-delivery.  Skip and let the default
        ``+0Hz`` apply so the user still hears something."""
        for bad in ("5Hz", "+5%", "+5", "5", "high", "+5hz", "+ 5Hz"):
            comm_cls = _run({"edge": {"pitch": bad}}, tmp_path)
            kwargs = comm_cls.call_args[1]
            assert "pitch" not in kwargs, f"bad pitch {bad!r} leaked through"

    def test_pitch_combines_with_speed(self, tmp_path):
        """Pitch and speed are independent knobs — both must reach
        Communicate together (this is the actual recipe Xiaoyi gets:
        rate=+5%, pitch=+5Hz)."""
        comm_cls = _run(
            {"edge": {"pitch": "+5Hz", "speed": 1.05}}, tmp_path,
        )
        kwargs = comm_cls.call_args[1]
        assert kwargs["pitch"] == "+5Hz"
        assert kwargs["rate"] == "+5%"

    def test_voice_kwarg_still_passed(self, tmp_path):
        """Sanity: voice selection still works alongside the new
        pitch knob (regression guard for the kwargs assembly)."""
        comm_cls = _run(
            {"edge": {"voice": "zh-CN-XiaoyiNeural", "pitch": "+5Hz"}},
            tmp_path,
        )
        kwargs = comm_cls.call_args[1]
        assert kwargs["voice"] == "zh-CN-XiaoyiNeural"
        assert kwargs["pitch"] == "+5Hz"

    def test_zero_pitch_explicit_still_passed(self, tmp_path):
        """``+0Hz`` is the same as Edge's native default but still a
        valid value — pass it through if the operator wrote it
        explicitly (don't second-guess intent)."""
        comm_cls = _run({"edge": {"pitch": "+0Hz"}}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert kwargs["pitch"] == "+0Hz"

    def test_pitch_none_treated_as_unset(self, tmp_path):
        """YAML ``pitch:`` with no value parses to ``None`` — must
        be treated as "not configured", same as omitting the key."""
        comm_cls = _run({"edge": {"pitch": None}}, tmp_path)
        kwargs = comm_cls.call_args[1]
        assert "pitch" not in kwargs
