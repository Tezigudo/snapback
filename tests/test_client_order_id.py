"""Tests for the clientOrderId tagging scheme (Option A: bot-side).

The scheme: snap-v1-<root>-<leg>
  root = millisecond signal_id
  leg  = e | s | t | x | bf | h | k | c

Investing-consolidate's importer parses these to attribute Binance trades
back to specific bot signals.
"""

from __future__ import annotations

import re

from exchange.binance_client import _coid, _COID_VALID


class TestCoidFormat:
    def test_none_root_returns_none(self) -> None:
        assert _coid(None, "e") is None

    def test_basic_entry(self) -> None:
        assert _coid("1716120000000", "e") == "snap-v1-1716120000000-e"

    def test_all_legs_valid(self) -> None:
        for leg in ("e", "s", "t", "x", "bf", "h", "k", "c"):
            coid = _coid("1716120000000", leg)
            assert coid is not None, f"leg {leg!r} produced None"
            assert coid.endswith(f"-{leg}"), f"leg {leg!r} not in suffix"

    def test_length_within_binance_36_char_cap(self) -> None:
        # 13-digit ms timestamp + longest leg "bf" = "snap-v1-1716120000000-bf" = 24 chars
        for leg in ("e", "s", "t", "x", "bf", "h", "k", "c"):
            coid = _coid("1716120000000", leg)
            assert coid is not None
            assert len(coid) <= 36, f"{coid!r} too long ({len(coid)} > 36)"

    def test_rejects_invalid_chars_in_root(self) -> None:
        # Spaces, slashes, etc. would break Binance's regex.
        assert _coid("with space", "e") is None
        assert _coid("with/slash", "e") is None
        assert _coid("with#hash", "e") is None

    def test_accepts_alnum_dot_underscore_hyphen(self) -> None:
        # If ever we change root format, these chars must stay safe.
        assert _coid("v2_1716120000000", "e") is not None
        assert _coid("v2.1716120000000", "e") is not None
        assert _coid("v2-1716120000000", "e") is not None

    def test_rejects_overlong_root(self) -> None:
        # snap-v1- prefix is 8 chars, "-e" suffix is 2 → root max 26 chars
        long_root = "x" * 30
        assert _coid(long_root, "e") is None


class TestCoidRoundTrip:
    """Parsing a COID back to (version, root, leg) — mirrors what
    consolidate-investment's importer will do."""

    @staticmethod
    def _parse(coid: str) -> tuple[str, str, str] | None:
        m = re.match(r"^snap-(v\d+)-(\d+)-([a-z]+)$", coid)
        return (m[1], m[2], m[3]) if m else None

    def test_parse_entry(self) -> None:
        assert self._parse("snap-v1-1716120000000-e") == ("v1", "1716120000000", "e")

    def test_parse_multi_char_leg(self) -> None:
        assert self._parse("snap-v1-1716120000000-bf") == ("v1", "1716120000000", "bf")

    def test_parse_returns_none_for_non_snap(self) -> None:
        assert self._parse("manual-trade-12345") is None
        assert self._parse("") is None

    def test_round_trip_all_legs(self) -> None:
        for leg in ("e", "s", "t", "x", "bf", "h", "k", "c"):
            coid = _coid("1716120000000", leg)
            assert coid is not None
            parsed = self._parse(coid)
            assert parsed == ("v1", "1716120000000", leg)


class TestValidatorRegex:
    """The validator regex protects against silent failures if we ever
    accidentally construct an invalid COID."""

    def test_allows_min_length(self) -> None:
        assert _COID_VALID.fullmatch("a") is not None

    def test_allows_max_length(self) -> None:
        assert _COID_VALID.fullmatch("a" * 36) is not None

    def test_rejects_too_long(self) -> None:
        assert _COID_VALID.fullmatch("a" * 37) is None

    def test_rejects_empty(self) -> None:
        assert _COID_VALID.fullmatch("") is None
