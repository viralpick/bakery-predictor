import sys
import pytest
from bakery.harness.event_priors import STORE_EVENT_PRIORS, resolve_event_priors


def test_gwangyo_preset_matches_script():
    """harness 프리셋이 원본 scripts 정의와 동일해야 한다(단일 출처 승격)."""
    sys.path.insert(0, "scripts")
    import store_predictive_power as s
    script_gw = s.STORE_EVENT_PRIORS["광교"]
    events, lunar = resolve_event_priors("gwangyo")
    assert events == script_gw["events"]
    assert lunar == script_gw["lunar_events"]


def test_resolve_none_returns_none_pair():
    assert resolve_event_priors(None) == (None, None)


def test_unknown_key_raises():
    with pytest.raises(KeyError):
        resolve_event_priors("nonexistent")
