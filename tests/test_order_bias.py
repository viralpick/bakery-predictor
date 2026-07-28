"""order_bias 프리미티브 — 동결 preds 캐시 golden 대조.

golden은 2026-07-28에 `scripts/weekday_bias_isowaste.py`의 함수를
2026-07-18 생성 동결 preds(현재 `tests/fixtures/frozen/track3_fresh_preds.parquet`, git 추적)에
직접 돌려 캡처했다.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakery.analysis.order_bias import (
    E_TRIMS,
    N_BOOT,
    SEED,
    TARGET_DOWS,
    W_TARGETS,
    bootstrap_gap_ci,
    dow_trimmed_order,
    isowaste_dow_gap,
    isowaste_grid,
    soldout_freq,
    soldout_mag,
    solve_base_for_waste,
    waste_rate_of,
)

FROZEN_PREDS = Path("tests/fixtures/frozen/track3_fresh_preds.parquet")   # git 추적(Task 8에서 이동)


@pytest.fixture(scope="module")
def frozen_preds():
    # 추적 fixture다 — 없으면 체크아웃 파손이므로 skip이 아니라 실패여야 한다
    # (gitignore된 reports/를 읽던 옛 경로에선 CI가 조용히 skip하고 green으로 보였다).
    assert FROZEN_PREDS.exists(), f"{FROZEN_PREDS} 없음 — 추적 fixture가 누락됐다"
    preds = pd.read_parquet(FROZEN_PREDS)
    preds["date"] = pd.to_datetime(preds["date"])
    return preds


def test_constants():
    assert W_TARGETS == (0.06, 0.08, 0.10)
    assert E_TRIMS == (0.02, 0.03, 0.04)
    assert N_BOOT == 2000
    assert SEED == 42
    assert TARGET_DOWS == (0, 2)          # 월=0, 수=2 (연도별 부호 안정한 요일만)


def test_metric_formulas_exact():
    order = np.array([10.0, 10.0])
    actual = np.array([8.0, 12.0])
    # 초과 2 / Σactual 20 = 0.1
    assert waste_rate_of(order, actual) == 0.1
    # 부족일 1/2
    assert soldout_freq(order, actual) == 0.5
    # 부족량 2 / 20
    assert soldout_mag(order, actual) == 0.1


def test_metrics_zero_denominator_guard():
    zeros = np.array([0.0, 0.0])
    assert waste_rate_of(np.array([1.0, 1.0]), zeros) == 0.0
    assert soldout_mag(np.array([1.0, 1.0]), zeros) == 0.0


def test_dow_trimmed_order_trims_only_target_dows():
    expected = np.array([100.0, 100.0])
    is_target = np.array([True, False])
    order = dow_trimmed_order(expected, base=0.10, trim=0.04, is_target_dow=is_target)
    # 1+0.10−0.04 / 1+0.10. approx: 100.0*(1.0+0.10)는 부동소수점으로 110.00000000000001.
    assert order.tolist() == pytest.approx([106.0, 110.0])


def test_solve_base_hits_waste_target():
    preds = pd.DataFrame({"expected": [100.0] * 10, "actual": [100.0] * 10})
    is_target = np.zeros(10, dtype=bool)
    base = solve_base_for_waste(preds["expected"].to_numpy(), preds["actual"].to_numpy(),
                                is_target, trim=0.0, w_target=0.05)
    order = dow_trimmed_order(preds["expected"].to_numpy(), base, 0.0, is_target)
    assert waste_rate_of(order, preds["actual"].to_numpy()) == pytest.approx(0.05, abs=1e-6)


def test_frozen_preds_shape_golden(frozen_preds):
    """golden: n=1090일, 월·수 비중 0.285321, base(expected) waste=0.047616."""
    assert len(frozen_preds) == 1090
    is_monwed = frozen_preds["date"].dt.dayofweek.isin(TARGET_DOWS)
    assert round(float(is_monwed.mean()), 6) == 0.285321
    base_waste = waste_rate_of(frozen_preds["expected"].to_numpy(),
                               frozen_preds["actual"].to_numpy())
    assert round(base_waste, 6) == 0.047616


@pytest.mark.parametrize("trim,expected_freq,expected_mag", [
    (0.02, -0.0009174311926605228, -0.0001567963417563184),
    (0.03, 0.00458715596330278, -4.3062759390345706e-05),
    (0.04, 0.007339449541284404, 0.0002116974752705003),
])
def test_isowaste_gap_golden_at_w006(frozen_preds, trim, expected_freq, expected_mag):
    """golden(2026-07-28, w_target=0.06). 음수=DOW 트림 우위."""
    gap_freq, gap_mag = isowaste_dow_gap(frozen_preds, w_target=0.06, trim=trim)
    assert gap_freq == pytest.approx(expected_freq, rel=1e-9)
    assert gap_mag == pytest.approx(expected_mag, rel=1e-9)


def test_bootstrap_ci_is_deterministic_for_fixed_seed(frozen_preds):
    first = bootstrap_gap_ci(frozen_preds, w_target=0.06, trim=0.03, n_boot=50, seed=SEED)
    second = bootstrap_gap_ci(frozen_preds, w_target=0.06, trim=0.03, n_boot=50, seed=SEED)
    assert first["freq"].tolist() == second["freq"].tolist()
    assert first["freq"].shape == (3,)          # [2.5, 50, 97.5] 백분위


def test_isowaste_grid_covers_full_cross_product(frozen_preds):
    grid = isowaste_grid(frozen_preds, n_boot=20)
    assert len(grid) == len(W_TARGETS) * len(E_TRIMS)
    assert grid.columns.tolist() == [
        "w_target", "trim", "gap_freq", "gap_mag",
        "freq_ci_low", "freq_median", "freq_ci_high", "winner"]


def test_script_delegates_to_primitive():
    import sys
    sys.path.insert(0, "scripts")
    import weekday_bias_isowaste

    assert weekday_bias_isowaste.isowaste_grid is isowaste_grid
