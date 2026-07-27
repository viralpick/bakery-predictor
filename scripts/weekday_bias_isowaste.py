"""실제 레버 검증 — 평일(월·수) 과대예측 트림이 전역 균일 하향을 iso-waste에서 이기는가.

계산은 `bakery.analysis.order_bias`로 옮겼다(Phase 6). 이 스크립트는 동결 fixture
tests/fixtures/frozen/track3_fresh_preds.parquet(git 추적)를 읽어 격자 결과를 print하는 wrapper다.
현 vintage/canonical preds 실행은 `bakery analysis-run`(hypotheses.weekday_bias)을 쓴다.

실행: PYTHONPATH=scripts uv run python scripts/weekday_bias_isowaste.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from bakery.analysis.order_bias import TARGET_DOWS, isowaste_grid, waste_rate_of

CACHE = Path("tests/fixtures/frozen/track3_fresh_preds.parquet")   # git 추적 동결 fixture


def _load() -> pd.DataFrame:
    preds = pd.read_parquet(CACHE)
    preds["date"] = pd.to_datetime(preds["date"])
    return preds


def main() -> None:
    preds = _load()
    is_target = preds["date"].dt.dayofweek.isin(TARGET_DOWS)
    base_waste = waste_rate_of(preds["expected"].to_numpy(), preds["actual"].to_numpy())
    print(f"[광교 3년 OOS] {len(preds)}일 · 월·수 {is_target.mean()*100:.1f}%  "
          f"base(expected) waste={base_waste*100:.1f}%")
    print("판정: 동일 waste에서 DOW(월·수 트림)−GLOBAL 매진 gap; 음수+CI0배제=DOW 우위")
    print(isowaste_grid(preds).to_string(index=False))


if __name__ == "__main__":
    main()
