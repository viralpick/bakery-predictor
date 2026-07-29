"""운영 예측 패널 — 행 하나 = (원점, 대상일) 쌍.

현 feature 조립(`category_aggregate.build_features`)은 자기회귀(AR) feature를 **대상일
기준**으로 만든다. 그래서 리드타임이 있는 운영 시나리오에서 원점 이후 실측을 보게 되고,
그걸 지우면 train/test 가용성이 어긋난다(공변량 shift). 여기서는 AR feature를 **원점 기준**
으로 다시 정의해 그 두 문제를 동시에 없앤다.

설계 근거·대안 비교는 docs/superpowers/specs/2026-07-29-operational-panel-design.md.

★핵심 = `y_same_dow_latest`: 원점 이전 마지막 같은 요일 값. 대상일 요일이 offset으로
결정되므로 `back = (-offset) % 7` 일만 되돌아가면 되고, **offset과 무관하게 항상 가용**하다
(화요일 원점 → 월요일 대상은 원점−1, 일요일 대상은 원점−2). 현 lag 집합 {1,7,14,28}은
일요일 대상일 때 lag7이 원점 이후라 그냥 버려진다.
"""
from __future__ import annotations

import pandas as pd

from bakery.features.category_aggregate import FEATURE_GROUPS, CategoryDaily

DAYS_PER_WEEK = 7
ORIGIN_LAGS = (0, 1, 2, 3)          # 0 = 원점 당일 실측(원점에서 이미 아는 값)
ROLLING_WINDOWS = (7, 28)
EWMA_HALFLIVES = (7, 28)
SAME_DOW_COUNT = 4                  # 같은 요일 최근 N회 평균
DEFAULT_OFFSETS = tuple(range(6, 13))   # 화요일 원점 → 다음주 월(+6)~일(+12)


def _require_gapless(base: pd.DataFrame) -> pd.DataFrame:
    """원점 기준 AR은 위치 shift라 날짜가 연속이어야 한다 — 아니면 조용히 틀린다."""
    out = base.sort_values("date").reset_index(drop=True)
    gaps = out["date"].diff().dt.days.dropna()
    bad = int((gaps != 1).sum())
    if bad:
        raise ValueError(
            f"패널 입력에 날짜 gap {bad}건 — 원점 기준 shift가 어긋난다. "
            "build_category_daily 출력(dropna 이전, 날짜 연속)을 넘겨라."
        )
    return out


def _origin_features(base: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """원점(행의 date)까지 가용한 값만으로 만든 AR feature.

    ★대상일 기준 조립과의 결정적 차이: rolling/ewma에 shift(1)을 걸지 않는다.
    원점 당일 실측은 원점에서 이미 아는 값이므로 포함하는 것이 맞다.
    """
    y = base[target_col]
    out = pd.DataFrame({"origin_date": base["date"]})
    for lag in ORIGIN_LAGS:
        out[f"y_origin_lag{lag}"] = y.shift(lag)
    for window in ROLLING_WINDOWS:
        min_periods = max(2, window // 3)
        out[f"y_origin_rmean{window}"] = y.rolling(window, min_periods=min_periods).mean()
        out[f"y_origin_rstd{window}"] = y.rolling(window, min_periods=min_periods).std()
    for halflife in EWMA_HALFLIVES:
        out[f"y_origin_ewma{halflife}"] = y.ewm(
            halflife=halflife, min_periods=max(2, halflife // 3)
        ).mean()
    return out


def same_dow_back_days(offset: int) -> int:
    """대상일과 같은 요일인, 원점 이전 가장 가까운 날까지의 일수.

    대상일 요일 = (원점 요일 + offset) mod 7 이므로 되돌아갈 일수는 offset만으로 정해진다.
    offset=6 → 1일(화요일 원점의 직전 월요일) / offset=7 → 0일(원점 당일) / offset=12 → 2일.
    """
    return (-offset) % DAYS_PER_WEEK


def _same_dow_features(base: pd.DataFrame, target_col: str, offset: int) -> pd.DataFrame:
    """대상일과 같은 요일의 최근 실측(원점 이전) — 항상 가용한 요일 레벨 앵커."""
    y = base[target_col]
    back = same_dow_back_days(offset)
    shifted = [y.shift(back + DAYS_PER_WEEK * k) for k in range(SAME_DOW_COUNT)]
    return pd.DataFrame({
        "y_same_dow_latest": shifted[0],
        "y_same_dow_mean4": pd.concat(shifted, axis=1).mean(axis=1),
    })


def _target_date_features(target_dates: pd.Series) -> pd.DataFrame:
    """대상일 기준 캘린더·외부 feature — 타깃과 무관해 사전에 알 수 있다(fold-invariant).

    ⚠️ 날씨는 관측값을 쓴다(현 헤드라인과 동일). D+6~D+12를 예측하며 그날 관측 기온·강수를
    쓰는 것이므로 이 축은 아직 낙관적이다 — 중기예보 정렬은 별도 단계(스펙 §6).
    """
    frame = pd.DataFrame({"date": pd.Series(sorted(target_dates.unique()))})
    for add_group in FEATURE_GROUPS.values():
        frame = add_group(frame)
    return frame


def build_forecast_panel(
    cd: CategoryDaily,
    *,
    target_col: str = "adjusted_demand_unit",
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
) -> pd.DataFrame:
    """(원점, 대상일) 패널. 모든 feature가 원점 시점에 가용하다.

    원점은 모든 날짜로 생성한다(daily origins) — offset별 오차 구조를 전 범위에서 학습하기
    위해서다. 평가 시 원점을 요일로 걸러내는 것은 백테스트(`harness/panel_backtest.py`) 몫.
    같은 실측이 offset 수만큼 중복 등장하므로 유효 표본은 행수보다 작다.

    반환 컬럼: origin_date / target_date / horizon_offset / y_origin_* / y_same_dow_* /
    대상일 캘린더·외부 feature / `target_col`(대상일 실측).
    """
    base = _require_gapless(cd.df[["date", target_col]].copy())
    origin_feats = _origin_features(base, target_col)
    actual = base.rename(columns={"date": "target_date"})

    chunks = []
    for offset in offsets:
        chunk = origin_feats.copy()
        chunk["horizon_offset"] = offset
        chunk["target_date"] = chunk["origin_date"] + pd.Timedelta(days=offset)
        same_dow = _same_dow_features(base, target_col, offset)
        for col in same_dow.columns:
            chunk[col] = same_dow[col].to_numpy()
        chunks.append(chunk)

    panel = pd.concat(chunks, ignore_index=True)
    # inner join: 대상일 실측이 있는 행만 남긴다(데이터 끝을 넘는 원점은 자동 탈락).
    panel = panel.merge(actual, on="target_date", how="inner")
    tgt_feats = _target_date_features(panel["target_date"]).rename(
        columns={"date": "target_date"}
    )
    panel = panel.merge(tgt_feats, on="target_date", how="left")
    return panel.sort_values(["origin_date", "horizon_offset"]).reset_index(drop=True)


def origin_feature_columns(panel: pd.DataFrame) -> list[str]:
    """원점 기준 AR feature 컬럼(leakage 회귀 테스트가 지목하는 집합)."""
    return [c for c in panel.columns if c.startswith(("y_origin_", "y_same_dow_"))]
