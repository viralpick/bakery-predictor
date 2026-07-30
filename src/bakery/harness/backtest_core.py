"""windowed_backtest 코어 — forecaster-general expanding backtest + event_prior.

forecaster 인자로 category_total(LightGBM, 기본) / distributional_total(NGBoost) 등을
태운다(Forecaster 어댑터). default=None→CategoryTotalForecaster라 category 경로는
scripts/store_predictive_power.py 원본과 바이트 동등(엔진 동등성 게이트로 회귀 방지).

leakage-safe: event_prior는 cutoff 이전 전체 history로 fit(train window보다 김), 예측 이후
post-model 블렌드라 forecaster 무관하게 균일 적용. 원본 모듈 상수(ALPHA=0.8, PROD_Q=0.85,
HORIZON=7, MIN_TRAIN_ROWS=60)는 함수 인자 기본값으로 흡수했다(값 동일).

운영 정렬 opt-in(lead_days / anchor_dow): 발주 리드타임과 요일 앵커링을 표현한다.
train과 event_prior history 둘 다 동일한 cutoff(= test_start − lead_days)를 쓰므로
원점 이후 데이터는 어느 경로로도 새지 않는다. 기본값(0 / None)은 헤드라인 동작과
완전히 동일하다 — cutoff == test_start_date, fold 경계는 기존 인덱스 산술 그대로.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bakery.features.category_aggregate import add_lag_rolling_ewma
from bakery.harness.forecasters import CategoryTotalForecaster, Forecaster
from bakery.models.category_total import BacktestResult
from bakery.models.event_prior import EventLevelPrior
from bakery.models.item_proportion import distribute_total

MIN_TRAIN_ROWS = 60
DAYS_PER_WEEK = 7
ORDER_LEVEL_CATEGORY = "category"
ORDER_LEVEL_ITEM = "item"


def _distribute_fold_orders(
    item_history: pd.DataFrame,
    dates: np.ndarray,
    production: np.ndarray,
    *,
    fold: int,
    lead_days: int,
) -> pd.DataFrame:
    """카테고리 총 발주량을 품목별로 배분 — models.item_proportion 호출만(재구현 없음).

    ★`lead_days` 를 그대로 넘긴다. 총량만 리드타임을 지키고 배분이 대상일 직전 실적을
    보면 파이프라인 전체가 여전히 leaky다(PR#74에서 이 축을 막았다).
    """
    totals = pd.Series(np.asarray(production, dtype=float), index=pd.to_datetime(dates))
    result = distribute_total(item_history, totals, lead_days=lead_days)
    out = result.quantities.rename(columns={"qty": "order_qty"})
    out["fold"] = fold
    return out[["date", "item_id", "fold", "order_qty"]]


def _fold_starts_by_dow(
    df: pd.DataFrame, anchor_dow: int, horizon_days: int, n_folds: int,
) -> list[pd.Timestamp]:
    """요일 앵커 fold 시작일 목록(최신 → 과거, horizon_days 간격).

    데이터 마지막 날짜 이하에서 horizon 블록이 완전히 들어가는 마지막 anchor_dow
    시작일을 찾고, 거기서 horizon_days씩 과거로 n_folds개를 만든다. 데이터 범위를
    벗어난 시작일은 test_df가 비어 호출부에서 skip된다.
    """
    span = pd.Timedelta(days=horizon_days - 1)
    latest = pd.Timestamp(df["date"].max()) - span
    back = (int(latest.dayofweek) - anchor_dow) % DAYS_PER_WEEK
    latest -= pd.Timedelta(days=back)
    step = pd.Timedelta(days=horizon_days)
    return [latest - k * step for k in range(n_folds)]


def _iter_test_folds(
    df: pd.DataFrame, *, n_folds: int, horizon_days: int, anchor_dow: int | None,
):
    """fold별 (index, test_df)를 최신→과거 순으로 낸다.

    anchor_dow=None이면 기존 인덱스 기반 연속 블록(헤드라인 경로, 산술 불변).
    anchor_dow가 주어지면 날짜 기반 요일 앵커 블록.
    """
    if anchor_dow is None:
        total = len(df)
        test_size = horizon_days
        for k in range(n_folds):
            test_end = total - k * test_size
            test_start = test_end - test_size
            yield k, df.iloc[test_start:test_end]
        return
    span = pd.Timedelta(days=horizon_days - 1)
    starts = _fold_starts_by_dow(df, anchor_dow, horizon_days, n_folds)
    for k, start in enumerate(starts):
        yield k, df[(df["date"] >= start) & (df["date"] <= start + span)]


def _ar_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """타깃 유래 자기회귀 feature 컬럼(lag/rmean/rstd/ewma)."""
    return [c for c in df.columns if c.startswith(f"{target_col}_")]


def _require_gapless(ar_history: pd.DataFrame) -> pd.DataFrame:
    """AR 재계산은 위치 기반 shift라 날짜가 연속이어야 한다 — 아니면 조용히 틀린다.

    backtest가 받는 프레임은 dropna로 gap이 생겨 있으므로(광교 실측 7건) 재계산에
    쓸 수 없다. 그래서 gapless history를 따로 받고, 여기서 fails-loud로 검증한다.
    """
    hist = ar_history.sort_values("date").reset_index(drop=True)
    gaps = hist["date"].diff().dt.days.dropna()
    bad = int((gaps != 1).sum())
    if bad:
        raise ValueError(
            f"ar_history에 날짜 gap {bad}건 — 자기회귀 재계산이 위치 shift로 어긋난다. "
            "dropna 이전의 날짜 연속 프레임을 넘겨라."
        )
    return hist


def _blind_ar_features(
    test_df: pd.DataFrame, *, ar_history: pd.DataFrame, cutoff: pd.Timestamp, target_col: str,
) -> pd.DataFrame:
    """원점(cutoff) 이후 실측을 보는 자기회귀 feature를 가린다(운영 feature 가용성 정렬).

    lead_days는 학습 시점만 옮기므로 test 행의 lag/rolling/ewma는 여전히 원점 이후를
    본다(예: 월요일 블록의 lag1 = 전날 일요일). 여기서 타깃을 cutoff 이후로 마스킹한
    뒤 AR feature만 재계산해 그 경로를 끊는다. 캘린더·날씨·경쟁점 feature는 타깃과
    무관해 fold-invariant이므로 재계산 대상이 아니다.

    ★타깃 컬럼 자체는 건드리지 않는다 — 평가(actual)에 쓰인다.
    """
    hist = _require_gapless(ar_history[["date", target_col]].copy())
    hist[target_col] = hist[target_col].mask(hist["date"] >= cutoff)
    blinded = add_lag_rolling_ewma(hist, target_col)
    cols = _ar_columns(test_df, target_col)
    patch = test_df[["date"]].merge(blinded[["date", *cols]], on="date", how="left")
    out = test_df.copy()
    out[cols] = patch[cols].to_numpy()
    return out


def windowed_backtest(
    df: pd.DataFrame, *, window_days: int,
    target_col: str = "adjusted_demand_unit", n_folds: int = 52,
    horizon_days: int = 7, production_q: float = 0.85, alpha: float = 0.8,
    events: dict | None = None, lunar_events: dict | None = None,
    min_train_rows: int = MIN_TRAIN_ROWS,
    forecaster: Forecaster | None = None,
    lead_days: int = 0, anchor_dow: int | None = None,
    ar_history: pd.DataFrame | None = None,
    order_level: str = ORDER_LEVEL_CATEGORY,
    item_history: pd.DataFrame | None = None,
) -> BacktestResult:
    """카테고리 총량 백테스트. order_level="item"이면 품목 배분까지 함께 낸다.

    order_level: "category"(기본, 헤드라인) | "item". "item"이면 `item_history`
      (품목 일별 프레임)가 **필수** — 없으면 fails-loud. 배분은 가법 출력이며
      folds/predictions는 배분 여부와 무관하게 동일하다(엔진 동등성 hard gate).
    """
    if order_level not in (ORDER_LEVEL_CATEGORY, ORDER_LEVEL_ITEM):
        raise ValueError(f"order_level must be 'category' or 'item', got {order_level!r}")
    if order_level == ORDER_LEVEL_ITEM and item_history is None:
        raise ValueError("order_level='item' requires item_history (품목 일별 프레임)")
    fc = forecaster if forecaster is not None else CategoryTotalForecaster()
    df = df.sort_values("date").reset_index(drop=True).dropna(subset=[target_col]).copy()
    df = df.dropna().reset_index(drop=True)
    total = len(df)
    test_size = horizon_days
    if total <= n_folds * test_size + min_train_rows:
        raise ValueError(f"Not enough data: total={total}, folds={n_folds}")

    window = pd.Timedelta(days=window_days)
    folds, preds = [], []
    item_orders: list[pd.DataFrame] = []
    for k, test_df in _iter_test_folds(
        df, n_folds=n_folds, horizon_days=horizon_days, anchor_dow=anchor_dow,
    ):
        if len(test_df) == 0:      # 요일 앵커 fold가 데이터 범위를 벗어난 경우
            continue
        test_start_date = test_df["date"].iloc[0]
        # 운영 리드타임: 원점 이후 데이터는 train/prior 어디에도 넣지 않는다(leakage 차단).
        # lead_days=0이면 cutoff == test_start_date → 기존 산술과 완전 동일.
        cutoff = test_start_date - pd.Timedelta(days=lead_days)
        if ar_history is not None:
            # 운영 feature 가용성 정렬: 원점 이후를 보는 AR feature를 가린다.
            # train 행은 전부 cutoff 미만이라 손대지 않는다(feature 불변).
            test_df = _blind_ar_features(
                test_df, ar_history=ar_history, cutoff=cutoff, target_col=target_col,
            )
        # === 유일한 변경점(원본 대비): train slice 를 날짜 기반 rolling window 로 ===
        train_df = df[(df["date"] < cutoff) & (df["date"] >= cutoff - window)]
        if len(train_df) < min_train_rows:
            continue
        model = fc.fit(
            train_df, target_col=target_col, alpha=alpha, production_q=production_q,
        )
        exp_pred = model.predict_expected(test_df)
        prod_pred = model.predict_production(test_df)
        # 특수일 레벨-앵커 prior: cutoff 이전 전체 history로 fit (train window보다 길게, leakage-safe)
        hist = df[df["date"] < cutoff]
        if events or lunar_events:   # event_prior 명시적으로 있을 때만 적용 (None/{}=비활성, EventLevelPrior의 xmas default fallback 방지)
            prior = EventLevelPrior(events=events, lunar_events=lunar_events).fit(hist, target_col=target_col)
            exp_pred, prod_pred = prior.blend(test_df["date"].values, exp_pred, prod_pred)
        actual = test_df[target_col].values
        wape = np.abs(actual - exp_pred).sum() / max(np.abs(actual).sum(), 1)
        folds.append(dict(
            fold=k, n_train=len(train_df), n_test=len(test_df),
            test_start=test_start_date, test_end=test_df["date"].iloc[-1],
            wape=wape,
            wpe=(exp_pred - actual).sum() / max(actual.sum(), 1),
            prod_pct_under=(prod_pred < actual).mean(),
        ))
        preds.append(pd.DataFrame({
            "date": test_df["date"].values, "fold": k,
            "actual": actual, "expected": exp_pred, "production": prod_pred,
        }))
        if order_level == ORDER_LEVEL_ITEM:
            item_orders.append(_distribute_fold_orders(
                item_history, test_df["date"].values, prod_pred,
                fold=k, lead_days=lead_days,
            ))
    return BacktestResult(
        folds=pd.DataFrame(folds).sort_values("fold").reset_index(drop=True),
        predictions=pd.concat(preds, ignore_index=True),
        item_orders=pd.concat(item_orders, ignore_index=True) if item_orders else None,
    )


def metrics_from_preds(p: pd.DataFrame) -> dict:
    actual, expected, prod = p["actual"], p["expected"], p["production"]
    surplus = (prod - actual).clip(lower=0)
    return {
        "n_test": int(len(p)),
        "wape": float(np.abs(actual - expected).sum() / max(np.abs(actual).sum(), 1)),
        "wpe": float((expected - actual).sum() / max(actual.sum(), 1)),
        "stockout_risk": float((prod < actual).mean()),
        "surplus_mean_units": float(surplus.mean()),
        "surplus_rate": float(surplus.sum() / max(actual.sum(), 1)),
    }
