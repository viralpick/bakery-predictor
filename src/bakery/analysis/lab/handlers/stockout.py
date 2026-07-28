"""매진 추정 손실 규모(하한) 보고 / 매진 부스트 ablation이 배분 순위를 흔드는가.

계산은 `bakery.analysis.{self_fulfillment, popularity}` 프리미티브 호출.
출처 스크립트: verify_stockout_revenue_4stores(_fixed), revalidate_popularity_stockout.

측정 헌장: 품절일 판매량은 censored — 추정 손실은 하한이다.

★`stockout_lost_demand`는 "매진의 매장 매출 영향(무영향 가정)" 가설 검정이 **아니다**.
그 가설은 traffic 통제 4-layer OLS(scripts/verify_stockout_revenue_4stores.py)로
2026-06-03/07-10 두 차례 검증됐고(3/4 매장 무영향·메세나만 약신호), 그 OLS는 아직
프리미티브로 이식되지 않았다. 여기서는 그와 다른 것 — 대체구매를 통제하지 않은
품목단위 손실 합산 규모 — 만 보고한다. 자세한 사유는 `_NOTE_NOT_THE_HYPOTHESIS_TEST`.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from scipy.stats import spearmanr

from bakery.analysis.lab.inputs import AnalysisInputs
from bakery.analysis.lab.registry import register_hypothesis
from bakery.analysis.lab.result import KIND_HYPOTHESIS, AnalysisResult
from bakery.analysis.self_fulfillment import (
    estimated_lost_demand,
    stockout_hour_distribution,
    top_self_fulfilling_items,
)

LOST_UNITS_COLUMN = "lost_units"                 # estimated_lost_demand의 손실량 컬럼
_LOCAL_ESTIMATE_COLUMNS = ("potential_demand",)  # 함수 내부 추정치 — 출력에서 드롭

LOST_SHARE_THRESHOLD = 0.02      # 보고 임계(가설 게이트 아님) — 이 비중 이상이면 표에 표시
POPULARITY_CORR_THRESHOLD = 0.8  # 순위 상관 0.8 이상 = 신호 안정
_TOP_ITEMS = 15
_TOP_RANK_N = 10  # top-N 멤버십 교체건수 보고 기준 — 통계적 임계가 아니라 리포팅 관례상 선택
_NOTE_CENSORED = ("품절일 판매량은 censored — 추정 손실은 하한이다. "
                  "실제 손실은 이 수치 이상일 수 있다.")
_NOTE_LOST_MODEL = ("손실 추정은 features/potential_demand와 같은 시간가중 공식 "
                    "(estimated_lost_demand) — 모델 예측이 아니라 관측 기반 산식이다.")
_NOTE_NOT_THE_HYPOTHESIS_TEST = (
    "이 항목은 매진으로 인한 추정 손실 **규모**만 보고한다 — '매진이 매장 매출에 영향 없다'는 "
    "가설의 검정이 아니다. 그 검정은 traffic 통제 4-layer OLS"
    "(scripts/verify_stockout_revenue_4stores.py, log_rev ~ n_stockout + C(dow) + C(month) + yr)이며 "
    "아직 프리미티브로 이식되지 않았다. 기존 확정 결론(2026-06-03, 2026-07-10 재검증)은 "
    "3/4 매장(광교·삼성·광화문) 무영향 / 메세나폴리스 약신호이고, 여기 수치와 상충하지 않는다 "
    "— 서로 다른 것을 재기 때문이다."
)


def lost_demand_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """매장별 매진 품목-일 건수 + 추정 손실 수량 + sold 대비 비중.

    `n_stockout_item_days`는 캘린더 일수가 아니라 (품목×일) 단위로 `is_stockout`이
    True인 행수다 — 한 매장이 하루에 여러 품목이 동시에 매진되면 그만큼 여러 번 잡힌다.
    """
    lost = estimated_lost_demand(daily)
    per_store_lost = lost.groupby("store_id")[LOST_UNITS_COLUMN].sum()
    rows = []
    for store, group in daily.groupby("store_id", observed=True):
        sold = float(group["sold_units"].sum())
        est_lost = float(per_store_lost.get(store, 0.0))
        rows.append({"store_id": store,
                     "n_stockout_item_days": int(group["is_stockout"].sum()),
                     "est_lost_units": est_lost,
                     "lost_share_of_sold": est_lost / sold if sold else 0.0})
    return pd.DataFrame(rows)


def lost_demand_verdict(summary: pd.DataFrame) -> str:
    """규모 보고 — 무영향 가설의 판정이 아니다(그 검정은 이식되지 않았다)."""
    worst = summary.loc[summary["lost_share_of_sold"].idxmax()]
    return (f"추정 손실 비중(하한) 최대 {worst['store_id']} "
            f"{worst['lost_share_of_sold'] * 100:.1f}%, "
            f"매장 {len(summary)}곳 중 "
            f"{int((summary['lost_share_of_sold'] >= LOST_SHARE_THRESHOLD).sum())}곳이 "
            f"보고 임계 {LOST_SHARE_THRESHOLD * 100:.0f}% 이상")


def popularity_boost_correlation(daily: pd.DataFrame, *,
                                 target_date: pd.Timestamp) -> pd.DataFrame:
    """매진 부스트 ablation — 실제 배분 vs 부스트만 제거한 반사실 배분의 순위 상관.

    proportion은 base_sold × adj_trend × adj_stockout × adj_closing × adj_new를 정규화한
    값이라, 원시 인기와 직접 비교하면 (1) base_sold에 지배되고 (2) 네 조정계수가 한 rho에
    섞여 매진 부스트의 기여를 분리할 수 없다. 그래서 adj_stockout만 1.0으로 되돌린
    반사실 배분을 만들어 비교한다 — 나머지 계수가 동일하므로 순위 차이는 부스트 효과다.
    """
    from bakery.models.item_proportion import compute_proportions

    proportions = compute_proportions(daily, target_date)
    pair = proportions[["item_id", "proportion", "adj_stockout"]].dropna().copy()
    # 반사실: adj_stockout을 나눠 제거하고 다시 정규화(Σ=1)
    ablated_weight = pair["proportion"] / pair["adj_stockout"]
    total = ablated_weight.sum()
    pair["proportion_no_boost"] = ablated_weight / total if total else 0.0
    rho = float(spearmanr(pair["proportion"], pair["proportion_no_boost"]).statistic)
    top_actual = pair.nlargest(_TOP_RANK_N, "proportion")["item_id"].tolist()
    top_ablated = pair.nlargest(_TOP_RANK_N, "proportion_no_boost")["item_id"].tolist()
    return pd.DataFrame([{
        "pair": "proportion_vs_no_stockout_boost", "spearman": rho, "n": int(len(pair)),
        "n_top_changed": int(len(set(top_actual) ^ set(top_ablated)) // 2),
        "max_abs_share_delta": float(
            (pair["proportion"] - pair["proportion_no_boost"]).abs().max()),
        "adj_stockout_min": float(pair["adj_stockout"].min()),
        "adj_stockout_max": float(pair["adj_stockout"].max()),
        "adj_stockout_std": float(pair["adj_stockout"].std(ddof=0)),
    }])


def popularity_verdict(corr: pd.DataFrame) -> str:
    row = corr.iloc[0]
    rho, n = float(row["spearman"]), int(row["n"])
    churn, delta = int(row["n_top_changed"]), float(row["max_abs_share_delta"])
    prefix = ("매진 부스트가 배분 순위를 거의 바꾸지 않음" if rho >= POPULARITY_CORR_THRESHOLD
              else "매진 부스트가 배분 순위를 크게 재배열")
    return (f"{prefix} — ablation spearman {rho:.3f} (n={n}), "
            f"top{_TOP_RANK_N} 교체 {churn}건, 최대 비중 변화 {delta:.4f}")


def _lost_fig(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(x=summary["store_id"], y=summary["lost_share_of_sold"]))
    fig.add_hline(y=LOST_SHARE_THRESHOLD, line_dash="dash")
    fig.update_layout(title="매장별 추정 손실 비중(점선=2% 임계)",
                      xaxis_title="매장", yaxis_title="sold 대비 비중")
    return fig


def _hour_fig(hours: pd.DataFrame) -> go.Figure:
    """품목×요일 평균 매진시각의 요일별 분포(히스토그램이 아니라 평균값의 분포)."""
    fig = go.Figure()
    for dow, group in hours.groupby("dow", observed=True):
        fig.add_trace(go.Box(y=group["stockout_hour_mean"], name=str(dow)))
    fig.update_layout(title="요일별 평균 매진시각 분포(품목 단위)",
                      xaxis_title="요일(월=0)", yaxis_title="평균 매진시각(시)")
    return fig


@register_hypothesis("stockout_lost_demand", "매진 추정 손실 규모(하한)")
def stockout_lost_demand(inputs: AnalysisInputs) -> AnalysisResult:
    daily = inputs.daily
    summary = lost_demand_summary(daily)
    hours = stockout_hour_distribution(daily)
    return AnalysisResult(
        name="stockout_lost_demand", kind=KIND_HYPOTHESIS,
        title="매진 추정 손실 규모(하한)",
        tables=[("summary", summary),
                ("top_self_fulfilling", top_self_fulfilling_items(daily, n=_TOP_ITEMS)),
                ("hour_distribution", hours)],
        figures=[_lost_fig(summary), _hour_fig(hours)],
        verdict=lost_demand_verdict(summary),
        notes=[_NOTE_CENSORED, _NOTE_LOST_MODEL, _NOTE_NOT_THE_HYPOTHESIS_TEST],
    )


@register_hypothesis("popularity_stockout", "매진 부스트가 배분 순위를 재배열하는가",
                     needs_single_store=True)
def popularity_stockout(inputs: AnalysisInputs) -> AnalysisResult:
    from bakery.analysis.popularity import compute_popularity_signals
    from bakery.models.item_proportion import STOCKOUT_MAX_BOOST

    closing = inputs.discount_rows
    closing = closing[closing["label"] == "closing"][["item_id", "date", "qty"]]
    target_date = pd.Timestamp(inputs.daily["date"].max())
    signals = compute_popularity_signals(inputs.daily, closing, today=target_date)
    corr = popularity_boost_correlation(inputs.daily, target_date=target_date)
    fig = go.Figure(go.Bar(x=corr["pair"], y=corr["spearman"]))
    fig.add_hline(y=POPULARITY_CORR_THRESHOLD, line_dash="dash")
    fig.update_layout(title="인기 신호 순위 상관(점선=0.8 임계)", yaxis_title="spearman")
    return AnalysisResult(
        name="popularity_stockout", kind=KIND_HYPOTHESIS,
        title="매진 부스트가 배분 순위를 재배열하는가",
        tables=[("rank_correlation", corr), ("signals", signals)],
        figures=[fig], verdict=popularity_verdict(corr),
        notes=["매진 라벨은 재정의(폐기0=완판) 반영본 — 옛 92.7% 정의가 아니다.",
               ("출처 스크립트의 옛/새 라벨 A/B는 canonical에 옛(오염) 라벨이 없어 "
                "불가 — 실제 배분(proportion) vs adj_stockout만 제거한 반사실 배분의 "
                "순위 ablation으로 재정의했다."),
               f"부스트 상한 STOCKOUT_MAX_BOOST={STOCKOUT_MAX_BOOST}, "
               f"기준일={target_date.date()}"],
    )
