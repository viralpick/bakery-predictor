"""매장별 '매진 무영향' 가정 심층 검증 — 요약만 출력.

가정(모델링 전제): 매진(재고 소진)은 매출/수요 추정에 체계적 손실을 주지 않는다.
  → 그래서 censored sold 를 수요 proxy 로 써도 편향이 작다.

검증의 본질적 난제: 매진은 수요에 내생적이다.
  수요 높은 날 = 더 많이 팔림(rev↑) + 더 많이/일찍 매진(stockout↑).
  따라서 rev~stockout 단순 상관은 (+)가 나오는 게 정상이며, 그 자체로
  '매진이 매출에 도움된다'거나 '무영향'을 증명하지 못한다. 실제 lost-sales 는
  관측되지 않고 검열(censored)된다.

식별 전략 — 매장별로 아래 4개 레이어를 본다:
  L1 (control 가용성): 무매진일이 존재하는가? 없으면 within-store 비교 불가.
  L2 (혼동된 상관): rev~매진강도 Welch t / 상관 — 부호만 기록 (내생성 주의).
  L3 (calendar 제거 FE): log(rev) ~ 매진강도 + dow + month + year_trend.
  L4 (lost-sales 핵심): 트래픽(receipts) 통제 후 '이른 매진'이 basket/rev 를
       추가로 깎는가? rev = receipts × basket 분해.
       log(rev) ~ log(receipts) + early_stockout_share + 달력.
       early share 계수가 유의하게 (-) → 같은 트래픽인데 일찍 동나서 매출 손실 신호.
       ~0 → 무영향 가정과 일관 (남은 게 없거나 대체 구매로 흡수).

매장별 평결: [무영향과 일관] / [lost-sales 신호] / [식별 불가].
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

V2 = Path("data/internal/v2")
STORE_MAP = {
    "1000000047": "광교",
    "1000000009": "삼성타운",
    "1000000029": "메세나폴리스",
    "1000000485": "광화문",
}
EARLY_HOUR = 15  # 15:00 이전 매진 = '이른 매진' (영업 후반 손실 가능성↑)


def parse_hhmm(v: str) -> float:
    try:
        s = str(int(float(v))).zfill(4)
        h, m = int(s[:-2]), int(s[-2:])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return h * 60 + m
    except (ValueError, TypeError):
        return np.nan
    return np.nan


def build_stockout_daily() -> pd.DataFrame:
    so = pd.read_parquet(V2 / "stockout.parquet")
    so["store"] = so["CD_PARTNER"].astype(str).map(STORE_MAP)
    so["date"] = pd.to_datetime(so["DT_SALE"].astype(str), errors="coerce")
    so["sold_min"] = so["SOLD_TIME"].apply(parse_hhmm)
    so = so.dropna(subset=["store", "date", "sold_min"])
    # 품목이 하루 여러 번 동날 수 있음 → 그날 '처음 동난 시각'으로 품목 단위 축약
    item_day = so.groupby(["store", "date", "CD_ITEM"]).agg(
        first_stockout_min=("sold_min", "min"),
    ).reset_index()
    item_day["is_early"] = item_day["first_stockout_min"] < EARLY_HOUR * 60
    daily = item_day.groupby(["store", "date"]).agg(
        n_stockout=("CD_ITEM", "nunique"),
        n_early_stockout=("is_early", "sum"),
        median_stockout_min=("first_stockout_min", "median"),
    ).reset_index()
    daily["early_share"] = daily["n_early_stockout"] / daily["n_stockout"]
    return daily


def build_panel() -> pd.DataFrame:
    daily = pd.read_parquet(V2 / "daily_4stores.parquet")[
        ["store", "DT_SALE", "qty", "rev", "receipts", "dow", "year"]
    ].rename(columns={"DT_SALE": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.month
    so = build_stockout_daily()
    panel = daily.merge(so, on=["store", "date"], how="left")
    for c in ["n_stockout", "n_early_stockout"]:
        panel[c] = panel[c].fillna(0)
    panel["has_stockout"] = panel["n_stockout"] > 0
    panel = panel[(panel["rev"] > 0) & (panel["receipts"] > 0)].copy()
    return panel


def layer1_control(g: pd.DataFrame) -> str:
    n_zero = int((~g["has_stockout"]).sum())
    n_tot = len(g)
    return (
        f"  L1 control: 무매진일 {n_zero}/{n_tot}일 ({n_zero/n_tot:.1%}) | "
        f"매진일 평균 동난 품목수 {g.loc[g['has_stockout'],'n_stockout'].mean():.0f}종"
    )


def layer2_confounded(g: pd.DataFrame) -> str:
    hi = g[g["n_stockout"] >= g["n_stockout"].median()]
    lo = g[g["n_stockout"] < g["n_stockout"].median()]
    if len(lo) < 5 or len(hi) < 5:
        return "  L2 혼동상관: 분할 불가 (한쪽 표본<5)"
    t, p = stats.ttest_ind(hi["rev"], lo["rev"], equal_var=False)
    diff = (hi["rev"].mean() - lo["rev"].mean()) / lo["rev"].mean() * 100
    r = g["n_stockout"].corr(g["rev"])
    return (
        f"  L2 혼동상관(내생성↑): 매진多日 rev {diff:+.1f}% vs 매진少日 "
        f"(Welch p={p:.3f}) | corr(n_stockout,rev)={r:+.3f}  [부호 참고용]"
    )


def layer3_fe(g: pd.DataFrame) -> str:
    d = g.copy()
    d["log_rev"] = np.log(d["rev"])
    d["yr"] = d["year"] - d["year"].min()
    try:
        m = smf.ols("log_rev ~ n_stockout + C(dow) + C(month) + yr", data=d).fit()
        b = m.params.get("n_stockout", np.nan)
        p = m.pvalues.get("n_stockout", np.nan)
        return f"  L3 FE(달력제거): 매진1종↑당 rev {b*100:+.2f}%/종 (p={p:.3f})  [여전히 내생]"
    except Exception as e:  # noqa: BLE001
        return f"  L3 FE: 실패 ({e})"


def layer4_lostsales(g: pd.DataFrame) -> tuple[str, float, float, int]:
    """핵심: 트래픽(receipts) 통제 후 early_share 가 rev 를 추가로 깎는가."""
    d = g[g["has_stockout"]].copy()
    d = d.dropna(subset=["early_share"])
    if len(d) < 30:
        return ("  L4 lost-sales: 표본부족(<30 매진일)", np.nan, np.nan, len(d))
    d["log_rev"] = np.log(d["rev"])
    d["log_rcpt"] = np.log(d["receipts"])
    d["yr"] = d["year"] - d["year"].min()
    try:
        m = smf.ols(
            "log_rev ~ log_rcpt + early_share + C(dow) + C(month) + yr", data=d
        ).fit()
        b = m.params.get("early_share", np.nan)
        p = m.pvalues.get("early_share", np.nan)
        msg = (
            f"  L4 lost-sales(트래픽통제): early_share 계수 {b*100:+.2f}% (p={p:.3f}) | "
            f"n={len(d)}매진일  [<0 & 유의 → 손실신호]"
        )
        return (msg, b, p, len(d))
    except Exception as e:  # noqa: BLE001
        return (f"  L4 lost-sales: 실패 ({e})", np.nan, np.nan, len(d))


def layer4b_robustness(g: pd.DataFrame) -> str:
    """교차검증: 매진 중앙시각이 늦을수록(=덜 일찍 동남) rev 가 높은가.

    early_share 와 반대 방향 변수. lost-sales 라면 median_stockout_min 계수가
    (+) 여야 부호 일관 (늦게 동남 = 손실 적음 = rev↑).
    """
    d = g[g["has_stockout"]].copy()
    d = d.dropna(subset=["median_stockout_min"])
    if len(d) < 30:
        return "  L4b 강건성: 표본부족"
    d["log_rev"] = np.log(d["rev"])
    d["log_rcpt"] = np.log(d["receipts"])
    d["yr"] = d["year"] - d["year"].min()
    d["med_h"] = d["median_stockout_min"] / 60.0
    try:
        m = smf.ols(
            "log_rev ~ log_rcpt + med_h + C(dow) + C(month) + yr", data=d
        ).fit()
        b = m.params.get("med_h", np.nan)
        p = m.pvalues.get("med_h", np.nan)
        sign = "일관(+,손실)" if (b > 0 and p < 0.05) else ("역방향" if b < 0 and p < 0.05 else "무의미")
        return f"  L4b 강건성: 매진 1h 늦어질수록 rev {b*100:+.2f}% (p={p:.3f}) → {sign}"
    except Exception as e:  # noqa: BLE001
        return f"  L4b 강건성: 실패 ({e})"


def verdict(g: pd.DataFrame, b4: float, p4: float, n4: int) -> str:
    n_zero = int((~g["has_stockout"]).sum())
    if n4 < 30:
        return "  ⇒ 평결: 식별 불가 (매진일 표본 부족)"
    if np.isnan(b4):
        return "  ⇒ 평결: 식별 불가 (L4 추정 실패)"
    if p4 < 0.05 and b4 < 0:
        return f"  ⇒ 평결: ⚠️ lost-sales 신호 (트래픽 같아도 이른매진→rev {b4*100:.1f}%, p={p4:.3f})"
    ctrl = "무매진일 없음→within-store control 부재" if n_zero == 0 else f"무매진일 {n_zero}일"
    return f"  ⇒ 평결: 무영향 가정과 일관 (early_share 효과 무의미 p={p4:.3f}; {ctrl})"


def main() -> None:
    panel = build_panel()
    print("=" * 74)
    print(f"매장별 '매진 무영향' 심층 검증 (EARLY_HOUR={EARLY_HOUR}:00, 5년 일단위)")
    print("=" * 74)
    for store in ["광교", "삼성타운", "메세나폴리스", "광화문"]:
        g = panel[panel["store"] == store]
        if g.empty:
            continue
        print(f"\n[{store}]  n={len(g)}일, 평균 rev {g['rev'].mean():,.0f}원")
        print(layer1_control(g))
        print(layer2_confounded(g))
        print(layer3_fe(g))
        msg4, b4, p4, n4 = layer4_lostsales(g)
        print(msg4)
        print(layer4b_robustness(g))
        print(verdict(g, b4, p4, n4))


if __name__ == "__main__":
    main()
