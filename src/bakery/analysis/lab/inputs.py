"""canonical 입력 lazy 로더 — 핸들러가 공유하는 단일 IO 지점.

모든 경로는 `bakery.data.paths.dataset()`으로만 도달한다(레거시 data/internal/v2
직독 금지). 측정 헌장: bulk는 canonical 빌드에서 이미 제외됐고, potential_demand는
오염 소스라 여기서 컬럼째 제거해 소비 자체를 막는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import pandas as pd

from bakery.analysis.lab.spec import MULTISTORE, AnalysisSpec
from bakery.data import paths

STORE_CODES: dict[str, str] = {
    "store_gw01": "1000000047",
    "store_ss01": "1000000009",
    "store_mp01": "1000000029",
    "store_gh01": "1000000485",
}
STORE_NAMES: dict[str, str] = {
    "store_gw01": "광교",
    "store_ss01": "삼성타운",
    "store_mp01": "메세나폴리스",
    "store_gh01": "광화문",
}
# harness.event_priors.STORE_EVENT_PRIORS의 키(영문 라벨) — 한글명과 다르다.
STORE_PRIOR_KEYS: dict[str, str] = {
    "store_gw01": "gwangyo",
    "store_ss01": "samsung",
    "store_mp01": "mecenatpolis",
    "store_gh01": "gwanghwamun",
}
GWANGYO = "store_gw01"
_POLLUTED_COLUMNS = ("potential_demand",)   # 오염 소스 — 측정 헌장상 사용 금지


@dataclass
class AnalysisInputs:
    """spec이 가리키는 입력 묶음. 속성 최초 접근 시에만 IO."""

    store: str
    alpha: float
    predictions_path: Path | None = None
    params: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: AnalysisSpec) -> AnalysisInputs:
        return cls(store=spec.data.store, alpha=spec.alpha,
                   predictions_path=spec.predictions, params=dict(spec.params))

    # ---------- 메타 ----------

    @property
    def is_multistore(self) -> bool:
        return self.store == MULTISTORE

    @property
    def store_code(self) -> str:
        """단매장 CD_PARTNER. multistore면 광교(참조 매장) 코드."""
        return STORE_CODES[GWANGYO if self.is_multistore else self.store]

    @property
    def prior_key(self) -> str:
        """이벤트 prior 프리셋 키. multistore면 광교(참조 매장)."""
        return STORE_PRIOR_KEYS[GWANGYO if self.is_multistore else self.store]

    @property
    def has_predictions(self) -> bool:
        return self.predictions_path is not None and Path(self.predictions_path).exists()

    def params_for(self, name: str) -> dict:
        return self.params.get(name, {})

    # ---------- 입력 ----------

    @cached_property
    def daily(self) -> pd.DataFrame:
        """item×day 관측 daily. bulk 제외·매진 재정의 반영(canonical)."""
        key = "multistore_daily" if self.is_multistore else "bonavi_daily"
        df = pd.read_parquet(paths.dataset(key))
        df = df.drop(columns=[c for c in _POLLUTED_COLUMNS if c in df.columns])
        df["item_id"] = df["item_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        if not self.is_multistore:
            df = df[df["store_id"] == self.store]
        return df.reset_index(drop=True)

    @cached_property
    def category_daily(self) -> pd.DataFrame:
        """날짜별 카테고리 합 daily(adjusted_demand_unit 포함) — 광교 전용."""
        from bakery.features.category_aggregate import build_category_daily

        return build_category_daily(alpha=self.alpha).df

    @cached_property
    def receipts(self) -> pd.DataFrame:
        """라인레벨 영수증(광교). is_bulk는 진단용 컬럼이며 이미 필터된 프레임."""
        df = pd.read_parquet(paths.dataset("bonavi_receipts"))
        df["item_id"] = df["item_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def discount_rows(self) -> pd.DataFrame:
        from bakery.analysis.discount import load_sales_with_discount_v2

        return load_sales_with_discount_v2(store_code=self.store_code).rows

    @cached_property
    def closing_returns(self) -> pd.DataFrame:
        from bakery.analysis.discount import load_closing_returns_v2

        return load_closing_returns_v2(store_code=self.store_code)

    @cached_property
    def waste(self) -> pd.DataFrame:
        """생산/폐기/마감 실측(4매장). production_qty=made, waste_qty=out.

        waste_qty 음수는 전일 재고 이월(carry-in)로 판매가 당일 생산을 초과한 경우다.
        값을 clip하지 않는다 — clip하면 made−(normal+closing)−out 항등식이 더 깨지고
        폐기율(1차 KPI)이 부풀려진다(광교 0.12532→0.12933).
        참고(2026-07-28 실측): 항등식 잔차 0 비율은 전체 91.83%, out<0 행 88.80%
        (8,108행 중 908행 위반, |잔차| max 25.0) — 음수 행이 오히려 약간 더 나쁘다.
        항등식 자체의 진단은 analysis/lab/handlers/waste.py(waste_alpha_identity)가 담당한다.
        """
        df = pd.read_parquet(paths.dataset("waste_alpha_4stores"))
        df["cd"] = df["cd"].astype(str)
        df["item_id"] = df["item_id"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        if not self.is_multistore:
            df = df[df["cd"] == self.store_code]
        df = df.rename(columns={"made": "production_qty", "out": "waste_qty"})
        return df.reset_index(drop=True)

    @cached_property
    def item_to_category(self) -> pd.Series:
        """item_id → category_id (canonical daily에서 유도)."""
        pairs = self.daily[["item_id", "category_id"]].drop_duplicates("item_id")
        return pairs.set_index("item_id")["category_id"]

    @cached_property
    def predictions(self) -> pd.DataFrame | None:
        """harness-run 산출 predictions.csv(date/fold/actual/expected/production). 읽기 전용."""
        if not self.has_predictions:
            return None
        df = pd.read_csv(self.predictions_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def calendar(self) -> pd.DataFrame:
        from bakery.data.calendar import build_calendar_daily

        dates = self.daily["date"]
        return build_calendar_daily(dates.min(), dates.max())
