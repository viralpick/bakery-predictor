"""raw → interim → processed 단일진입 오케스트레이터. 기존 loader 호출만."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bakery.data import bonavi_loader_v2 as v2
from bakery.data import paths


def build_internal(reconvert: bool = False,
                   out_root: Path | None = None) -> dict[str, Path]:
    """내부 결정적 테이블 재생성. out_root 주면 그 밑에(테스트/진단용), 없으면 registry 위치."""
    clean = paths.dataset("sales_lines_clean")
    if reconvert or not clean.exists():
        v2.convert_sales_to_parquet(paths.dataset("sales_xlsx"), clean)
    daily = (out_root / "bonavi_daily.parquet") if out_root else paths.dataset("bonavi_daily")
    receipts = (out_root / "bonavi_receipts.parquet") if out_root else paths.dataset("bonavi_receipts")
    v2.build_v2(clean_parquet=clean, master_xlsx=paths.dataset("master_xlsx"),
                out_path=daily, receipts_path=receipts)
    return {"bonavi_daily": daily, "bonavi_receipts": receipts}


def equivalence_diff(rebuilt: dict[str, Path],
                     reference: dict[str, Path]) -> dict[str, float]:
    """재생성 vs 참조 테이블의 최대 수치 diff (0이면 완전 일치)."""
    out: dict[str, float] = {}
    for name, path in rebuilt.items():
        r = pd.read_parquet(path)
        d = pd.read_parquet(reference[name])
        keys = [c for c in ("date", "item_id", "store_id") if c in r.columns]
        r = r.sort_values(keys).reset_index(drop=True)
        d = d.sort_values(keys).reset_index(drop=True)
        max_diff = 0.0
        for c in r.select_dtypes(include=[np.number]).columns:
            max_diff = max(max_diff,
                           float((r[c].fillna(-9e9) - d[c].fillna(-9e9)).abs().max()))
        out[name] = max_diff
    return out
