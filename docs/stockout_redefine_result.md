# is_stockout/stockout_time 재정의 결과 (하위1)

재생성: `uv run bakery format-bonavi`. bonavi_loader가 폐기(재고정보)+마지막판매 기반으로
is_stockout/stockout_time 산출(첫 순간품절 이벤트 버그 대체).

재생성 과정에서 **배선 버그**(Critical)를 발견해 수정: `build`가 재고정보를 `xlsx_path`
기본값(`보나비 데이터_20260520.xlsx`)에서 읽으려 했으나 이 파일엔 `재고정보` 시트가
없어 crash(`ValueError`)했다. `재고정보`는 `보나비 데이터_20260526.xlsx`에만 존재 —
`build`에 별도 `inventory_xlsx_path` 파라미터(기본값 0526)를 추가해 items/sales/receipts는
0520, inventory는 0526에서 각각 읽도록 분리했다. (커밋 `b3b9c21`)

- is_stockout: 92% → **60.4%** (made>0 & waste≤0)
- stockout_time: 09:05 등 이른 아티팩트 소멸, 마지막 실판매와 정합 —
  hour 분위수 {25%: 16시, 50%(중앙값): 19시, 75%: 20시}. 07~09시 이른 비율 **0.5%**
  (과거 개점 직후 고정값 아티팩트가 저녁 시간대 실제 소진 시각으로 대체됨을 확인).
- potential_demand: is_stockout False 행의 **100%**가 `== sold_units`
  (전체 행 기준으로는 40.4% — is_stockout 비율 60.4%와 상보적으로 정합).
  `potential_demand >= sold_units`는 전 행(65,452건)에서 성립 — 물리적으로 불가능한
  하한 위반 없음.
- 재고정보 결측: sales(date, item_id) 조합 65,452건 중 **65건(0.1%)**이 inventory와
  매치되지 않음 → `assign_stockout_fields`의 NaN 비교 자동 False 규칙에 따라
  `is_stockout=False`로 처리됨(계획서 예상치 0.1%와 정확히 일치).

## 전체 스위트

```
uv run pytest --color=no
400 passed in 165.79s (0:02:45)
```

베이스라인(400 passed) 유지 — 회귀 없음.

## caveat

- 흡수(W0) 하 item-level 복원은 수요 target 부적합(double-count) — 수요 target=adjusted_demand.
- 광교 단독. 재검증(하위2)에서 W0/timing KPI 등 이 데이터로 재실행 예정.
- 이번 재생성으로 `is_stockout` 정의 자체가 바뀌었으므로(92%→60.4%), 이 컬럼에 의존하는
  다운스트림 리포트/모델(예: 매진 KPI, stockout_classifier 학습)은 재실행 전까지는
  구버전 정의 기준 수치임에 유의.
