"""보나비 20260526 xlsx → parquet 변환.

- 시트별 분리 저장
- 1행 영문 코드명을 컬럼명으로 사용 (한글 헤더는 row 0)
- 판매정보 + 판매정보2 합산
- data/internal/v2/ 디렉토리에 저장
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import pandas as pd

SRC = Path('data/internal/보나비 데이터_20260526.xlsx')
OUT = Path('data/internal/v2')
OUT.mkdir(parents=True, exist_ok=True)

# (시트명, 출력 파일명, 컬럼 정의)
SHEET_SPECS = {
    '판매정보': 'sales_p1',
    '판매정보2': 'sales_p2',
    '재고정보': 'inventory',
    '품목정보': 'items',
    '점포정보': 'stores',
    '영업시간': 'hours',
    '품절정보': 'stockout',
    '할인코드': 'discount_codes',
}


def load_sheet(sheet: str) -> pd.DataFrame:
    """Row 0 = 한글 헤더, Row 1 = 영문 코드, Row 2+ = 데이터.
    영문 코드를 컬럼명으로 사용 (한글은 \n 포함 등 dirty)."""
    df = pd.read_excel(SRC, sheet_name=sheet, header=None)
    cols = df.iloc[1].astype(str).str.strip().tolist()
    df = df.iloc[2:].reset_index(drop=True)
    df.columns = cols
    return df


def main():
    for sheet, name in SHEET_SPECS.items():
        print(f'[{sheet}] loading...')
        df = load_sheet(sheet)
        print(f'  {df.shape} cols={list(df.columns)}')
        path = OUT / f'{name}.parquet'
        df.to_parquet(path, index=False)
        print(f'  saved: {path} ({path.stat().st_size / 1024 / 1024:.1f}MB)\n')

    # 판매정보 + 판매정보2 통합
    p1 = pd.read_parquet(OUT / 'sales_p1.parquet')
    p2 = pd.read_parquet(OUT / 'sales_p2.parquet')
    sales = pd.concat([p1, p2], ignore_index=True)
    sales.to_parquet(OUT / 'sales.parquet', index=False)
    print(f'sales merged: {sales.shape}, '
          f'saved {(OUT / "sales.parquet").stat().st_size / 1024 / 1024:.1f}MB')

    # 분포 sanity check
    print('\n=== sales 매장별 분포 ===')
    print(sales['CD_PARTNER'].value_counts())
    print(f'\n=== sales 기간 ===')
    sales['DT_SALE'] = pd.to_datetime(sales['DT_SALE'].astype(str), errors='coerce')
    print(f'{sales["DT_SALE"].min()} ~ {sales["DT_SALE"].max()}')


if __name__ == '__main__':
    main()
