"""커버리지 매트릭스 → 자기포함 HTML. 크로스소스 갭/충돌을 표면화(탐지만, 교정 없음)."""
from __future__ import annotations

import pandas as pd


def render_coverage_matrix(
    cells: pd.DataFrame,
    conflicts: list[str] | None = None,
) -> str:
    pivot = cells.pivot_table(
        index=["source", "store", "field"],
        columns="month",
        values="present",
        aggfunc="first",
    )
    rows_html = []
    for idx, row in pivot.iterrows():
        tds = []
        for month, value in row.items():
            # pivot_table은 (source,store,field) x month의 완전 교차곱을 만든다 —
            # 예: display_xls는 날짜 month가 없고 date-row는 "static" month가 없어
            # 그 조합은 NaN(구조적으로 해당 없음)이 된다. NaN은 bool(NaN)==True라
            # missing 판정에서 반드시 pd.notna로 먼저 걸러야 "존재함"으로 오표시되지 않는다.
            if pd.isna(value):
                tds.append('<td class="na">·</td>')
            elif value:
                tds.append(f"<td>{month}</td>")
            else:
                tds.append('<td class="missing" style="background:#fdd">—</td>')
        rows_html.append(f"<tr><th>{' / '.join(map(str, idx))}</th>{''.join(tds)}</tr>")
    conflict_html = ""
    if conflicts:
        items = "".join(f"<li>{c}</li>" for c in conflicts)
        conflict_html = f"<h2>규명된 크로스소스 불일치(탐지)</h2><ul>{items}</ul>"
    return (
        "<style>td,th{border:1px solid #ccc;padding:4px}</style>"
        f"{conflict_html}<table>{''.join(rows_html)}</table>"
    )
