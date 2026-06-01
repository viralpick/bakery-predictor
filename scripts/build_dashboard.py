"""광교/광화문/메세나/삼성타운 4매장 종합 EDA dashboard (HTML).

탭 구조:
- Total: 4매장 비교 차트
- 광교 / 광화문 / 메세나폴리스 / 삼성타운: 매장별 12개 차트

데이터: data/internal/v2/*.parquet + external (weather, calendar)
출력: reports/dashboard.html
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots

from bakery.data.bonavi_loader import map_category

V2 = Path('data/internal/v2')
EXT = Path('data/external')
OUT = Path('reports/dashboard.html')

STORE_MAP = {
    '1000000047': '광교',
    '1000000009': '삼성타운',
    '1000000029': '메세나폴리스',
    '1000000485': '광화문',
}
STORE_ORDER = ['광교', '광화문', '메세나폴리스', '삼성타운']
STORE_WEATHER = {  # 매장 → 기상관측소
    '광교': 119,        # 수원
    '광화문': 108,      # 서울
    '메세나폴리스': 108,
    '삼성타운': 108,
}
STORE_COLORS = {
    '광교': '#1f77b4',
    '광화문': '#ff7f0e',
    '메세나폴리스': '#2ca02c',
    '삼성타운': '#d62728',
}

CLOSING_CODES = {'0069', '0077'}
DN = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}
DOW_ORDER = ['월', '화', '수', '목', '금', '토', '일']

# 특일 정의
EVENTS = {
    'xmas':     [(y, 12, 25) for y in range(2021, 2026)],
    'valentine': [(y, 2, 14) for y in range(2021, 2026)],
    'white_day': [(y, 3, 14) for y in range(2021, 2026)],
    'children': [(y, 5, 5) for y in range(2021, 2026)],
}
LUNAR_EVENTS = {
    'chuseok': ['2021-09-21', '2022-09-10', '2023-09-29', '2024-09-17', '2025-10-06'],
    'seollal': ['2021-02-12', '2022-02-01', '2023-01-22', '2024-02-10', '2025-01-29'],
}


# =============================================================================
# Data prep
# =============================================================================

def prep_data() -> dict:
    print('[prep] loading sales/items/inventory/stockout/hours/weather/calendar...')
    items = pd.read_parquet(V2 / 'items.parquet')
    items['item_id'] = items['CD_ITEM'].astype(str)
    items['NM_ITEM'] = items['NM_ITEM'].astype(str)
    items['category'] = items['NM_ITEM'].apply(map_category)
    items['UM_SO'] = pd.to_numeric(items['UM_SO'], errors='coerce').fillna(4000)

    sales = pd.read_parquet(V2 / 'sales.parquet')
    sales = sales[sales['CD_PARTNER'].astype(str).isin(STORE_MAP.keys())]
    sales = sales[sales['SALES_FG'].astype(str) == '0']
    sales = sales[sales['CD_USERDEF2'].astype(str) == 'SS']
    sales['date'] = pd.to_datetime(sales['DT_SALE'].astype(str))
    sales['QT_SALE'] = pd.to_numeric(sales['QT_SALE'], errors='coerce').fillna(0)
    sales['AM_PAYMENT'] = pd.to_numeric(sales['AM_PAYMENT'], errors='coerce').fillna(0)
    sales['AM_DC'] = pd.to_numeric(sales['AM_DC'], errors='coerce').fillna(0)
    sales['store'] = sales['CD_PARTNER'].astype(str).map(STORE_MAP)
    sales['CD_USERDEF1'] = sales['CD_USERDEF1'].astype(str)
    sales['is_closing'] = sales['CD_USERDEF1'].isin(CLOSING_CODES)
    sales['CD_ITEM'] = sales['CD_ITEM'].astype(str)
    sales = sales.merge(items[['item_id', 'category']], left_on='CD_ITEM', right_on='item_id', how='left')
    sales['category'] = sales['category'].fillna('etc')
    # 시각
    sales['hhmm'] = sales['SALES_TIME'].astype(str).str.slice(8, 12)
    sales['hour'] = pd.to_numeric(sales['hhmm'].str.slice(0, 2), errors='coerce').astype('Int64')
    sales['dow'] = sales['date'].dt.dayofweek
    sales['month'] = sales['date'].dt.month
    sales['year'] = sales['date'].dt.year
    print(f'  sales prepared: {len(sales):,}')

    # daily aggregate per store
    print('[prep] daily aggregate...')
    daily = sales.groupby(['store', 'date']).agg(
        qty=('QT_SALE', 'sum'),
        rev=('AM_PAYMENT', 'sum'),
        closing_qty=('QT_SALE', lambda x: x[sales.loc[x.index, 'is_closing']].sum()),
        n_items=('CD_ITEM', 'nunique'),
    ).reset_index()
    daily['closing_pct'] = daily['closing_qty'] / daily['qty'].replace(0, np.nan) * 100
    daily['dow'] = daily['date'].dt.dayofweek
    daily['month'] = daily['date'].dt.month
    daily['year'] = daily['date'].dt.year
    daily['dow_name'] = daily['dow'].map(DN)

    # category daily
    daily_cat = sales.groupby(['store', 'date', 'category'])['QT_SALE'].sum().reset_index()
    daily_cat = daily_cat.rename(columns={'QT_SALE': 'qty'})

    # hourly aggregate per store (×dow). Int64 NA → drop, then int cast.
    sales_h = sales.dropna(subset=['hour']).copy()
    sales_h['hour'] = sales_h['hour'].astype(int)
    hourly = sales_h.groupby(['store', 'dow', 'hour'])['QT_SALE'].sum().reset_index()
    hourly = hourly.rename(columns={'QT_SALE': 'qty'})
    hourly['dow_name'] = hourly['dow'].map(DN)

    # inventory (생산/폐기)
    inv = pd.read_parquet(V2 / 'inventory.parquet')
    inv['date'] = pd.to_datetime(inv['DT_SALE'].astype(str))
    inv['cd'] = inv['CD_PARTNER'].astype(str)
    inv['made'] = pd.to_numeric(inv['QT_MADE'], errors='coerce').fillna(0)
    inv['out'] = pd.to_numeric(inv['QT_OUT'], errors='coerce').fillna(0)
    inv['store'] = inv['cd'].map(STORE_MAP)
    inv['item_id'] = inv['CD_ITEM'].astype(str)
    inv = inv.merge(items[['item_id', 'NM_ITEM', 'category', 'UM_SO']], on='item_id', how='left')
    inv['waste_cost'] = inv['out'] * inv['UM_SO'].fillna(4000)
    inv = inv.dropna(subset=['store'])

    inv_daily = inv.groupby(['store', 'date']).agg(
        made=('made', 'sum'),
        out=('out', 'sum'),
        waste_cost=('waste_cost', 'sum'),
    ).reset_index()
    inv_daily['waste_rate'] = inv_daily['out'] / inv_daily['made'].replace(0, np.nan)
    inv_daily['month'] = inv_daily['date'].dt.month
    inv_daily['year'] = inv_daily['date'].dt.year

    # weather
    weather = pd.read_parquet(EXT / 'weather_observed.parquet')
    weather['date'] = pd.to_datetime(weather['date'])
    weather['sumRn'] = pd.to_numeric(weather['sumRn'], errors='coerce').fillna(0)
    weather['avgTa'] = pd.to_numeric(weather['avgTa'], errors='coerce')
    weather['maxTa'] = pd.to_numeric(weather['maxTa'], errors='coerce')
    weather['minTa'] = pd.to_numeric(weather['minTa'], errors='coerce')
    weather['avgRhm'] = pd.to_numeric(weather['avgRhm'], errors='coerce')

    # calendar
    cal = pd.read_parquet(EXT / 'calendar_raw.parquet')
    cal['date'] = pd.to_datetime(cal['date'])
    holiday_set = set(cal.loc[cal['is_holiday'] == True, 'date'])

    # daily + weather merge per store
    daily_full = []
    for st in STORE_ORDER:
        stn = STORE_WEATHER[st]
        d = daily[daily['store'] == st].copy()
        w = weather[weather['station_id'] == stn][['date', 'avgTa', 'maxTa', 'minTa', 'sumRn', 'avgRhm']]
        d = d.merge(w, on='date', how='left')
        d['is_holiday'] = d['date'].isin(holiday_set).astype(int)
        d['is_weekend'] = (d['dow'] >= 5).astype(int)
        daily_full.append(d)
    daily_full = pd.concat(daily_full, ignore_index=True)

    # top items per store
    top_items = sales.groupby(['store', 'CD_ITEM']).agg(
        qty=('QT_SALE', 'sum'),
        rev=('AM_PAYMENT', 'sum'),
    ).reset_index()
    top_items = top_items.merge(items[['item_id', 'NM_ITEM', 'category']],
                                left_on='CD_ITEM', right_on='item_id', how='left')

    # 신규 SKU 도입 시점 (per store)
    first_sale = sales.groupby(['store', 'CD_ITEM'])['date'].min().reset_index()
    first_sale = first_sale.rename(columns={'date': 'first_date'})
    # 매장당 도입일 분포
    item_intro = first_sale.merge(items[['item_id', 'NM_ITEM', 'category']],
                                  left_on='CD_ITEM', right_on='item_id', how='left')
    item_intro['intro_month'] = item_intro['first_date'].dt.to_period('M').dt.to_timestamp()

    # normal vs bulk daily (bulk_detector.py 산출)
    try:
        nb = pd.read_parquet(V2 / 'daily_normal_vs_bulk.parquet')
        nb['date'] = pd.to_datetime(nb['date'])
        daily_full = daily_full.merge(nb[['store', 'date', 'normal_qty', 'bulk_qty', 'bulk_pct']],
                                       on=['store', 'date'], how='left')
        daily_full[['normal_qty', 'bulk_qty', 'bulk_pct']] = daily_full[['normal_qty', 'bulk_qty', 'bulk_pct']].fillna(
            {'normal_qty': daily_full['qty'], 'bulk_qty': 0, 'bulk_pct': 0})
        print(f'  daily + normal/bulk merged')
    except FileNotFoundError:
        print('  daily_normal_vs_bulk.parquet 없음 - bulk_detector.py 먼저 실행')

    print('[prep] done')
    return {
        'sales': sales,
        'daily': daily_full,
        'daily_cat': daily_cat,
        'hourly': hourly,
        'inv_daily': inv_daily,
        'inv': inv,
        'weather': weather,
        'holiday_set': holiday_set,
        'top_items': top_items,
        'item_intro': item_intro,
        'items': items,
    }


# =============================================================================
# Chart helpers
# =============================================================================

_PLOTLY_INCLUDED = [False]


def fig_to_div(fig: go.Figure, div_id: str, height: int = 450) -> str:
    """Plotly fig → HTML div. First chart embeds plotly.js via cdn; rest skip."""
    fig.update_layout(margin=dict(l=50, r=20, t=50, b=50), height=height,
                      autosize=True, hovermode='x unified')
    if not _PLOTLY_INCLUDED[0]:
        _PLOTLY_INCLUDED[0] = True
        return pio.to_html(fig, include_plotlyjs='cdn', div_id=div_id, full_html=False)
    return pio.to_html(fig, include_plotlyjs=False, div_id=div_id, full_html=False)


# =============================================================================
# Store charts (12 per store)
# =============================================================================

def make_store_charts(store: str, data: dict) -> list[tuple[str, str, str]]:
    """Return list of (title, description, html_div) for given store."""
    daily = data['daily'][data['daily']['store'] == store]
    daily_cat = data['daily_cat'][data['daily_cat']['store'] == store]
    hourly = data['hourly'][data['hourly']['store'] == store]
    inv_daily = data['inv_daily'][data['inv_daily']['store'] == store]
    top_items = data['top_items'][data['top_items']['store'] == store]
    inv = data['inv'][data['inv']['store'] == store]
    item_intro = data['item_intro'][data['item_intro']['store'] == store]
    sales = data['sales'][data['sales']['store'] == store]

    charts = []
    color = STORE_COLORS[store]

    # 1. 일별 판매량 시계열 + closing% overlay
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['qty'].rolling(7).mean(),
                              name='일판매량 (7d MA)', line=dict(color=color)),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=daily['date'], y=daily['closing_pct'].rolling(7).mean(),
                              name='closing %', line=dict(color='gray', dash='dot'), opacity=0.5),
                  secondary_y=True)
    fig.update_layout(title=f'{store} 일별 판매량 (7d 이동평균) + closing % 추이')
    fig.update_yaxes(title_text='판매량', secondary_y=False)
    fig.update_yaxes(title_text='closing %', secondary_y=True)
    charts.append(('일별 판매량 시계열',
                   '일별 판매량(7일 이동평균)에 closing% (마감 할인 비중)를 overlay. 추세 + 마감세일 패턴 동시 확인.',
                   fig_to_div(fig, f'{store}_chart1')))

    # 2. 월×요일 heatmap (raw avg qty)
    pivot = daily.groupby(['month', 'dow'])['qty'].mean().unstack('dow')
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [DN[d] for d in pivot.columns]
    fig = px.imshow(pivot.values, labels=dict(x='요일', y='월', color='평균 판매량'),
                    x=pivot.columns, y=[f'{m}월' for m in pivot.index],
                    color_continuous_scale='YlOrRd', aspect='auto',
                    text_auto='.0f')
    fig.update_layout(title=f'{store} 월×요일 평균 판매량 heatmap')
    charts.append(('월×요일 평균 판매량',
                   '12 × 7 cells. 색이 진할수록 판매량 큼. 매장별 dow/월 패턴 동시 확인.',
                   fig_to_div(fig, f'{store}_chart2')))

    # 3. 월×요일 heatmap (closing %)
    pivot = daily.groupby(['month', 'dow'])['closing_pct'].mean().unstack('dow')
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [DN[d] for d in pivot.columns]
    fig = px.imshow(pivot.values, labels=dict(x='요일', y='월', color='closing %'),
                    x=pivot.columns, y=[f'{m}월' for m in pivot.index],
                    color_continuous_scale='Blues', aspect='auto',
                    text_auto='.1f')
    fig.update_layout(title=f'{store} 월×요일 closing % heatmap')
    charts.append(('월×요일 closing % (마감 비중)',
                   '마감 할인 사용 비중. 높을수록 잉여 발주 → 마감세일 ↑. 9월(추석) / 8월 패턴 주목.',
                   fig_to_div(fig, f'{store}_chart3')))

    # 4. 시각×요일 heatmap (시간별 매출)
    hp = hourly.dropna(subset=['hour']).copy()
    pivot = hp.pivot_table(index='dow_name', columns='hour', values='qty', aggfunc='sum')
    pivot = pivot.reindex(DOW_ORDER)
    fig = px.imshow(pivot.values, labels=dict(x='시각', y='요일', color='5년 누적 판매량'),
                    x=[f'{int(h)}시' for h in pivot.columns], y=pivot.index,
                    color_continuous_scale='Viridis', aspect='auto')
    fig.update_layout(title=f'{store} 시각×요일 누적 판매량 heatmap')
    charts.append(('시각×요일 판매량',
                   '시간대 trade-off + 요일 패턴. 점심러시/오후/마감 등 hour-of-day pattern 확인.',
                   fig_to_div(fig, f'{store}_chart4')))

    # 5. 카테고리 매출 비중 월별 stacked area
    cat_m = daily_cat.copy()
    cat_m['ym'] = cat_m['date'].dt.to_period('M').dt.to_timestamp()
    cat_pivot = cat_m.groupby(['ym', 'category'])['qty'].sum().unstack('category').fillna(0)
    # bread/pastry/sandwich/cake/etc 순으로
    cat_order = [c for c in ['bread', 'pastry', 'sandwich', 'cake', 'etc'] if c in cat_pivot.columns]
    cat_pivot = cat_pivot[cat_order]
    cat_pct = cat_pivot.div(cat_pivot.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    for c in cat_order:
        fig.add_trace(go.Scatter(x=cat_pct.index, y=cat_pct[c], name=c, stackgroup='one',
                                 hovertemplate=f'{c}: %{{y:.1f}}%<extra></extra>'))
    fig.update_layout(title=f'{store} 카테고리 매출 비중 (월별)', yaxis_title='비중 %', yaxis_range=[0, 100])
    charts.append(('카테고리 매출 비중 시계열',
                   '월별 카테고리 비중. bread 비중 매년 증가 확인 (brand-wide trend).',
                   fig_to_div(fig, f'{store}_chart5')))

    # 6. 온도 vs 일판매량 scatter
    d = daily.dropna(subset=['avgTa']).copy()
    fig = go.Figure()
    for dow in range(7):
        sub = d[d['dow'] == dow]
        fig.add_trace(go.Scatter(x=sub['avgTa'], y=sub['qty'], mode='markers',
                                 name=DN[dow], opacity=0.4, marker=dict(size=5)))
    # smoothed trend line
    d_sorted = d.sort_values('avgTa')
    smoothed = d_sorted['qty'].rolling(50, center=True).mean()
    fig.add_trace(go.Scatter(x=d_sorted['avgTa'], y=smoothed, mode='lines',
                             name='전체 추세', line=dict(color='black', width=3)))
    fig.update_layout(title=f'{store} 평균기온 vs 일판매량 (요일별)',
                      xaxis_title='평균기온 (°C)', yaxis_title='일판매량')
    charts.append(('기온 vs 판매량',
                   '온도-판매 상관. 요일별 색 분리. 추세선으로 전체 패턴.',
                   fig_to_div(fig, f'{store}_chart6')))

    # 7. 강수량 vs 판매량 (binned bar)
    d = daily.dropna(subset=['sumRn']).copy()
    d['rain_bin'] = pd.cut(d['sumRn'], bins=[-0.1, 0, 5, 20, 50, 999],
                            labels=['비없음', '0-5mm', '5-20mm', '20-50mm', '50+mm'])
    by_rain = d.groupby(['rain_bin', 'dow'], observed=True).agg(
        avg_qty=('qty', 'mean'),
        n=('qty', 'size'),
    ).reset_index()
    by_rain['dow_name'] = by_rain['dow'].map(DN)
    rain_order = ['비없음', '0-5mm', '5-20mm', '20-50mm', '50+mm']
    fig = go.Figure()
    dow_colors = px.colors.qualitative.Set2
    for i, dn_name in enumerate(DOW_ORDER):
        sub = by_rain[by_rain['dow_name'] == dn_name]
        sub_sorted = sub.set_index('rain_bin').reindex(rain_order)['avg_qty']
        fig.add_trace(go.Bar(x=rain_order, y=sub_sorted.values, name=dn_name,
                              marker_color=dow_colors[i % len(dow_colors)]))
    fig.update_layout(title=f'{store} 강수량 × 요일별 평균 판매량',
                      barmode='group',
                      yaxis_title='평균 판매량', xaxis_title='강수량 범위')
    charts.append(('강수량 영향',
                   '강수량 binning. 비 오는 날 판매량 감소 패턴 확인.',
                   fig_to_div(fig, f'{store}_chart7')))

    # 8. 특일 효과 — D-7 ~ D+7 평균 매출 (전 매출 대비)
    d = daily.copy().set_index('date')
    overall_mean = d['qty'].mean()
    event_rows = []
    for ev_name, dates in EVENTS.items():
        ev_dates = [pd.Timestamp(*t) for t in dates]
        for delta in range(-7, 8):
            qtys = []
            for ed in ev_dates:
                td = ed + pd.Timedelta(days=delta)
                if td in d.index:
                    qtys.append(d.loc[td, 'qty'])
            if qtys:
                event_rows.append({'event': ev_name, 'delta': delta, 'avg_qty': np.mean(qtys),
                                   'pct_vs_normal': (np.mean(qtys) / overall_mean - 1) * 100})
    for ev_name, dstrs in LUNAR_EVENTS.items():
        ev_dates = [pd.Timestamp(s) for s in dstrs]
        for delta in range(-7, 8):
            qtys = []
            for ed in ev_dates:
                td = ed + pd.Timedelta(days=delta)
                if td in d.index:
                    qtys.append(d.loc[td, 'qty'])
            if qtys:
                event_rows.append({'event': ev_name, 'delta': delta, 'avg_qty': np.mean(qtys),
                                   'pct_vs_normal': (np.mean(qtys) / overall_mean - 1) * 100})
    ev_df = pd.DataFrame(event_rows)
    if len(ev_df):
        fig = go.Figure()
        ev_colors = px.colors.qualitative.Set2
        for i, ev in enumerate(ev_df['event'].unique()):
            sub = ev_df[ev_df['event'] == ev].sort_values('delta')
            fig.add_trace(go.Scatter(x=sub['delta'], y=sub['pct_vs_normal'], name=ev,
                                     mode='lines+markers',
                                     line=dict(color=ev_colors[i % len(ev_colors)], width=2)))
        fig.add_hline(y=0, line_dash='dot', line_color='gray')
        fig.update_layout(title=f'{store} 특일 ±7일 매출 효과 (평시 대비 %)',
                          xaxis_title='D±n일', yaxis_title='평시 대비 %',
                          xaxis=dict(tickmode='linear', tick0=-7, dtick=1))
        charts.append(('특일 ±7일 효과',
                       '발렌타인/화이트/어린이날/크리스마스 + 추석/설날 전후 7일 매출 변화. 평시 평균 대비 %.',
                       fig_to_div(fig, f'{store}_chart8')))

    # 9. 폐기율 시계열 (월별)
    if len(inv_daily):
        inv_m = inv_daily.copy()
        inv_m['ym'] = inv_m['date'].dt.to_period('M').dt.to_timestamp()
        m_agg = inv_m.groupby('ym').agg(made=('made', 'sum'), out=('out', 'sum')).reset_index()
        m_agg['waste_rate'] = m_agg['out'] / m_agg['made'] * 100
        m_agg = m_agg.sort_values('ym')
        fig = go.Figure(go.Scatter(x=m_agg['ym'], y=m_agg['waste_rate'], mode='lines+markers',
                                    line=dict(color=color, width=2),
                                    marker=dict(size=6)))
        fig.update_layout(title=f'{store} 월별 폐기율 추이',
                          xaxis_title='월', yaxis_title='폐기율 %')
        charts.append(('월별 폐기율',
                       '폐기량 / 생산량. 2025년 급증 확인 (brand-wide signal).',
                       fig_to_div(fig, f'{store}_chart9')))

    # 10. 마감회수율 시계열 (월별)
    if len(inv_daily):
        # closing_qty (sales daily에 있음) + out (inv_daily) merge
        merge_daily = daily.merge(inv_daily[['date', 'out']], on='date', how='left')
        merge_daily['ym'] = merge_daily['date'].dt.to_period('M').dt.to_timestamp()
        m_agg = merge_daily.groupby('ym').agg(closing_qty=('closing_qty', 'sum'), out=('out', 'sum')).reset_index()
        m_agg['recovery'] = m_agg['closing_qty'] / (m_agg['closing_qty'] + m_agg['out']) * 100
        m_agg = m_agg.sort_values('ym')
        fig = go.Figure(go.Scatter(x=m_agg['ym'], y=m_agg['recovery'], mode='lines+markers',
                                    line=dict(color=color, width=2),
                                    marker=dict(size=6)))
        fig.update_layout(title=f'{store} 월별 마감 회수율 추이',
                          xaxis_title='월', yaxis_title='마감 회수율 %')
        charts.append(('월별 마감 회수율',
                       ('<b>정의</b>: 마감 회수율 = 마감 할인 판매량 / (마감 할인 판매량 + 폐기량) × 100<br>'
                        '<b>의미</b>: 운영 종료 시점에 남았던 잉여 빵 중 마감 할인(closing 코드 0077/0069)으로 판매 회수한 비율. '
                        '나머지는 그대로 폐기.<br>'
                        '<b>해석</b>: 높을수록 잉여를 마감세일로 잘 회수 (예: 광교 48% = 폐기될 뻔한 빵의 절반 회수). '
                        '낮으면 마감 시간대 손님이 적거나 마감 운영 효율 낮음 (예: 삼성타운 25% — 오피스가 = 19시 이후 손님 적음).<br>'
                        '<b>주의</b>: 회수율 높다고 무조건 좋은 것 X — 정상가 매출이 마감세일에 잠식되면 손익은 별개. '
                        '본질적으로는 "잉여 발주를 줄이는 것"이 1순위.'),
                       fig_to_div(fig, f'{store}_chart10')))

    # 11. Top 10 품목 매출 (go.Bar)
    top10 = top_items.sort_values('qty', ascending=False).head(10).sort_values('qty')
    cat_colors = {'bread': '#e74c3c', 'pastry': '#3498db', 'sandwich': '#2ecc71',
                  'cake': '#f39c12', 'etc': '#95a5a6'}
    fig = go.Figure(go.Bar(
        x=top10['qty'], y=top10['NM_ITEM'], orientation='h',
        marker_color=[cat_colors.get(c, '#666') for c in top10['category']],
        text=top10['category'], textposition='inside',
    ))
    fig.update_layout(title=f'{store} top 10 판매 품목 (5년 누적)',
                      xaxis_title='5년 누적 판매량', yaxis_title='품목')
    charts.append(('Top 10 품목',
                   '5년 누적 판매량 상위 10. 카테고리 색으로 분류.',
                   fig_to_div(fig, f'{store}_chart11')))

    # 12. 요일별 매출 분포 box plot
    fig = go.Figure()
    for dow in range(7):
        sub = daily[daily['dow'] == dow]
        fig.add_trace(go.Box(y=sub['qty'], name=DN[dow], marker_color=color))
    fig.update_layout(title=f'{store} 요일별 판매량 분포', yaxis_title='일판매량', showlegend=False)
    charts.append(('요일별 판매량 분포',
                   'Box plot. 요일별 median + 25/75/이상치 분포 확인.',
                   fig_to_div(fig, f'{store}_chart12')))

    # 13. 연도별 매출 trend
    yr = daily.groupby(['year', 'month'])['qty'].mean().reset_index()
    fig = go.Figure()
    for y in sorted(yr['year'].unique()):
        sub = yr[yr['year'] == y]
        fig.add_trace(go.Scatter(x=sub['month'], y=sub['qty'], name=f'{y}', mode='lines+markers'))
    fig.update_layout(title=f'{store} 연도별 월평균 판매량 비교',
                      xaxis_title='월', yaxis_title='일평균 판매량',
                      xaxis=dict(tickmode='linear', tick0=1, dtick=1))
    charts.append(('연도별 월평균 비교',
                   '연 단위 매출 trend 비교. 광교 2022 정점 + 2025 patterns.',
                   fig_to_div(fig, f'{store}_chart13')))

    # 14. 신규 SKU 도입 시점 (월별 build-up)
    intro = item_intro.groupby('intro_month').size().reset_index(name='n_new')
    intro['cumulative'] = intro['n_new'].cumsum()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=intro['intro_month'], y=intro['n_new'], name='월 신규 SKU 수',
                          marker_color=color, opacity=0.5),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=intro['intro_month'], y=intro['cumulative'], name='누적 SKU 수',
                              line=dict(color='black', width=2)),
                  secondary_y=True)
    fig.update_layout(title=f'{store} 신규 SKU 도입 추이')
    fig.update_yaxes(title_text='월 신규 SKU 수', secondary_y=False)
    fig.update_yaxes(title_text='누적 SKU 수', secondary_y=True)
    charts.append(('신규 SKU 도입',
                   '월별 신규 도입 + 누적. SKU 확장 시점 파악.',
                   fig_to_div(fig, f'{store}_chart14')))

    # ========================================================================
    # 추가 차트 15~22 (8개 신규)
    # ========================================================================

    # 15. 카테고리 × 시간대 heatmap
    sales_h = sales.dropna(subset=['hour']).copy()
    sales_h['hour'] = sales_h['hour'].astype(int)
    cat_hr = sales_h.groupby(['category', 'hour'])['QT_SALE'].sum().unstack('hour').fillna(0)
    cat_order_present = [c for c in ['bread', 'pastry', 'sandwich', 'cake', 'etc'] if c in cat_hr.index]
    cat_hr = cat_hr.reindex(cat_order_present)
    # row 정규화 (카테고리 내 시간 분포 %)
    cat_hr_pct = cat_hr.div(cat_hr.sum(axis=1), axis=0) * 100
    fig = px.imshow(cat_hr_pct.values,
                    labels=dict(x='시각', y='카테고리', color='카테고리 내 시간 분포 %'),
                    x=[f'{int(h)}시' for h in cat_hr_pct.columns], y=cat_hr_pct.index,
                    color_continuous_scale='Viridis', aspect='auto',
                    text_auto='.1f')
    fig.update_layout(title=f'{store} 카테고리 × 시간대 분포 (행 정규화 %)')
    charts.append(('카테고리 × 시간대',
                   '각 카테고리가 하루 어느 시간대에 팔리는지. sandwich 점심, cake 오후, bread 아침 패턴 확인.',
                   fig_to_div(fig, f'{store}_chart15')))

    # 16. 카테고리별 매진 시각 box plot
    so = pd.read_parquet(V2 / 'stockout.parquet')
    so = so[so['CD_PARTNER'].astype(str) == [k for k, v in STORE_MAP.items() if v == store][0]].copy()
    so['item_id'] = so['CD_ITEM'].astype(str)
    so['SOLD_TIME'] = pd.to_numeric(so['SOLD_TIME'], errors='coerce')
    so['hour'] = so['SOLD_TIME'] // 100
    so = so.dropna(subset=['hour'])
    so['hour'] = so['hour'].astype(int)
    so = so.merge(data['items'][['item_id', 'category']], on='item_id', how='left')
    so['category'] = so['category'].fillna('etc')
    fig = go.Figure()
    cat_colors = {'bread': '#e74c3c', 'pastry': '#3498db', 'sandwich': '#2ecc71',
                  'cake': '#f39c12', 'etc': '#95a5a6'}
    for cat in [c for c in ['bread', 'pastry', 'sandwich', 'cake'] if c in so['category'].unique()]:
        sub = so[so['category'] == cat]
        fig.add_trace(go.Box(y=sub['hour'], name=cat, marker_color=cat_colors.get(cat, '#666')))
    fig.update_layout(title=f'{store} 카테고리별 매진 시각 분포',
                      yaxis_title='매진 시각 (시)', showlegend=False)
    charts.append(('카테고리별 매진 시각',
                   '어느 카테고리가 일찍 매진? 발주 수량 우선순위 단서.',
                   fig_to_div(fig, f'{store}_chart16')))

    # 17. 휴일/주말/평일 매출 분포 box plot
    d = daily.copy()
    def day_type(r):
        if r['is_holiday'] == 1:
            return '공휴일'
        if r['is_weekend'] == 1:
            return '주말'
        return '평일'
    d['day_type'] = d.apply(day_type, axis=1)
    fig = go.Figure()
    type_colors = {'평일': '#3498db', '주말': '#e74c3c', '공휴일': '#f39c12'}
    for t in ['평일', '주말', '공휴일']:
        sub = d[d['day_type'] == t]
        fig.add_trace(go.Box(y=sub['qty'], name=f'{t} (n={len(sub)})',
                              marker_color=type_colors[t]))
    fig.update_layout(title=f'{store} 휴일/주말/평일 일판매량 분포',
                      yaxis_title='일판매량', showlegend=False)
    charts.append(('휴일/주말/평일 분포',
                   '평일 vs 주말 vs 공휴일 매출 분포 비교. 공휴일이 주말과 비슷한지/다른지 확인.',
                   fig_to_div(fig, f'{store}_chart17')))

    # 18. 매출 이상치 (z-score > 2) day highlight
    d = daily.copy().sort_values('date')
    d['z'] = (d['qty'] - d['qty'].mean()) / d['qty'].std()
    outliers = d[d['z'].abs() > 2]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d['date'], y=d['qty'], mode='lines', name='일판매량',
                              line=dict(color=color, width=1.2), opacity=0.5))
    fig.add_trace(go.Scatter(x=outliers['date'], y=outliers['qty'], mode='markers',
                              name=f'outlier (|z|>2, n={len(outliers)})',
                              marker=dict(size=8, color='red')))
    fig.update_layout(title=f'{store} 매출 이상치 일자 (z-score > 2)',
                      xaxis_title='날짜', yaxis_title='일판매량')
    charts.append(('매출 이상치 일자',
                   '평균 ±2σ 벗어난 일자. 특별한 이벤트/promo/공휴일/이상 패턴 inspection.',
                   fig_to_div(fig, f'{store}_chart18')))

    # 19. YoY 매출 변화 heatmap (월 × 연도)
    y_avg = daily.groupby(['year', 'month'])['qty'].mean().unstack('year')
    # YoY % change per month
    yoy = y_avg.pct_change(axis=1) * 100
    yoy = yoy.dropna(axis=1, how='all')
    if yoy.shape[1] >= 1:
        fig = px.imshow(yoy.values,
                        labels=dict(x='연도', y='월', color='YoY 변화 %'),
                        x=yoy.columns.astype(str), y=[f'{m}월' for m in yoy.index],
                        color_continuous_scale='RdBu_r', color_continuous_midpoint=0,
                        aspect='auto', text_auto='.1f')
        fig.update_layout(title=f'{store} 월별 YoY 매출 변화 (%)')
        charts.append(('YoY 매출 변화',
                       '월별 전년 대비 매출 변화 %. 빨강=증가, 파랑=감소. 어느 월이 회복/저성장인지.',
                       fig_to_div(fig, f'{store}_chart19')))

    # 20. 연휴 길이 × 매출 box plot
    # 연속 휴일/주말 묶기
    d = daily.sort_values('date').copy()
    d['off'] = ((d['is_holiday'] == 1) | (d['is_weekend'] == 1)).astype(int)
    # 연속된 off block 길이 계산
    d['block'] = (d['off'] != d['off'].shift()).cumsum()
    block_len = d[d['off'] == 1].groupby('block')['date'].count()
    d = d.merge(block_len.rename('block_len'), left_on='block', right_index=True, how='left')
    d['block_len'] = d['block_len'].fillna(0).astype(int)

    def bucket_len(n):
        if n == 0: return '평일'
        if n == 1: return '1일 off'
        if n == 2: return '2일 off (주말)'
        if n == 3: return '3일 연휴'
        return '4일+ 연휴'
    d['len_bucket'] = d['block_len'].apply(bucket_len)
    order = ['평일', '1일 off', '2일 off (주말)', '3일 연휴', '4일+ 연휴']
    fig = go.Figure()
    bucket_colors = ['#7f8c8d', '#1abc9c', '#3498db', '#e67e22', '#e74c3c']
    for i, b in enumerate(order):
        sub = d[d['len_bucket'] == b]
        if len(sub) == 0: continue
        fig.add_trace(go.Box(y=sub['qty'], name=f'{b} (n={len(sub)})',
                              marker_color=bucket_colors[i]))
    fig.update_layout(title=f'{store} 연휴 길이별 매출 분포',
                      yaxis_title='일판매량', showlegend=False)
    charts.append(('연휴 길이별 매출',
                   '연속된 휴일 블록의 길이 (평일 vs 1일 off vs 2일 주말 vs 3일+ 연휴) 별 매출.',
                   fig_to_div(fig, f'{store}_chart20')))

    # 21. 기온 binning + 요일 평균 매출
    d = daily.dropna(subset=['avgTa']).copy()
    d['temp_bin'] = pd.cut(d['avgTa'], bins=[-30, 0, 10, 20, 30, 50],
                            labels=['<0°C', '0-10°C', '10-20°C', '20-30°C', '30+°C'])
    by_temp = d.groupby(['temp_bin', 'dow'], observed=True)['qty'].mean().reset_index()
    by_temp['dow_name'] = by_temp['dow'].map(DN)
    fig = go.Figure()
    dow_colors = px.colors.qualitative.Set2
    temp_order = ['<0°C', '0-10°C', '10-20°C', '20-30°C', '30+°C']
    for i, dn_name in enumerate(DOW_ORDER):
        sub = by_temp[by_temp['dow_name'] == dn_name]
        sub_sorted = sub.set_index('temp_bin').reindex(temp_order)['qty']
        fig.add_trace(go.Bar(x=temp_order, y=sub_sorted.values, name=dn_name,
                              marker_color=dow_colors[i % len(dow_colors)]))
    fig.update_layout(title=f'{store} 기온 × 요일별 평균 판매량',
                      barmode='group', yaxis_title='평균 판매량', xaxis_title='평균기온 구간')
    charts.append(('기온 구간 × 요일',
                   '5단계 기온 binning. 요일별 색. 추운 날/더운 날 패턴 명확히.',
                   fig_to_div(fig, f'{store}_chart21')))

    # 23. 일별 normal vs total + 예약 일자 highlight
    if 'normal_qty' in daily.columns:
        d = daily.copy().sort_values('date')
        d_bulk_only = d[d['bulk_qty'] > 0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d['date'], y=d['qty'].rolling(14, min_periods=1).mean(),
                                  name='전체 매출 (14d MA)', line=dict(color=color, width=1.5),
                                  opacity=0.5))
        fig.add_trace(go.Scatter(x=d['date'], y=d['normal_qty'].rolling(14, min_periods=1).mean(),
                                  name='평소 수요 (14d MA, 예약 제외)',
                                  line=dict(color='#2c3e50', width=2)))
        fig.add_trace(go.Scatter(x=d_bulk_only['date'], y=d_bulk_only['bulk_qty'], mode='markers',
                                  name=f'예약 발생일 ({len(d_bulk_only)}일)',
                                  marker=dict(size=6, color='red', opacity=0.6)))
        fig.update_layout(title=f'{store} 일별 매출 — 평소 수요 + 예약 분리',
                          xaxis_title='날짜', yaxis_title='일판매량 / 예약 수량')
        charts.append(('일별 평소 수요 vs 예약 분리',
                       '<b>회색</b>: 전체 매출 (14d MA) — 예약 포함<br>'
                       '<b>검정</b>: 평소 수요 (14d MA) — 예약 제외 → 모델 학습 target<br>'
                       '<b>빨간 점</b>: 예약 발생일 + 그 날 예약 수량 (단발 outlier 시각화)<br>'
                       '평소 수요는 예측 대상이고, 예약은 매장이 사전 알아서 처리.',
                       fig_to_div(fig, f'{store}_chart23', height=450)))

    # 22. 요일별 시간대 비중 (100% stacked)
    if len(hourly):
        h_pct = hourly.groupby(['dow', 'hour'])['qty'].sum().unstack('hour').fillna(0)
        h_pct = h_pct.div(h_pct.sum(axis=1), axis=0) * 100
        h_pct.index = [DN[d] for d in h_pct.index]
        h_pct = h_pct.reindex(DOW_ORDER)
        fig = go.Figure()
        hour_palette = px.colors.sequential.Viridis
        for i, h in enumerate(h_pct.columns):
            color_i = hour_palette[int(i / len(h_pct.columns) * (len(hour_palette) - 1))]
            fig.add_trace(go.Bar(x=h_pct.index, y=h_pct[h], name=f'{int(h)}시',
                                  marker_color=color_i))
        fig.update_layout(title=f'{store} 요일별 시간대 매출 비중 (100% stacked)',
                          barmode='stack', yaxis_title='시간대 비중 %', xaxis_title='요일',
                          yaxis_range=[0, 100])
        charts.append(('요일별 시간대 비중',
                       '요일별 일매출의 시간 분포. 주말/평일 시간 분배 차이.',
                       fig_to_div(fig, f'{store}_chart22')))

    return charts


# =============================================================================
# Total tab charts (4매장 비교)
# =============================================================================

def make_total_charts(data: dict) -> list[tuple[str, str, str]]:
    daily = data['daily']
    daily_cat = data['daily_cat']
    hourly = data['hourly']
    inv_daily = data['inv_daily']
    sales = data['sales']
    charts = []

    # T1. 4매장 일별 매출 (14d MA로 변동성 보존)
    fig = go.Figure()
    for st in STORE_ORDER:
        sub = daily[daily['store'] == st].sort_values('date').copy()
        sub['ma'] = sub['qty'].rolling(14, min_periods=1).mean()
        fig.add_trace(go.Scatter(x=sub['date'], y=sub['ma'], name=st, mode='lines',
                                 line=dict(color=STORE_COLORS[st], width=1.8)))
    fig.update_layout(title='4매장 일별 판매량 (14일 이동평균)',
                      yaxis_title='일판매량 (14d MA)', xaxis_title='날짜',
                      yaxis=dict(rangemode='tozero'))
    charts.append(('4매장 일별 판매량 (14일 MA)',
                   '매장 간 매출 절대값 + 추세 비교. 광교 2022 정점 + 2025 회복, 광화문 꾸준 성장.',
                   fig_to_div(fig, 'total_chart1', height=500)))

    # T2. 4매장 연도별 일평균 매출 (line + markers — 트렌드 강조)
    fig = go.Figure()
    for st in STORE_ORDER:
        sub = daily[daily['store'] == st].groupby('year')['qty'].mean()
        fig.add_trace(go.Scatter(x=sub.index.astype(int), y=sub.values, name=st,
                                 mode='lines+markers',
                                 line=dict(color=STORE_COLORS[st], width=2.5),
                                 marker=dict(size=12)))
    fig.update_layout(title='4매장 연도별 일평균 판매량 (트렌드)',
                      xaxis=dict(title='연도', tickmode='linear', tick0=2021, dtick=1),
                      yaxis=dict(title='일평균 판매량', rangemode='tozero'))
    charts.append(('4매장 연도별 매출',
                   '연단위 비교. 2022 광교 정점. 광화문 2022~ 만 데이터.',
                   fig_to_div(fig, 'total_chart2')))

    # T3. 4매장 요일 패턴 비교 (go.Scatter 직접 사용으로 정렬 안정성 확보)
    fig = go.Figure()
    for st in STORE_ORDER:
        sub = daily[daily['store'] == st].groupby('dow')['qty'].mean().reindex(range(7))
        fig.add_trace(go.Scatter(x=DOW_ORDER, y=sub.values, name=st, mode='lines+markers',
                                 line=dict(color=STORE_COLORS[st], width=2),
                                 marker=dict(size=10)))
    fig.update_layout(title='4매장 요일별 판매 패턴',
                      xaxis_title='요일', yaxis_title='일평균 판매량',
                      yaxis=dict(rangemode='tozero'))
    charts.append(('4매장 요일 패턴',
                   '광교 주말 강세 vs 삼성타운 토/일 절벽. 매장 특성 명확히 다름.',
                   fig_to_div(fig, 'total_chart3')))

    # T4. 4매장 시간대 패턴 비교 (go.Scatter)
    fig = go.Figure()
    for st in STORE_ORDER:
        sub = hourly[hourly['store'] == st].groupby('hour')['qty'].sum().sort_index()
        fig.add_trace(go.Scatter(x=sub.index.astype(int), y=sub.values, name=st,
                                 mode='lines+markers',
                                 line=dict(color=STORE_COLORS[st], width=2),
                                 marker=dict(size=8)))
    fig.update_layout(title='4매장 시간대별 판매 패턴',
                      xaxis=dict(tickmode='linear', tick0=0, dtick=1, title='시각'),
                      yaxis=dict(title='5년 누적 판매량', rangemode='tozero'))
    charts.append(('4매장 시간대 패턴',
                   '시간대별 매출 곡선. 삼성타운 점심 러시 / 광교 오후 / 광화문 평탄.',
                   fig_to_div(fig, 'total_chart4')))

    # T5. 4매장 카테고리 비중 비교 (stacked bar, 5년 누적)
    cat_summary = sales.groupby(['store', 'category'])['QT_SALE'].sum().reset_index()
    cat_pivot = cat_summary.pivot(index='store', columns='category', values='QT_SALE').fillna(0)
    cat_pct = cat_pivot.div(cat_pivot.sum(axis=1), axis=0) * 100
    cat_order = [c for c in ['bread', 'pastry', 'sandwich', 'cake', 'etc'] if c in cat_pct.columns]
    fig = go.Figure()
    for c in cat_order:
        fig.add_trace(go.Bar(x=cat_pct.index, y=cat_pct[c], name=c,
                              text=cat_pct[c].round(1), textposition='inside'))
    fig.update_layout(title='4매장 카테고리 매출 비중 (5년 누적)',
                      barmode='stack', yaxis_title='비중 %',
                      xaxis={'categoryorder': 'array', 'categoryarray': STORE_ORDER})
    charts.append(('4매장 카테고리 비중',
                   '5년 누적 카테고리 매출 비중. pastry 60~72% 모든 매장 core.',
                   fig_to_div(fig, 'total_chart5')))

    # T6. 4매장 월별 폐기율 비교 (go.Scatter)
    if len(inv_daily):
        inv_m = inv_daily.copy()
        inv_m['ym'] = inv_m['date'].dt.to_period('M').dt.to_timestamp()
        m_agg = inv_m.groupby(['store', 'ym']).agg(made=('made', 'sum'), out=('out', 'sum')).reset_index()
        m_agg['waste_rate'] = m_agg['out'] / m_agg['made'] * 100
        fig = go.Figure()
        for st in STORE_ORDER:
            sub = m_agg[m_agg['store'] == st].sort_values('ym')
            fig.add_trace(go.Scatter(x=sub['ym'], y=sub['waste_rate'], name=st, mode='lines',
                                     line=dict(color=STORE_COLORS[st], width=2)))
        fig.update_layout(title='4매장 월별 폐기율 추이',
                          xaxis_title='월', yaxis_title='폐기율 %')
        charts.append(('4매장 폐기율 추이',
                       '2025년 4매장 모두 폐기율 +2~5pp 급증. brand-wide signal.',
                       fig_to_div(fig, 'total_chart6')))

    # T7. 4매장 마감 회수율 비교
    merged_full = daily.merge(inv_daily[['store', 'date', 'out']], on=['store', 'date'], how='left')
    merged_full['ym'] = merged_full['date'].dt.to_period('M').dt.to_timestamp()
    m_agg = merged_full.groupby(['store', 'ym']).agg(closing_qty=('closing_qty', 'sum'), out=('out', 'sum')).reset_index()
    m_agg['recovery'] = m_agg['closing_qty'] / (m_agg['closing_qty'] + m_agg['out']).replace(0, np.nan) * 100
    fig = go.Figure()
    for st in STORE_ORDER:
        sub = m_agg[m_agg['store'] == st].sort_values('ym')
        fig.add_trace(go.Scatter(x=sub['ym'], y=sub['recovery'], name=st, mode='lines',
                                 line=dict(color=STORE_COLORS[st], width=2)))
    fig.update_layout(title='4매장 월별 마감 회수율 추이',
                      xaxis_title='월', yaxis_title='마감 회수율 %')
    charts.append(('4매장 마감 회수율',
                   ('<b>정의</b>: 마감 회수율 = 마감 할인 판매량 / (마감 할인 판매량 + 폐기량) × 100<br>'
                    '<b>의미</b>: 영업 종료 시점에 남았던 잉여 빵 중 마감 할인(closing 코드 0077/0069)으로 판매 회수한 비율. '
                    '나머지는 그대로 폐기.<br>'
                    '<b>매장 차이 해석</b>:<br>'
                    '· 광교 ~48% (높음) — 마감 시간대(20-21시)에 손님 많음. 주거+상권 매장 특성<br>'
                    '· 광화문 ~43% — 평이한 수준<br>'
                    '· 메세나 ~37% — 비교적 낮음<br>'
                    '· 삼성타운 ~25% (낮음) — 오피스가, 19시 이후 손님 급감 → 마감 판매 어려움<br>'
                    '<b>주의</b>: 회수율은 보조 지표. 본질은 "잉여 발주를 줄이는 것" (폐기율 차트 참조).'),
                   fig_to_div(fig, 'total_chart7')))

    # T8. 4매장 특일 효과 비교 (heatmap)
    rows = []
    for st in STORE_ORDER:
        d = daily[daily['store'] == st].set_index('date')
        overall = d['qty'].mean()
        all_events = {**{k: [pd.Timestamp(*t) for t in v] for k, v in EVENTS.items()},
                      **{k: [pd.Timestamp(s) for s in v] for k, v in LUNAR_EVENTS.items()}}
        for ev, dates in all_events.items():
            qtys = []
            for ed in dates:
                if ed in d.index:
                    qtys.append(d.loc[ed, 'qty'])
            if qtys:
                rows.append({'store': st, 'event': ev,
                              'pct_vs_normal': (np.mean(qtys) / overall - 1) * 100})
    ev_df = pd.DataFrame(rows)
    if len(ev_df):
        pivot = ev_df.pivot(index='event', columns='store', values='pct_vs_normal')
        pivot = pivot[STORE_ORDER]
        fig = px.imshow(pivot.values, labels=dict(x='매장', y='특일', color='평시 대비 %'),
                        x=pivot.columns, y=pivot.index,
                        color_continuous_scale='RdBu_r', aspect='auto',
                        text_auto='.1f', color_continuous_midpoint=0)
        fig.update_layout(title='4매장 특일 당일 매출 효과 (평시 대비 %)')
        charts.append(('4매장 특일 효과 비교',
                       '특일 당일 매출의 평시 대비 변화율. 빨강 = 매출 증가, 파랑 = 감소.',
                       fig_to_div(fig, 'total_chart8')))

    # T9. 4매장 closing % 시계열
    cl = daily.copy()
    cl['ym'] = cl['date'].dt.to_period('M').dt.to_timestamp()
    m_agg = cl.groupby(['store', 'ym']).agg(closing_qty=('closing_qty', 'sum'), qty=('qty', 'sum')).reset_index()
    m_agg['closing_pct'] = m_agg['closing_qty'] / m_agg['qty'] * 100
    fig = go.Figure()
    for st in STORE_ORDER:
        sub = m_agg[m_agg['store'] == st].sort_values('ym')
        fig.add_trace(go.Scatter(x=sub['ym'], y=sub['closing_pct'], name=st, mode='lines',
                                 line=dict(color=STORE_COLORS[st], width=2)))
    fig.update_layout(title='4매장 월별 closing % 추이',
                      xaxis_title='월', yaxis_title='closing %')
    charts.append(('4매장 closing % 추이',
                   '월별 closing 비중. 9월 추석 시즌 peak 매장별 차이.',
                   fig_to_div(fig, 'total_chart9')))

    # T10. 4매장 2025 월별 비교 (전체 일평균 + 평일만 별도)
    fig = make_subplots(rows=1, cols=2,
                         subplot_titles=('2025 전체 일평균 (평일+주말)', '2025 평일만 일평균'))
    for st in STORE_ORDER:
        sub_all = daily[(daily['store'] == st) & (daily['year'] == 2025)].groupby('month')['qty'].mean().sort_index()
        sub_wk = daily[(daily['store'] == st) & (daily['year'] == 2025) & (daily['is_weekend'] == 0)
                       & (daily['is_holiday'] == 0)].groupby('month')['qty'].mean().sort_index()
        fig.add_trace(go.Scatter(x=sub_all.index.astype(int), y=sub_all.values, name=st,
                                 mode='lines+markers',
                                 line=dict(color=STORE_COLORS[st], width=2),
                                 marker=dict(size=8), legendgroup=st), row=1, col=1)
        fig.add_trace(go.Scatter(x=sub_wk.index.astype(int), y=sub_wk.values, name=st,
                                 mode='lines+markers',
                                 line=dict(color=STORE_COLORS[st], width=2),
                                 marker=dict(size=8), legendgroup=st, showlegend=False), row=1, col=2)
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1, title='월', row=1, col=1)
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1, title='월', row=1, col=2)
    fig.update_yaxes(title='일평균 판매량', rangemode='tozero', row=1, col=1)
    fig.update_yaxes(title='평일 일평균 판매량', rangemode='tozero', row=1, col=2)
    fig.update_layout(title='4매장 2025년 월별 일평균 판매량 (전체 vs 평일만)')
    charts.append(('4매장 2025 월별 비교 (전체 vs 평일)',
                   ('<b>왼쪽</b>: 전체 일평균 (평일+주말+공휴일). 광교 11-12월 drop이 주말 강세로 흡수돼 덜 보임.<br>'
                    '<b>오른쪽</b>: 평일만 일평균 (주말/공휴일 제외). 광교 11-12월 평일 매출 -9.8%/-10.7% drop 명확히 보임.<br>'
                    '<b>인사이트</b>: 광교 drop은 평일 한정 — 주말은 정상 → 평일 외부 이벤트 (직장 환경/학교 방학 등) 가설 강화.'),
                   fig_to_div(fig, 'total_chart10', height=450)))

    # T11. 4매장 매장×카테고리 폐기율 heatmap
    inv = data['inv']
    inv_cat = inv.groupby(['store', 'category']).agg(made=('made', 'sum'), out=('out', 'sum')).reset_index()
    inv_cat['waste_rate'] = inv_cat['out'] / inv_cat['made'].replace(0, np.nan) * 100
    pivot = inv_cat.pivot(index='store', columns='category', values='waste_rate')
    pivot = pivot.reindex(STORE_ORDER)
    cat_order = [c for c in ['bread', 'pastry', 'sandwich', 'cake', 'etc'] if c in pivot.columns]
    pivot = pivot[cat_order]
    fig = px.imshow(pivot.values, labels=dict(x='카테고리', y='매장', color='폐기율 %'),
                    x=pivot.columns, y=pivot.index,
                    color_continuous_scale='Reds', aspect='auto',
                    text_auto='.1f')
    fig.update_layout(title='4매장 × 카테고리 폐기율 heatmap')
    charts.append(('4매장 × 카테고리 폐기율',
                   '매장 × 카테고리 폐기율 매트릭스. 삼성타운 cake 25.2% / 메세나 sandwich 25.5%.',
                   fig_to_div(fig, 'total_chart11')))

    # ========================================================================
    # 신규 Total 차트 T12~T13
    # ========================================================================

    # T12. 매장 × 카테고리 폐기율 vs 매진율 사분면 산점도
    inv = data['inv']
    inv_cat = inv.groupby(['store', 'category']).agg(made=('made', 'sum'), out=('out', 'sum')).reset_index()
    inv_cat['waste_rate'] = inv_cat['out'] / inv_cat['made'].replace(0, np.nan) * 100

    # 매진 빈도 (stockout count per item-day → category level)
    so = pd.read_parquet(V2 / 'stockout.parquet')
    so['cd'] = so['CD_PARTNER'].astype(str)
    so['store'] = so['cd'].map(STORE_MAP)
    so['item_id'] = so['CD_ITEM'].astype(str)
    so = so.merge(data['items'][['item_id', 'category']], on='item_id', how='left')
    so['category'] = so['category'].fillna('etc')
    so = so.dropna(subset=['store'])
    # 매진율 = 매진 발생 일 / 판매 일 (item-day 기준)
    sales_id = data['sales'].groupby(['store', 'category']).agg(item_days=('date', 'nunique')).reset_index()
    so_count = so.groupby(['store', 'category']).size().reset_index(name='stockout_days')
    merged = inv_cat.merge(so_count, on=['store', 'category'], how='left')
    merged['stockout_days'] = merged['stockout_days'].fillna(0)
    merged = merged.merge(sales_id, on=['store', 'category'], how='left')
    merged['stockout_rate'] = merged['stockout_days'] / merged['item_days'].replace(0, np.nan) * 100

    fig = go.Figure()
    for st in STORE_ORDER:
        sub = merged[merged['store'] == st]
        fig.add_trace(go.Scatter(x=sub['waste_rate'], y=sub['stockout_rate'],
                                 mode='markers+text',
                                 marker=dict(size=18, color=STORE_COLORS[st]),
                                 text=sub['category'], textposition='top center',
                                 name=st))
    # 사분면 가이드 라인 (평균값)
    wr_mean = merged['waste_rate'].mean()
    sr_mean = merged['stockout_rate'].mean()
    fig.add_vline(x=wr_mean, line_dash='dot', line_color='gray', opacity=0.5)
    fig.add_hline(y=sr_mean, line_dash='dot', line_color='gray', opacity=0.5)
    fig.update_layout(title='매장 × 카테고리 폐기율 vs 매진율 사분면',
                      xaxis_title='폐기율 % (생산가중)', yaxis_title='매진 발생률 %')
    charts.append(('폐기율 vs 매진율 사분면',
                   '20개 점 (4매장 × 5카테고리). 우상=발주 과잉+매진多, 좌하=정상 운영. 점선=평균.',
                   fig_to_div(fig, 'total_chart12', height=500)))

    # T14. 4매장 월별 예약(bulk) qty 시계열
    if 'bulk_qty' in daily.columns:
        d = daily.copy()
        d['ym'] = d['date'].dt.to_period('M').dt.to_timestamp()
        m_agg = d.groupby(['store', 'ym']).agg(
            bulk_qty=('bulk_qty', 'sum'),
            total_qty=('qty', 'sum'),
        ).reset_index()
        m_agg['bulk_pct'] = m_agg['bulk_qty'] / m_agg['total_qty'].replace(0, np.nan) * 100
        fig = make_subplots(rows=1, cols=2, subplot_titles=('월별 예약 수량 (절대값)', '월별 예약 비중 %'))
        for st in STORE_ORDER:
            sub = m_agg[m_agg['store']==st].sort_values('ym')
            fig.add_trace(go.Scatter(x=sub['ym'], y=sub['bulk_qty'], name=st,
                                     mode='lines', line=dict(color=STORE_COLORS[st], width=2),
                                     legendgroup=st), row=1, col=1)
            fig.add_trace(go.Scatter(x=sub['ym'], y=sub['bulk_pct'], name=st,
                                     mode='lines', line=dict(color=STORE_COLORS[st], width=2),
                                     legendgroup=st, showlegend=False), row=1, col=2)
        fig.update_layout(title='4매장 월별 예약 주문 추이')
        fig.update_yaxes(title='예약 수량', row=1, col=1)
        fig.update_yaxes(title='예약 비중 %', row=1, col=2)
        charts.append(('4매장 예약 주문 추이',
                       ('<b>정의</b>: 예약 = (단일 품목 ≥ 5 AND 평소×2.5) OR (다양 품목 영수증 total ≥ 15)<br>'
                        '<b>좌</b>: 월별 예약 절대 수량 (매장 매출 절대값 영향)<br>'
                        '<b>우</b>: 월별 예약 비중 % (정규화 비교)<br>'
                        '<b>해석</b>: 삼성타운 3.5% (오피스가 회의 케이터링 多) > 광화문 1.8% > 메세나 0.9% > 광교 0.6%'),
                       fig_to_div(fig, 'total_chart14', height=450)))

    # T15. 매장별 5년 예약 빈도 + qty 비중 비교
    if 'bulk_qty' in daily.columns:
        summary_rows = []
        for st in STORE_ORDER:
            sub = daily[daily['store']==st]
            n_days = len(sub)
            n_weeks = n_days / 7
            bulk_qty = sub['bulk_qty'].sum()
            total_qty = sub['qty'].sum()
            bulk_days = (sub['bulk_qty'] > 0).sum()
            summary_rows.append({
                'store': st,
                'bulk_pct': bulk_qty / total_qty * 100,
                'bulk_days_per_week': bulk_days / n_weeks,
                'avg_bulk_qty_per_event': bulk_qty / bulk_days if bulk_days else 0,
            })
        summary = pd.DataFrame(summary_rows)
        fig = make_subplots(rows=1, cols=3,
                             subplot_titles=('예약 qty 비중 %', '주당 예약 발생일 수', '예약일 평균 qty'))
        for i, (col, name) in enumerate([('bulk_pct', '비중 %'), ('bulk_days_per_week', '주당 발생일'),
                                          ('avg_bulk_qty_per_event', '평균 qty')]):
            fig.add_trace(go.Bar(x=summary['store'], y=summary[col],
                                  marker_color=[STORE_COLORS[s] for s in summary['store']],
                                  text=summary[col].round(2), textposition='outside',
                                  showlegend=False), row=1, col=i+1)
        fig.update_layout(title='4매장 예약 주문 빈도 비교 (5년 누적)')
        charts.append(('4매장 예약 빈도 비교',
                       ('매장 특성 정량 비교. 삼성타운/광화문 오피스가 = 회의용 케이터링 빈도 높음.<br>'
                        '광교/메세나 = 주거+상권 = 예약 빈도 낮지만 발생 시 평균 qty 큼 (가족 모임/행사).'),
                       fig_to_div(fig, 'total_chart15', height=400)))

    # T13. 4매장 5년 매출 추세 (linear fit slope)
    fig = go.Figure()
    slopes = []
    for st in STORE_ORDER:
        sub = daily[daily['store'] == st].sort_values('date').copy()
        sub['day_idx'] = (sub['date'] - sub['date'].min()).dt.days
        # 30d MA
        sub['ma'] = sub['qty'].rolling(30, min_periods=1).mean()
        # linear fit
        x = sub['day_idx'].values
        y = sub['qty'].values
        slope, intercept = np.polyfit(x, y, 1)
        trend = slope * x + intercept
        slopes.append((st, slope * 365))   # 연간 변화율
        fig.add_trace(go.Scatter(x=sub['date'], y=sub['ma'], name=f'{st} (30d MA)',
                                 line=dict(color=STORE_COLORS[st], width=1.5),
                                 opacity=0.5))
        fig.add_trace(go.Scatter(x=sub['date'], y=trend, name=f'{st} 추세선',
                                 line=dict(color=STORE_COLORS[st], width=3, dash='dash'),
                                 showlegend=False))
    fig.update_layout(title='4매장 5년 매출 추세 + 선형 fit',
                      xaxis_title='날짜', yaxis_title='일판매량')
    slope_text = ' / '.join(f'{st}: {s:+.1f}/년' for st, s in slopes)
    charts.append(('4매장 5년 trend + slope',
                   f'30d MA + 선형 fit. 연간 변화율: {slope_text}.',
                   fig_to_div(fig, 'total_chart13', height=500)))

    return charts


# =============================================================================
# 광교 전용 차트 (모델 진단)
# =============================================================================

def make_gwangyo_extra_charts(data: dict) -> list[tuple[str, str, str]]:
    charts = []
    # G1. v4 fair backtest 시계열 (광교, sumRn fix + bulk 제외 = 최신 모델)
    try:
        bt = pd.read_csv('reports/v4_new_data_bulk_excl.csv')
        bt['D'] = pd.to_datetime(bt['D'])
        bt['test_date'] = bt.apply(lambda r: r['D'] + pd.Timedelta(days=int(r['h'])), axis=1)
        bt = bt.sort_values('test_date')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt['test_date'], y=bt['actual'], mode='lines+markers',
                                  name='실제', line=dict(color='#2c3e50', width=2)))
        fig.add_trace(go.Scatter(x=bt['test_date'], y=bt['production'], mode='lines+markers',
                                  name='예측 (production)', line=dict(color='#e74c3c', width=2)))
        fig.add_trace(go.Scatter(x=bt['test_date'], y=bt['baseline'], mode='lines',
                                  name='baseline', line=dict(color='#95a5a6', width=1, dash='dot')))
        fig.update_layout(title='광교 v4 backtest — 예측 vs 실제 (sumRn fix + 예약 제외, 최신)',
                          xaxis_title='날짜', yaxis_title='판매량 (adjusted_demand, normal only)')
        charts.append(('v4 backtest 시계열 (최신: sumRn fix + 예약 제외)',
                       '<b>최신 모델</b>: sumRn 버그 fix (N=44 → 108) + 예약(bulk) 영수증 제외.<br>'
                       'WAPE 17.29% / 매진율 8.3% / 폐기/일 37.2 / 부족/일 1.2 (4 metric).<br>'
                       '예측 (빨강) vs 실제 (검정) vs baseline (회색 점선). 광교 11/12월 D+7 outlier 확인.',
                       fig_to_div(fig, 'gwangyo_extra1', height=500)))

        # G2. v4 error 분포 (error vs dow + error vs month)
        bt['err'] = bt['actual'] - bt['production']
        bt['err_pct'] = bt['err'] / bt['actual'] * 100
        bt['dow'] = bt['test_date'].dt.dayofweek
        bt['dow_name'] = bt['dow'].map(DN)
        bt['month'] = bt['test_date'].dt.month

        fig = make_subplots(rows=1, cols=2, subplot_titles=('요일별 오차 분포', '월별 오차 분포'))
        for i, dn_name in enumerate(DOW_ORDER):
            sub = bt[bt['dow_name'] == dn_name]
            fig.add_trace(go.Box(y=sub['err_pct'], name=dn_name, showlegend=False,
                                  marker_color='#3498db'), row=1, col=1)
        months_present = sorted(bt['month'].unique())
        for m in months_present:
            sub = bt[bt['month'] == m]
            fig.add_trace(go.Box(y=sub['err_pct'], name=f'{m}월', showlegend=False,
                                  marker_color='#e67e22'), row=1, col=2)
        fig.add_hline(y=0, line_dash='dot', line_color='gray', row=1, col=1)
        fig.add_hline(y=0, line_dash='dot', line_color='gray', row=1, col=2)
        fig.update_layout(title='광교 v4 모델 오차 분포 (요일/월별)')
        fig.update_yaxes(title_text='오차 % (실제 - 예측)/실제 ×100', row=1, col=1)
        charts.append(('v4 모델 오차 분포',
                       '예측 - 실제 오차의 요일/월별 분포. 음수 = over-prediction, 양수 = under-prediction.',
                       fig_to_div(fig, 'gwangyo_extra2', height=500)))
    except FileNotFoundError:
        pass

    return charts


# =============================================================================
# HTML template
# =============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>보나비 4매장 EDA Dashboard</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          margin: 0; padding: 0; background: #f5f5f5; color: #333; }}
  header {{ background: #2c3e50; color: white; padding: 20px 30px; }}
  header h1 {{ margin: 0; font-size: 22px; }}
  header .meta {{ font-size: 14px; opacity: 0.8; margin-top: 5px; }}
  .tabs {{ display: flex; background: #34495e; padding: 0 20px; position: sticky; top: 0; z-index: 100; }}
  .tabs button {{ background: none; border: none; color: white; padding: 14px 24px;
                  cursor: pointer; font-size: 15px; font-weight: 500; }}
  .tabs button.active {{ background: #1abc9c; }}
  .tabs button:hover {{ background: #16a085; }}
  .tab-content {{ display: none; padding: 30px; max-width: 1400px; margin: 0 auto; }}
  .tab-content.active {{ display: block; }}
  .chart-block {{ background: white; padding: 20px; margin-bottom: 25px; border-radius: 8px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .chart-block h2 {{ margin: 0 0 8px 0; font-size: 18px; color: #2c3e50; }}
  .chart-block .desc {{ color: #666; font-size: 13px; margin-bottom: 15px; line-height: 1.5; }}
  .intro {{ background: white; padding: 25px; border-radius: 8px; margin-bottom: 25px; }}
  .intro h2 {{ margin-top: 0; color: #2c3e50; }}
  .intro ul {{ line-height: 1.7; }}
  .badge {{ display: inline-block; padding: 3px 8px; background: #1abc9c; color: white;
             border-radius: 3px; font-size: 11px; margin-right: 5px; }}
</style>
</head>
<body>
<header>
  <h1>보나비 4매장 EDA Dashboard</h1>
  <div class="meta">2021-01-01 ~ 2025-12-31 · 광교 / 광화문 / 메세나폴리스 / 삼성타운 · sales 1.47M rows + inventory + weather + calendar</div>
</header>

<div class="tabs">
  <button class="active" onclick="showTab('total')">Total (4매장 비교)</button>
  {tab_buttons}
</div>

{tab_contents}

<script>
function showTab(name) {{
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(e => e.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>
"""


def render_html(total_charts: list, store_charts: dict) -> str:
    tab_buttons = ''
    for st in STORE_ORDER:
        safe = st.replace('폴리스', 'p').replace('타운', 't').replace('교', 'g').replace('문', 'm')
        tab_buttons += f'<button onclick="showTab(\'{safe}\')">{st}</button>\n'

    # Total tab content
    total_blocks = '<div class="intro"><h2>📊 4매장 비교 분석</h2><p>4매장 매출/카테고리/폐기/closing/특일 영향 종합 비교. 매장 특성 + brand-wide signal 동시 확인.</p></div>\n'
    for title, desc, div in total_charts:
        total_blocks += f'<div class="chart-block"><h2>{title}</h2><div class="desc">{desc}</div>{div}</div>\n'

    contents = f'<div id="tab-total" class="tab-content active">\n{total_blocks}\n</div>\n'

    for st in STORE_ORDER:
        safe = st.replace('폴리스', 'p').replace('타운', 't').replace('교', 'g').replace('문', 'm')
        store_blocks = f'<div class="intro"><h2>🏪 {st} 상세 분석</h2></div>\n'
        for title, desc, div in store_charts[st]:
            store_blocks += f'<div class="chart-block"><h2>{title}</h2><div class="desc">{desc}</div>{div}</div>\n'
        contents += f'<div id="tab-{safe}" class="tab-content">\n{store_blocks}\n</div>\n'

    return HTML_TEMPLATE.format(tab_buttons=tab_buttons, tab_contents=contents)


# =============================================================================
# Main
# =============================================================================

def main():
    data = prep_data()
    print('\n[charts] total tab...')
    total_charts = make_total_charts(data)
    print(f'  total: {len(total_charts)} charts')

    store_charts = {}
    for st in STORE_ORDER:
        print(f'[charts] {st}...')
        store_charts[st] = make_store_charts(st, data)
        if st == '광교':
            extras = make_gwangyo_extra_charts(data)
            store_charts[st].extend(extras)
            print(f'  광교 extras: +{len(extras)}')
        print(f'  {st}: {len(store_charts[st])} charts')

    print('\n[html] rendering...')
    html = render_html(total_charts, store_charts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding='utf-8')
    print(f'\nsaved: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
