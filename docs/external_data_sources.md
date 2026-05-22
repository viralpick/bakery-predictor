# 외부 데이터 출처 및 ingest 가이드

> 크롤러·운영 인프라 담당자용. 7개 외부 데이터의 정확한 endpoint, 인증, 호출 방법,
> 응답 컬럼 매핑, 적재 위치, 한계까지 단일 문서로.

---

## 인증키 종합

| 환경변수 | 발급처 | 발급 방법 | 주의 |
|---|---|---|---|
| `DATA_GO_KR_API_KEY` | https://data.go.kr | 회원가입 → 마이페이지 → 인증키 신청 (즉시 발급) | **활용신청한 API에만 권한 부여**. 신규 API 사용 전 활용신청 필수 |
| `SEOUL_OPEN_API_KEY` | https://data.seoul.go.kr | 회원가입 → 마이페이지 → 인증키 신청 (즉시 발급) | data.go.kr 키와 **별개**. 모든 서울 OpenAPI에 한 키 공통 사용 |

`.env` 위치: 프로젝트 root.

---

## 1. 캘린더 (특일 + 24절기)

| 항목 | 값 |
|---|---|
| **이름** | 천문연 특일정보 |
| **출처** | data.go.kr `B090041/SpcdeInfoService` |
| **인증** | `DATA_GO_KR_API_KEY` (별도 활용신청 필요) |
| **호출 단위** | 월별 (year + month) |
| **갱신 주기** | 연 1회 갱신 (다음 해 공휴일 추가) |
| **적재 위치** | `data/external/calendar_raw.parquet` |

### Endpoint

```
GET https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo
GET https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/get24DivisionsInfo
```

### 필수 파라미터

| 파라미터 | 값 | 예 |
|---|---|---|
| `serviceKey` | DATA_GO_KR_API_KEY | (env) |
| `solYear` | 연 | `2024` |
| `solMonth` | 월 (2자리) | `01` |
| `numOfRows` | 페이지 크기 | `100` |
| `_type` | 응답 포맷 | `json` |

### 응답 → 우리 schema

| API 컬럼 | 우리 컬럼 | 비고 |
|---|---|---|
| `locdate` (YYYYMMDD) | `date` | normalize to midnight |
| `dateName` | `name` | 공휴일/절기 이름 |
| `isHoliday` ("Y"/"N") | `is_holiday` (boolean) | |
| (별도 계산) | `is_public_holiday`, `is_substitute_holiday`, `off_streak_length`, `off_position_in_streak`, `is_day_before_off`, `is_day_after_off`, 이벤트별 boolean | `build_calendar_daily()`이 도출 |

### CLI

```bash
uv run bakery ingest-calendar --start-year 2024 --end-year 2026
```

### 한계

- 매년 12월에 다음 해 공휴일 데이터 갱신 — 운영 시 매년 1월 ingest 권장

---

## 2. 날씨 ASOS 관측

| 항목 | 값 |
|---|---|
| **이름** | 기상청 ASOS 일자료 조회 |
| **출처** | data.go.kr `1360000/AsosDalyInfoService/getWthrDataList` |
| **인증** | `DATA_GO_KR_API_KEY` (별도 활용신청 필요) |
| **호출 단위** | 매장 매핑된 station × 연도 chunk |
| **갱신 주기** | 일별 (전일 데이터 익일 적재) |
| **적재 위치** | `data/external/weather_observed.parquet` (long-form station × date) |

### Endpoint

```
GET https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList
```

### 필수 파라미터

| 파라미터 | 값 |
|---|---|
| `ServiceKey` | DATA_GO_KR_API_KEY |
| `pageNo`, `numOfRows` | 페이지네이션 (999 max) |
| `dataType` | `JSON` |
| `dataCd` | `ASOS` |
| `dateCd` | `DAY` |
| `startDt`, `endDt` | YYYYMMDD |
| `stnIds` | ASOS station_id (서울 108, 수원 119, 인천 112, ...) |

### 매장 → station 매핑

매장 좌표에서 가장 가까운 ASOS station 선택:

| 매장 위치 | station_id |
|---|---|
| 서울 | 108 |
| 수원 (광교 포함) | 119 |
| 인천 | 112 |
| 동두천 | 98 |
| 대구 | 143 |
| 부산 | 159 |
| 광주 | 156 |

자동 매핑은 PoC 미구현. `store_mapping.yaml`에 매장별 `station_id` 명시.

### 응답 → 우리 schema

KMA 응답의 핵심 컬럼:

| API 컬럼 | 우리 컬럼 | 단위 |
|---|---|---|
| `tm` | `date` | YYYY-MM-DD |
| `avgTa` | `avg_temp` | °C |
| `maxTa` / `minTa` | `max_temp` / `min_temp` | °C |
| `avgRhm` | `humidity` | % |
| `sumRn` | `precipitation_mm` | mm |
| `ddMes` (or `ddMefs`) | `snow_depth_cm` | cm |
| `sumSsHr` | `sunshine_hours` | hour |
| (도출) | `diurnal_range`, `is_rain`, `is_snow` | |

매장별 fan-out: long-form `(store_id, date)` schema로 매장 station 공유 시에도 store_id 부착.

### CLI

```bash
uv run bakery ingest-weather --start 2024-01-01 --end 2025-12-31
```

매장 mapping 변경 후 새 station 자동 fetch (DEFAULT_STATIONS 또는 yaml).

### 한계

- ASOS는 station 단위. 매장 좌표 기반 자동 nearest는 미구현 (yaml hardcode)
- 일부 station에서 humidity / snow_depth가 결측. 우리 loader가 interpolate + fillna 처리

---

## 3. 날씨 단기/중기예보

| 항목 | 값 |
|---|---|
| **이름** | KMA 단기예보 + 중기예보 |
| **출처** | data.go.kr `1360000/VilageFcstInfoService_2.0/getVilageFcst` + `MidFcstInfoService/getMidLandFcst`, `getMidTa` |
| **인증** | `DATA_GO_KR_API_KEY` (별도 활용신청 필요) |
| **호출 단위** | 매장 매핑된 격자(nx/ny) + 예보구역(mid_reg) × 최신 발표시각 |
| **갱신 주기** | 단기 8회/일 (02/05/08/11/14/17/20/23시), 중기 2회/일 (06/18시) |
| **적재 위치** | `data/external/forecast_short_term*.parquet`, `forecast_mid_term_daily.parquet` |

### Endpoint

```
GET https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst    (단기 D+0~D+3)
GET https://apis.data.go.kr/1360000/MidFcstInfoService/getMidLandFcst         (중기 강수확률·날씨)
GET https://apis.data.go.kr/1360000/MidFcstInfoService/getMidTa                (중기 기온)
```

### 필수 파라미터 (단기)

| 파라미터 | 값 |
|---|---|
| `ServiceKey` | DATA_GO_KR_API_KEY |
| `pageNo`, `numOfRows` | (1000) |
| `dataType` | `JSON` |
| `base_date` | YYYYMMDD (발표일) |
| `base_time` | HHMM (가장 최근 02/05/.../23) |
| `nx`, `ny` | KMA Lambert 격자 (서울 60/127, 수원 60/121) |

### 필수 파라미터 (중기)

| 파라미터 | 값 |
|---|---|
| `regId` | 예보구역 코드 (육상 11B00000 = 서울·인천·경기, 기온 11B10101 = 서울, 11B20601 = 경기남부) |
| `tmFc` | YYYYMMDDHHmm (최신 발표시각) |

### 매장 → 격자/예보구역 매핑

| 매장 위치 | nx, ny | mid_land_reg_id | mid_ta_reg_id |
|---|---|---|---|
| 서울 | 60, 127 | 11B00000 | 11B10101 |
| 수원 (광교) | 60, 121 | 11B00000 | 11B20601 |
| 인천 | 55, 124 | 11B00000 | 11B20201 |
| 대전 | 67, 100 | 11C20000 | 11C20401 |
| ... | | | |

좌표 → nx/ny 자동 변환은 KMA DFS_XY_CONV 공식 사용 가능 (PoC 미구현, hardcode).

### CLI

```bash
uv run bakery ingest-forecast    # 최신 발표시각 자동 결정 + 매장별 fetch
```

매번 호출 시 latest base_time 자동 계산 (단기 30분 지연, 중기 60분 지연 반영).

### 한계

- 운영 시 `predict-next-week --use-forecast`로 horizon 날씨 사용
- humidity / sunshine은 중기예보 미제공 → 최근 28일 ASOS 평균 fallback

---

## 4. 경쟁점 — Bakery (LOCALDATA)

| 항목 | 값 |
|---|---|
| **이름** | 행정안전부_식품 제과점영업 |
| **출처** | data.go.kr **15155252** (`apis.data.go.kr/1741000/bakeries/info`) |
| **인증** | `DATA_GO_KR_API_KEY` (별도 활용신청 필요) |
| **호출 단위** | 시군구 (OPN_ATMY_GRP_CD) |
| **갱신 주기** | 매일 (2일전 기준 현행화) |
| **적재 위치** | `data/external/competitor_raw.parquet` (bakery + cafe 통합) |

### Endpoint

```
GET https://apis.data.go.kr/1741000/bakeries/info
```

### 필수 파라미터

| 파라미터 | 값 |
|---|---|
| `serviceKey` | DATA_GO_KR_API_KEY |
| `returnType` | `json` |
| `pageNo`, `numOfRows` | (numOfRows max 100) |
| `cond[OPN_ATMY_GRP_CD::EQ]` | 시군구 개방자치단체코드 |

선택 cond: `cond[LCPMT_YMD::GTE]/LT`, `cond[SALS_STTS_CD::EQ]`, `cond[BPLC_NM::LIKE]` 등.

### 매장 시군구 → OPN_ATMY_GRP_CD 매핑

| 시군구 | OPN_ATMY_GRP_CD |
|---|---|
| 서울 강남구 | 3220000 |
| 서울 마포구 | 3130000 |
| 서울 영등포구 | 3180000 |
| 서울 강서구 | 3150000 |
| 서울 영등포구 | 3180000 |
| 수원시 영통구 (광교) | 3740000 |

매핑 코드: `ingest/competitor_api.py::SIGUNGU_BY_DONG_PREFIX`. 신규 시군구는 ROAD_NM_ADDR LIKE로 사전 검색해서 추가.

### 응답 → 우리 schema

응답이 풍부 (40+ 컬럼). 우리가 사용하는 컬럼:

| API 컬럼 | 우리 컬럼 |
|---|---|
| `MNG_NO` | `business_id` |
| (고정) | `category = "bakery"` |
| `LCPMT_YMD` | `license_date` |
| `CLSBIZ_YMD` (빈 string → NaT) | `close_date` |
| `CRD_INFO_X`, `CRD_INFO_Y` | `lat`, `lon` (TM EPSG:5181 → WGS84 변환) |
| `SALS_STTS_CD` ("01" = 영업/정상) | `business_status` |

### 좌표 변환

응답 좌표는 **TM 중부원점 (EPSG:5181 KATEC)**. WGS84로 변환 필요:

```python
from pyproj import Transformer
tf = Transformer.from_crs("epsg:5181", "epsg:4326", always_xy=True)
lon, lat = tf.transform(crd_info_x, crd_info_y)
```

### CLI

```bash
uv run bakery ingest-competitor    # 모든 매장 시군구 + 카페까지 통합 fetch
```

### 한계

- LOCALDATA 휴게음식점 데이터셋(15155330)이 별도 ID로 검색 어려움 → cafe는 SBIZ snapshot 사용 (다음 source)
- 좌표가 일부 결측 또는 0/0 sentinel → 우리 normalize에서 위경도 한국 범위(33~39°N, 124~132°E) 필터링

---

## 5. 경쟁점 — Cafe (SBIZ)

| 항목 | 값 |
|---|---|
| **이름** | 소상공인진흥공단 상가(상권)정보 |
| **출처** | data.go.kr `B553077/api/open/sdsc2/storeListInRadius` |
| **인증** | `DATA_GO_KR_API_KEY` (별도 활용신청 필요) |
| **호출 단위** | 매장 lat/lon × 반경 1km |
| **갱신 주기** | 분기 1회 (snapshot, license/close date 없음) |
| **적재 위치** | `data/external/competitor_raw.parquet` (bakery와 통합) |

### Endpoint

```
GET https://apis.data.go.kr/B553077/api/open/sdsc2/storeListInRadius
```

### 필수 파라미터

| 파라미터 | 값 | 예 |
|---|---|---|
| `serviceKey` | DATA_GO_KR_API_KEY | |
| `type` | `json` | |
| `radius` | 반경 (m, max 1000) | `1000` |
| `cx`, `cy` | 중심 lon/lat (WGS84) | `126.9240, 37.5563` |
| `indsSclsCd` | 업종 소분류 코드 | `I21201` (카페) |
| `numOfRows`, `pageNo` | (1000 max) | |

### 업종 코드

다른 카테고리는 `/smallUpjongList`로 검색:

| 코드 | 의미 |
|---|---|
| `I21001` | 음식 > 기타 간이 > 빵/도넛 |
| `I21201` | 음식 > 비알코올 > 카페 |
| `I21002` | 휴게음식점 일부 |

### 응답 → 우리 schema

| API 컬럼 | 우리 컬럼 |
|---|---|
| `bizesId` | `business_id` |
| (고정) | `category = "cafe"` |
| (없음 → fixed 2019-01-01) | `license_date` |
| (없음 → NaT) | `close_date` |
| `lat`, `lon` (WGS84) | `lat`, `lon` |
| (고정) | `business_status = "active"` |

### 한계

- **License/close date 정보 없음** — 시간 변동 features `new_90d` / `closed_90d` cafe 부분에서 동작 안 함 (분산 0, LightGBM 무시)
- 카페 trend 신호 필요시 LOCALDATA 휴게음식점 데이터셋 별도 신청

### CLI

`ingest-competitor` 통합 명령에 포함됨 (bakery LOCALDATA + cafe SBIZ).

---

## 6. 생활인구 (서울)

| 항목 | 값 |
|---|---|
| **이름** | 서울시 행정동 단위 생활인구 (KT 통신 기반 추계) |
| **출처** | 서울 열린데이터광장 OA-14991 `SPOP_LOCAL_RESD_DONG` |
| **인증** | `SEOUL_OPEN_API_KEY` |
| **호출 단위** | 일자 (YYYYMMDD) × pageNo (numOfRows max 1000) |
| **갱신 주기** | 일별. **OpenAPI retention 최근 2개월만**. 전체 history는 zip 다운로드 |
| **적재 위치** | `data/external/living_population.parquet` (long-form) |

### Endpoint (최근 2개월만)

```
GET http://openapi.seoul.go.kr:8088/{KEY}/json/SPOP_LOCAL_RESD_DONG/{start}/{end}/{USE_DT}
```

### 필수 파라미터

| 위치 | 값 |
|---|---|
| `{KEY}` | SEOUL_OPEN_API_KEY |
| `{start}/{end}` | 페이지 범위 (예: 1/1000) |
| `{USE_DT}` | YYYYMMDD |

한 일자에 ~10,176 rows (서울 424 행정동 × 24시간) → 11 페이지.

### 응답 → 우리 schema

| API 컬럼 | 우리 컬럼 |
|---|---|
| `STDR_DE_ID` | `date` |
| `TMZON_PD_SE` (`"00"`~`"23"`) | `hour` (int8) |
| `ADSTRD_CODE_SE` (8자리) | `admin_dong_code` |
| `TOT_LVPOP_CO` | `total_pop` (float32) |
| + 성연령별 ~16 컬럼 | (현재 미사용) |

### CLI (OpenAPI, 최근 2개월)

```bash
uv run bakery ingest-living-population --start 2026-04-01 --end 2026-05-15
```

### CSV history 다운로드 (전체 기간)

데이터셋 페이지의 "파일" 탭에서 월별 zip 다운로드:

- 2017-01 ~ 2022-12: 반기 zip (`LOCAL_PEOPLE_DONG_2020_상반기.zip` 등, 250-330MB)
- 2023-01 ~ : 월별 zip (`LOCAL_PEOPLE_DONG_YYYYMM.zip`, 50-60MB)

### CSV → schema 매핑

CSV 컬럼명 (한글) → 우리 schema:

| CSV | 우리 |
|---|---|
| `기준일ID` | `date` (YYYYMMDD) |
| `시간대구분` | `hour` |
| `행정동코드` | `admin_dong_code` |
| `총생활인구수` | `total_pop` |
| + 성연령별 ~28 컬럼 | (미사용) |

CSV 파일 특이사항:
- 인코딩: 대부분 UTF-8 BOM, 일부 cp949
- pandas `index_col=False` 필수 (pandas의 "ragged header" 자동 인덱스 처리로 컬럼 right-shift 버그 회피)

### CLI

```bash
# zip 다운로드 후 data/external/living_pop_zips/ 에 넣고 실행
uv run bakery ingest-living-pop-csv
```

자동: 매장 dong filter + dedup + 기존 parquet과 merge.

### 한계

- **서울만 cover**. 경기 매장은 default fallback (광교 매장)
- 경기도 대체 source: 경기데이터드림(data.gg.go.kr) 별도 검토 필요

---

## 7. 연령대 (행정동 주민등록)

| 항목 | 값 |
|---|---|
| **이름** | 행정안전부 행정동별 성·연령별 주민등록 인구수 |
| **출처** | data.go.kr **15108072** (`apis.data.go.kr/1741000/admmSexdAgePpltn/selectAdmmSexdAgePpltn`) |
| **인증** | `DATA_GO_KR_API_KEY` (별도 활용신청 필요. 발급된 키는 일반 키와 동일값) |
| **호출 단위** | 시군구 admmCd (10자리) + 월별 |
| **갱신 주기** | 월별 (매월 말일 집계) |
| **적재 위치** | `data/external/population.parquet` |

### Endpoint

```
GET https://apis.data.go.kr/1741000/admmSexdAgePpltn/selectAdmmSexdAgePpltn
```

### 필수 파라미터

| 파라미터 | 값 |
|---|---|
| `serviceKey` | DATA_GO_KR_API_KEY (활용신청한 키) |
| `type` | `json` |
| `admmCd` | 10자리 행안부 행정동코드 (시군구만 매핑되면 그 시군구 전체 dong 응답) |
| `srchFrYm`, `srchToYm` | YYYYMM (3개월 윈도우 내, 2022.10부터) |
| `lv` | `3` (읍면동 단위) 또는 `7` (단일 dong) |
| `numOfRows` | 1~100 (max) |
| `pageNo` | |
| `regSeCd` | `1` (전체, 기본값) |

### 매장 시군구 prefix → admmCd

매장 admin_dong_code(8자리)의 앞 5자리가 시군구 prefix. 10자리 admmCd로 변환:

```
prefix + "00000"  (e.g. 11680 + 00000 = 1168000000)
```

`lv=3`에서 admmCd는 그 시군구의 어느 dong이든 OK — 응답이 시군구 전체로 옴.

### 응답 → 우리 schema

응답 row 단위: (admmCd, statsYm, ctpvNm, sggNm, dongNm, tong, ban, totNmprCnt, maleNmprCnt, femlNmprCnt, male0AgeNmprCnt, ..., feml100AgeNmprCnt)

**10세 단위 cohort** (1세 cohort 아님):
- `male0AgeNmprCnt` = 0~9세 남성
- `male10AgeNmprCnt` = 10~19세 남성
- ... `male100AgeNmprCnt` = 100+ 남성

| API 컬럼 | 우리 컬럼 |
|---|---|
| `admmCd` (10자리) → 첫 8자리 매장과 매칭 | `admin_dong_code` (8자리) |
| `statsYm` (YYYYMM) | `ym` (YYYY-MM) |
| `male{0,10,20,...}AgeNmprCnt` | `age_bin = "0_9"/"10_19"/...` |
| `feml{0,10,...}AgeNmprCnt` | 동일, sex="F" |
| 합산 | `population` |

### Dong matching

서울 SPOP의 8자리 ADSTRD_CODE_SE와 행안부 10자리 admmCd는 **체계가 다름**. 매핑은 dongNm 기반:
- store_mapping의 `admin_dong_name` (예: "광교2동") 
- API 응답의 `dongNm` 매칭
- → 매장의 8자리 admin_dong_code 부여

### CLI

```bash
uv run bakery ingest-population    # snapshot 최근 월
```

### 한계

- 1개월 snapshot. PoC는 매장 정적 baseline만 사용 → snapshot으로 충분
- 3개월 윈도우 제약. 더 긴 history 필요 시 시기별 호출 누적

---

## 8. 소비 (서울 상권분석)

| 항목 | 값 |
|---|---|
| **이름** | 서울시 상권분석서비스(소비-행정동) |
| **출처** | 서울 열린데이터광장 OA-22166 `VwsmAdstrdNcmCnsmpW` |
| **인증** | `SEOUL_OPEN_API_KEY` |
| **호출 단위** | pageNo (전체 데이터 다운) |
| **갱신 주기** | 분기 1회 |
| **적재 위치** | `data/external/consumption.parquet` |

### Endpoint

```
GET http://openapi.seoul.go.kr:8088/{KEY}/json/VwsmAdstrdNcmCnsmpW/{start}/{end}/
```

### 필수 파라미터

| 위치 | 값 |
|---|---|
| `{KEY}` | SEOUL_OPEN_API_KEY |
| `{start}/{end}` | 페이지 (1/1000) |

### 응답 컬럼

| API 컬럼 | 의미 |
|---|---|
| `STDR_YYQU_CD` | 기준 연분기 ("20241" = 2024Q1) |
| `ADSTRD_CD` | 행정동코드 8자리 |
| `ADSTRD_CD_NM` | 행정동명 |
| `EXPNDTR_TOTAMT` | 총 지출 |
| `FDSTFFS_EXPNDTR_TOTAMT` | 식료품 지출 |
| `FD_EXPNDTR_TOTAMT` | 음식점 지출 |
| `CLTHS_FTWR_EXPNDTR_TOTAMT` | 의류/신발 |
| (외 9개 카테고리) | |

### 응답 → 우리 schema

| API | 우리 |
|---|---|
| `STDR_YYQU_CD` ("20241") | `quarter` ("2024Q1") |
| `ADSTRD_CD` | `admin_dong_code` |
| `EXPNDTR_TOTAMT` | `total_spend` |
| `FDSTFFS_EXPNDTR_TOTAMT + FD_EXPNDTR_TOTAMT` | `food_retail_spend` |

### CLI

```bash
uv run bakery ingest-consumption    # 전체 history (7년+)
```

### 한계

- **서울만 cover** (광교는 default fallback)
- 분기 단위라 시간 변동 features는 제한적 — 정적 baseline 용도

---

## 9. 매장 데이터 (보나비, 자체 시스템)

> 외부 API는 아니지만 ingest 흐름 정리.

| 항목 | 값 |
|---|---|
| **이름** | 보나비 데이터 (xlsx) |
| **출처** | 보나비 본사 직접 전달 |
| **적재 위치** | `data/internal/보나비 데이터_YYYYMMDD.xlsx` |
| **변환 위치** | `data/internal/bonavi_daily.parquet` |

### Excel 시트 5개

| 시트 | 의미 | rows (광교 5년) |
|---|---|---|
| `판매정보` | 영수증 line item (점포·일자·POS·영수증·품목·수량·단가·할인·결제) | 458,366 |
| `품목정보` | 품목 마스터 (코드·메뉴명·단가·당일폐기여부·브랜드) | 417 |
| `점포정보` | 점포 마스터 (코드·점포명·주소·상태) | 121 |
| `품절정보` | 일별 품절 시각 (점포·일자·품목·시분) | 300,801 |
| `할인코드` | 할인 마스터 | 314 |

### 변환 흐름 (`data/bonavi_loader.py`)

1. 시트 4개 로드 + dummy 첫 row 제거 (`CD_PARTNER`, `CD_ITEM` 등)
2. 광교점(`점포코드=1000000047`) 필터
3. `셋트상품구분=SS` (단품), `판매구분=0` (정상) 필터
4. `POS메뉴명` 키워드 기반 카테고리 매핑 (bread/pastry/cake/sandwich/sweets/beverage)
5. 일별 집계: sum `판매수량` over (store, item, date)
6. 품절정보 join → `is_stockout` + `stockout_time` (시분 → datetime)
7. capacity = running max (proxy)
8. `potential_demand = sold_units` (1차 PoC; v2가 모델 학습 시 자체 재보정)
9. → `bonavi_daily.parquet` (`DAILY_COLUMNS` 준수)

### CLI

```bash
uv run bakery format-bonavi    # xlsx → daily parquet 변환
```

---

## 10. 외부 데이터 backfill 순서 (신규 매장 도착 시)

```
1. store_mapping yaml 작성 (lat/lon/admin_dong/station_id/nx/ny/mid_reg)
2. SIGUNGU_BY_DONG_PREFIX에 신규 시군구 매핑 추가 (필요시)
3. bakery format-bonavi              # 매장 매출 schema 변환
4. bakery ingest-calendar             # 학습 윈도우 cover
5. bakery ingest-weather              # 매장 station 추가
6. bakery ingest-competitor           # LOCALDATA bakery + SBIZ cafe
7. bakery ingest-population           # 행안부 매장 시군구
8. bakery ingest-consumption          # 서울 매장만 (경기는 default)
9. (선택) bakery ingest-living-pop-csv  # zip 다운로드 후
10. bakery backtest --source real --variants v0,v1,v2,v3 --include-production
11. bakery business-report           # 사업 임팩트 종합
```

매장 수가 늘어도 mapping 추가 + ingest 명령 재실행 (단위는 매장별 자동 union).

---

## 11. 데이터 갱신 주기 운영 권장

| Source | 갱신 명령 | 권장 주기 |
|---|---|---|
| 캘린더 | `ingest-calendar` | 연 1회 (1월) |
| 날씨 ASOS | `ingest-weather` | 매일 또는 매주 (전일 누적) |
| 날씨 예보 | `ingest-forecast` | 매번 predict 전에 (단기 발표 8회/일) |
| 경쟁점 | `ingest-competitor` | 분기 1회 |
| 생활인구 (OpenAPI) | `ingest-living-population` | 매일 (최근 2개월 retention) |
| 생활인구 (CSV) | `ingest-living-pop-csv` | 월 1회 (새 zip 다운로드 후) |
| 연령대 | `ingest-population` | 월 1회 |
| 소비 | `ingest-consumption` | 분기 1회 |

---

## 12. 트러블슈팅

### 403 Forbidden (data.go.kr)

- 활용신청 안 된 API. 마이페이지 → 활용 신청 → 즉시 승인 → 1시간 내 활성화
- 발급된 인증키 값은 일반 `DATA_GO_KR_API_KEY`와 동일 (활용신청 단위로 API 권한만 분리)

### INFO-200 "해당 데이터 없음" (서울 OpenAPI)

- 데이터셋 retention 범위 외 (예: LOCAL_PEOPLE_DONG OpenAPI는 최근 2개월만)
- SERVICE명 오타 (`SPOP_LOCAL_RESD_DONG` vs `LOCAL_PEOPLE_DONG` — 정확한 이름은 데이터셋 페이지 확인)

### 좌표 변환 실패 (LOCALDATA TM → WGS84)

- LOCALDATA는 EPSG:5181 (KATEC 중부원점). EPSG:5174나 5179 사용 시 결과 어긋남
- `pyproj` 패키지 필수 (`uv add pyproj`)

### CSV BOM 깨짐 (서울 생활인구 zip)

- UTF-8 BOM 바이트 시퀀스 `\xef\xbb\xbf` 직접 strip (byte-level)
- `pd.read_csv(index_col=False)` 명시 — pandas의 "ragged header" 자동 인덱스 처리 회피

### LOCALDATA OPN_ATMY_GRP_CD 찾기

매장이 새 시군구일 때:

```bash
# 매장 시군구명으로 LOCALDATA API에 LIKE 검색
curl "https://apis.data.go.kr/1741000/bakeries/info?\
serviceKey=$KEY&returnType=json&pageNo=1&numOfRows=1&\
cond[ROAD_NM_ADDR::LIKE]=용인시 수지구"
# → 응답의 첫 row "OPN_ATMY_GRP_CD" 값을 SIGUNGU_BY_DONG_PREFIX에 매핑
```

---

## 13. 보안 — 인증키 관리

- `.env`는 `.gitignore`에 포함 (절대 commit 금지)
- 채팅·문서·PR description에 평문 노출 금지
- 키 노출 의심 시 즉시 재발급 (마이페이지 → 인증키 재발급)
- 운영 시 vault / secret manager 사용 권장

---

## 부록 — 의존성

```
httpx        # API 호출
openpyxl     # 보나비 xlsx 읽기
pyproj       # TM → WGS84 좌표 변환
python-dotenv # .env 로딩
pandas       # 데이터 처리
```

`pyproject.toml` 참조. `uv sync`로 일괄 설치.
