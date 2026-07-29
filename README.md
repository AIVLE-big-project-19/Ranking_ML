# AI 기반 공공 유휴공간 태양광 사전검토 시스템

공공 유휴 토지·건물을 대상으로 지자체 규제, 입지 특성, 발전환경을 분석해 태양광 설치 적합도와 후보지 우선순위를 제공하는 프로젝트입니다.

## 주요 기능

- 지자체 조례 기반 Rule-based 사전검토
- 토지형·건물형 ML 적합도 예측
- Score, Rank, Percentile, Grade 산출
- SHAP 기반 후보지별 주요 요인 설명
- Vision AI 기반 면적·장애물·수목·음영 분석 예정
- 지원사업 추천, 민원 가능성 분석, 보고서 생성 Agent 예정

## 처리 흐름

```text
후보 데이터
→ Rule-based 검토
→ Vision AI 분석
→ ML 적합도 예측
→ Score / Rank / Grade / SHAP
→ 지원사업·민원·보고서 생성
```

## 주요 파일

```text
Solar_RuleBase_Review_Aligned.ipynb
Common_Test_Ranking_Aligned.ipynb
태양광_RuleBase_실행용_수정본.xlsx
Land_model_bundle.pkl
Building_model_bundle.pkl
Land_Test_Chungcheong_Uninstalled.csv
Building_Test_Chungcheong_Uninstalled.csv
api/main.py
api/pipeline.py
api/config.py
requirements.txt
```

## 1. Rule-based 검토 실행

### 필요한 파일

토지형:

```text
Solar_RuleBase_Review_Aligned.ipynb
태양광_RuleBase_실행용_수정본.xlsx
Land_Test_Chungcheong_Uninstalled.csv
```

건물형:

```text
Solar_RuleBase_Review_Aligned.ipynb
태양광_RuleBase_실행용_수정본.xlsx
Building_Test_Chungcheong_Uninstalled.csv
```

### 실행 방법

노트북에서 분석 대상을 선택합니다.

```python
DATASET_TYPE = "land"
# DATASET_TYPE = "building"
```

입력 파일명을 실제 업로드한 파일명과 맞춘 뒤 셀을 위에서부터 실행합니다.

```python
RULE_XLSX_FILENAME = "태양광_RuleBase_실행용_수정본.xlsx"
LAND_TEST_FILENAME = "Land_Test_Chungcheong_Uninstalled.csv"
BUILDING_TEST_FILENAME = "Building_Test_Chungcheong_Uninstalled.csv"
```

### 출력 파일

결과는 `/content/rulebase_results`에 저장됩니다.

토지형:

```text
Land_Test_Chungcheong_RuleReviewed.csv
Land_Test_Chungcheong_RulePassed.csv
Land_Test_Chungcheong_RuleAudit.csv
```

건물형:

```text
Building_Test_Chungcheong_RuleReviewed.csv
Building_Test_Chungcheong_RulePassed.csv
Building_Test_Chungcheong_RuleAudit.csv
```

| 파일 | 내용 |
|---|---|
| `RuleReviewed` | 전체 후보지와 최종 Rule 판정 |
| `RulePassed` | Vision AI 또는 ML 단계에 전달할 후보 |
| `RuleAudit` | 후보별 규칙 적용 여부와 미평가 사유 |

## 2. ML 예측 및 랭킹 실행

### 필요한 파일

토지형:

```text
Common_Test_Ranking_Aligned.ipynb
Land_model_bundle.pkl
Land_Test_Chungcheong_RulePassed.csv
```

건물형:

```text
Common_Test_Ranking_Aligned.ipynb
Building_model_bundle.pkl
Building_Test_Chungcheong_RulePassed.csv
```

### 실행 방법

분석 대상과 입력 단계를 설정합니다.

```python
DATASET_TYPE = "land"
# DATASET_TYPE = "building"

INPUT_STAGE = "rule"
```

파일명을 실제 업로드 파일명과 맞춘 뒤 셀을 위에서부터 실행합니다.

```python
FILE_CONFIG = {
    "land": {
        "model": "Land_model_bundle.pkl",
        "rule": "Land_Test_Chungcheong_RulePassed.csv",
    },
    "building": {
        "model": "Building_model_bundle.pkl",
        "rule": "Building_Test_Chungcheong_RulePassed.csv",
    },
}
```

### 출력 항목

- 설치 적합 확률
- Solar Readiness Score
- 전체 및 지역별 Rank
- Percentile
- A/B/C Grade
- SHAP 주요 요인
- Rule 검토 결과
- 후보지별 분석 JSON

### 출력 파일

토지형 기준:

```text
Land_Test_candidate_ranking.csv
Land_Test_candidate_ranking_with_shap.csv
Land_Test_candidate_shap_details.csv
Land_Test_ranking_results.xlsx
Land_Top20_Candidate_Analysis.json
```

건물형 실행 시 파일명의 `Land`가 `Building`으로 변경됩니다.

| 파일 | 내용 |
|---|---|
| `candidate_ranking.csv` | 점수·등급·순위 결과 |
| `candidate_ranking_with_shap.csv` | SHAP 설명이 결합된 결과 |
| `candidate_shap_details.csv` | Feature별 SHAP 세부값 |
| `ranking_results.xlsx` | 예측·랭킹·SHAP 통합 결과 |
| `Top20_Candidate_Analysis.json` | 상위 후보의 서비스 연계용 JSON |

## 3. FastAPI로 실행

`Common_Test_Ranking_Aligned.ipynb`의 예측·랭킹·SHAP·JSON 생성 로직을 `api/` 아래 FastAPI 서버로 옮겨 두었습니다. 노트북을 열지 않고도 후보지 CSV/XLSX를 업로드해 같은 결과를 받을 수 있습니다.

### 필요한 파일

```text
api/main.py
api/pipeline.py
api/config.py
Land_model_bundle.pkl
Building_model_bundle.pkl
requirements.txt
```

### 설치 및 실행

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8002
```

서버가 뜨면 `http://127.0.0.1:8002/docs`에서 Swagger UI로 바로 테스트할 수 있습니다.

### 엔드포인트

| Method | Path | 설명 |
|---|---|---|
| GET | `/health` | 헬스체크 |
| GET | `/models` | land/building 모델 로드 상태, feature 목록 |
| GET | `/candidates/{dataset_type}/search` | 서버에 상주하는 `Rule_base/*_RulePassed.csv` 전체 후보군에서 주소로 검색 |
| POST | `/rank/{dataset_type}` | 후보지 파일 업로드 → 예측·랭킹·SHAP·상위 후보 JSON 반환 |
| POST | `/rank/{dataset_type}/export` | 위와 동일한 처리 후 CSV/XLSX/JSON을 zip으로 다운로드 |

`dataset_type`은 `land` 또는 `building`. 파일은 `multipart/form-data`의 `file` 필드로 업로드하며, `Rule_Pass_For_Next_Step` 컬럼이 있는 Rule-based 통과 결과(`*_RulePassed.csv`)를 그대로 넣으면 됩니다.

주요 폼 필드(전부 선택, 기본값은 노트북과 동일):

- `filter_rule_excluded` (기본 `true`): `Rule_Pass_For_Next_Step`이 true인 행만 사용
- `rank_filter_mode` (`all` | `label_0`, 기본 `all`)
- `create_region_ranks` (기본 `true`): 시도/시군구 랭크 생성
- `include_shap` (기본 `true`)
- `shap_top_n` (기본 `1000`)
- `ml_weight` / `policy_weight` (기본 `1.0` / `0.0`)
- `policy_weight_config` (JSON 문자열, 예: `{"col": {"weight": 1, "direction": "higher"}}`)
- `top_n_json` (기본 `20`): 상세 JSON을 생성할 상위 후보 수

예시:

```bash
curl -X POST http://127.0.0.1:8002/rank/land \
  -F "file=@Rule_base/Land_Test_Chungcheong_RulePassed.csv" \
  -F "top_n_json=20"
```

### 주소 검색 (`/candidates/{dataset_type}/search`)

파일을 매번 업로드하지 않고, 서버가 들고 있는 `Rule_base/*_RulePassed.csv` 전체 후보군에서 주소로 검색합니다. 서버 시작 시 이 전체 후보군에 대한 예측·랭킹·SHAP 결과를 한 번 계산해 캐싱해 둡니다.

쿼리 파라미터:

- `q`: 검색어. 공백으로 구분한 각 단어가 주소(`address_ml` 우선, 없으면 `주소`/`소재지`/`도로명주소`/`지번주소`)에 모두 포함된 행만 매칭 (AND 조건, 부분 일치)
- `top_n` (기본 `20`): 검색 결과 중 상세 JSON을 반환할 상위 개수

응답의 각 결과는 순위를 두 가지 기준으로 함께 제공합니다:

- `overall_rank`: 전체 후보군(532건/60건) 대비 순위
- `search_rank`: 검색된 결과들 사이에서의 순위

```bash
curl "http://127.0.0.1:8002/candidates/land/search?q=대전%20유성구&top_n=10"
```

```json
{
  "dataset_type": "land",
  "query": "대전 유성구",
  "universe_size": 532,
  "matched_count": 1,
  "results": [
    {
      "overall_rank": 149,
      "search_rank": 1,
      "candidate": { "target_type": "LAND", "1_site_info": { "...": "..." }, "...": "..." }
    }
  ]
}
```

## 데이터 범위

- 학습: 전국 설치·미설치 태양광 데이터
- 외부 Test: 충남·충북·대전·세종
- 토지형 Test 후보: 532건
- 건물형 Test 후보: 60건

## 참고

모델 파일과 대용량 CSV는 저장소 용량을 고려해 `.gitignore`로 제외하고 별도로 관리하는 것을 권장합니다.
