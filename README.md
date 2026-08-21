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
태양광_RuleBase_조건.xlsx
```

## 1. Rule-based 검토 실행

### 필요한 파일

토지형:

```text
Solar_RuleBase_Review_Aligned.ipynb
태양광_RuleBase_조건.xlsx
Land_Test_Chungcheong_Uninstalled.csv
```

건물형:

```text
Solar_RuleBase_Review_Aligned.ipynb
태양광_RuleBase_조건.xlsx
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
RULE_XLSX_FILENAME = "태양광_RuleBase_조건.xlsx"
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

## 데이터 범위

- 학습: 전국 설치·미설치 태양광 데이터
- 외부 Test: 충남·충북·대전·세종
- 토지형 Test 후보: 532건
- 건물형 Test 후보: 60건

## 참고

모델 파일과 대용량 CSV는 저장소 용량을 고려해 `.gitignore`로 제외하고 별도로 관리하는 것을 권장합니다.


## 부록

## ML 모델

### 모델 구성

후보지 유형에 따라 서로 다른 모델 번들을 사용합니다.

| 구분 | 토지형 모델 | 건물형 모델 |
|---|---|---|
| 모델 파일 | `Land_model_bundle.pkl` | `Building_model_bundle.pkl` |
| 적용 대상 | 유휴 토지, 나대지, 농지, 주차장 등 | 공공건축물 및 건물 지붕 |
| 기본 Feature | 일사량·기상·지형 | 일사량·기상·전력계통 |
| 제외 Feature | 없음 | `slope_avg`, `slope_dir`, `elevation_avg`, `Hillshade`, `Southness` |
| 예측값 | 태양광 설치 적합 확률 | 태양광 설치 적합 확률 |
| 최종 알고리즘 | LightGBM | LightGBM |

### 학습 코드

모델 학습 코드는 [`LightGBM model - training code`](./LightGBM%20model%20-%20training%20code/) 폴더에서 확인할 수 있습니다.

