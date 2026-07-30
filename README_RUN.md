# Vision POST → Rule/ML 통합 API 실행

## 1. 프로젝트 배치

```text
project/
├─ api/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ pipeline.py
│  └─ config.py
├─ Rule_base/
│  ├─ Land_Test_Chungcheong_RulePassed.csv
│  └─ Building_Test_Chungcheong_RulePassed.csv
├─ Merged_Test_Data.csv
├─ 태양광_RuleBase_조건.xlsx
├─ Land_model_bundle.pkl
├─ Building_model_bundle.pkl
├─ requirements.txt
└─ test_vision_post.py
```

`Merged_Test_Data.csv`에는 `source_id_ml`, `address_ml`, `longitude`, `latitude`와 모델 Feature가 필요합니다.

## 2. 설치

Windows PowerShell 기준:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 서버 실행

프로젝트 최상위 폴더에서:

```powershell
uvicorn api.main:app --reload --port 8002
```

Swagger:

```text
http://127.0.0.1:8002/docs
```

## 4. Vision JSON 직접 POST 테스트

새 터미널에서:

```powershell
.venv\Scripts\activate
python test_vision_post.py
```

호출 엔드포인트:

```text
POST http://127.0.0.1:8002/analyze/vision-json
```

## 5. 좌표 처리 방식

Vision payload에 `longitude`, `latitude`가 없으면 `polygon`의 EPSG:3857 좌표를 사용합니다.

- 유효한 점이 2개: 두 점 중점 계산
- 유효한 점이 3개 이상: 전체 점 평균 계산
- 대표점을 `pyproj.Transformer`로 EPSG:3857 → EPSG:4326 변환
- `Merged_Test_Data.csv`의 longitude/latitude와 Haversine 거리 계산
- `config.py`의 `MAX_MATCH_DISTANCE_M` 이내 최근접 후보만 매칭

예시 입력:

```json
[
  {
    "candidate_type": "land",
    "confidence": 0.9231,
    "polygon": [
      [14135200.5, 4518750.2],
      [14135220.8, 4518761.4]
    ],
    "pixel_area": 1223.5,
    "real_area": 186.69,
    "distance_to_road_px": 18.42,
    "distance_to_building_px": 35.71,
    "distance_to_road_m": 14.02,
    "distance_to_building_m": 27.19,
    "model_version": "solar-yolov8-seg-v2"
  }
]
```

매칭 실패 시 `MAX_MATCH_DISTANCE_M`를 무작정 크게 늘리지 말고 CSV 좌표 및 polygon 좌표계를 먼저 확인하세요.
