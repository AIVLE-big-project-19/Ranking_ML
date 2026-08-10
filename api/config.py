from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_BUNDLE_PATHS = {
    "land": BASE_DIR / "Land_model_bundle.pkl",
    "building": BASE_DIR / "Building_model_bundle.pkl",
}

CANDIDATE_UNIVERSE_PATHS = {
    "land": BASE_DIR / "Rule_base" / "Land_Test_Chungcheong_RulePassed.csv",
    "building": BASE_DIR / "Rule_base" / "Building_Test_Chungcheong_RulePassed.csv",
}

# Vision → Rule → ML 통합 분석 리소스
MERGED_TEST_DATA_PATH = BASE_DIR / "Merged_Test_Data.csv"
RULE_XLSX_PATH = BASE_DIR / "태양광_RuleBase_조건.xlsx"
RULE_SHEET_NAME = "Rule_Data"

VISION_MIN_CONFIDENCE = 0.10
MAX_MATCH_DISTANCE_M = 300.0

INTEGRATED_ML_WEIGHT = 0.50
INTEGRATED_AREA_WEIGHT = 0.50


# 사전검토용 공통 대표 태양광 모듈
PANEL_SPEC = {
    "reference_product": "Qcells Q.TRON XL-G2.13/BFG",
    "power_w": 640.0,
    "length_m": 2.465,
    "width_m": 1.134,
    "area_m2": 2.7953,
}


# Vision의 패널 수가 없을 때만 사용하는 면적 기반 fallback
AREA_PER_KW_M2 = {
    "building": 6.2,
    "land": 8.7,
}


# PVOUT이 없을 때만 사용하는 기본 연간 단위발전량
DEFAULT_SPECIFIC_YIELD_KWH_PER_KW_YEAR = 1200.0

# 수익 단가 미설정
ENERGY_VALUE_KRW_PER_KWH = None
