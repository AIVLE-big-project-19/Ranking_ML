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
INTEGRATED_ML_WEIGHT = 0.85
INTEGRATED_AREA_WEIGHT = 0.15
AREA_PER_KW_M2 = {"building": 7.5, "land": 10.0}
DEFAULT_SPECIFIC_YIELD_KWH_PER_KW_YEAR = 1200.0
ENERGY_VALUE_KRW_PER_KWH = None
