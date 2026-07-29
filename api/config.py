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
