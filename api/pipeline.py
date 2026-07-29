"""Common_Test_Ranking_Aligned.ipynb 노트북의 예측·랭킹·SHAP 로직을 API에서 재사용할 수 있게 옮긴 모듈."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from .config import CANDIDATE_UNIVERSE_PATHS, MODEL_BUNDLE_PATHS

_BUNDLE_CACHE: dict[str, dict[str, Any]] = {}
_UNIVERSE_CACHE: dict[str, "RankingResult"] = {}

ADDRESS_COLUMNS = ["address_ml", "주소", "소재지", "도로명주소", "지번주소"]


class PipelineError(ValueError):
    """입력 데이터나 설정이 잘못되어 API가 4xx로 응답해야 하는 경우."""


def load_bundle(dataset_type: str) -> dict[str, Any]:
    dataset_type = dataset_type.lower()
    if dataset_type not in MODEL_BUNDLE_PATHS:
        raise PipelineError(f"dataset_type은 'land' 또는 'building'이어야 합니다: {dataset_type}")

    if dataset_type in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[dataset_type]

    path = MODEL_BUNDLE_PATHS[dataset_type]
    if not path.exists():
        raise PipelineError(f"모델 번들을 찾을 수 없습니다: {path}")

    bundle = joblib.load(path)
    bundle_dataset_type = str(bundle.get("dataset_type", dataset_type)).lower()
    if bundle_dataset_type != dataset_type:
        raise PipelineError(f"번들의 dataset_type({bundle_dataset_type})이 요청({dataset_type})과 다릅니다.")

    _BUNDLE_CACHE[dataset_type] = bundle
    return bundle


def read_table(filename: str, content: bytes) -> pd.DataFrame:
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    buffer = io.BytesIO(content)

    if suffix in ("xlsx", "xls"):
        return pd.read_excel(buffer)

    if suffix == "csv":
        for encoding in ("utf-8-sig", "cp949"):
            buffer.seek(0)
            try:
                return pd.read_csv(buffer, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise PipelineError("CSV 인코딩을 인식할 수 없습니다 (utf-8-sig / cp949 모두 실패).")

    raise PipelineError("업로드 파일은 csv, xlsx 또는 xls 형식이어야 합니다.")


def predict_and_score(bundle: dict[str, Any], test_df: pd.DataFrame, *, filter_rule_excluded: bool) -> pd.DataFrame:
    feature_columns = bundle["feature_columns"]
    train_medians = pd.Series(bundle["train_medians"])
    model = bundle["model"]

    test_df = test_df.copy()

    if filter_rule_excluded and "Rule_Pass_For_Next_Step" in test_df.columns:
        rule_pass_mask = (
            test_df["Rule_Pass_For_Next_Step"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )
        test_df = test_df[rule_pass_mask].copy().reset_index(drop=True)

    if test_df.empty:
        raise PipelineError("Rule 필터 이후 남은 후보지가 없습니다.")

    missing_features = [col for col in feature_columns if col not in test_df.columns]
    if missing_features:
        raise PipelineError(f"업로드 데이터에 모델 Feature가 없습니다: {missing_features}")

    for col in feature_columns:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

    test_df = test_df.reset_index(drop=True)
    test_df["test_row_id"] = np.arange(len(test_df))

    X_test = test_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(train_medians)

    remaining_missing = X_test.columns[X_test.isna().any()].tolist()
    if remaining_missing:
        raise PipelineError(f"Train 중앙값 적용 후에도 결측치가 남았습니다: {remaining_missing}")

    probability = model.predict_proba(X_test)[:, 1]

    scored_test = test_df.copy()
    scored_test["Solar_Readiness_Probability"] = probability
    scored_test["ML_Score"] = (scored_test["Solar_Readiness_Probability"] * 100).round(4)

    return scored_test


def calculate_policy_score(data: pd.DataFrame, weight_config: dict, train_medians: pd.Series) -> pd.Series:
    if not weight_config:
        return pd.Series(0.0, index=data.index, dtype=float)

    total_weight = sum(config["weight"] for config in weight_config.values())
    if total_weight <= 0:
        raise PipelineError("정책 Feature 가중치 합은 0보다 커야 합니다.")

    policy_score = pd.Series(0.0, index=data.index, dtype=float)

    for column, config in weight_config.items():
        if column not in data.columns:
            raise PipelineError(f"정책 가중치 컬럼이 없습니다: {column}")

        values = pd.to_numeric(data[column], errors="coerce")
        fallback = train_medians.get(column, values.median())
        values = values.fillna(fallback)

        normalized_score = values.rank(pct=True, method="average")

        direction = config["direction"]
        if direction == "lower":
            normalized_score = 1 - normalized_score
        elif direction != "higher":
            raise PipelineError(f"{column} direction은 'higher' 또는 'lower'여야 합니다.")

        normalized_weight = config["weight"] / total_weight
        data[f"{column}_Policy_Score"] = normalized_score
        policy_score += normalized_score * normalized_weight

    return policy_score


def build_ranking(
    scored_test: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    rank_filter_mode: str,
    ml_weight: float,
    policy_weight: float,
    policy_weight_config: dict,
    create_region_ranks: bool,
) -> pd.DataFrame:
    target_column = bundle.get("target_column", "label")

    if rank_filter_mode == "label_0":
        if target_column not in scored_test.columns:
            raise PipelineError("label_0 모드는 데이터에 label 컬럼이 필요합니다.")
        candidate_ranking = scored_test[pd.to_numeric(scored_test[target_column], errors="coerce") == 0].copy()
    elif rank_filter_mode == "all":
        candidate_ranking = scored_test.copy()
    else:
        raise PipelineError("rank_filter_mode는 'label_0' 또는 'all'이어야 합니다.")

    if candidate_ranking.empty:
        raise PipelineError("랭킹 대상 후보지가 없습니다.")

    train_medians = pd.Series(bundle["train_medians"])

    candidate_ranking["Policy_Feature_Score"] = calculate_policy_score(
        candidate_ranking, policy_weight_config, train_medians
    )

    candidate_ranking["Final_Readiness_Probability"] = (
        ml_weight * candidate_ranking["Solar_Readiness_Probability"]
        + policy_weight * candidate_ranking["Policy_Feature_Score"]
    )

    candidate_ranking["Solar_Readiness_Score"] = (candidate_ranking["Final_Readiness_Probability"] * 100).round(4)

    candidate_ranking["Candidate_Rank"] = (
        candidate_ranking["Solar_Readiness_Score"].rank(method="min", ascending=False).astype(int)
    )

    candidate_ranking["Score_Percentile"] = candidate_ranking["Solar_Readiness_Score"].rank(pct=True, ascending=True)

    candidate_ranking["Solar_Readiness_Grade"] = pd.cut(
        candidate_ranking["Score_Percentile"],
        bins=[0, 0.50, 0.80, 1.00],
        labels=["C", "B", "A"],
        include_lowest=True,
    ).astype(str)

    if create_region_ranks:
        if "시도" in candidate_ranking.columns:
            candidate_ranking["Province_Rank"] = (
                candidate_ranking.groupby("시도", dropna=False)["Solar_Readiness_Score"]
                .rank(method="min", ascending=False)
                .astype(int)
            )

        if "시도" in candidate_ranking.columns and "시군구" in candidate_ranking.columns:
            candidate_ranking["Local_Rank"] = (
                candidate_ranking.groupby(["시도", "시군구"], dropna=False)["Solar_Readiness_Score"]
                .rank(method="min", ascending=False)
                .astype(int)
            )

    candidate_ranking = candidate_ranking.sort_values(
        ["Solar_Readiness_Score", "Candidate_Rank"], ascending=[False, True]
    ).reset_index(drop=True)

    return candidate_ranking


def _extract_positive_class_values(shap_result) -> np.ndarray:
    values = shap_result.values

    if isinstance(values, list):
        values = values[-1]

    values = np.asarray(values)

    if values.ndim == 3:
        values = values[:, :, 1]

    return values


def _build_reason_sentence(feature_name: str, feature_value: float, percentile: float, shap_value: float) -> str:
    return (
        f"{feature_name} 값이 {feature_value:.3f}이며 "
        f"현재 후보지 기준 {percentile:.1f}백분위입니다. "
        f"이 값은 모델의 설치 가능성 점수를 높인 주요 요인으로 "
        f"분석되었습니다 (SHAP {shap_value:+.4f})."
    )


def compute_shap_details(
    scored_test: pd.DataFrame,
    candidate_ranking: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    shap_top_n: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = bundle["feature_columns"]
    feature_korean_names = bundle.get("feature_korean_names", {})
    train_medians = pd.Series(bundle["train_medians"])
    model = bundle["model"]

    top_n = min(shap_top_n, len(candidate_ranking))
    top_candidates = candidate_ranking.head(top_n).copy()

    top_test_row_ids = top_candidates["test_row_id"].astype(int).tolist()

    X_top = (
        scored_test.set_index("test_row_id")
        .loc[top_test_row_ids, feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
    )

    explainer = shap.TreeExplainer(model)
    shap_result = explainer(X_top)
    shap_values = _extract_positive_class_values(shap_result)

    reference_test_row_ids = candidate_ranking["test_row_id"].astype(int).tolist()

    X_reference = (
        scored_test.set_index("test_row_id")
        .loc[reference_test_row_ids, feature_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(train_medians)
    )

    percentile_reference = X_reference.rank(pct=True) * 100

    reason_rows = []

    for row_position in range(len(top_candidates)):
        candidate = top_candidates.iloc[row_position]
        test_row_id = int(candidate["test_row_id"])

        one_candidate = pd.DataFrame(
            {
                "Feature": feature_columns,
                "Feature_Value": X_top.iloc[row_position].values,
                "SHAP_Value": shap_values[row_position],
            }
        )

        one_candidate = (
            one_candidate[one_candidate["SHAP_Value"] > 0]
            .assign(Abs_SHAP=lambda data: data["SHAP_Value"].abs())
            .sort_values("Abs_SHAP", ascending=False)
            .head(3)
            .copy()
        )

        for reason_rank, (_, reason) in enumerate(one_candidate.iterrows(), start=1):
            feature = reason["Feature"]
            feature_name = feature_korean_names.get(feature, feature)
            percentile = float(percentile_reference.loc[test_row_id, feature])

            reason_rows.append(
                {
                    "test_row_id": test_row_id,
                    "Candidate_Rank": int(candidate["Candidate_Rank"]),
                    "Reason_Rank": reason_rank,
                    "Feature": feature,
                    "Feature_Korean": feature_name,
                    "Feature_Value": float(reason["Feature_Value"]),
                    "SHAP_Value": float(reason["SHAP_Value"]),
                    "Percentile": percentile,
                    "Reason_Text": _build_reason_sentence(
                        feature_name=feature_name,
                        feature_value=float(reason["Feature_Value"]),
                        percentile=percentile,
                        shap_value=float(reason["SHAP_Value"]),
                    ),
                }
            )

    candidate_shap_details = pd.DataFrame(reason_rows)

    if not candidate_shap_details.empty:
        feature_wide = (
            candidate_shap_details.pivot(index="test_row_id", columns="Reason_Rank", values="Feature_Korean")
            .rename(columns={1: "추천요인_1", 2: "추천요인_2", 3: "추천요인_3"})
            .reset_index()
        )

        reason_wide = (
            candidate_shap_details.pivot(index="test_row_id", columns="Reason_Rank", values="Reason_Text")
            .rename(columns={1: "추천이유_1", 2: "추천이유_2", 3: "추천이유_3"})
            .reset_index()
        )

        candidate_ranking_with_shap = candidate_ranking.merge(feature_wide, on="test_row_id", how="left").merge(
            reason_wide, on="test_row_id", how="left"
        )
    else:
        candidate_ranking_with_shap = candidate_ranking.copy()

    for number in [1, 2, 3]:
        for prefix in ["추천요인", "추천이유"]:
            column = f"{prefix}_{number}"
            if column not in candidate_ranking_with_shap.columns:
                candidate_ranking_with_shap[column] = pd.NA

    return candidate_ranking_with_shap, candidate_shap_details


def _is_valid_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def _safe_value(value: Any, default: Any = None) -> Any:
    if not _is_valid_value(value):
        return default
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _safe_float(value: Any, digits: int = 2, default: Any = None) -> float | None:
    if not _is_valid_value(value):
        return default
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: Any = None) -> int | None:
    if not _is_valid_value(value):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _first_available_value(row: pd.Series, candidate_columns: list[str], default: Any = None) -> Any:
    for column in candidate_columns:
        if column in row.index and _is_valid_value(row[column]):
            return _safe_value(row[column])
    return default


def _degree_to_direction(value: Any) -> str | None:
    degree = _safe_float(value)
    if degree is None:
        return None

    directions = ["북향", "북동향", "동향", "남동향", "남향", "남서향", "서향", "북서향"]
    index = int(((degree + 22.5) % 360) // 45)
    return directions[index]


def _make_status(grade: Any) -> str:
    grade = str(grade)
    if grade in ["A", "B"]:
        return "통과"
    if grade == "C":
        return "재검토"
    return "검토 필요"


def _make_grid_description(row: pd.Series) -> str | None:
    substation_distance = _safe_float(row.get("distance_to_substation_km"))
    powerline_distance = _safe_float(row.get("distance_to_powerline_km"))

    descriptions = []
    if substation_distance is not None:
        descriptions.append(f"최근접 변전소 약 {substation_distance:.2f}km")
    if powerline_distance is not None:
        descriptions.append(f"최근접 전력선 약 {powerline_distance:.2f}km")

    if not descriptions:
        return None

    return ", ".join(descriptions) + " 거리로 분석됨 (공개 공간정보 기반 접근성)"


def _collect_text_values(row: pd.Series, columns: list[str]) -> list[str]:
    result = []
    for column in columns:
        if column in row.index and _is_valid_value(row[column]):
            text = str(row[column]).strip()
            if text:
                result.append(text)
    return result


def _get_type_settings(dataset_type: str) -> dict[str, Any]:
    if dataset_type == "building":
        return {
            "target_type": "BUILDING",
            "id_prefix": "BUILDING",
            "default_space_type": "건물형 후보지",
            "checklist": [
                {
                    "item": "지붕 구조안전성 및 적재하중 확인",
                    "note": "구조검토를 통해 태양광 모듈·구조물 하중 수용 가능 여부 확인",
                },
                {
                    "item": "옥상 방수 상태 및 보수 필요 여부",
                    "note": "누수 이력과 방수층 수명을 확인하고 필요 시 설치 전 보수",
                },
                {
                    "item": "옥상 장애물·설비 및 피난동선 확인",
                    "note": "냉난방기, 물탱크, 피뢰설비와 소방 피난동선 간섭 여부 확인",
                },
                {
                    "item": "계통연계 및 건축·소방 기준 검토",
                    "note": "한전 계통연계 가능 여부와 건축물·소방 관련 기준 확인",
                },
            ],
        }

    return {
        "target_type": "LAND",
        "id_prefix": "LAND",
        "default_space_type": "토지형 후보지",
        "checklist": [
            {
                "item": "진입로 확보 및 공사차량 진입 가능 여부",
                "note": "현장 진입로 폭과 도로점용 허가 필요 여부 확인",
            },
            {
                "item": "토질 상태 및 배수시설 설치 가능 여부",
                "note": "우수 배제와 토사 유출 가능성 현장 확인",
            },
            {
                "item": "최근접 전력선 및 계통연계 가능 여부",
                "note": "한전 계통 여유 용량과 선로 신설 비용 별도 확인",
            },
            {
                "item": "개발행위허가 및 이격거리 충족 여부",
                "note": "해당 지자체의 현행 조례와 부지 경계를 기준으로 재검토",
            },
        ],
    }


def build_candidate_json(
    row: pd.Series,
    *,
    dataset_type: str,
    policy_weight: float,
    policy_weight_config: dict,
) -> dict[str, Any]:
    type_settings = _get_type_settings(dataset_type)

    ml_score = _safe_float(row.get("ML_Score"), digits=2)
    total_score = _safe_float(row.get("Solar_Readiness_Score"), digits=2)

    policy_score = None
    if policy_weight > 0 and policy_weight_config:
        policy_raw = _safe_float(row.get("Policy_Feature_Score"), digits=4)
        if policy_raw is not None:
            policy_score = round(policy_raw * 100, 2)

    priority_rank = _first_available_value(row, ["Local_Rank", "Province_Rank", "Candidate_Rank"])
    if priority_rank is not None:
        priority_rank = str(_safe_int(priority_rank))

    bonus_reasons = _collect_text_values(row, ["추천이유_1", "추천이유_2", "추천이유_3"])
    penalty_reasons = _collect_text_values(row, ["감점이유_1", "감점이유_2", "감점이유_3"])

    if bonus_reasons:
        ml_reason = bonus_reasons[0]
    else:
        probability = _safe_float(row.get("Solar_Readiness_Probability"), digits=4)
        ml_reason = (
            f"저장된 ML 모델이 산출한 설치사례 유사 확률은 {probability * 100:.2f}%입니다."
            if probability is not None
            else None
        )

    rule_reason = _first_available_value(row, ["Rule_Final_Message", "rule_final_message"])
    if rule_reason is None:
        rule_reason = (
            "사용자가 설정한 정책 Feature 가중치를 최종 점수에 반영함"
            if policy_score is not None
            else "Rule-based 검토 결과가 연결되지 않음"
        )

    site_id = _first_available_value(
        row,
        ["source_id_ml", "후보ID", "site_id"],
        default=f"{type_settings['id_prefix']}_{_safe_int(row.name, 0):05d}",
    )

    address = _first_available_value(row, ["address_ml", "주소", "소재지", "도로명주소", "지번주소"])

    site_name = _first_available_value(row, ["site_name", "재산명", "시설명", "자산명", "발전소명", "건물명"])
    if site_name is None:
        site_name = f"{address} 태양광 후보지" if address else f"{site_id} 태양광 후보지"

    space_type = _first_available_value(
        row,
        ["space_type", "자산구분_ML", "설치구분", "재산구분", "지목", "건물용도"],
        default=type_settings["default_space_type"],
    )

    total_area = _safe_float(
        _first_available_value(
            row, ["total_area", "전체면적", "토지면적", "연면적", "옥상면적", "면적", "공유재산면적", "대지면적"]
        )
    )

    available_area = _safe_float(
        _first_available_value(row, ["available_area", "설치가능면적", "가용면적", "유효면적", "옥상가용면적"])
    )

    availability_rate = None
    if total_area is not None and available_area is not None and total_area > 0:
        availability_rate = round(available_area / total_area * 100, 2)

    owner_agency = _first_available_value(row, ["owner_agency", "소유기관", "관리기관", "기관명", "소관기관"])

    grade = _safe_value(row.get("Solar_Readiness_Grade"))

    if dataset_type == "land":
        slope_degree = _safe_float(row.get("slope_avg"), digits=2)
        aspect_direction = _degree_to_direction(row.get("slope_dir"))
    else:
        slope_degree = None
        aspect_direction = None

    return {
        "target_type": type_settings["target_type"],
        "1_site_info": {
            "site_id": str(site_id),
            "site_name": site_name,
            "address": address,
            "space_type": space_type,
            "total_area": total_area,
            "available_area": available_area,
            "availability_rate_percent": availability_rate,
            "owner_agency": owner_agency,
            "created_at": datetime.now().strftime("%Y년 %m월 %d일"),
        },
        "2_scores_and_evaluation": {
            "grade": str(grade) if grade is not None else None,
            "total_score": total_score,
            "priority_rank": priority_rank,
            "status": _make_status(grade),
            "detail_scores": {
                "ml_technical_score": ml_score,
                "ml_reason": ml_reason,
                "vision_ai_score": _safe_float(
                    _first_available_value(row, ["Vision_Score", "vision_score"]), digits=2
                ),
                "vision_reason": _first_available_value(
                    row,
                    ["Vision_Final_Message", "vision_reason"],
                    default="Vision AI 분석 결과가 아직 연결되지 않음",
                ),
                "rule_based_score": policy_score,
                "rule_reason": rule_reason,
            },
            "xai_explanation": {
                "bonus_reason": bonus_reasons,
                "penalty_reason": penalty_reasons,
            },
        },
        "3_vision_ai_simulation": {
            "vision_analysis": {
                "slope_degree": slope_degree,
                "aspect_direction": aspect_direction,
                "vegetation_coverage_percent": None,
                "has_access_road": None,
                "access_road_width_m": None,
                "recommended_orientation": None,
                "recommended_tilt_angle_deg": None,
            },
            "simulation": {
                "recommended_capacity_kw": None,
                "annual_generation_kwh": None,
                "annual_revenue_krw": None,
                "roi_percent": None,
                "payback_years": None,
            },
        },
        "4_risk_and_support": {
            "rule_based_risk_check": {
                "grid_connection": _make_grid_description(row),
                "regulation": _first_available_value(row, ["Rule_Final_Message", "rule_final_message"]),
                "public_complaint": None,
            },
            "recommended_subsidies": [],
        },
        "5_pre_investigation_checklist": type_settings["checklist"],
    }


@dataclass
class RankOptions:
    filter_rule_excluded: bool = True
    rank_filter_mode: str = "all"
    create_region_ranks: bool = True
    include_shap: bool = True
    shap_top_n: int = 1000
    ml_weight: float = 1.0
    policy_weight: float = 0.0
    policy_weight_config: dict = None
    top_n_json: int = 20

    def __post_init__(self):
        if self.policy_weight_config is None:
            self.policy_weight_config = {}


@dataclass
class RankingResult:
    dataset_type: str
    model_name: str
    input_rows: int
    ranked_rows: int
    scored_test: pd.DataFrame
    candidate_ranking: pd.DataFrame
    candidate_ranking_with_shap: pd.DataFrame
    candidate_shap_details: pd.DataFrame
    top_candidates_json: list[dict]


def run_pipeline(dataset_type: str, filename: str, content: bytes, options: RankOptions) -> RankingResult:
    dataset_type = dataset_type.lower()
    bundle = load_bundle(dataset_type)
    raw_df = read_table(filename, content)
    input_rows = len(raw_df)

    scored_test = predict_and_score(bundle, raw_df, filter_rule_excluded=options.filter_rule_excluded)

    candidate_ranking = build_ranking(
        scored_test,
        bundle,
        rank_filter_mode=options.rank_filter_mode,
        ml_weight=options.ml_weight,
        policy_weight=options.policy_weight,
        policy_weight_config=options.policy_weight_config,
        create_region_ranks=options.create_region_ranks,
    )

    if options.include_shap:
        candidate_ranking_with_shap, candidate_shap_details = compute_shap_details(
            scored_test, candidate_ranking, bundle, shap_top_n=options.shap_top_n
        )
    else:
        candidate_ranking_with_shap = candidate_ranking.copy()
        candidate_shap_details = pd.DataFrame()

    top_source = candidate_ranking_with_shap if options.include_shap else candidate_ranking
    top_df = top_source.sort_values(
        ["Solar_Readiness_Score", "Candidate_Rank"], ascending=[False, True]
    ).head(options.top_n_json)

    top_candidates_json = [
        build_candidate_json(
            row,
            dataset_type=dataset_type,
            policy_weight=options.policy_weight,
            policy_weight_config=options.policy_weight_config,
        )
        for _, row in top_df.iterrows()
    ]

    return RankingResult(
        dataset_type=dataset_type,
        model_name=str(bundle.get("model_name", "")),
        input_rows=input_rows,
        ranked_rows=len(candidate_ranking),
        scored_test=scored_test,
        candidate_ranking=candidate_ranking,
        candidate_ranking_with_shap=candidate_ranking_with_shap,
        candidate_shap_details=candidate_shap_details,
        top_candidates_json=top_candidates_json,
    )


def _find_address_column(df: pd.DataFrame) -> str | None:
    for column in ADDRESS_COLUMNS:
        if column in df.columns:
            return column
    return None


def get_universe_ranking(dataset_type: str, *, force_refresh: bool = False) -> RankingResult:
    """Rule_base/*_RulePassed.csv 전체를 기본 옵션으로 예측·랭킹·SHAP 계산한 결과를 캐싱해 반환."""
    dataset_type = dataset_type.lower()
    if dataset_type not in CANDIDATE_UNIVERSE_PATHS:
        raise PipelineError(f"dataset_type은 'land' 또는 'building'이어야 합니다: {dataset_type}")

    if not force_refresh and dataset_type in _UNIVERSE_CACHE:
        return _UNIVERSE_CACHE[dataset_type]

    path = CANDIDATE_UNIVERSE_PATHS[dataset_type]
    if not path.exists():
        raise PipelineError(f"후보지 데이터셋을 찾을 수 없습니다: {path}")

    result = run_pipeline(dataset_type, path.name, path.read_bytes(), RankOptions())
    _UNIVERSE_CACHE[dataset_type] = result
    return result


def search_candidates(dataset_type: str, query: str, *, top_n: int = 20) -> dict[str, Any]:
    """주소 검색어로 후보지를 필터링하고, 전체 순위(Candidate_Rank)와 검색결과 내 순위를 함께 반환."""
    tokens = [token for token in query.strip().split() if token]
    if not tokens:
        raise PipelineError("검색어를 입력해 주세요.")

    result = get_universe_ranking(dataset_type)
    df = result.candidate_ranking_with_shap

    address_column = _find_address_column(df)
    if address_column is None:
        raise PipelineError("후보지 데이터에 주소 컬럼이 없습니다.")

    address_series = df[address_column].astype(str)
    mask = pd.Series(True, index=df.index)
    for token in tokens:
        mask &= address_series.str.contains(token, case=False, na=False, regex=False)

    matched = df[mask].copy()

    if matched.empty:
        return {
            "dataset_type": dataset_type,
            "query": query,
            "universe_size": len(df),
            "matched_count": 0,
            "results": [],
        }

    matched["Search_Rank"] = matched["Solar_Readiness_Score"].rank(method="min", ascending=False).astype(int)
    matched = matched.sort_values("Search_Rank")

    top_matched = matched.head(top_n)

    results = []
    for _, row in top_matched.iterrows():
        candidate_json = build_candidate_json(
            row,
            dataset_type=dataset_type,
            policy_weight=0.0,
            policy_weight_config={},
        )
        results.append(
            {
                "overall_rank": int(row["Candidate_Rank"]),
                "search_rank": int(row["Search_Rank"]),
                "candidate": candidate_json,
            }
        )

    return {
        "dataset_type": dataset_type,
        "query": query,
        "universe_size": len(df),
        "matched_count": len(matched),
        "results": results,
    }
