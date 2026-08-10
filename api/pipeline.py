"""Common_Test_Ranking_Aligned.ipynb 노트북의 예측·랭킹·SHAP 로직을 API에서 재사용할 수 있게 옮긴 모듈."""

from __future__ import annotations

import json
import re
import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from .config import CANDIDATE_UNIVERSE_PATHS, MODEL_BUNDLE_PATHS, PANEL_SPEC

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


# ============================================================
# Vision JSON → 후보 매칭 → Rule → 유형별 ML → 통합 랭킹
# ============================================================
from pathlib import Path
from pyproj import Transformer

from .config import (
    MERGED_TEST_DATA_PATH, RULE_XLSX_PATH, RULE_SHEET_NAME,
    VISION_MIN_CONFIDENCE, MAX_MATCH_DISTANCE_M,
    INTEGRATED_ML_WEIGHT as ML_WEIGHT,
    INTEGRATED_AREA_WEIGHT as AREA_WEIGHT,
    AREA_PER_KW_M2, DEFAULT_SPECIFIC_YIELD_KWH_PER_KW_YEAR,
    ENERGY_VALUE_KRW_PER_KWH,
)

MERGED_CSV_PATH = MERGED_TEST_DATA_PATH
LAND_MODEL_PATH = MODEL_BUNDLE_PATHS["land"]
BUILDING_MODEL_PATH = MODEL_BUNDLE_PATHS["building"]

# 태양광 사업성 단순 추정용 기준값
# Vision AI가 패널 수를 제공하지 못한 경우, 가용면적을 설비용량으로 환산할 때 사용합니다.
BUSINESS_AREA_PER_KW_M2 = {
    'land': 10.0,      # 토지형: 통로·배치 간격을 포함해 1kW당 10㎡ 적용
    'building': 7.0,   # 건물형: 평지붕 기준 1kW당 7㎡ 적용
}

# 발전 전량을 판매한다고 가정한 평균 전력 가치입니다.
BUSINESS_ENERGY_VALUE_KRW_PER_KWH = 160.0

# 유형별 초기 설치비 단가입니다. 주차장은 캐노피 구조물 비용을 고려해 별도 적용합니다.
CAPEX_PER_KW_KRW = {
    'land': 1_200_000.0,
    'building': 1_300_000.0,
    'parking_lot': 1_500_000.0,
}

# 연간 유지관리비를 초기 투자비의 1.5%로 단순 추정합니다.
ANNUAL_OPEX_RATE = 0.015

# 화면과 보고서에 계산 범위를 명시하기 위한 설명입니다.
BUSINESS_ESTIMATE_SCOPE = '보조금·부가세·금융비용·계통보강비 제외 단순 추정'

@dataclass
class IntegratedResult:
    summary: dict[str, Any]
    reviewed_df: pd.DataFrame
    passed_df: pd.DataFrame
    ranking_df: pd.DataFrame
    results: list[dict[str, Any]]
    errors: list[dict[str, Any]]

EXCLUDE_DECISIONS = {'EXCLUDE', 'EXCLUDE_CANDIDATE'}
DECISION_PRIORITY = {'EXCLUDE': 100, 'EXCLUDE_CANDIDATE': 95, 'LEGAL_REVIEW': 70, 'UNKNOWN': 60, 'PASS_EXCEPTION': 30, 'PASS': 10, 'NO_APPLICABLE_RULE': 0}

def iv_read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ['utf-8-sig', 'cp949', 'utf-8']:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f'CSV 인코딩을 읽지 못했습니다: {path}')

def iv_to_optional_float(value: Any) -> float | None:
    try:
        number = float(value)
        if np.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None

# vision 으로 받아온 값의 추천 전력양 구하는 함수
# 추천용량 500kW가 100점.
VISION_CAPACITY_FULL_SCORE_KW = 500.0

def iv_calculate_recommended_capacity_kw(row) -> float:
    model_type = str(
        row.get('model_type', row.get('Model_Type', 'land')) or 'land'
    ).strip().lower()

    try:
        estimated_panel_count = float(row.get('estimated_panel_count') or 0)
    except (TypeError, ValueError):
        estimated_panel_count = 0.0

    try:
        usable_area = float(row.get('usable_area') or 0)
    except (TypeError, ValueError):
        usable_area = 0.0

    try:
        real_area = float(row.get('real_area') or 0)
    except (TypeError, ValueError):
        real_area = 0.0

    # 1순위: Vision 패널 수
    if estimated_panel_count > 0:
        return (
            int(estimated_panel_count)
            * PANEL_SPEC['power_w']
            / 1000.0
        )

    # 2순위: usable_area, 없으면 real_area
    capacity_base_area = usable_area if usable_area > 0 else real_area

    area_per_kw = BUSINESS_AREA_PER_KW_M2.get(
        model_type,
        BUSINESS_AREA_PER_KW_M2['land']
    )

    if capacity_base_area > 0:
        return capacity_base_area / area_per_kw

    return 0.0


def iv_is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False

def iv_normalize_address(value: Any) -> str:
    if iv_is_missing(value):
        return ''
    return re.sub('[^0-9가-힣a-zA-Z]', '', str(value).lower())

def iv_json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def iv_load_resources():
    merged_df = iv_read_csv_flexible(MERGED_CSV_PATH)
    required_lookup_columns = {'source_id_ml', 'address_ml', 'longitude', 'latitude'}
    missing_lookup_columns = required_lookup_columns - set(merged_df.columns)
    if missing_lookup_columns:
        raise KeyError(f'Merged CSV 조회 필수 컬럼 누락: {sorted(missing_lookup_columns)}')
    merged_df['source_id_ml'] = merged_df['source_id_ml'].astype(str).str.strip()
    merged_df['_address_normalized'] = merged_df['address_ml'].map(iv_normalize_address)
    land_bundle = joblib.load(LAND_MODEL_PATH)
    building_bundle = joblib.load(BUILDING_MODEL_PATH)
    rules_df = pd.read_excel(RULE_XLSX_PATH, sheet_name=RULE_SHEET_NAME)
    if 'enabled' in rules_df.columns:
        enabled_mask = rules_df['enabled'].astype(str).str.strip().str.lower().isin(['true', '1', 'yes', 'y'])
        rules_df = rules_df[enabled_mask].copy()
    rules_df = rules_df.reset_index(drop=True)
    return (merged_df, land_bundle, building_bundle, rules_df)
VISION_NUMERIC_COLUMNS = ['confidence', 'pixel_area', 'real_area', 'distance_to_road_px', 'distance_to_building_px', 'distance_to_road_m', 'distance_to_building_m']

def iv_normalize_detection(raw: dict[str, Any]) -> dict[str, Any]:
    detection = dict(raw)
    detection['candidate_type'] = str(detection.get('candidate_type', '')).strip().lower()
    for column in VISION_NUMERIC_COLUMNS:
        detection[column] = iv_to_optional_float(detection.get(column))
    polygon = detection.get('polygon')
    detection['polygon'] = polygon if isinstance(polygon, list) else []
    return detection

def iv_unpack_payload(payload: Any) -> list[dict[str, Any]]:
    """
    지원 형식

    1. {"items": [{...}, {...}]}
    2. {"source_id_ml": ..., "detections": [...]}
    3. [{wrapper}, {wrapper}]
    4. detection 배열
    """
    if isinstance(payload, dict):
        if 'items' in payload:
            return payload['items']
        if 'detections' in payload:
            return [payload]
        if 'predictions' in payload:
            predictions = payload.get('predictions', [])

            first_prediction = (
                predictions[0]
                if predictions
                and isinstance(predictions[0], dict)
                else {}
            )

            return [{
                'source_id_ml': (
                    payload.get('source_id_ml')
                    or first_prediction.get('source_id_ml')
                    or first_prediction.get('candidate_id')
                ),
                'address': (
                    payload.get('address')
                    or payload.get('address_ml')
                    or first_prediction.get('address')
                    or first_prediction.get('address_ml')
                ),
                'longitude': (
                    payload.get('longitude')
                    if payload.get('longitude') is not None
                    else first_prediction.get('longitude')
                ),
                'latitude': (
                    payload.get('latitude')
                    if payload.get('latitude') is not None
                    else first_prediction.get('latitude')
                ),
                'detections': predictions,
            }]

        return [{
            'source_id_ml': payload.get('source_id_ml'),
            'address': payload.get('address') or payload.get('address_ml'),
            'longitude': payload.get('longitude'),
            'latitude': payload.get('latitude'),
            'detections': [payload],
        }]
        # return [{'source_id_ml': payload.get('source_id_ml'), 'address': payload.get('address') or payload.get('address_ml'), 'longitude': payload.get('longitude'), 'latitude': payload.get('latitude'), 'detections': [payload]}]
    if isinstance(payload, list):
        if not payload:
            return []
        if all((isinstance(item, dict) and 'detections' in item for item in payload)):
            return payload
        grouped = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_id = item.get('source_id_ml')
            address = item.get('address') or item.get('address_ml')
            longitude = item.get('longitude')
            latitude = item.get('latitude')
            group_key = source_id or address or (f'{longitude},{latitude}' if longitude is not None and latitude is not None else '__single_candidate__')
            grouped.setdefault(group_key, []).append(item)
        wrappers = []
        for group_key, detections in grouped.items():
            first = detections[0]
            wrappers.append({'source_id_ml': first.get('source_id_ml'), 'address': first.get('address') or first.get('address_ml'), 'longitude': first.get('longitude'), 'latitude': first.get('latitude'), 'detections': detections})
        return wrappers
    raise TypeError('Vision JSON 최상위 형식은 dict 또는 list여야 합니다.')

def iv_polygon_centroid(polygon: list) -> tuple[float | None, float | None]:
    """Vision polygon의 대표 EPSG:3857 좌표를 계산합니다.

    - 유효한 점이 정확히 2개이면 두 점의 중점을 사용합니다.
    - 점이 3개 이상이면 전체 꼭짓점 좌표의 평균점을 사용합니다.
    - polygon 면적 계산용 centroid가 아니라 CSV 최근접 후보 조회용 대표점입니다.
    """
    points: list[tuple[float, float]] = []
    for point in polygon or []:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = iv_to_optional_float(point[0])
        y = iv_to_optional_float(point[1])
        if x is not None and y is not None:
            points.append((x, y))

    if len(points) < 2:
        return (None, None)

    if len(points) == 2:
        return (
            float((points[0][0] + points[1][0]) / 2.0),
            float((points[0][1] + points[1][1]) / 2.0),
        )

    return (
        float(np.mean([point[0] for point in points])),
        float(np.mean([point[1] for point in points])),
    )

def iv_aggregate_vision_candidate(wrapper: dict[str, Any]) -> dict[str, Any]:
    detections = [iv_normalize_detection(item) for item in wrapper.get('detections', []) if isinstance(item, dict)]
    detections = [item for item in detections if item.get('confidence') is not None and item['confidence'] >= VISION_MIN_CONFIDENCE]
    if not detections:
        raise ValueError('신뢰도 기준을 통과한 Vision detection이 없습니다.')
    valid_types = {'building', 'parking_lot', 'land'}
    detections = [item for item in detections if item['candidate_type'] in valid_types]
    if not detections:
        raise ValueError('지원되는 candidate_type이 없습니다.')
    area_by_type = {'building': 0.0, 'parking_lot': 0.0, 'land': 0.0}
    for detection in detections:
        area_by_type[detection['candidate_type']] += detection.get('real_area') or 0.0
    total_real_area = sum((detection.get('real_area') or 0.0 for detection in detections))
    total_pixel_area = sum((detection.get('pixel_area') or 0.0 for detection in detections))
    if total_real_area > 0:
        candidate_type = max(area_by_type, key=area_by_type.get)
    else:
        candidate_type = detections[0]['candidate_type']
    model_type = 'building' if candidate_type == 'building' else 'land'

    # 대표 detection:
    # 최종 candidate_type과 같은 detection 중 면적이 가장 큰 항목
    representative_detection = max(
        (
            detection
            for detection in detections
            if detection.get('candidate_type') == candidate_type
        ),
        key=lambda detection: detection.get('real_area') or 0.0,
    )
    all_polygon_points = [
        point
        for detection in detections
        for point in detection.get('polygon', [])
    ]
    polygon_x, polygon_y = iv_polygon_centroid(all_polygon_points)

    def minimum(
        column: str,
        default: float | None = None,
    ) -> float | None:
        values = [
            detection[column]
            for detection in detections
            if detection.get(column) is not None
        ]
        return min(values) if values else default

    road_detected = any(
        detection.get('distance_to_road_m') is not None
        for detection in detections
    )

    building_detected = any(
        detection.get('distance_to_building_m') is not None
        for detection in detections
    )

    return {
        'source_id_ml_json': wrapper.get('source_id_ml'),
        'address_json': wrapper.get('address') or wrapper.get('address_ml'),
        'longitude_json': iv_to_optional_float(wrapper.get('longitude')),
        'latitude_json': iv_to_optional_float(wrapper.get('latitude')),
        'polygon_centroid_x': polygon_x,
        'polygon_centroid_y': polygon_y,
        'candidate_type': candidate_type,
        'model_type': model_type,
        'confidence': max(
            detection['confidence']
            for detection in detections
        ),
        'pixel_area': total_pixel_area,
        'real_area': total_real_area,
        'shape_score': iv_to_optional_float(
            representative_detection.get('shape_score')
        ),
        'shape_grade': representative_detection.get('shape_grade'),
        'shape_efficiency': iv_to_optional_float(
            representative_detection.get('shape_efficiency')
        ),
        'recommended_layout': representative_detection.get(
            'recommended_layout'
        ),

        # 여러 detection이 있으면 가용면적과 패널 수는 합산
        'usable_area': sum(
            iv_to_optional_float(detection.get('usable_area')) or 0.0
            for detection in detections
        ),
        'estimated_panel_count': sum(
            int(iv_to_optional_float(
                detection.get('estimated_panel_count')
            ) or 0)
            for detection in detections
        ),
        'distance_to_road_px': minimum(
            'distance_to_road_px',
            9999.0,
        ),
        'distance_to_building_px': minimum(
            'distance_to_building_px',
            9999.0,
        ),
        'distance_to_road_m': minimum(
            'distance_to_road_m',
            999.0,
        ),
        'distance_to_building_m': minimum(
            'distance_to_building_m',
            999.0,
        ),

        'road_detected': road_detected,
        'building_detected': building_detected,
        'road_distance_missing_pass_applied': not road_detected,
        'building_distance_missing_pass_applied': not building_detected,

        'model_version': (
            wrapper.get('model_version')
            or next(
                (
                    detection.get('model_version')
                    for detection in detections
                    if detection.get('model_version')
                ),
                None,
            )
        ),
        'vision_detection_count': len(detections),
        'vision_detections_json': json.dumps(
            detections,
            ensure_ascii=False,
        ),
    }

EPSG3857_TO_4326 = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)

def iv_haversine_m(lon1: float, lat1: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    earth_radius = 6371000.0
    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius * np.arcsin(np.sqrt(a))

def iv_get_query_lon_lat(vision: dict[str, Any]) -> tuple[float | None, float | None]:
    """Vision 위치를 EPSG:4326 경도·위도로 반환합니다.

    우선순위
    1. Vision payload에 longitude/latitude가 있으면 그대로 사용
    2. 없으면 polygon 대표점(EPSG:3857)을 pyproj로 EPSG:4326 변환
    """
    longitude = vision.get('longitude_json')
    latitude = vision.get('latitude_json')
    if longitude is not None and latitude is not None:
        longitude = float(longitude)
        latitude = float(latitude)
        if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
            raise ValueError(
                f'longitude/latitude 범위가 올바르지 않습니다: {longitude}, {latitude}'
            )
        return (longitude, latitude)

    x = vision.get('polygon_centroid_x')
    y = vision.get('polygon_centroid_y')
    if x is None or y is None:
        return (None, None)

    x = float(x)
    y = float(y)
    longitude, latitude = EPSG3857_TO_4326.transform(x, y)
    return (round(float(longitude), 7), round(float(latitude), 7))

def iv_match_candidate_row(merged_df: pd.DataFrame, vision: dict[str, Any]) -> dict[str, Any]:
    source_id = vision.get('source_id_ml_json')
    if source_id is not None:
        mask = merged_df['source_id_ml'] == str(source_id).strip()
        if mask.any():
            row = merged_df[mask].iloc[0].to_dict()
            row['match_method'] = 'source_id_ml'
            row['match_distance_m'] = 0.0
            return row
    address = iv_normalize_address(vision.get('address_json'))
    if address:
        exact_mask = merged_df['_address_normalized'] == address
        if exact_mask.any():
            row = merged_df[exact_mask].iloc[0].to_dict()
            row['match_method'] = 'address_exact'
            row['match_distance_m'] = 0.0
            return row
        contains_mask = merged_df['_address_normalized'].str.contains(address, regex=False, na=False)
        if contains_mask.any():
            row = merged_df[contains_mask].iloc[0].to_dict()
            row['match_method'] = 'address_partial'
            row['match_distance_m'] = 0.0
            return row
    query_lon, query_lat = iv_get_query_lon_lat(vision)
    if query_lon is None or query_lat is None:
        raise ValueError('source_id_ml·주소 조회 실패 후 사용할 좌표가 없습니다.')
    candidate_lon = pd.to_numeric(merged_df['longitude'], errors='coerce')
    candidate_lat = pd.to_numeric(merged_df['latitude'], errors='coerce')
    valid_mask = candidate_lon.notna() & candidate_lat.notna()
    if not valid_mask.any():
        raise ValueError('Merged CSV에 유효한 경도·위도가 없습니다.')
    distances = iv_haversine_m(query_lon, query_lat, candidate_lon[valid_mask].to_numpy(), candidate_lat[valid_mask].to_numpy())
    nearest_position = int(np.argmin(distances))
    nearest_distance = float(distances[nearest_position])
    if nearest_distance > MAX_MATCH_DISTANCE_M:
        raise ValueError(f'최근접 후보가 허용 거리보다 멉니다: {nearest_distance:.2f}m')
    nearest_index = candidate_lon[valid_mask].index[nearest_position]
    row = merged_df.loc[nearest_index].to_dict()
    row['match_method'] = 'coordinate_nearest'
    row['match_distance_m'] = nearest_distance
    return row

def iv_normalize_jurisdiction(candidate: dict[str, Any]) -> str | None:
    sigungu = str(candidate.get('시군구', '') or '').strip()
    if sigungu:
        return sigungu
    address = str(candidate.get('address_ml', '') or '').strip()
    match = re.search('([가-힣]+(?:시|군|구))', address)
    return match.group(1) if match else None

def iv_parse_rule_value(value: Any) -> Any:
    if iv_is_missing(value):
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if text.lower() == 'true':
        return True
    if text.lower() == 'false':
        return False
    try:
        number = float(text)
        return int(number) if number.is_integer() else number
    except ValueError:
        return text

def iv_compare_rule_value(value: Any, operator: Any, threshold: Any) -> bool:
    operator = str(operator or '').strip().upper()
    if operator == 'IS_NULL':
        return iv_is_missing(value)
    if operator == 'NOT_NULL':
        return not iv_is_missing(value)
    if iv_is_missing(value):
        return False
    parsed_threshold = iv_parse_rule_value(threshold)
    if operator == 'EQ':
        return str(value) == str(parsed_threshold)
    if operator == 'NE':
        return str(value) != str(parsed_threshold)
    if operator in {'LT', 'LTE', 'GT', 'GTE'}:
        try:
            left = float(value)
            right = float(parsed_threshold)
        except (TypeError, ValueError):
            return False
        return {'LT': left < right, 'LTE': left <= right, 'GT': left > right, 'GTE': left >= right}[operator]
    if operator in {'IN', 'NOT_IN'}:
        options = [item.strip() for item in re.split('[|,]', str(parsed_threshold)) if item.strip()]
        matched = str(value) in options
        return matched if operator == 'IN' else not matched
    raise ValueError(f'지원하지 않는 Rule operator: {operator}')

def iv_scope_matches(candidate: dict[str, Any], rule: pd.Series) -> bool:
    candidate_asset_type = 'BUILDING' if candidate['model_type'] == 'building' else 'LAND'
    candidate_installation_type = 'ROOFTOP' if candidate_asset_type == 'BUILDING' else 'GROUND'
    rule_asset_type = str(rule.get('asset_type', 'ALL') or 'ALL').strip().upper()
    rule_installation_type = str(rule.get('installation_type', 'ALL') or 'ALL').strip().upper()
    rule_jurisdiction = str(rule.get('jurisdiction', 'ALL') or 'ALL').strip()
    if rule_asset_type != 'ALL' and rule_asset_type != candidate_asset_type:
        return False
    if rule_installation_type != 'ALL' and rule_installation_type != candidate_installation_type:
        return False
    if rule_jurisdiction != 'ALL':
        jurisdiction = str(candidate.get('jurisdiction_norm', '') or '').strip()
        address = str(candidate.get('address_ml', '') or '')
        if rule_jurisdiction not in jurisdiction and rule_jurisdiction not in address:
            return False
    return True

def iv_safe_rule_int(value: Any, default: int=0) -> int:
    if iv_is_missing(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def iv_evaluate_candidate_rules(candidate: dict[str, Any], rules_df: pd.DataFrame) -> dict[str, Any]:
    matched_rules = []
    audit = []
    for _, rule in rules_df.iterrows():
        if not iv_scope_matches(candidate, rule):
            continue
        target_feature = str(rule.get('target_feature', '') or '').strip()
        if not target_feature:
            continue
        condition_feature = str(rule.get('condition_feature', '') or '').strip()
        condition_operator = rule.get('condition_operator')
        condition_value = rule.get('condition_value')
        if condition_feature:
            condition_matched = iv_compare_rule_value(candidate.get(condition_feature), condition_operator, condition_value)
            if not condition_matched:
                continue
        operator = rule.get('operator')
        threshold = rule.get('threshold_value')
        feature_value = candidate.get(target_feature)
        matched = iv_compare_rule_value(feature_value, operator, threshold)
        audit.append({'rule_id': rule.get('rule_id'), 'target_feature': target_feature, 'feature_value': iv_json_safe(feature_value), 'operator': operator, 'threshold_value': iv_json_safe(threshold), 'matched': matched})
        if matched:
            decision = str(rule.get('decision', 'PASS') or 'PASS').strip().upper()
            matched_rules.append({'rule_id': rule.get('rule_id'), 'decision': decision, 'severity': iv_safe_rule_int(rule.get('severity')), 'message': rule.get('message')})
    if matched_rules:
        matched_rules.sort(key=lambda item: (DECISION_PRIORITY.get(item['decision'], 0), item['severity']), reverse=True)
        final_rule = matched_rules[0]
        decision = final_rule['decision']
        message = final_rule['message']
    else:
        decision = 'PASS'
        message = '활성 Rule 위반 조건 없음'
    return {'Rule_Final_Decision': decision, 'Rule_Final_Message': message, 'Rule_Pass_For_Next_Step': decision not in EXCLUDE_DECISIONS, 'Rule_Matched_Count': len(matched_rules), 'Rule_Matched_JSON': json.dumps(matched_rules, ensure_ascii=False), 'Rule_Audit_JSON': json.dumps(audit, ensure_ascii=False)}
SHAP_TOP_K = 3
FEATURE_DISPLAY_NAMES = {'ghi_avg_daily': '수평면 일사량', 'pvout_avg_daily': '예상 태양광 발전량', 'dni_avg_daily': '직달 일사량', 'dif_avg_daily': '산란일사량', 'gti_avg_daily': '경사면 일사량', 'temp_avg': '평균 기온', 'wind_speed_10m': '10m 높이 풍속', 'wind_speed_50m': '50m 높이 풍속', 'wind_speed_100m': '100m 높이 풍속', 'slope_avg': '평균 경사도', 'slope_dir': '경사 방향', 'elevation_avg': '평균 고도', 'Hillshade': '지형 음영도', 'Southness': '남향성', 'distance_to_substation_km': '변전소 이격거리', 'distance_to_powerline_km': '전력선 이격거리', 'substation_count_5km': '5km 이내 변전소 수', 'powerline_length_5km_km': '5km 이내 전력선 길이', 'high_voltage_line_nearby_5km': '5km 이내 고압선 존재 여부', 'substation_max_voltage_kv': '인근 변전소 최대 전압', 'powerline_max_voltage_kv': '인근 전력선 최대 전압'}

def iv_extract_positive_class_shap(shap_result: Any) -> np.ndarray:
    """SHAP 결과를 양성 클래스 기준 2차원 배열로 변환합니다."""
    if isinstance(shap_result, list):
        values = np.asarray(shap_result[1] if len(shap_result) > 1 else shap_result[0])
    elif hasattr(shap_result, 'values'):
        values = np.asarray(shap_result.values)
    else:
        values = np.asarray(shap_result)
    if values.ndim == 3 and values.shape[-1] == 2:
        values = values[:, :, 1]
    elif values.ndim == 3 and values.shape[0] == 2:
        values = values[1, :, :]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f'지원하지 않는 SHAP 결과 형태: {values.shape}')
    return values

def iv_add_shap_reasons(scored: pd.DataFrame, model: Any, X: pd.DataFrame, top_k: int=SHAP_TOP_K) -> pd.DataFrame:
    """
    후보별 SHAP 양수 상위 3개와 음수 상위 3개를
    값·후보군 백분위·SHAP 기여도가 포함된 문장으로 생성합니다.
    """
    result = scored.copy()
    for rank in range(1, top_k + 1):
        result[f'추천이유_{rank}'] = None
        result[f'감점이유_{rank}'] = None
    if X.empty:
        return result
    try:
        try:
            explainer = shap.TreeExplainer(model)
            raw_shap = explainer.shap_values(X)
        except Exception:
            explainer = shap.Explainer(model, X)
            raw_shap = explainer(X)
        shap_values = iv_extract_positive_class_shap(raw_shap)
        if shap_values.shape != X.shape:
            raise ValueError(f'SHAP 결과 크기와 모델 입력 크기가 다릅니다. SHAP={shap_values.shape}, X={X.shape}')
        feature_percentiles = X.rank(pct=True, method='average') * 100
        for row_position, row_index in enumerate(X.index):
            records = []
            for feature_position, feature_name in enumerate(X.columns):
                records.append({'feature': feature_name, 'feature_value': float(X.iloc[row_position, feature_position]), 'percentile': float(feature_percentiles.iloc[row_position, feature_position]), 'shap_value': float(shap_values[row_position, feature_position])})
            positive_records = sorted([record for record in records if record['shap_value'] > 0], key=lambda record: record['shap_value'], reverse=True)[:top_k]
            negative_records = sorted([record for record in records if record['shap_value'] < 0], key=lambda record: record['shap_value'])[:top_k]
            for rank, record in enumerate(positive_records, start=1):
                display_name = FEATURE_DISPLAY_NAMES.get(record['feature'], record['feature'])
                result.loc[row_index, f'추천이유_{rank}'] = f"{display_name} 값이 {record['feature_value']:.3f}이며 현재 평가 후보지 기준 {record['percentile']:.1f}백분위입니다. 이 값은 모델의 설치 가능성 점수를 높인 주요 요인으로 분석되었습니다 (SHAP {record['shap_value']:+.4f})."
            for rank, record in enumerate(negative_records, start=1):
                display_name = FEATURE_DISPLAY_NAMES.get(record['feature'], record['feature'])
                result.loc[row_index, f'감점이유_{rank}'] = f"{display_name} 값이 {record['feature_value']:.3f}이며 현재 평가 후보지 기준 {record['percentile']:.1f}백분위입니다. 이 값은 모델의 설치 가능성 점수를 낮춘 주요 요인으로 분석되었습니다 (SHAP {record['shap_value']:+.4f})."
    except Exception as exc:
        print('[경고] SHAP 설명 생성 실패:', str(exc))
    return result

def iv_predict_group(data: pd.DataFrame, bundle: dict[str, Any], model_type: str) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    feature_columns = bundle['feature_columns']
    missing_features = [column for column in feature_columns if column not in data.columns]
    if missing_features:
        raise KeyError(f'{model_type} 모델 Feature 누락: {missing_features}')
    train_medians = pd.Series(bundle['train_medians'])
    scored = data.copy()
    for column in feature_columns:
        scored[column] = pd.to_numeric(scored[column], errors='coerce')
    X = scored[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(train_medians)
    remaining_missing = X.columns[X.isna().any()].tolist()
    if remaining_missing:
        raise ValueError(f'{model_type} 모델 입력 결측: {remaining_missing}')
    model = bundle['model']
    scored['Solar_Readiness_Probability'] = model.predict_proba(X)[:, 1]
    scored['ML_Score'] = (scored['Solar_Readiness_Probability'] * 100).round(4)
    scored['Model_Type'] = model_type
    scored['Model_Name'] = bundle.get('model_name', type(model).__name__)
    scored = iv_add_shap_reasons(scored=scored, model=model, X=X, top_k=SHAP_TOP_K)
    return scored

def iv_estimate_business_metrics(row: pd.Series) -> dict[str, Any]:
    """설비용량, 발전량, 매출, 단순 ROI와 회수기간을 계산합니다."""
    model_type = str(row.get('model_type', row.get('Model_Type', 'land')) or 'land').strip().lower()
    candidate_type = str(row.get('candidate_type', model_type) or model_type).strip().lower()

    # real_area는 검출된 전체 면적이고, usable_area는 배치 제약을 반영한 가용면적입니다.
    real_area = iv_to_optional_float(row.get('real_area'))
    usable_area = iv_to_optional_float(row.get('usable_area'))
    estimated_panel_count = iv_to_optional_float(
        row.get('estimated_panel_count')
    )

    # 설비용량 계산에는 usable_area를 우선 사용하고, 없을 때만 real_area를 사용합니다.
    capacity_base_area = (
        usable_area
        if usable_area is not None and usable_area > 0
        else real_area
    )
    area_per_kw = BUSINESS_AREA_PER_KW_M2.get(
        model_type,
        BUSINESS_AREA_PER_KW_M2['land'],
    )

    # 주차장은 land 모델을 사용하더라도 캐노피 설치비 단가를 별도로 적용합니다.
    capex_type = 'parking_lot' if candidate_type == 'parking_lot' else model_type
    capex_per_kw_krw = CAPEX_PER_KW_KRW.get(
        capex_type,
        CAPEX_PER_KW_KRW['land'],
    )

    capacity_kw = None
    annual_generation_kwh = None
    annual_revenue_krw = None
    specific_yield = None
    generation_basis = None
    total_investment_cost_krw = None
    annual_opex_krw = None
    annual_net_cashflow_krw = None
    roi_percent = None
    payback_years = None

    # Vision AI가 패널 수를 제공하면 모듈 정격출력으로 설비용량을 계산합니다.
    if estimated_panel_count is not None and estimated_panel_count > 0:
        capacity_kw = (
            int(estimated_panel_count)
            * PANEL_SPEC['power_w']
            / 1000.0
        )
    # 패널 수가 없으면 가용면적을 유형별 1kW당 필요 면적으로 나눕니다.
    elif (
        capacity_base_area is not None
        and capacity_base_area > 0
        and area_per_kw > 0
    ):
        capacity_kw = capacity_base_area / area_per_kw

    if capacity_kw is not None and capacity_kw > 0:
        pvout_daily = iv_to_optional_float(row.get('pvout_avg_daily'))

        # 후보지 PVOUT이 있으면 연간 환산하고, 없으면 config의 기본 발전량을 사용합니다.
        if pvout_daily is not None and pvout_daily > 0:
            specific_yield = pvout_daily * 365.0
            generation_basis = 'pvout_avg_daily × 365'
        else:
            specific_yield = DEFAULT_SPECIFIC_YIELD_KWH_PER_KW_YEAR
            generation_basis = 'DEFAULT_SPECIFIC_YIELD_KWH_PER_KW_YEAR'

        # 연간 발전량과 단순 연매출을 계산합니다.
        annual_generation_kwh = capacity_kw * specific_yield
        annual_revenue_krw = (
            annual_generation_kwh
            * BUSINESS_ENERGY_VALUE_KRW_PER_KWH
        )

        # 초기 투자비, 연간 운영비, 연간 순현금흐름을 계산합니다.
        total_investment_cost_krw = capacity_kw * capex_per_kw_krw
        annual_opex_krw = total_investment_cost_krw * ANNUAL_OPEX_RATE
        annual_net_cashflow_krw = annual_revenue_krw - annual_opex_krw

        # 단순 1년 차 ROI는 순현금흐름이 음수여도 계산합니다.
        # 음수 ROI는 해당 가정에서 1년 차 운영수익이 투자비 대비 손실임을 의미합니다.
        if total_investment_cost_krw > 0:
            roi_percent = (
                annual_net_cashflow_krw
                / total_investment_cost_krw
                * 100.0
            )

        # 단순 회수기간은 연간 순현금흐름이 양수일 때만 정의합니다.
        # 매년 동일한 순현금흐름이 발생한다는 단순 가정이며, 할인율과 열화는 반영하지 않습니다.
        if annual_net_cashflow_krw > 0:
            payback_years = (
                total_investment_cost_krw
                / annual_net_cashflow_krw
            )

    # 계산 결과는 기존 ranking DataFrame과 최종 JSON에 병합될 수 있도록 dict로 반환합니다.
    return {
        'recommended_capacity_kw': (
            round(capacity_kw, 2)
            if capacity_kw is not None
            else None
        ),
        'specific_yield_kwh_per_kw_year': (
            round(specific_yield, 2)
            if specific_yield is not None
            else None
        ),
        'annual_generation_kwh': (
            round(annual_generation_kwh, 2)
            if annual_generation_kwh is not None
            else None
        ),
        'annual_revenue_krw': (
            round(annual_revenue_krw, 0)
            if annual_revenue_krw is not None
            else None
        ),
        'capex_per_kw_krw': round(capex_per_kw_krw, 0),
        'total_investment_cost_krw': (
            round(total_investment_cost_krw, 0)
            if total_investment_cost_krw is not None
            else None
        ),
        'annual_opex_rate': ANNUAL_OPEX_RATE,
        'annual_opex_krw': (
            round(annual_opex_krw, 0)
            if annual_opex_krw is not None
            else None
        ),
        'annual_net_cashflow_krw': (
            round(annual_net_cashflow_krw, 0)
            if annual_net_cashflow_krw is not None
            else None
        ),
        'roi_percent': (
            round(roi_percent, 2)
            if roi_percent is not None
            else None
        ),
        'payback_years': (
            round(payback_years, 2)
            if payback_years is not None
            else None
        ),
        'generation_basis': generation_basis,
        'area_per_kw_m2': area_per_kw,
        'revenue_unit_price_krw_per_kwh': BUSINESS_ENERGY_VALUE_KRW_PER_KWH,
        'roi_method': 'simple_first_year_roi',
        'roi_formula': 'annual_net_cashflow_krw / total_investment_cost_krw × 100',
        'payback_method': 'constant_annual_cashflow_simple_payback',
        'payback_formula': 'total_investment_cost_krw / annual_net_cashflow_krw',
        'financial_model_level': 'preliminary_screening',
        'estimate_scope': BUSINESS_ESTIMATE_SCOPE,
    }

def iv_build_integrated_ranking(passed_scored_df: pd.DataFrame, failed_df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not passed_scored_df.empty:
        passed = passed_scored_df.copy()
        real_area = pd.to_numeric(passed['real_area'], errors='coerce').fillna(0.0).clip(lower=0.0)
        pixel_area = pd.to_numeric(passed['pixel_area'], errors='coerce').fillna(0.0).clip(lower=0.0)
        zero_area_mask = real_area.le(0.0) | pixel_area.le(0.0)
        # vision area score 추천 전력용량 처리
        passed['recommended_capacity_kw'] = passed.apply(iv_calculate_recommended_capacity_kw,axis=1)
        passed['Vision_Area_Score'] = (passed['recommended_capacity_kw'] / VISION_CAPACITY_FULL_SCORE_KW).clip(0.0, 1.0)
        # passed['Vision_Area_Score'] = np.log1p(estimated_panel_count).rank(pct=True, method='average').where(estimated_panel_count > 0, 0.0)
        # passed['Vision_Area_Score'] = np.log1p(usable_area).rank(pct=True, method='average').where(usable_area > 0, 0.0)
        # passed['Vision_Area_Score'] = np.log1p(real_area).rank(pct=True, method='average').where(real_area > 0, 0.0)
        passed['Final_Readiness_Probability'] = ML_WEIGHT * passed['Solar_Readiness_Probability'] + AREA_WEIGHT * passed['Vision_Area_Score']
        passed['Solar_Readiness_Score'] = (passed['Final_Readiness_Probability'] * 100).round(4)
        passed.loc[zero_area_mask, ['Solar_Readiness_Probability', 'ML_Score', 'Vision_Area_Score', 'Final_Readiness_Probability', 'Solar_Readiness_Score']] = 0.0
        passed['Suitability_Status'] = 'PASS'
        frames.append(passed)
    if not failed_df.empty:
        failed = failed_df.copy()
        failed['Solar_Readiness_Probability'] = 0.0
        failed['ML_Score'] = 0.0
        failed['Vision_Area_Score'] = 0.0
        failed['Final_Readiness_Probability'] = 0.0
        failed['Solar_Readiness_Score'] = 0.0
        failed['Model_Type'] = failed['model_type']
        failed['Model_Name'] = None
        failed['Suitability_Status'] = 'FAIL'
        frames.append(failed)
    if not frames:
        return pd.DataFrame()
    scored_df = pd.concat(frames, ignore_index=True, sort=False)
    scored_df['_pass_sort'] = scored_df['Suitability_Status'].eq('PASS').astype(int)
    scored_df = scored_df.sort_values(['_pass_sort', 'Solar_Readiness_Score', 'real_area'], ascending=[False, False, False]).reset_index(drop=True)
    scored_df['Candidate_Rank'] = np.arange(1, len(scored_df) + 1)
    scored_df['Score_Percentile'] = 1.0 - (scored_df['Candidate_Rank'] - 1) / max(len(scored_df), 1)

    def grade_row(row):
        if row['Suitability_Status'] == 'FAIL':
            return 'F'
        percentile = row['Score_Percentile']
        if percentile >= 0.8:
            return 'A'
        if percentile >= 0.5:
            return 'B'
        return 'C'
    scored_df['Solar_Readiness_Grade'] = scored_df.apply(grade_row, axis=1)
    business_metrics = scored_df.apply(iv_estimate_business_metrics, axis=1, result_type='expand')
    scored_df = pd.concat([scored_df, business_metrics], axis=1)
    return scored_df.drop(columns=['_pass_sort'], errors='ignore')

def run_integrated_pipeline(payload: Any) -> IntegratedResult:
    merged_df, land_bundle, building_bundle, rules_df = iv_load_resources()
    wrappers = iv_unpack_payload(payload)
    vision_rows = []
    vision_errors = []
    for index, wrapper in enumerate(wrappers):
        try:
            vision_rows.append(iv_aggregate_vision_candidate(wrapper))
        except Exception as exc:
            vision_errors.append({'vision_index': index, 'stage': 'vision_aggregate', 'error': str(exc)})
    vision_df = pd.DataFrame(vision_rows)
    combined_rows = []
    match_errors = []
    for index, vision in vision_df.iterrows():
        vision_dict = vision.to_dict()
        try:
            candidate = iv_match_candidate_row(merged_df, vision_dict)
            combined = dict(candidate)
            combined.update(vision_dict)
            combined['jurisdiction_norm'] = iv_normalize_jurisdiction(combined)
            rule_result = iv_evaluate_candidate_rules(combined, rules_df)
            combined.update(rule_result)
            combined_rows.append(combined)
        except Exception as exc:
            match_errors.append({'vision_index': index, 'stage': 'candidate_match_or_rule', 'candidate_type': vision_dict.get('candidate_type'), 'model_type': vision_dict.get('model_type'), 'error': str(exc)})
    reviewed_df = pd.DataFrame(combined_rows)
    reviewed_df = pd.DataFrame(combined_rows)

    if reviewed_df.empty:
        error_detail = {
            "message": "후보 매칭 및 Rule 처리에 성공한 데이터가 없습니다.",
            "vision_errors": vision_errors,
            "match_or_rule_errors": match_errors,
        }

        raise ValueError(
            json.dumps(
                error_detail,
                ensure_ascii=False,
                indent=2,
            )
        )
    passed_df = reviewed_df[reviewed_df['Rule_Pass_For_Next_Step']].copy().reset_index(drop=True)
    land_passed = passed_df[passed_df['model_type'] == 'land'].copy()
    building_passed = passed_df[passed_df['model_type'] == 'building'].copy()
    land_scored = iv_predict_group(land_passed, land_bundle, 'land')
    building_scored = iv_predict_group(building_passed, building_bundle, 'building')
    scored_df = pd.concat([land_scored, building_scored], ignore_index=True, sort=False)
    failed_df = reviewed_df[~reviewed_df['Rule_Pass_For_Next_Step']].copy().reset_index(drop=True)
    ranking_df = iv_build_integrated_ranking(scored_df, failed_df)

    def collect_text_values(row: pd.Series, columns: list[str]) -> list[str]:
        values = []
        for column in columns:
            value = row.get(column)
            if not iv_is_missing(value):
                text = str(value).strip()
                if text:
                    values.append(text)
        return values

    def first_available_value(row: pd.Series, columns: list[str], default: Any=None) -> Any:
        for column in columns:
            value = row.get(column)
            if not iv_is_missing(value):
                return value
        return default

    def make_status(row: pd.Series) -> str:
        if row.get('Suitability_Status') == 'FAIL':
            return '부적합'
        grade = row.get('Solar_Readiness_Grade')
        return {'A': '우선 검토', 'B': '검토 가능', 'C': '추가 검토'}.get(str(grade), '검토 필요')

    def get_type_settings(model_type: str) -> dict[str, Any]:
        if model_type == 'building':
            return {'target_type': 'BUILDING', 'default_space_type': '건물형 후보지', 'checklist': [{'item': '지붕 구조안전성 및 적재하중 확인', 'note': '구조검토를 통해 태양광 모듈·구조물 하중 수용 가능 여부 확인'}, {'item': '옥상 방수 상태 및 보수 필요 여부', 'note': '누수 이력과 방수층 수명을 확인하고 필요 시 설치 전 보수'}, {'item': '옥상 장애물·설비 및 피난동선 확인', 'note': '냉난방기, 물탱크, 피뢰설비와 소방 피난동선 간섭 여부 확인'}, {'item': '계통연계 및 건축·소방 기준 검토', 'note': '한전 계통연계 가능 여부와 건축물·소방 관련 기준 확인'}]}
        return {'target_type': 'LAND', 'default_space_type': '토지형 후보지', 'checklist': [{'item': '진입로 확보 및 공사차량 진입 가능 여부', 'note': '현장 진입로 폭과 도로점용 허가 필요 여부 확인'}, {'item': '토질 상태 및 배수시설 설치 가능 여부', 'note': '우수 배제와 토사 유출 가능성 현장 확인'}, {'item': '최근접 전력선 및 계통연계 가능 여부', 'note': '한전 계통 여유 용량과 선로 신설 비용 별도 확인'}, {'item': '개발행위허가 및 이격거리 충족 여부', 'note': '해당 지자체의 현행 조례와 부지 경계를 기준으로 재검토'}]}

    def build_candidate_json(row: pd.Series) -> dict[str, Any]:
        model_type = str(row.get('Model_Type', row.get('model_type', 'land')) or 'land').strip().lower()
        settings = get_type_settings(model_type)
        site_id = first_available_value(row, ['source_id_ml', '후보ID', 'site_id'])
        address = first_available_value(row, ['address_ml', '주소', '소재지', '도로명주소', '지번주소'])
        site_name = first_available_value(row, ['site_name', '재산명', '시설명', '자산명', '발전소명', '건물명'])
        if site_name is None:
            site_name = f'{address} 태양광 후보지' if address else f'{site_id} 태양광 후보지'
        real_area = iv_to_optional_float(row.get('real_area'))
        total_area = first_available_value(row, ['total_area', '전체면적', '토지면적', '연면적', '옥상면적', '면적', '공유재산면적', '대지면적'])
        total_area = iv_to_optional_float(total_area)
        availability_rate = None
        if total_area is not None and total_area > 0 and (real_area is not None):
            availability_rate = round(real_area / total_area * 100, 2)
        bonus_reasons = collect_text_values(row, ['추천이유_1', '추천이유_2', '추천이유_3'])
        penalty_reasons = collect_text_values(row, ['감점이유_1', '감점이유_2', '감점이유_3'])
        probability = iv_to_optional_float(row.get('Solar_Readiness_Probability'))
        if bonus_reasons:
            ml_reason = bonus_reasons[0]
        elif probability is not None and row.get('Suitability_Status') == 'PASS':
            ml_reason = f'저장된 ML 모델이 산출한 설치사례 유사 확률은 {probability * 100:.2f}%입니다.'
        else:
            ml_reason = 'Rule-based 검토에서 부적합으로 판정되어 ML 점수는 0점 처리했습니다.'
        rule_pass = bool(row.get('Rule_Pass_For_Next_Step'))
        candidate_type = str(row.get('candidate_type', model_type)).strip().lower()
        return {'target_type': settings['target_type'], '1_site_info': {'site_id': str(site_id) if site_id is not None else None, 'site_name': site_name, 'address': address, 'longitude': iv_json_safe(row.get('longitude')), 'latitude': iv_json_safe(row.get('latitude')), 'space_type': first_available_value(row, ['space_type', '자산구분_ML', '설치구분', '재산구분', '지목', '건물용도'], default=settings['default_space_type']), 'vision_candidate_type': candidate_type, 'total_area_m2': iv_json_safe(total_area), 'available_area_m2': iv_json_safe(real_area), 'availability_rate_percent': availability_rate, 'owner_agency': first_available_value(row, ['owner_agency', '소유기관', '관리기관', '기관명', '소관기관'])}, '2_scores_and_evaluation': {'grade': iv_json_safe(row.get('Solar_Readiness_Grade')), 'total_score': iv_json_safe(row.get('Solar_Readiness_Score')), 'priority_rank': iv_json_safe(row.get('Candidate_Rank')), 'status': make_status(row), 'suitability': {'rule_pass': rule_pass, 'rule_decision': iv_json_safe(row.get('Rule_Final_Decision')), 'rule_message': iv_json_safe(row.get('Rule_Final_Message')), 'suitability_status': iv_json_safe(row.get('Suitability_Status'))}, 'detail_scores': {'ml_technical_score': iv_json_safe(row.get('ML_Score')), 'ml_probability': iv_json_safe(probability), 'ml_reason': ml_reason, 'vision_area_score': iv_json_safe(row.get('Vision_Area_Score')), 'vision_confidence': iv_json_safe(row.get('confidence')), 'rule_based_score': 100.0 if rule_pass else 0.0}, 'xai_explanation': {'bonus_reason': bonus_reasons if rule_pass else [], 'penalty_reason': penalty_reasons if rule_pass else [str(row.get('Rule_Final_Message', 'Rule-based 검토 부적합'))]}}, '3_vision_ai_and_simulation': {'vision_analysis': {'candidate_type': candidate_type, 'confidence': iv_json_safe(row.get('confidence')), 'pixel_area_px2': iv_json_safe(row.get('pixel_area')), 'real_area_m2': iv_json_safe(real_area), 'shape_score': iv_json_safe(row.get('shape_score')),'shape_grade': iv_json_safe(row.get('shape_grade')),'shape_efficiency': iv_json_safe(row.get('shape_efficiency')),'recommended_layout': iv_json_safe(row.get('recommended_layout')),'usable_area_m2': iv_json_safe(row.get('usable_area')),'estimated_panel_count': iv_json_safe(row.get('estimated_panel_count')), 'distance_to_road_px': iv_json_safe(row.get('distance_to_road_px')), 'distance_to_building_px': iv_json_safe(row.get('distance_to_building_px')), 'distance_to_road_m': iv_json_safe(row.get('distance_to_road_m')), 'distance_to_building_m': iv_json_safe(row.get('distance_to_building_m')), 'model_version': iv_json_safe(row.get('model_version')), 'slope_degree': iv_json_safe(row.get('slope_avg')) if model_type == 'land' else None, 'aspect_direction_degree': iv_json_safe(row.get('slope_dir')) if model_type == 'land' else None}, 'simulation': {'recommended_capacity_kw': iv_json_safe(row.get('recommended_capacity_kw')), 'annual_generation_kwh': iv_json_safe(row.get('annual_generation_kwh')), 'specific_yield_kwh_per_kw_year': iv_json_safe(row.get('specific_yield_kwh_per_kw_year')), 'generation_basis': iv_json_safe(row.get('generation_basis')), 'area_per_kw_m2': iv_json_safe(row.get('area_per_kw_m2')), 'annual_revenue_krw': iv_json_safe(row.get('annual_revenue_krw')), 'revenue_unit_price_krw_per_kwh': iv_json_safe(row.get('revenue_unit_price_krw_per_kwh')), 'capex_per_kw_krw': iv_json_safe(row.get('capex_per_kw_krw')), 'total_investment_cost_krw': iv_json_safe(row.get('total_investment_cost_krw')), 'annual_opex_rate': iv_json_safe(row.get('annual_opex_rate')), 'annual_opex_krw': iv_json_safe(row.get('annual_opex_krw')), 'annual_net_cashflow_krw': iv_json_safe(row.get('annual_net_cashflow_krw')), 'roi_percent': iv_json_safe(row.get('roi_percent')), 'payback_years': iv_json_safe(row.get('payback_years')), 'roi_method': iv_json_safe(row.get('roi_method')), 'roi_formula': iv_json_safe(row.get('roi_formula')), 'payback_method': iv_json_safe(row.get('payback_method')), 'payback_formula': iv_json_safe(row.get('payback_formula')), 'financial_model_level': iv_json_safe(row.get('financial_model_level')), 'estimate_scope': iv_json_safe(row.get('estimate_scope'))}}, '4_risk_and_support': {'rule_based_risk_check': {'grid_connection': {'substation_distance_km': iv_json_safe(row.get('substation_dist_km')), 'powerline_distance_km': iv_json_safe(row.get('powerline_dist_km'))}, 'regulation': iv_json_safe(row.get('Rule_Final_Message')), 'distance_risk': {'distance_to_road_m': iv_json_safe(row.get('distance_to_road_m')), 'distance_to_building_m': iv_json_safe(row.get('distance_to_building_m'))}, 'public_complaint': None}, 'recommended_subsidies': []}, '5_pre_investigation_checklist': settings['checklist']}
    output_json = [build_candidate_json(row) for _, row in ranking_df.iterrows()]
    print('\n[RANKING] 최종 반환 JSON')
    print(
        json.dumps(
            output_json,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    summary = {'vision_received': len(wrappers), 'vision_aggregated': len(vision_df), 'matched_and_reviewed': len(reviewed_df), 'rule_passed': len(passed_df), 'ranked': len(ranking_df), 'rule_failed_included': int((ranking_df['Suitability_Status'] == 'FAIL').sum()) if not ranking_df.empty else 0, 'land_model_count': int((ranking_df['Model_Type'] == 'land').sum()) if not ranking_df.empty else 0, 'building_model_count': int((ranking_df['Model_Type'] == 'building').sum()) if not ranking_df.empty else 0, 'error_count': len(vision_errors + match_errors)}
    return IntegratedResult(summary=summary, reviewed_df=reviewed_df, passed_df=passed_df, ranking_df=ranking_df, results=output_json, errors=vision_errors + match_errors)
