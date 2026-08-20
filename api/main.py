from __future__ import annotations

import json
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Annotated

import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from . import pipeline
from .config import MODEL_BUNDLE_PATHS

VALID_DATASET_TYPES = set(MODEL_BUNDLE_PATHS)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for dataset_type in VALID_DATASET_TYPES:
        try:
            pipeline.load_bundle(dataset_type)
        except pipeline.PipelineError:
            pass
        try:
            pipeline.get_universe_ranking(dataset_type)
        except pipeline.PipelineError:
            pass
    yield


app = FastAPI(
    title="태양광 후보지 랭킹 API",
    description="Rule-based 검토를 통과한 토지형/건물형 후보지 CSV·XLSX를 업로드하면 ML 예측·랭킹·SHAP 결과를 반환합니다.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청한 데이터 유형이 토지형 또는 건물형인지 확인합니다.
def _validate_dataset_type(dataset_type: str) -> str:
    dataset_type = dataset_type.lower()
    if dataset_type not in VALID_DATASET_TYPES:
        raise HTTPException(status_code=404, detail=f"dataset_type은 'land' 또는 'building'이어야 합니다: {dataset_type}")
    return dataset_type


# 문자열로 전달된 정책 가중치 설정을 JSON 객체로 변환합니다.
def _parse_policy_weight_config(raw: str | None) -> dict:
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"policy_weight_config가 올바른 JSON이 아닙니다: {exc}") from exc

    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="policy_weight_config는 JSON 객체여야 합니다.")

    for column, config in parsed.items():
        if not isinstance(config, dict) or "weight" not in config or "direction" not in config:
            raise HTTPException(
                status_code=400, detail=f"policy_weight_config.{column}에는 weight와 direction이 필요합니다."
            )

    return parsed


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", force_ascii=False))


# 업로드 파일과 옵션을 단독 ML 랭킹 파이프라인에 전달합니다.
async def _run_pipeline_from_upload(
    dataset_type: str,
    file: UploadFile,
    *,
    filter_rule_excluded: bool,
    rank_filter_mode: str,
    create_region_ranks: bool,
    include_shap: bool,
    shap_top_n: int,
    ml_weight: float,
    policy_weight: float,
    policy_weight_config: str | None,
    top_n_json: int,
) -> pipeline.RankingResult:
    dataset_type = _validate_dataset_type(dataset_type)
    content = await file.read()

    options = pipeline.RankOptions(
        filter_rule_excluded=filter_rule_excluded,
        rank_filter_mode=rank_filter_mode,
        create_region_ranks=create_region_ranks,
        include_shap=include_shap,
        shap_top_n=shap_top_n,
        ml_weight=ml_weight,
        policy_weight=policy_weight,
        policy_weight_config=_parse_policy_weight_config(policy_weight_config),
        top_n_json=top_n_json,
    )

    try:
        return pipeline.run_pipeline(dataset_type, file.filename or "", content, options)
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"필수 값이 없습니다: {exc}") from exc


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# 유형별 모델 정보와 학습 Feature 목록을 반환합니다.
@app.get("/models")
def list_models() -> dict:
    result = {}
    for dataset_type in sorted(VALID_DATASET_TYPES):
        try:
            bundle = pipeline.load_bundle(dataset_type)
            result[dataset_type] = {
                "available": True,
                "model_name": bundle.get("model_name"),
                "feature_count": len(bundle.get("feature_columns", [])),
                "feature_columns": bundle.get("feature_columns", []),
                "metadata_columns": bundle.get("metadata_columns", []),
            }
        except pipeline.PipelineError as exc:
            result[dataset_type] = {"available": False, "error": str(exc)}
    return result


# 전체 후보군에서 주소를 검색하고 전체 순위와 검색 결과 내 순위를 반환합니다.
@app.get("/candidates/{dataset_type}/search")
def search_candidates(
    dataset_type: str,
    q: str,
    top_n: int = 20,
) -> dict:
    """Rule_base/*_RulePassed.csv 전체 후보군(서버에 상주)에서 주소로 검색합니다.

    각 결과에 `overall_rank`(전체 후보 대비 순위)와 `search_rank`(검색 결과 내 순위)를
    함께 반환합니다.
    """
    dataset_type = _validate_dataset_type(dataset_type)
    try:
        return pipeline.search_candidates(dataset_type, q, top_n=top_n)
    except pipeline.PipelineError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# 업로드한 후보지 파일의 ML 점수, 순위, SHAP 결과를 반환합니다.
@app.post("/rank/{dataset_type}")
async def rank(
    dataset_type: str,
    file: Annotated[UploadFile, File(description="Rule-based 검토를 통과한 후보지 CSV 또는 XLSX")],
    filter_rule_excluded: Annotated[bool, Form()] = True,
    rank_filter_mode: Annotated[str, Form()] = "all",
    create_region_ranks: Annotated[bool, Form()] = True,
    include_shap: Annotated[bool, Form()] = True,
    shap_top_n: Annotated[int, Form()] = 1000,
    ml_weight: Annotated[float, Form()] = 1.0,
    policy_weight: Annotated[float, Form()] = 0.0,
    policy_weight_config: Annotated[
        str, Form(description='JSON 객체 문자열, 예: {"col": {"weight": 1, "direction": "higher"}}')
    ] = "{}",
    top_n_json: Annotated[int, Form()] = 20,
) -> dict:
    result = await _run_pipeline_from_upload(
        dataset_type,
        file,
        filter_rule_excluded=filter_rule_excluded,
        rank_filter_mode=rank_filter_mode,
        create_region_ranks=create_region_ranks,
        include_shap=include_shap,
        shap_top_n=shap_top_n,
        ml_weight=ml_weight,
        policy_weight=policy_weight,
        policy_weight_config=policy_weight_config,
        top_n_json=top_n_json,
    )

    ranking_df = result.candidate_ranking_with_shap if include_shap else result.candidate_ranking

    return {
        "dataset_type": result.dataset_type,
        "model_name": result.model_name,
        "input_rows": result.input_rows,
        "ranked_rows": result.ranked_rows,
        "ranking": _df_to_records(ranking_df),
        "top_candidates": result.top_candidates_json,
    }


# 단독 ML 랭킹 결과를 CSV, Excel, JSON이 포함된 ZIP으로 반환합니다.
@app.post("/rank/{dataset_type}/export")
async def rank_export(
    dataset_type: str,
    file: Annotated[UploadFile, File(description="Rule-based 검토를 통과한 후보지 CSV 또는 XLSX")],
    filter_rule_excluded: Annotated[bool, Form()] = True,
    rank_filter_mode: Annotated[str, Form()] = "all",
    create_region_ranks: Annotated[bool, Form()] = True,
    include_shap: Annotated[bool, Form()] = True,
    shap_top_n: Annotated[int, Form()] = 1000,
    ml_weight: Annotated[float, Form()] = 1.0,
    policy_weight: Annotated[float, Form()] = 0.0,
    policy_weight_config: Annotated[
        str, Form(description='JSON 객체 문자열, 예: {"col": {"weight": 1, "direction": "higher"}}')
    ] = "{}",
    top_n_json: Annotated[int, Form()] = 20,
) -> StreamingResponse:
    result = await _run_pipeline_from_upload(
        dataset_type,
        file,
        filter_rule_excluded=filter_rule_excluded,
        rank_filter_mode=rank_filter_mode,
        create_region_ranks=create_region_ranks,
        include_shap=include_shap,
        shap_top_n=shap_top_n,
        ml_weight=ml_weight,
        policy_weight=policy_weight,
        policy_weight_config=policy_weight_config,
        top_n_json=top_n_json,
    )

    output_prefix = "Land" if result.dataset_type == "land" else "Building"

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result.scored_test.to_excel(writer, sheet_name="전체_Test_예측", index=False)
        result.candidate_ranking_with_shap.to_excel(writer, sheet_name="후보지_랭킹_SHAP", index=False)
        result.candidate_shap_details.to_excel(writer, sheet_name="SHAP_세부", index=False)
    excel_buffer.seek(0)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            f"{output_prefix}_Test_candidate_ranking.csv",
            result.candidate_ranking.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            f"{output_prefix}_Test_candidate_ranking_with_shap.csv",
            result.candidate_ranking_with_shap.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            f"{output_prefix}_Test_candidate_shap_details.csv",
            result.candidate_shap_details.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(f"{output_prefix}_Test_ranking_results.xlsx", excel_buffer.getvalue())
        archive.writestr(
            f"{output_prefix}_Top{top_n_json}_Candidate_Analysis.json",
            json.dumps(result.top_candidates_json, ensure_ascii=False, indent=2),
        )
    zip_buffer.seek(0)

    filename = f"{output_prefix}_ranking_results.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

# Vision AI가 전송한 JSON Body를 받아 통합 분석을 수행합니다.
@app.post("/analyze/vision-json")
async def analyze_vision_json(
    payload: dict | list = Body(..., description="Vision AI 서버가 전송하는 분석 결과 JSON"),
) -> dict:
    """Vision AI 서버가 POST한 JSON Body를 직접 받아 통합 분석합니다."""
    try:
        result = pipeline.run_integrated_pipeline(payload)
    except (pipeline.PipelineError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Vision 연계 통합 분석 중 오류가 발생했습니다: {exc}",
        ) from exc

    return pipeline.iv_deep_json_safe({
        "success": True,
        "summary": result.summary,
        "results": result.results,
        "errors": result.errors,
    })


# Vision JSON 파일을 후보 데이터와 매칭해 Rule, ML, SHAP 분석을 수행합니다.
@app.post("/analyze/vision")
async def analyze_vision(
    file: Annotated[UploadFile, File(description="Vision AI 분석 결과 JSON")],
) -> dict:
    """Vision JSON을 후보 데이터와 매칭한 뒤 Rule → 유형별 ML → SHAP → 통합 랭킹을 수행합니다."""
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Vision 입력 파일은 JSON 형식이어야 합니다.")

    content = await file.read()
    try:
        payload = json.loads(content.decode("utf-8-sig"))
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="JSON 파일 인코딩을 읽을 수 없습니다.") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"올바른 JSON 형식이 아닙니다: {exc}") from exc

    try:
        result = pipeline.run_integrated_pipeline(payload)
    except (pipeline.PipelineError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "summary": result.summary,
        "results": result.results,
        "errors": result.errors,
    }


# Vision 통합 분석 결과를 CSV, Excel, JSON이 포함된 ZIP으로 반환합니다.
@app.post("/analyze/vision/export")
async def analyze_vision_export(
    file: Annotated[UploadFile, File(description="Vision AI 분석 결과 JSON")],
) -> StreamingResponse:
    """통합 분석 결과를 CSV·XLSX·JSON이 포함된 ZIP으로 반환합니다."""
    if not (file.filename or "").lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Vision 입력 파일은 JSON 형식이어야 합니다.")

    content = await file.read()
    try:
        payload = json.loads(content.decode("utf-8-sig"))
        result = pipeline.run_integrated_pipeline(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"올바른 JSON 형식이 아닙니다: {exc}") from exc
    except (pipeline.PipelineError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result.reviewed_df.to_excel(writer, sheet_name="RuleReviewed", index=False)
        result.passed_df.to_excel(writer, sheet_name="RulePassed", index=False)
        result.ranking_df.to_excel(writer, sheet_name="ML_Ranking", index=False)
    excel_buffer.seek(0)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "Vision_RuleReviewed.csv",
            result.reviewed_df.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "Vision_RulePassed.csv",
            result.passed_df.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "Vision_ML_Ranking.csv",
            result.ranking_df.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr("Vision_Rule_ML_Ranking.xlsx", excel_buffer.getvalue())
        archive.writestr(
            "Vision_Rule_ML_Ranking.json",
            json.dumps(result.results, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "Vision_Processing_Errors.json",
            json.dumps(result.errors, ensure_ascii=False, indent=2),
        )
        archive.writestr(
            "Vision_Processing_Summary.json",
            json.dumps(result.summary, ensure_ascii=False, indent=2),
        )
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="Vision_integrated_results.zip"'},
    )
