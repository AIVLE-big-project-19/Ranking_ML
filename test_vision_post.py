from __future__ import annotations

import json

import requests

API_URL = "http://127.0.0.1:8002/analyze/vision-json"

payload = [
    {
        "candidate_type": "land",
        "confidence": 0.9231,
        "polygon": [
            [14135200.5, 4518750.2],
            [14135220.8, 4518761.4],
        ],
        "pixel_area": 1223.5,
        "real_area": 186.69,
        "distance_to_road_px": 18.42,
        "distance_to_building_px": 35.71,
        "distance_to_road_m": 14.02,
        "distance_to_building_m": 27.19,
        "model_version": "solar-yolov8-seg-v2",
    }
]

response = requests.post(API_URL, json=payload, timeout=180)
print("status:", response.status_code)
try:
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
except ValueError:
    print(response.text)
response.raise_for_status()
