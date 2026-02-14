# backend/app.py
"""
Flood Prediction API - updated with /recent-predictions endpoint.

Features:
- /recent-predictions?limit=N returns last N rows from Supabase (ordered by timestamp DESC)
  and includes model prediction label + probabilities for each row.
- /latest-prediction remains (single latest row prediction).
- Falls back to mock rows (Dehradun + Roorkee) when Supabase fetch fails if configured.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import logging
import traceback
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np
import requests

# Load .env
load_dotenv()

# Logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("flood-api")

# Config (from env or defaults)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "").strip()
TABLE_NAME = os.getenv("SUPABASE_TABLE", "sensor_readings").strip()

MODEL_PATH = os.getenv("MODEL_PATH", "svm_flood_model.joblib").strip()
SCALER_PATH = os.getenv("SCALER_PATH", "svm_scaler.joblib").strip()

USE_MOCK_ON_SUPABASE_FAIL = os.getenv("USE_MOCK_ON_SUPABASE_FAIL", "true").lower() in ("1", "true", "yes")

# === Feature names (order matters) ===
FEATURE_COLUMNS = [
    "rainfall_mm",
    "rain_3h_sum",
    "rain_6h_sum",
    "rain_12h_sum",
    "water_level_m",
    "level_prev",
    "level_change",
    "discharge_m3s",
    "humidity_percent",
    "temperature_c"
]

# Legacy alias mapping (old_column -> new_feature_name)
ALIAS_MAP = {
    "rainfall": "rainfall_mm",
    "humidity": "humidity_percent",
    "temperature": "temperature_c",
    "river_level": "water_level_m",
}

# Basic config warnings
if not SUPABASE_URL:
    logger.warning("SUPABASE_URL not set! Set SUPABASE_URL in .env (https://<ref>.supabase.co).")
if not SUPABASE_KEY:
    logger.warning("SUPABASE_KEY not set! Set SUPABASE_KEY in .env with your anon/service_role key.")

# Load artifacts
model = None
scaler = None

def load_artifacts():
    global model, scaler
    try:
        logger.info("Loading model from: %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded. Type: %s, has_predict_proba: %s", type(model), hasattr(model, "predict_proba"))
    except Exception as e:
        logger.error("Failed to load model from %s: %s", MODEL_PATH, e)
        logger.error(traceback.format_exc())
        model = None

    try:
        logger.info("Loading scaler from: %s", SCALER_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("Scaler loaded. Type: %s, n_features_in_: %s, feature_names_in_: %s",
                    type(scaler), getattr(scaler, "n_features_in_", None), getattr(scaler, "feature_names_in_", None))
    except Exception as e:
        logger.error("Failed to load scaler from %s: %s", SCALER_PATH, e)
        logger.error(traceback.format_exc())
        scaler = None

load_artifacts()

app = FastAPI(title="Flood-Prediction-API")

# CORS for dev (change in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

def get_supabase_rest_url_latest() -> str:
    base = SUPABASE_URL.rstrip("/")
    # returns single latest row by timestamp.desc limit=1
    return f"{base}/rest/v1/{TABLE_NAME}?select=*&order=timestamp.desc&limit=1"

def get_supabase_rest_url_recent(limit: int = 12) -> str:
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/rest/v1/{TABLE_NAME}?select=*&order=timestamp.desc&limit={limit}"

def make_supabase_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Accept": "application/json"
    }

# Mock rows for local/dev fallback (two rows: Dehradun + Roorkee) to show multiple markers and updates
def mock_row_dehradun() -> Dict[str, Any]:
    return {
        "id": -1,
        "timestamp": None,
        "rainfall_mm": 85.3,
        "rain_3h_sum": 85.3,
        "rain_6h_sum": 120.0,
        "rain_12h_sum": 200.0,
        "water_level_m": 4.2,
        "level_prev": 4.0,
        "level_change": 0.2,
        "discharge_m3s": 30.0,
        "humidity_percent": 92.1,
        "temperature_c": 24.6,
        "latitude": 29.9457,   # Dehradun-like
        "longitude": 78.1642,
        "meta": {"note": "mock dehradun row (dev fallback)"}
    }

def mock_row_roorkee() -> Dict[str, Any]:
    return {
        "id": -2,
        "timestamp": None,
        "rainfall_mm": 6.0,
        "rain_3h_sum": 12.0,
        "rain_6h_sum": 18.0,
        "rain_12h_sum": 30.0,
        "water_level_m": 2.45,
        "level_prev": 2.30,
        "level_change": 0.15,
        "discharge_m3s": 25.0,
        "humidity_percent": 70.0,
        "temperature_c": 24.0,
        "latitude": 29.8665,   # Roorkee-like
        "longitude": 77.8980,
        "meta": {"note": "mock roorkee row (dev fallback)"}
    }

def fetch_latest_row():
    rest_url = get_supabase_rest_url_latest()
    headers = make_supabase_headers()

    logger.debug("Fetching latest row from Supabase: %s", rest_url)
    try:
        resp = requests.get(rest_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Requests exception contacting Supabase: %s", e)
        logger.error(traceback.format_exc())
        if USE_MOCK_ON_SUPABASE_FAIL:
            logger.warning("Returning mock row due to Supabase fetch failure (dev fallback enabled).")
            return mock_row_dehradun()
        raise HTTPException(status_code=502, detail=f"Error contacting Supabase: {e}")

    try:
        data = resp.json()
    except Exception as e:
        logger.error("Failed to parse JSON from Supabase response: %s", e)
        logger.error("Response text (truncated): %s", resp.text[:2000])
        logger.error(traceback.format_exc())
        if USE_MOCK_ON_SUPABASE_FAIL:
            return mock_row_dehradun()
        raise HTTPException(status_code=502, detail="Invalid JSON from Supabase")

    if not data:
        logger.info("Supabase returned empty data.")
        if USE_MOCK_ON_SUPABASE_FAIL:
            return mock_row_dehradun()
        raise HTTPException(status_code=404, detail="No rows found in Supabase table")

    logger.debug("Supabase returned row: %s", data[0])
    return data[0]

def fetch_recent_rows(limit: int = 12) -> List[Dict[str, Any]]:
    rest_url = get_supabase_rest_url_recent(limit)
    headers = make_supabase_headers()
    logger.debug("Fetching recent rows from Supabase: %s", rest_url)
    try:
        resp = requests.get(rest_url, headers=headers, timeout=12)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Requests exception contacting Supabase for recent rows: %s", e)
        logger.error(traceback.format_exc())
        if USE_MOCK_ON_SUPABASE_FAIL:
            logger.warning("Returning mock recent rows due to Supabase fetch failure (dev fallback enabled).")
            # return two mock rows (newest first)
            return [mock_row_dehradun(), mock_row_roorkee()]
        raise HTTPException(status_code=502, detail=f"Error contacting Supabase: {e}")

    try:
        data = resp.json()
    except Exception as e:
        logger.error("Failed to parse JSON from Supabase response: %s", e)
        logger.error("Response text (truncated): %s", resp.text[:2000])
        logger.error(traceback.format_exc())
        if USE_MOCK_ON_SUPABASE_FAIL:
            return [mock_row_dehradun(), mock_row_roorkee()]
        raise HTTPException(status_code=502, detail="Invalid JSON from Supabase")

    if not data:
        logger.info("Supabase returned empty recent rows.")
        if USE_MOCK_ON_SUPABASE_FAIL:
            return [mock_row_dehradun(), mock_row_roorkee()]
        return []

    # data should be a list (newest first due to order=timestamp.desc)
    return data

def build_feature_vector_from_row(row: Dict[str, Any]) -> np.ndarray:
    """
    Build feature vector in EXACT order of FEATURE_COLUMNS.
    Strategy:
      1) If FEATURE_COLUMNS key exists in row -> use it
      2) Else if any alias exists (ALIAS_MAP) -> use mapped value
      3) Else fill with 0.0 (and log warning)
    """
    vals = []
    for feat in FEATURE_COLUMNS:
        if feat in row and row[feat] is not None:
            try:
                vals.append(float(row[feat]))
                continue
            except Exception:
                logger.warning("Feature %s exists but cannot be cast to float: %s", feat, row[feat])

        # try aliases (old_key -> new_feature)
        found = False
        for oldk, newk in ALIAS_MAP.items():
            if newk == feat and oldk in row and row[oldk] is not None:
                try:
                    vals.append(float(row[oldk]))
                    found = True
                    break
                except Exception:
                    logger.warning("Alias %s present but cannot cast to float: %s", oldk, row[oldk])
        if found:
            continue

        # Not present -> pad with 0.0 but log
        logger.warning("Feature '%s' not found in row; padding with 0.0 (dev fallback).", feat)
        vals.append(0.0)

    X = np.array(vals).reshape(1, -1)

    # If scaler expects more features (rare), pad accordingly
    expected = getattr(scaler, "n_features_in_", None)
    if expected is not None:
        if X.shape[1] < expected:
            pad = expected - X.shape[1]
            logger.warning("After building vector: have %d features, scaler expects %d; padding %d zeros.",
                           X.shape[1], expected, pad)
            X = np.hstack([X, np.zeros((1, pad))])
        elif X.shape[1] > expected:
            logger.error("Built feature vector has %d features but scaler expects %d.", X.shape[1], expected)
            raise HTTPException(status_code=500, detail=f"Feature mismatch: built {X.shape[1]}, scaler expects {expected}")

    return X

def softmax(arr: np.ndarray) -> np.ndarray:
    e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def predict_for_features(X: np.ndarray) -> Tuple[str, Dict[str, float]]:

    """
    Given a 2D feature vector X (1 x n), scale and run model -> return (label, probabilities_dict)
    If model or scaler missing or any error -> return ("Unknown", {})
    """
    if scaler is None:
        logger.warning("Scaler not loaded; cannot scale features for prediction.")
        return "Unknown", {}

    if model is None:
        logger.warning("Model not loaded; cannot produce prediction.")
        return "Unknown", {}

    try:
        Xs = scaler.transform(X)
    except Exception as e:
        logger.error("Error transforming features with scaler: %s", e)
        logger.error(traceback.format_exc())
        return "Unknown", {}

    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(Xs)[0].tolist()
            classes = model.classes_.tolist() if hasattr(model, "classes_") else list(range(len(probs)))
            proba_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
            pred_label = str(model.predict(Xs)[0])
        else:
            df = model.decision_function(Xs)
            probs = softmax(df)[0].tolist()
            classes = getattr(model, "classes_", list(range(len(probs))))
            proba_dict = {str(classes[i]): float(probs[i]) for i in range(len(classes))}
            pred_label = str(classes[int(np.argmax(probs))])
        return pred_label, proba_dict
    except Exception as e:
        logger.error("Model prediction error: %s", e)
        logger.error(traceback.format_exc())
        return "Unknown", {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/latest-prediction")
def latest_prediction():
    try:
        row = fetch_latest_row()

        lat = float(row.get("latitude") or row.get("lat") or 0.0)
        lon = float(row.get("longitude") or row.get("lon") or 0.0)

        X = build_feature_vector_from_row(row)

        pred_label, proba_dict = predict_for_features(X)

        return {
            "label": pred_label,
            "probabilities": proba_dict,
            "latitude": lat,
            "longitude": lon,
            "raw_row": row
        }

    except HTTPException as he:
        logger.error("HTTPException in /latest-prediction: %s", he.detail)
        raise he
    except Exception as e:
        logger.error("Unhandled exception in /latest-prediction: %s", e)
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={
            "error": "Internal server error",
            "exception_type": type(e).__name__,
            "message": str(e)
        })

@app.get("/recent-predictions")
def recent_predictions(limit: int = Query(12, ge=1, le=200)):
    """
    Return last `limit` rows (newest first) with model predictions included.
    Response: list of objects:
      {
        "id": <id>,
        "timestamp": <timestamp string or None>,
        "latitude": <float>,
        "longitude": <float>,
        "label": <pred_label string>,
        "probabilities": { class: prob, ... },
        "raw_row": { ... }  # original DB row
      }
    """
    try:
        rows = fetch_recent_rows(limit)
    except HTTPException as he:
        # bubble up Supabase connectivity issues
        logger.error("HTTPException fetching recent rows: %s", he.detail)
        raise he
    except Exception as e:
        logger.error("Unhandled error in fetch_recent_rows: %s", e)
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to fetch recent rows")

    results: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            # id/timestamp normalization
            row_id = row.get("id", None)
            ts = row.get("timestamp") or row.get("time") or row.get("created_at") or None

            # lat/lon fallback keys (lat/lon or latitude/longitude)
            try:
                lat = float(row.get("latitude") or row.get("lat") or 0.0)
                lon = float(row.get("longitude") or row.get("lon") or 0.0)
            except Exception:
                lat, lon = 0.0, 0.0

            # build features & predict
            X = build_feature_vector_from_row(row)
            label, proba_dict = predict_for_features(X)

            results.append({
                "id": row_id,
                "timestamp": ts,
                "latitude": lat,
                "longitude": lon,
                "label": label,
                "probabilities": proba_dict,
                "raw_row": row
            })
        except Exception as e:
            # per-row failure: include row but mark unknown
            logger.error("Failed to process row index %s: %s", idx, e)
            logger.error(traceback.format_exc())
            results.append({
                "id": row.get("id", None),
                "timestamp": row.get("timestamp", None),
                "latitude": row.get("latitude") or row.get("lat") or 0.0,
                "longitude": row.get("longitude") or row.get("lon") or 0.0,
                "label": "Unknown",
                "probabilities": {},
                "raw_row": row
            })

    # API contract: return newest-first (Supabase already returned order=timestamp.desc)
    return results

@app.get("/__debug/config")
def debug_config():
    return {
        "supabase_url": SUPABASE_URL if SUPABASE_URL else "<not-set>",
        "supabase_table": TABLE_NAME,
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "use_mock_on_supabase_fail": USE_MOCK_ON_SUPABASE_FAIL,
        "feature_columns": FEATURE_COLUMNS,
        "scaler_n_features_in": getattr(scaler, "n_features_in_", None),
        "scaler_feature_names_in": getattr(scaler, "feature_names_in_", None),
    }
