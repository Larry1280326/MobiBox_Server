"""
Test HAR pipeline: fetch IMU from Supabase, run model inference, save result to CSV.

Run from project root:
  pytest:  python -m pytest src/test/test_imu.py -v -s
  script:  python src/test/test_imu.py
Requires .env with Supabase credentials and HAR model checkpoint configured.
"""

import csv
from pathlib import Path

import pytest

from src.database import get_supabase_client
from src.celery_app.services.har_service import (
    _get_imu_model,
    _imu_data_to_tensor,
    _run_imu_model_sync,
    HAR_LABEL_BY_INDEX,
    HAR_IMU_WINDOW_SIZE,
)

# Query window and user for this test
USER = "samsumg_test"
START_TS = "2026-03-02T22:20:54.905+00:00"
END_TS = "2026-03-02T22:21:36.280+00:00"


def fetch_imu_in_range(user: str, start_iso: str, end_iso: str) -> list[dict]:
    """Fetch IMU rows from Supabase for user in [start_iso, end_iso] (sync)."""
    client = get_supabase_client()
    response = (
        client.table("imu")
        .select("*")
        .eq("user", user)
        .gte("timestamp", start_iso)
        .lte("timestamp", end_iso)
        .order("timestamp", desc=False)
        .execute()
    )
    return response.data if response.data else []


def run_inference_on_windows(imu_data: list[dict]) -> list[tuple[str, str, str]]:
    """
    Split IMU data into non-overlapping windows of HAR_IMU_WINDOW_SIZE,
    run model on each window, return list of (start_ts, end_ts, label).
    """
    model, available = _get_imu_model()
    if not available or model is None:
        raise RuntimeError(
            "IMU model not available; set HAR_IMU_MODEL_CHECKPOINT and ensure checkpoint exists"
        )

    results = []
    n = len(imu_data)
    if n < HAR_IMU_WINDOW_SIZE:
        tensor = _imu_data_to_tensor(imu_data)
        pred_idx, _ = _run_imu_model_sync(tensor)
        label = HAR_LABEL_BY_INDEX[pred_idx] if pred_idx < len(HAR_LABEL_BY_INDEX) else "unknown"
        start_ts = imu_data[0].get("timestamp", "") if imu_data else ""
        end_ts = imu_data[-1].get("timestamp", "") if imu_data else ""
        results.append((start_ts, end_ts, label))
        return results

    for start in range(0, n - HAR_IMU_WINDOW_SIZE + 1, HAR_IMU_WINDOW_SIZE):
        end = start + HAR_IMU_WINDOW_SIZE
        window = imu_data[start:end]
        tensor = _imu_data_to_tensor(window)
        pred_idx, _ = _run_imu_model_sync(tensor)
        label = HAR_LABEL_BY_INDEX[pred_idx] if pred_idx < len(HAR_LABEL_BY_INDEX) else "unknown"
        start_ts = window[0].get("timestamp", "")
        end_ts = window[-1].get("timestamp", "")
        results.append((start_ts, end_ts, label))

    return results


def save_results_csv(rows: list[tuple[str, str, str]], out_path: Path) -> None:
    """Write (start_timestamp, end_timestamp, label) to CSV."""
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start_timestamp", "end_timestamp", "label"])
        w.writerows(rows)


def test_har_fetch_infer_save_csv():
    """
    1. Fetch IMU from Supabase (user=samsumg_test, 2026-03-02 22:20:54.905 to 22:21:36.28).
    2. Run HAR model inference on windows.
    3. Save (start_timestamp, end_timestamp, label) to CSV.
    """
    imu_data = fetch_imu_in_range(USER, START_TS, END_TS)
    assert len(imu_data) > 0, "No IMU data in range; check Supabase and time range"

    results = run_inference_on_windows(imu_data)

    out_path = Path(__file__).resolve().parent / "har_results.csv"
    save_results_csv(results, out_path)

    assert len(results) >= 1
    assert out_path.is_file()


if __name__ == "__main__":
    imu_data = fetch_imu_in_range(USER, START_TS, END_TS)
    if not imu_data:
        raise SystemExit("No IMU data in range; check Supabase and time range")
    results = run_inference_on_windows(imu_data)
    out_path = Path(__file__).resolve().parent / "har_results.csv"
    save_results_csv(results, out_path)
    print(f"Saved {len(results)} rows to {out_path}")
