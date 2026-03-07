"""Atomic Activity Encoding Service.

Implements Level-1 and Level-2 encoding for 6-dimensional atomic activities:
- Format A (time-sliced joint view): Timeline slices showing all dimensions together
- Format B (per-dimension RLE): Run-length encoding for each dimension independently

Dimensions:
1. HAR (har_label): Sitting, Walking, Running, etc.
2. Location: home, office, shopping_mall, etc.
3. Phone Usage: idle, low, medium, high, very high
4. Steps (step_count): almost stationary, low, medium, high, very high
5. Movement: stationary, slow, medium, fast
6. App Category: social communication app, video and music app, etc.
"""

import logging
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

CHINA_TZ = ZoneInfo("Asia/Shanghai")


def run_length_encode(labels: list[str], timestamps: list[datetime]) -> list[tuple[str, float, float]]:
    """
    Perform run-length encoding on a sequence of labels with timestamps.

    Args:
        labels: List of label strings
        timestamps: List of corresponding timestamps

    Returns:
        List of (label, start_minute, end_minute) tuples
    """
    if not labels or not timestamps:
        return []

    if len(labels) != len(timestamps):
        logger.warning("Labels and timestamps length mismatch")
        return []

    # Sort by timestamp
    sorted_pairs = sorted(zip(timestamps, labels), key=lambda x: x[0])

    if not sorted_pairs:
        return []

    # Get the start time for relative minutes
    start_time = sorted_pairs[0][0]

    rle_segments = []
    current_label = sorted_pairs[0][1]
    segment_start = 0.0  # minutes from start

    for i, (ts, label) in enumerate(sorted_pairs):
        # Calculate minutes from start
        minutes_from_start = (ts - start_time).total_seconds() / 60.0

        if label != current_label:
            # End previous segment
            rle_segments.append((current_label, segment_start, minutes_from_start))
            # Start new segment
            current_label = label
            segment_start = minutes_from_start

    # Add the last segment
    if sorted_pairs:
        last_ts = sorted_pairs[-1][0]
        end_minute = (last_ts - start_time).total_seconds() / 60.0
        rle_segments.append((current_label, segment_start, end_minute))

    return rle_segments


def format_rle_string(rle_segments: list[tuple[str, float, float]]) -> str:
    """Format RLE segments into a compact string like 'Sitting(0-2)|Walking(2-3)'."""
    if not rle_segments:
        return ""

    parts = []
    for label, start, end in rle_segments:
        parts.append(f"{label}({start:.0f}-{end:.0f})")

    return "|".join(parts)


def format_timeline_compact(rle_segments: list[tuple[str, float, float]], prefix: str) -> str:
    """Format RLE segments into a timeline string with prefix."""
    if not rle_segments:
        return ""

    parts = []
    for label, start, end in rle_segments:
        parts.append(f"{label}({start:.0f}-{end:.0f})")

    return f"{prefix}: " + " -> ".join(parts)


def aggregate_top_labels(labels: list[str], top_n: int = 3) -> list[tuple[str, float]]:
    """
    Get top N labels with their total duration (in minutes).
    Since we don't have duration per label, we count occurrences and estimate.

    Args:
        labels: List of label strings
        top_n: Number of top labels to return

    Returns:
        List of (label, estimated_minutes) tuples
    """
    if not labels:
        return []

    counter = Counter(labels)
    total_count = len(labels)

    # Estimate minutes based on proportion
    # Assuming atomic activities are generated every ~10 seconds
    estimated_total_minutes = total_count * (10 / 60)  # 10 seconds per sample

    result = []
    for label, count in counter.most_common(top_n):
        estimated_minutes = (count / total_count) * estimated_total_minutes
        result.append((label, round(estimated_minutes, 3)))

    return result


def generate_macro_timeline(
    har_rle: list[tuple[str, float, float]],
    location_rle: list[tuple[str, float, float]],
    phone_rle: list[tuple[str, float, float]],
    steps_rle: list[tuple[str, float, float]],
    movement_rle: list[tuple[str, float, float]],
    app_rle: list[tuple[str, float, float]],
    window_duration_min: float,
    slot_size_min: float = 5.0
) -> list[str]:
    """
    Generate macro timeline with time slots showing all dimensions together.

    Args:
        *_rle: RLE segments for each dimension
        window_duration_min: Total window duration in minutes
        slot_size_min: Size of each time slot (default 5 minutes)

    Returns:
        List of human-readable timeline strings
    """
    if window_duration_min <= 0:
        return []

    timeline = []
    num_slots = max(1, int(window_duration_min / slot_size_min))

    def get_label_at_time(rle_segments: list[tuple[str, float, float]], time_min: float) -> str:
        """Get the label at a specific time."""
        if not rle_segments:
            return "unknown"
        for label, start, end in rle_segments:
            if start <= time_min < end:
                return label
        # If past the last segment, return the last label
        if rle_segments:
            return rle_segments[-1][0]
        return "unknown"

    def get_dominant_label(rle_segments: list[tuple[str, float, float]],
                          start_min: float, end_min: float) -> str:
        """Get the dominant label in a time range."""
        if not rle_segments:
            return "unknown"

        label_durations = {}
        for label, seg_start, seg_end in rle_segments:
            # Calculate overlap with the time range
            overlap_start = max(seg_start, start_min)
            overlap_end = min(seg_end, end_min)
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                label_durations[label] = label_durations.get(label, 0) + duration

        if not label_durations:
            return "unknown"

        return max(label_durations, key=label_durations.get)

    for slot_idx in range(num_slots):
        slot_start = slot_idx * slot_size_min
        slot_end = min((slot_idx + 1) * slot_size_min, window_duration_min)

        # Get dominant labels for each dimension
        har = get_dominant_label(har_rle, slot_start, slot_end)
        loc = get_dominant_label(location_rle, slot_start, slot_end)
        phone = get_dominant_label(phone_rle, slot_start, slot_end)
        steps = get_dominant_label(steps_rle, slot_start, slot_end)
        movement = get_dominant_label(movement_rle, slot_start, slot_end)
        app = get_dominant_label(app_rle, slot_start, slot_end)

        # Format the timeline entry
        entry = (f"{slot_start:.0f}-{slot_end:.0f} min | "
                f"act={har} | loc={loc} | phone={phone} | "
                f"steps={steps} | disp={movement} | app={app}")
        timeline.append(entry)

    return timeline


def encode_atomic_activities(
    records: list[dict],
    window_duration_min: Optional[float] = None
) -> dict:
    """
    Encode atomic activities into Level-1 and Level-2 formats.

    Args:
        records: List of atomic activity records from database
        window_duration_min: Optional window duration (calculated from data if not provided)

    Returns:
        Dictionary containing level1_temporal_view and level2_compact_view
    """
    if not records:
        return {
            "window_meta": {
                "duration_min": 0,
                "token_minutes": 1.0
            },
            "level2_compact_view": {},
            "level1_temporal_view": {
                "timeline_compact": [],
                "macro_timeline": [],
                "rle_exact_compact": {}
            }
        }

    # Extract data by dimension with timestamps
    har_data = []  # (timestamp, label)
    location_data = []
    phone_data = []
    steps_data = []
    movement_data = []
    app_data = []

    # Also collect raw labels for aggregation
    har_labels = []
    location_labels = []
    phone_labels = []
    steps_labels = []
    movement_labels = []
    app_labels = []

    for record in records:
        ts = record.get("timestamp")
        if ts is None:
            continue

        # Ensure timestamp is datetime
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        elif isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=CHINA_TZ)

        har_label = record.get("har_label")
        if har_label and har_label != "unknown":
            har_data.append((ts, har_label))
            har_labels.append(har_label)

        location = record.get("location")
        if location:
            location_data.append((ts, location))
            location_labels.append(location)

        phone_usage = record.get("phone_usage")
        if phone_usage:
            phone_data.append((ts, phone_usage))
            phone_labels.append(phone_usage)

        step_count = record.get("step_count")
        if step_count:
            steps_data.append((ts, step_count))
            steps_labels.append(step_count)

        movement = record.get("movement")
        if movement:
            movement_data.append((ts, movement))
            movement_labels.append(movement)

        app_category = record.get("app_category")
        if app_category:
            app_data.append((ts, app_category))
            app_labels.append(app_category)

    # Calculate window duration
    if window_duration_min is None:
        all_timestamps = [ts for ts, _ in har_data + location_data + phone_data +
                         steps_data + movement_data + app_data]
        if all_timestamps:
            duration = (max(all_timestamps) - min(all_timestamps)).total_seconds() / 60.0
            window_duration_min = max(duration, 1.0)  # At least 1 minute
        else:
            window_duration_min = 0

    # Perform RLE for each dimension
    har_rle = run_length_encode([l for _, l in har_data], [t for t, _ in har_data])
    location_rle = run_length_encode([l for _, l in location_data], [t for t, _ in location_data])
    phone_rle = run_length_encode([l for _, l in phone_data], [t for t, _ in phone_data])
    steps_rle = run_length_encode([l for _, l in steps_data], [t for t, _ in steps_data])
    movement_rle = run_length_encode([l for _, l in movement_data], [t for t, _ in movement_data])
    app_rle = run_length_encode([l for _, l in app_data], [t for t, _ in app_data])

    # Generate Level-1 Temporal View
    timeline_compact = [
        format_timeline_compact(location_rle, "LOC"),
        format_timeline_compact(har_rle, "HAR"),
        format_timeline_compact(phone_rle, "PHONE"),
        format_timeline_compact(app_rle, "APP(cat)"),
    ]

    macro_timeline = generate_macro_timeline(
        har_rle, location_rle, phone_rle, steps_rle, movement_rle, app_rle,
        window_duration_min
    )

    rle_exact_compact = {
        "HAR": format_rle_string(har_rle),
        "Location": format_rle_string(location_rle),
        "Phone": format_rle_string(phone_rle),
        "Steps": format_rle_string(steps_rle),
        "Displacement": format_rle_string(movement_rle),
        "App": format_rle_string(app_rle),
    }

    # Generate Level-2 Compact View
    level2_compact_view = {
        "activity_top": aggregate_top_labels(har_labels, 3),
        "place_top": aggregate_top_labels(location_labels, 3),
        "steps_distribution": aggregate_top_labels(steps_labels, 5),
        "movement_distribution": aggregate_top_labels(movement_labels, 4),
        "phone_distribution": aggregate_top_labels(phone_labels, 5),
        "app_top": aggregate_top_labels(app_labels, 5),
    }

    return {
        "window_meta": {
            "duration_min": round(window_duration_min, 3),
            "token_minutes": 1.0
        },
        "level2_compact_view": level2_compact_view,
        "level1_temporal_view": {
            "timeline_compact": timeline_compact,
            "macro_timeline": macro_timeline,
            "rle_exact_compact": rle_exact_compact
        }
    }