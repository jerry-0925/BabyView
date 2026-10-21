#!/usr/bin/env python3
"""Track hands and export 21-point MediaPipe landmarks from a video."""
from __future__ import annotations

import argparse
import csv
import json
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Detection:
    landmarks: list[tuple[float, float, float]]
    handedness: str
    handedness_score: float

    @property
    def center(self) -> tuple[float, float]:
        return (sum(point[0] for point in self.landmarks) / len(self.landmarks),
                sum(point[1] for point in self.landmarks) / len(self.landmarks))


@dataclass
class Track:
    id: int
    last_center: tuple[float, float]
    last_frame: int


def assign_tracks(detections, tracks, frame, next_id, max_gap=12, max_distance=0.18):
    """Greedily match the small number of hands by normalized image distance."""
    for track_id in [key for key, track in tracks.items() if frame - track.last_frame > max_gap]:
        del tracks[track_id]
    pairs = sorted((math.dist(detection.center, track.last_center), detection_index, track_id)
                   for detection_index, detection in enumerate(detections) for track_id, track in tracks.items())
    assigned_detections, assigned_tracks, ids = set(), set(), [None] * len(detections)
    for distance, detection_index, track_id in pairs:
        if distance > max_distance or detection_index in assigned_detections or track_id in assigned_tracks:
            continue
        ids[detection_index] = track_id
        assigned_detections.add(detection_index); assigned_tracks.add(track_id)
    for detection_index, detection in enumerate(detections):
        track_id = ids[detection_index]
        if track_id is None:
            track_id, next_id = next_id, next_id + 1
            tracks[track_id] = Track(track_id, detection.center, frame)
        tracks[track_id].last_center, tracks[track_id].last_frame = detection.center, frame
        ids[detection_index] = track_id
    return ids, next_id


CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
               (13,17),(17,18),(18,19),(19,20),(0,17))
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def resolve_model(model_path: Path) -> Path:
    if not model_path.exists():
        print(f"Downloading MediaPipe hand model to {model_path} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
        except OSError as exc:
            raise SystemExit(f"Download {MODEL_URL} to {model_path} and retry.") from exc
    return model_path


def draw_hand(frame, detection, track_id, cv2):
    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in detection.landmarks]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 255), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, (0, 220, 255), -1, cv2.LINE_AA)
    x, y = points[0]
    cv2.putText(frame, f"T{track_id}", (x + 6, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2, cv2.LINE_AA)


def process(args):
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise SystemExit("Install dependencies: python -m pip install -r requirements.txt") from exc
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output / "hand_tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (width, height))
    tracks, next_id, frame_index, rows = {}, 1, 0, []
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(resolve_model(args.model))),
        running_mode=vision.RunningMode.VIDEO, num_hands=args.max_hands,
        min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55, min_tracking_confidence=0.50,
    )
    with vision.HandLandmarker.create_from_options(options) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % args.stride:
                frame_index += 1; continue
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = hands.detect_for_video(image, round(frame_index * 1000 / fps))
            detections = [Detection([(point.x, point.y, point.z) for point in landmarks], categories[0].category_name, categories[0].score)
                          for landmarks, categories in zip(result.hand_landmarks or [], result.handedness or [])]
            ids, next_id = assign_tracks(detections, tracks, frame_index, next_id)
            for detection, track_id in zip(detections, ids):
                draw_hand(frame, detection, track_id, cv2)
                rows.append({"frame": frame_index, "time_s": frame_index / fps, "track_id": track_id,
                             "handedness": detection.handedness, "handedness_confidence": detection.handedness_score,
                             "landmarks_normalized": detection.landmarks})
            writer.write(frame); frame_index += 1
    cap.release(); writer.release()
    (args.output / "hand_tracks.json").write_text(json.dumps({"input": str(args.video), "fps": fps, "stride": args.stride, "rows": rows}))
    fields = ["frame", "time_s", "track_id", "handedness", "handedness_confidence"]
    with (args.output / "hand_tracks.csv").open("w", newline="") as handle:
        output = csv.DictWriter(handle, fieldnames=fields); output.writeheader()
        output.writerows({field: row[field] for field in fields} for row in rows)
    print(f"Wrote {len(rows)} hand observations to {args.output}")


def self_test():
    detection = Detection([(0.2, 0.2, 0)] * 21, "Left", 0.9)
    ids, _ = assign_tracks([detection], {}, 0, 1)
    assert ids == [1]
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/mediapipe"))
    parser.add_argument("--model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.video is None:
        parser.error("Pass a video path.")
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    process(args)


if __name__ == "__main__":
    main()
