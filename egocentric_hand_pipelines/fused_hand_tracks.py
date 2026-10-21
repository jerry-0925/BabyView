#!/usr/bin/env python3
"""Fuse per-frame MediaPipe hand tracks with sparse YOLO self/other labels."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from mediapipe_hand_tracks import CONNECTIONS, Detection, assign_tracks, resolve_model


def hand_box(detection, padding=0.04):
    xs, ys = zip(*(point[:2] for point in detection.landmarks))
    return max(0, min(xs) - padding), max(0, min(ys) - padding), min(1, max(xs) + padding), min(1, max(ys) + padding)


def iou(first, second):
    left, top, right, bottom = max(first[0], second[0]), max(first[1], second[1]), min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0, right - left) * max(0, bottom - top)
    return overlap / ((first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - overlap) if overlap else 0


def center_distance(first, second):
    return math.dist(((first[0] + first[2]) / 2, (first[1] + first[3]) / 2), ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2))


def match_detections(hands, yolo_hands, min_iou, max_distance):
    pairs = []
    for yolo_index, (yolo_box, _, _) in enumerate(yolo_hands):
        for hand_index, hand in enumerate(hands):
            overlap, distance = iou(yolo_box, hand_box(hand)), center_distance(yolo_box, hand_box(hand))
            if overlap >= min_iou or distance <= max_distance:
                pairs.append((overlap - distance, yolo_index, hand_index))
    matches, used_yolo, used_hands = {}, set(), set()
    for _, yolo_index, hand_index in sorted(pairs, reverse=True):
        if yolo_index not in used_yolo and hand_index not in used_hands:
            matches[yolo_index] = hand_index
            used_yolo.add(yolo_index); used_hands.add(hand_index)
    return matches


def owner_label(state, frame, max_age, threshold):
    if state is None or frame - state["last_yolo_frame"] > max_age:
        return "unknown", 0.0
    if state["self_probability"] >= threshold:
        return "self", state["self_probability"]
    if state["self_probability"] <= 1 - threshold:
        return "other", 1 - state["self_probability"]
    return "unknown", 0.0


def draw_hand(frame, detection, track_id, label, confidence, cv2):
    height, width = frame.shape[:2]
    points = [(int(x * width), int(y * height)) for x, y, _ in detection.landmarks]
    color = {"self": (60, 220, 60), "other": (30, 160, 255), "unknown": (150, 150, 150)}[label]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
    x, y = points[0]
    text = f"T{track_id} {label}" + (f" {confidence:.0%}" if label != "unknown" else "")
    cv2.putText(frame, text, (x + 6, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def yolo_hands(model, frame, device, confidence):
    result = model.predict(frame, device=device, conf=confidence, verbose=False)[0]
    if result.boxes is None:
        return []
    hands = []
    for box in result.boxes:
        label = str(result.names[int(box.cls[0])]).lower()
        if label not in {"self", "other"}:
            raise ValueError(f"YOLO model must use self/other labels, got {label!r}")
        hands.append((tuple(float(value) for value in box.xyxyn[0]), label, float(box.conf[0])))
    return hands


def process(args):
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies: python -m pip install -r requirements.txt") from exc
    if not args.yolo_model.exists():
        raise SystemExit(f"YOLO weights not found: {args.yolo_model}")
    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(args.output / "fused_hand_tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    model, tracks, owners = YOLO(str(args.yolo_model)), {}, {}
    next_id, frame_index, rows = 1, 0, []
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(resolve_model(args.mediapipe_model))),
        running_mode=vision.RunningMode.VIDEO, num_hands=args.max_hands,
        min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55, min_tracking_confidence=0.50,
    )
    with vision.HandLandmarker.create_from_options(options) as hands:
        while True:
            if args.max_frames and frame_index >= args.max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = hands.detect_for_video(image, round(frame_index * 1000 / fps))
            detections = [Detection([(point.x, point.y, point.z) for point in landmarks], categories[0].category_name, categories[0].score)
                          for landmarks, categories in zip(result.hand_landmarks or [], result.handedness or [])]
            ids, next_id = assign_tracks(detections, tracks, frame_index, next_id)
            if frame_index % args.yolo_stride == 0:
                predictions = yolo_hands(model, frame, args.device, args.yolo_confidence)
                for yolo_index, hand_index in match_detections(detections, predictions, args.match_iou, args.match_distance).items():
                    _, label, confidence = predictions[yolo_index]
                    state = owners.setdefault(ids[hand_index], {"self_probability": 0.5, "last_yolo_frame": -10**9})
                    measurement = confidence if label == "self" else 1 - confidence
                    state["self_probability"] = (1 - args.smoothing) * state["self_probability"] + args.smoothing * measurement
                    state["last_yolo_frame"] = frame_index
            for detection, track_id in zip(detections, ids):
                state = owners.get(track_id)
                label, confidence = owner_label(state, frame_index, args.max_label_age, args.label_threshold)
                draw_hand(frame, detection, track_id, label, confidence, cv2)
                rows.append({"frame": frame_index, "time_s": frame_index / fps, "track_id": track_id, "agent": label,
                             "agent_confidence": confidence, "self_probability": state["self_probability"] if state else None,
                             "yolo_age_frames": frame_index - state["last_yolo_frame"] if state else None,
                             "handedness": detection.handedness, "handedness_confidence": detection.handedness_score,
                             "landmarks_normalized": detection.landmarks})
            writer.write(frame); frame_index += 1
    cap.release(); writer.release()
    (args.output / "fused_hand_tracks.json").write_text(json.dumps({"input": str(args.video), "fps": fps, "yolo_model": str(args.yolo_model), "yolo_stride": args.yolo_stride, "rows": rows}))
    fields = ["frame", "time_s", "track_id", "agent", "agent_confidence", "self_probability", "yolo_age_frames", "handedness", "handedness_confidence"]
    with (args.output / "fused_hand_tracks.csv").open("w", newline="") as handle:
        output = csv.DictWriter(handle, fieldnames=fields); output.writeheader()
        output.writerows({field: row[field] for field in fields} for row in rows)
    print(f"Wrote {len(rows)} hand observations to {args.output}")


def self_test():
    assert round(iou((0, 0, 1, 1), (0.5, 0.5, 1, 1)), 2) == 0.25
    detection = Detection([(0.4, 0.4, 0), (0.6, 0.6, 0)] * 11, "Left", 0.9)
    assert match_detections([detection], [((0.39, 0.39, 0.61, 0.61), "self", 0.9)], 0.1, 0.12) == {0: 0}
    assert owner_label({"self_probability": 0.9, "last_yolo_frame": 10}, 12, 10, 0.75)[0] == "self"
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--yolo-model", type=Path, help="Two-class YOLO checkpoint with self and other labels.")
    parser.add_argument("--output", type=Path, default=Path("outputs/fused"))
    parser.add_argument("--mediapipe-model", type=Path, default=Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--device", default="mps", help="Use cpu on machines without Apple MPS.")
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--yolo-stride", type=int, default=5)
    parser.add_argument("--yolo-confidence", type=float, default=0.35)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--match-distance", type=float, default=0.12)
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--label-threshold", type=float, default=0.55)
    parser.add_argument("--max-label-age", type=int, default=45)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.video is None or args.yolo_model is None:
        parser.error("Pass a video path and --yolo-model /path/to/best.pt.")
    if args.yolo_stride < 1 or args.max_frames < 0 or not 0 < args.smoothing <= 1 or not 0.5 <= args.label_threshold < 1:
        parser.error("Check --yolo-stride, --max-frames, --smoothing, and --label-threshold values.")
    process(args)


if __name__ == "__main__":
    main()
