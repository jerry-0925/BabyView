#!/usr/bin/env python3
"""Track YOLO hand boxes, smooth ownership votes, and attach optional MediaPipe landmarks."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path

if __package__:
    from .mediapipe_hand_tracks import CONNECTIONS, Detection, resolve_model
else:
    from mediapipe_hand_tracks import CONNECTIONS, Detection, resolve_model


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
            used_yolo.add(yolo_index)
            used_hands.add(hand_index)
    return matches


@dataclass
class BoxTrack:
    box: tuple
    last_frame: int
    label: str
    confidence: float
    velocity: tuple = (0.0, 0.0)
    owner: dict = field(default_factory=dict)

    def predicted_box(self, frame):
        dx, dy = (value * (frame - self.last_frame) for value in self.velocity)
        # Keep box size intact when a prediction reaches an image edge.
        dx = max(-self.box[0], min(dx, 1 - self.box[2]))
        dy = max(-self.box[1], min(dy, 1 - self.box[3]))
        return self.box[0] + dx, self.box[1] + dy, self.box[2] + dx, self.box[3] + dy


def assign_box_tracks(predictions, tracks, frame, next_id, max_gap=15):
    """Associate YOLO boxes independently of landmarks and ownership class."""
    for track_id in list(tracks):
        if frame - tracks[track_id].last_frame > max_gap:
            del tracks[track_id]
    pairs = []
    for index, (box, _, _) in enumerate(predictions):
        for track_id, track in tracks.items():
            predicted = track.predicted_box(frame)
            distance, overlap = center_distance(box, predicted), iou(box, predicted)
            scale = max(math.hypot(b[2] - b[0], b[3] - b[1]) for b in (box, predicted))
            if overlap >= 0.10 or distance <= min(0.12, scale / 2):
                pairs.append((overlap - distance, index, track_id))
    ids, used_tracks = [None] * len(predictions), set()
    # shortcut: geometric association can swap IDs at crossings; evaluate a dedicated tracker if this dominates errors.
    for _, index, track_id in sorted(pairs, reverse=True):
        if ids[index] is None and track_id not in used_tracks:
            ids[index] = track_id
            used_tracks.add(track_id)
    for index, (box, label, confidence) in enumerate(predictions):
        track_id = ids[index]
        if track_id is None:
            track_id, next_id = next_id, next_id + 1
            tracks[track_id] = BoxTrack(box, frame, label, confidence)
            ids[index] = track_id
        else:
            track = tracks[track_id]
            elapsed = max(1, frame - track.last_frame)
            movement = tuple(((box[axis] + box[axis + 2]) - (track.box[axis] + track.box[axis + 2])) / (2 * elapsed) for axis in (0, 1))
            track.velocity = tuple(0.5 * old + 0.5 * new for old, new in zip(track.velocity, movement))
            track.box, track.last_frame, track.label, track.confidence = box, frame, label, confidence
    return ids, next_id


def update_owner(state, label, confidence, frame, smoothing, threshold, max_age, stride, switch_updates):
    """Signed confidence-weighted votes; a weak vote never supports the opposite class."""
    if label not in {"self", "other"} or not 0 <= confidence <= 1:
        raise ValueError("Expected a self/other detection with confidence in [0, 1].")
    if not state or frame - state["last_yolo_frame"] > max_age:
        state.clear()
        state.update(evidence=0.0, last_yolo_frame=frame, label="unknown", pending=None, pending_count=0)
    gap = max(1.0, (frame - state["last_yolo_frame"]) / stride)
    vote = confidence if label == "self" else -confidence
    state["evidence"] = (1 - smoothing) ** gap * state["evidence"] + smoothing * vote
    state["last_yolo_frame"] = frame
    score = (1 + state["evidence"]) / 2
    candidate = "self" if score >= threshold else "other" if score <= 1 - threshold else "unknown"
    if candidate == "unknown" or candidate == state["label"]:
        state["pending"], state["pending_count"] = None, 0
    elif state["label"] == "unknown":
        state["label"] = candidate
    else:
        state["pending_count"] = state["pending_count"] + 1 if state["pending"] == candidate else 1
        state["pending"] = candidate
        if state["pending_count"] >= switch_updates:
            state["label"] = candidate
            state["pending"], state["pending_count"] = None, 0


def owner_label(state, frame, max_age):
    if not state or frame - state["last_yolo_frame"] > max_age:
        return "unknown", None
    label = state["label"]
    score = (1 + state["evidence"]) / 2
    return label, score if label == "self" else 1 - score if label == "other" else None


def observation(track, track_id, detection, frame, fps, max_age):
    label, score = owner_label(track.owner, frame, max_age)
    box = track.predicted_box(frame)
    return {"frame": frame, "time_s": frame / fps, "track_id": track_id,
            "agent": label, "agent_score": score,
            "self_vote_score": (1 + track.owner["evidence"]) / 2 if track.owner else None,
            "detection_confidence": track.confidence, "yolo_label": track.label,
            "yolo_age_frames": frame - track.last_frame,
            "observation_source": "yolo" if frame == track.last_frame else "prediction",
            "bbox_x1": box[0], "bbox_y1": box[1], "bbox_x2": box[2], "bbox_y2": box[3],
            "landmarks_available": detection is not None,
            "handedness": detection.handedness if detection else None,
            "handedness_confidence": detection.handedness_score if detection else None,
            "landmarks_normalized": detection.landmarks if detection else None}


def draw_hand(frame, row, cv2):
    height, width = frame.shape[:2]
    color = {"self": (60, 220, 60), "other": (30, 160, 255), "unknown": (150, 150, 150)}[row["agent"]]
    x, y = int(row["bbox_x1"] * width), int(row["bbox_y1"] * height)
    cv2.rectangle(frame, (x, y), (int(row["bbox_x2"] * width), int(row["bbox_y2"] * height)), color, 2)
    if row["landmarks_available"]:
        points = [(int(px * width), int(py * height)) for px, py, _ in row["landmarks_normalized"]]
        for start, end in CONNECTIONS:
            cv2.line(frame, points[start], points[end], color, 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)
    suffix = " predicted" if row["observation_source"] == "prediction" else ""
    suffix += " box only" if not row["landmarks_available"] else ""
    text = f"T{row['track_id']} {row['agent']}{suffix}"
    cv2.putText(frame, text, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def yolo_hands(model, frame, device, confidence, max_hands):
    result = model.predict(frame, device=device, conf=confidence, max_det=max_hands,
                           agnostic_nms=True, verbose=False)[0]
    if result.boxes is None:
        return []
    return [(tuple(float(value) for value in box.xyxyn[0]), str(result.names[int(box.cls[0])]).lower(),
             float(box.conf[0])) for box in result.boxes]


def process(args):
    try:
        import cv2
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install dependencies: python -m pip install -r requirements.txt") from exc
    if not args.yolo_model.exists():
        raise SystemExit(f"YOLO weights not found: {args.yolo_model}")
    model = YOLO(str(args.yolo_model))
    if {str(name).lower() for name in model.names.values()} != {"self", "other"} or len(model.names) != 2:
        raise SystemExit("YOLO model must have exactly two classes named self and other.")
    hand_context = nullcontext(None)
    if not args.no_landmarks:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise SystemExit("Install mediapipe or pass --no-landmarks for YOLO-only tracking.") from exc
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(resolve_model(args.mediapipe_model))),
            running_mode=vision.RunningMode.VIDEO, num_hands=args.max_hands,
            min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55, min_tracking_confidence=0.50,
        )
        hand_context = vision.HandLandmarker.create_from_options(options)
    with hand_context as hands:
        cap = cv2.VideoCapture(str(args.video))
        writer = None
        try:
            if not cap.isOpened():
                raise SystemExit(f"Cannot open video: {args.video}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if math.isfinite(fps) and fps > 0 else 30.0
            width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            args.output.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.output / "fused_hand_tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise SystemExit(f"Cannot write video in {args.output}")
            tracks, next_id, frame_index, rows, active_ids = {}, 1, 0, [], []
            while not args.max_frames or frame_index < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % args.yolo_stride == 0:
                    predictions = yolo_hands(model, frame, args.device, args.yolo_confidence, args.max_hands)
                    active_ids, next_id = assign_box_tracks(predictions, tracks, frame_index, next_id, args.max_track_age)
                    for track_id in active_ids:
                        track = tracks[track_id]
                        update_owner(track.owner, track.label, track.confidence, frame_index, args.smoothing,
                                     args.label_threshold, args.max_label_age, args.yolo_stride, args.switch_updates)
                # A failed YOLO pass clears active_ids; retained tracks can be reacquired but are not rendered as detections.
                active_ids = [track_id for track_id in active_ids if frame_index - tracks[track_id].last_frame <= args.max_track_age]
                detections = []
                if hands is not None:
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = hands.detect_for_video(image, round(frame_index * 1000 / fps))
                    detections = [Detection([(point.x, point.y, point.z) for point in landmarks], categories[0].category_name, categories[0].score)
                                  for landmarks, categories in zip(result.hand_landmarks or [], result.handedness or [])]
                boxes = [(tracks[track_id].predicted_box(frame_index), tracks[track_id].label, tracks[track_id].confidence) for track_id in active_ids]
                matches = match_detections(detections, boxes, args.match_iou, args.match_distance)
                for index, track_id in enumerate(active_ids):
                    detection = detections[matches[index]] if index in matches else None
                    row = observation(tracks[track_id], track_id, detection, frame_index, fps, args.max_label_age)
                    draw_hand(frame, row, cv2)
                    rows.append(row)
                writer.write(frame)
                frame_index += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
    metadata = {"schema_version": 2, "input": str(args.video), "fps": fps,
                "frames_processed": frame_index, "yolo_model": str(args.yolo_model),
                "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "score_semantics": "confidence-weighted ownership votes, not calibrated probabilities", "rows": rows}
    (args.output / "fused_hand_tracks.json").write_text(json.dumps(metadata, allow_nan=False))
    fields = ["frame", "time_s", "track_id", "agent", "agent_score", "self_vote_score", "detection_confidence",
              "yolo_label", "yolo_age_frames", "observation_source", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
              "landmarks_available", "handedness", "handedness_confidence"]
    with (args.output / "fused_hand_tracks.csv").open("w", newline="") as handle:
        output = csv.DictWriter(handle, fieldnames=fields)
        output.writeheader()
        output.writerows({key: row[key] for key in fields} for row in rows)
    print(f"Wrote {len(rows)} hand observations ({sum(not row['landmarks_available'] for row in rows)} without landmarks) to {args.output}")


def self_test():
    if __package__:
        from .test_fused_hand_tracks import run_tests
    else:
        from test_fused_hand_tracks import run_tests
    run_tests()


def main(default_mediapipe_model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", type=Path)
    parser.add_argument("--yolo-model", type=Path, default=Path(__file__).parent / "models/self_other_yolo11s_75e.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/fused"))
    parser.add_argument("--mediapipe-model", type=Path, default=default_mediapipe_model or Path(__file__).with_name("hand_landmarker.task"))
    parser.add_argument("--no-landmarks", action="store_true", help="Run YOLO boxes/ownership without loading MediaPipe.")
    parser.add_argument("--device", default="mps", help="Use cpu on machines without Apple MPS.")
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--yolo-stride", type=int, default=1, help="Detect every N frames; skipped frames contain explicitly marked predicted boxes.")
    parser.add_argument("--yolo-confidence", type=float, default=0.35)
    parser.add_argument("--match-iou", type=float, default=0.10, help="YOLO-to-landmark association only.")
    parser.add_argument("--match-distance", type=float, default=0.12, help="YOLO-to-landmark normalized center-distance gate.")
    parser.add_argument("--smoothing", type=float, default=0.35)
    parser.add_argument("--label-threshold", type=float, default=0.55, help="Ownership vote threshold, not a calibrated probability.")
    parser.add_argument("--switch-updates", type=int, default=3, help="Consecutive opposite vote-threshold crossings required to change a settled label.")
    parser.add_argument("--max-label-age", type=int, default=45)
    parser.add_argument("--max-track-age", type=int, default=15, help="Retain a missing track for this many source frames for reacquisition.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.video is None:
        parser.error("Pass a video path.")
    if (args.yolo_stride < 1 or args.max_frames < 0 or args.max_hands < 1 or args.switch_updates < 1
            or args.max_label_age < 0 or args.max_track_age < args.yolo_stride
            or not 0 < args.smoothing <= 1 or not 0.5 < args.label_threshold < 1
            or not 0 < args.yolo_confidence <= 1 or not 0 < args.match_iou <= 1 or not 0 <= args.match_distance <= 1):
        parser.error("Invalid thresholds/counts; --max-track-age must be >= --yolo-stride and --label-threshold must be in (0.5, 1).")
    process(args)


if __name__ == "__main__":
    main()
