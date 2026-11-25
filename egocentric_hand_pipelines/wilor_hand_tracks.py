#!/usr/bin/env python3
"""Experimental WiLoR MLX 2D detector with independent YOLO ownership evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

from fused_hand_tracks import assign_box_tracks, iou, owner_label, update_owner, yolo_hands
from mediapipe_hand_tracks import CONNECTIONS


def match_ownership(boxes, owners, threshold=0.2):
    pairs = sorted(((iou(box, owner[0]), index, other)
                    for index, box in enumerate(boxes)
                    for other, owner in enumerate(owners)), reverse=True)
    matches, used = {}, set()
    for overlap, index, other in pairs:
        if overlap >= threshold and index not in matches and other not in used:
            matches[index] = other
            used.add(other)
    return matches


def normalize(hand, width, height):
    box = tuple(max(0, min(1, value / scale)) for value, scale in zip(hand.bbox, (width, height, width, height)))
    points = [(x / width, y / height) for x, y in hand.keypoints_2d]
    if len(points) != 21 or not all(math.isfinite(v) for point in points for v in point):
        raise ValueError("Expected 21 finite 2D keypoints from WiLoR.")
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("WiLoR returned a degenerate box.")
    return box, points


def load_detector(path):
    import mlx.core as mx
    import numpy as np
    from safetensors.numpy import load_file
    from wilor_mlx.detector import HandDetector
    from wilor_mlx.pipeline import HandPosePipeline
    model = HandDetector()
    arrays = load_file(str(path))
    # The port computes DFL directly and defines strides as constants.
    dfl = arrays.pop("pose.dfl.conv.weight", None)
    if dfl is not None and not np.array_equal(dfl.reshape(-1), np.arange(16)):
        raise ValueError("Unexpected DFL projection weights.")
    arrays["pose.stride"] = np.array([8., 16., 32.], dtype=np.float32)
    model.load_weights([(key, mx.array(value)) for key, value in arrays.items()], strict=True)
    model.eval()
    return HandPosePipeline(model, None)


def draw(frame, row, cv2):
    h, w = frame.shape[:2]
    color = {"self": (60, 220, 60), "other": (30, 160, 255), "unknown": (150, 150, 150)}[row["agent"]]
    x1, y1, x2, y2 = row["bbox_normalized"]
    cv2.rectangle(frame, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), color, 2)
    points = [(int(x*w), int(y*h)) for x, y in row["keypoints_2d_normalized"]]
    for start, end in CONNECTIONS:
        cv2.line(frame, points[start], points[end], color, 1, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, 2, color, -1)
    cv2.putText(frame, f"T{row['track_id']} {row['agent']}", (int(x1*w), max(16, int(y1*h)-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def process(args):
    import cv2
    import numpy as np
    from ultralytics import YOLO
    detector = load_detector(args.detector)
    ownership = YOLO(str(args.yolo_model))
    if set(ownership.names.values()) != {"self", "other"}:
        raise ValueError("Ownership model must have exactly self/other classes.")
    cap = cv2.VideoCapture(str(args.video))
    baseline = cv2.VideoCapture(str(args.baseline_video)) if args.baseline_video else None
    writer = comparison = None
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be positive.")
        if baseline and (not baseline.isOpened() or abs(baseline.get(cv2.CAP_PROP_FPS)-fps) > 0.01):
            raise ValueError("Baseline video must be readable and use the same FPS.")
        args.output.mkdir(parents=True, exist_ok=False)
        writer = cv2.VideoWriter(str(args.output/"wilor_hand_tracks.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if baseline:
            comparison = cv2.VideoWriter(str(args.output/"comparison.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2*width, height+28))
        if not writer.isOpened() or (comparison and not comparison.isOpened()):
            raise ValueError("Cannot create video output.")
        tracks, next_id, rows, frame_index = {}, 1, [], 0
        started = time.perf_counter()
        while not args.max_frames or frame_index < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            hands = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf_threshold=args.confidence,
                             include_3d=False, include_vertices=False)[:args.max_hands]
            converted = [normalize(hand, width, height) for hand in hands]
            boxes = [box for box, _ in converted]
            predictions = [(box, "unknown", hand.confidence) for box, hand in zip(boxes, hands)]
            ids, next_id = assign_box_tracks(predictions, tracks, frame_index, next_id)
            owners = yolo_hands(ownership, frame, args.device, 0.35, args.max_hands)
            matches = match_ownership(boxes, owners, args.match_iou)
            for index, track_id in enumerate(ids):
                state = tracks[track_id].owner
                matched = owners[matches[index]] if index in matches else None
                if matched:
                    update_owner(state, matched[1], matched[2], frame_index, 0.35, 0.55, 45, 1, 3)
                label, score = owner_label(state, frame_index, 45)
                row = {"frame": frame_index, "time_s": frame_index/fps, "track_id": track_id,
                       "agent": label, "agent_score": score,
                       "ownership_matched": matched is not None,
                       "ownership_raw_label": matched[1] if matched else None,
                       "ownership_raw_confidence": matched[2] if matched else None,
                       "ownership_age_frames": frame_index-state["last_yolo_frame"] if state else None,
                       "hand_side": hands[index].hand_side, "detector_confidence": hands[index].confidence,
                       "bbox_normalized": boxes[index], "keypoints_2d_normalized": converted[index][1]}
                rows.append(row)
                draw(frame, row, cv2)
            writer.write(frame)
            if baseline:
                ok, left = baseline.read()
                if not ok:
                    raise ValueError("Baseline ended before the experiment; use matching clip lengths.")
                left = cv2.resize(left, (width, height))
                canvas = np.zeros((height+28, width*2, 3), dtype=np.uint8)
                canvas[28:, :width], canvas[28:, width:] = left, frame
                for x, title in [(4, "Current YOLO + MediaPipe"), (width+4, "WiLoR 2D + YOLO ownership")]:
                    cv2.putText(canvas, title, (x, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)
                comparison.write(canvas)
            frame_index += 1
            if frame_index % 150 == 0:
                print(f"Processed {frame_index} frames", flush=True)
        elapsed = time.perf_counter()-started
    finally:
        cap.release()
        if baseline:
            baseline.release()
        for video_writer in (writer, comparison):
            if video_writer is not None:
                video_writer.release()
    summary = {"frames": frame_index, "fps": fps, "processing_seconds": elapsed,
               "processing_fps": frame_index/elapsed, "observations": len(rows),
               "ownership_matched": sum(row["ownership_matched"] for row in rows),
               "unknown": sum(row["agent"] == "unknown" for row in rows),
               "first_detection_s": rows[0]["time_s"] if rows else None,
               "note": "Coverage and throughput only; no ground-truth accuracy evaluation."}
    metadata = {"schema_version": 1, "settings": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "detector_sha256": hashlib.sha256(args.detector.read_bytes()).hexdigest(),
                "ownership_sha256": hashlib.sha256(args.yolo_model.read_bytes()).hexdigest(),
                "keypoints": "2D normalized x,y only; no depth or keypoint confidence exported by upstream pipeline",
                "summary": summary, "rows": rows}
    (args.output/"wilor_hand_tracks.json").write_text(json.dumps(metadata, allow_nan=False))
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2))
    fields = ["frame", "time_s", "track_id", "agent", "agent_score", "ownership_matched",
              "ownership_raw_label", "ownership_raw_confidence", "ownership_age_frames",
              "hand_side", "detector_confidence", "bbox_normalized", "keypoints_2d_normalized"]
    with (args.output/"wilor_hand_tracks.csv").open("w", newline="") as stream:
        output = csv.DictWriter(stream, fieldnames=fields)
        output.writeheader()
        for row in rows:
            output.writerow({k: json.dumps(v) if isinstance(v, (list, tuple)) else v for k,v in row.items()})
    print(json.dumps(summary, indent=2))


def self_test():
    from types import SimpleNamespace
    first, second = (0., 0., .2, .2), (.8, .8, 1., 1.)
    assert match_ownership([first, second], [(second, "self", .8)]) == {1: 0}
    assert len(match_ownership([first, first], [(first, "other", .8)])) == 1
    assert match_ownership([first], []) == {}
    box, points = normalize(SimpleNamespace(bbox=[0, 0, 100, 50], keypoints_2d=[[50,25]]*21), 200, 100)
    assert box == (0,0,.5,.5) and points[0] == (.25,.25)
    print("WiLoR association and coordinate checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?")
    parser.add_argument("--detector", type=Path, default=Path(__file__).parent/"models/hand-detector.safetensors")
    parser.add_argument("--yolo-model", type=Path, default=Path(__file__).parent/"models/self_other_yolo11s_75e.pt")
    parser.add_argument("--output", type=Path, default=Path("outputs/wilor"))
    parser.add_argument("--baseline-video", type=Path, help="Overlay from the same source, starting at frame zero.")
    parser.add_argument("--device", default="mps", help="Device for ownership YOLO; WiLoR uses MLX.")
    parser.add_argument("--max-frames", type=int, default=900)
    parser.add_argument("--max-hands", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=.35)
    parser.add_argument("--match-iou", type=float, default=.2)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.video is None or args.max_frames < 0 or args.max_hands < 1 or not 0 < args.confidence <= 1 or not 0 < args.match_iou <= 1:
            parser.error("Provide a video and valid thresholds/counts.")
        if not args.detector.is_file() or not args.yolo_model.is_file():
            parser.error("Both model files are required; see the README.")
        if args.output.exists():
            parser.error("Choose a new output directory to preserve existing results.")
        process(args)
