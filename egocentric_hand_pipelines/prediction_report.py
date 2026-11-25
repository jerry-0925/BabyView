#!/usr/bin/env python3
"""Describe fused/WiLoR predictions without treating them as ground truth."""
import argparse
import base64
from collections import Counter, defaultdict
import html
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "p10": None,
                "p25": None, "median": None, "p75": None, "p90": None, "max": None}
    def percentile(p):
        pos = (len(values)-1)*p
        low = int(pos)
        return values[low] + (values[min(low+1, len(values)-1)]-values[low])*(pos-low)
    return dict(n=len(values), mean=statistics.mean(values), std=statistics.pstdev(values),
                min=values[0], p10=percentile(.1), p25=percentile(.25), median=percentile(.5),
                p75=percentile(.75), p90=percentile(.9), max=values[-1])


def summarize(data, name):
    wilor = "summary" in data and "detector_sha256" in data
    frames = data["summary"]["frames"] if wilor else data["frames_processed"]
    fps = data["summary"]["fps"] if wilor else data["fps"]
    if not isinstance(frames, int) or frames < 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Need a nonnegative frame count and positive finite FPS.")
    stride = 1 if wilor else data.get("settings", {}).get("yolo_stride", 1)
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("Invalid detector stride.")
    all_rows, observed, seen = data["rows"], [], set()
    for row in all_rows:
        frame, track_id = row["frame"], row["track_id"]
        if not isinstance(frame, int) or not 0 <= frame < frames or (frame, track_id) in seen:
            raise ValueError("Invalid frame or duplicate frame/track observation.")
        seen.add((frame, track_id))
        if row.get("observation_source", "yolo") == "prediction":
            continue
        if frame % stride:
            raise ValueError("Observed detection is outside the declared inference schedule.")
        box = row["bbox_normalized"] if wilor else [row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]]
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or not (0 <= box[0] <= box[2] <= 1 and 0 <= box[1] <= box[3] <= 1):
            raise ValueError("Invalid normalized bounding box.")
        confidence = row["detector_confidence"] if wilor else row["detection_confidence"]
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid detection confidence.")
        observed.append(dict(frame=frame, track_id=track_id, box=box, confidence=confidence,
                             agent=row["agent"], score=row.get("agent_score"),
                             raw_owner=row.get("ownership_raw_label") if wilor else row.get("yolo_label"),
                             fresh=row["ownership_matched"] if wilor else True,
                             owner_confidence=row.get("ownership_raw_confidence") if wilor else confidence,
                             landmarks=bool(row.get("keypoints_2d_normalized")) if wilor else row["landmarks_available"]))
    observed.sort(key=lambda r: (r["frame"], r["track_id"]))
    grouped = defaultdict(list)
    frame_counts = Counter(r["frame"] for r in observed)
    for row in observed:
        grouped[row["track_id"]].append(row)
    tracks, steps, label_transitions, known_flips, gaps, run_lengths = [], [], 0, 0, 0, []
    raw_flips = raw_pairs = 0
    for track_id, sequence in grouped.items():
        changes = flips = holes = 0
        run = 1
        for previous, current in zip(sequence, sequence[1:]):
            delta = current["frame"]-previous["frame"]
            if delta != stride:
                holes += delta//stride-1
                run_lengths.append(run)
                run = 1
                continue
            run += 1
            a, b = previous["box"], current["box"]
            steps.append(math.hypot((b[0]+b[2]-a[0]-a[2])/2, (b[1]+b[3]-a[1]-a[3])/2))
            changes += previous["agent"] != current["agent"]
            flips += {previous["agent"], current["agent"]} == {"self", "other"}
            if previous["raw_owner"] in {"self", "other"} and current["raw_owner"] in {"self", "other"} and previous["fresh"] and current["fresh"]:
                raw_pairs += 1
                raw_flips += previous["raw_owner"] != current["raw_owner"]
        run_lengths.append(run)
        span = (sequence[-1]["frame"]-sequence[0]["frame"])//stride+1
        tracks.append({"track_id": track_id, "observations": len(sequence),
                       "first_s": sequence[0]["frame"]/fps, "last_s": sequence[-1]["frame"]/fps,
                       "span_s": (sequence[-1]["frame"]-sequence[0]["frame"]+1)/fps,
                       "within_span_coverage": len(sequence)/span, "missing_scheduled_frames": holes,
                       "label_transitions": changes, "direct_self_other_flips": flips,
                       "confidence_mean": statistics.mean(r["confidence"] for r in sequence)})
        gaps += holes
        label_transitions += changes
        known_flips += flips
    evaluated = len(range(0, frames, stride))
    centers_x = [(r["box"][0]+r["box"][2])/2 for r in observed]
    centers_y = [(r["box"][1]+r["box"][3])/2 for r in observed]
    labels = Counter(r["agent"] for r in observed)
    result = {"name": name, "frames": frames, "fps": fps, "duration_s": frames/fps,
              "detector_stride": stride, "evaluated_frames": evaluated,
              "observations": len(observed), "prediction_only_rows_excluded": len(all_rows)-len(observed),
              "frames_with_detections": len(frame_counts),
              "frame_coverage": len(frame_counts)/evaluated if evaluated else None,
              "hands_per_evaluated_frame": distribution(frame_counts[f] for f in range(0, frames, stride)),
              "labels": {label: labels[label] for label in ("self", "other", "unknown")},
              "unknown_fraction": labels["unknown"]/len(observed) if observed else None,
              "fresh_ownership_fraction": sum(r["fresh"] for r in observed)/len(observed) if observed else None,
              "landmark_availability_fraction": sum(r["landmarks"] for r in observed)/len(observed) if observed else None,
              "detection_confidence": distribution(r["confidence"] for r in observed),
              "fresh_ownership_confidence": distribution(r["owner_confidence"] for r in observed if r["fresh"]),
              "displayed_ownership_score": distribution(r["score"] for r in observed if r["agent"] != "unknown"),
              "confidence_by_label": {label: distribution(r["confidence"] for r in observed if r["agent"] == label) for label in ("self", "other", "unknown")},
              "spatial": {"center_x": distribution(centers_x), "center_y": distribution(centers_y),
                          "box_area_fraction": distribution((r["box"][2]-r["box"][0])*(r["box"][3]-r["box"][1]) for r in observed)},
              "temporal": {"track_count": len(tracks), "single_observation_tracks": sum(t["observations"]==1 for t in tracks),
                           "track_span_s": distribution(t["span_s"] for t in tracks),
                           "detection_run_length_updates": distribution(run_lengths),
                           "within_track_missing_updates": gaps, "label_transitions": label_transitions,
                           "within_track_gap_fraction": gaps/(gaps+len(observed)) if observed else None,
                           "raw_ownership_flips": raw_flips, "raw_ownership_adjacent_pairs": raw_pairs,
                           "direct_self_other_flips": known_flips,
                           "center_step_per_detector_update": distribution(steps)},
              "tracks": tracks, "settings": data.get("settings", {})}
    plot_data = {"counts": [frame_counts[f] for f in range(0, frames, stride)],
                 "times": [f/fps for f in range(0, frames, stride)],
                 "confidence": [r["confidence"] for r in observed],
                 "ownership": [r["score"] for r in observed if r["agent"] != "unknown" and r["score"] is not None],
                 "x": centers_x, "y": centers_y}
    return result, plot_data


def chart(reports, plots, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(reports), 4, figsize=(16, 3.6*len(reports)), squeeze=False)
    bins = [i/20 for i in range(21)]
    for row, (report, values) in enumerate(zip(reports, plots)):
        ax = axes[row]
        ax[0].hist(values["confidence"], bins=bins, color="#2878a0")
        ax[0].set(title=report["name"]+"\nDetection confidence", xlim=(0,1), ylabel="Hand/frame observations")
        ax[1].hist(values["ownership"], bins=bins, color="#a36521")
        ax[1].set(title="Displayed ownership vote score", xlim=(0,1))
        if values["x"]:
            ax[2].hist2d(values["x"], values["y"], bins=12, range=[[0,1],[0,1]],
                         weights=[1/len(values["x"])]*len(values["x"]), vmin=0, vmax=.2, cmap="Blues")
        ax[2].set(title="Box centers (fraction per cell)", xlim=(0,1), ylim=(1,0), xlabel="Image x", ylabel="Image y")
        ax[3].step(values["times"], values["counts"], where="post")
        ax[3].set(title="Detections over time", xlabel="Source time (seconds)", ylabel="Hands", ylim=(0,None))
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)


def table(headers, rows):
    return "<table><thead><tr>"+"".join("<th>"+html.escape(str(h))+"</th>" for h in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join("<td>"+html.escape(fmt(v))+"</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>"


def write_report(paths, names, output):
    reports, plots = [], []
    for path, name in zip(paths, names):
        report, values = summarize(json.loads(path.read_text()), name)
        report["source_json"] = str(path.resolve())
        reports.append(report)
        plots.append(values)
    output.mkdir(parents=True, exist_ok=False)
    chart(reports, plots, output/"distributions.png")
    sections = ['<h1>Hand prediction diagnostics</h1>',
                '<p>These statistics describe model output, not accuracy. No ground-truth annotations were used. A confident or stable prediction can still be wrong. Scores from different models are not calibrated on a shared scale.</p>',
                '<p>Only current detector observations are counted; carried/predicted boxes are excluded. Frame coverage uses scheduled detector frames and includes frames with zero detections. Ownership scores are smoothed votes, not probabilities. Landmark availability does not measure keypoint quality.</p>']
    metrics = [
        ("Duration (seconds)", lambda r:r["duration_s"]),
        ("Evaluated frames", lambda r:r["evaluated_frames"]),
        ("Observed hand/frame pairs", lambda r:r["observations"]),
        ("Frame coverage (fraction)", lambda r:r["frame_coverage"]),
        ("Mean hands per evaluated frame", lambda r:r["hands_per_evaluated_frame"]["mean"]),
        ("Median detection confidence", lambda r:r["detection_confidence"]["median"]),
        ("Detection confidence p10–p90", lambda r:f'{fmt(r["detection_confidence"]["p10"])} – {fmt(r["detection_confidence"]["p90"])}'),
        ("Unknown ownership (fraction)", lambda r:r["unknown_fraction"]),
        ("Fresh ownership match (fraction)", lambda r:r["fresh_ownership_fraction"]),
        ("Landmarks available (fraction)", lambda r:r["landmark_availability_fraction"]),
        ("Track IDs", lambda r:r["temporal"]["track_count"]),
        ("Single-observation tracks", lambda r:r["temporal"]["single_observation_tracks"]),
        ("Median track span (seconds)", lambda r:r["temporal"]["track_span_s"]["median"]),
        ("Within-track missing updates", lambda r:r["temporal"]["within_track_missing_updates"]),
        ("Within-track gap fraction", lambda r:r["temporal"]["within_track_gap_fraction"]),
        ("Raw ownership flips / eligible pairs", lambda r:f'{r["temporal"]["raw_ownership_flips"]} / {r["temporal"]["raw_ownership_adjacent_pairs"]}'),
        ("Direct self/other flips", lambda r:r["temporal"]["direct_self_other_flips"]),
        ("All label transitions", lambda r:r["temporal"]["label_transitions"])]
    sections.append(table(["Metric"]+names, [[label]+[fn(r) for r in reports] for label,fn in metrics]))
    encoded = base64.b64encode((output/"distributions.png").read_bytes()).decode()
    sections.append(f'<img alt="Confidence distributions, spatial spread and detection timelines" src="data:image/png;base64,{encoded}">')
    for report in reports:
        sections.append("<h2>"+html.escape(report["name"])+"</h2>")
        sections.append(table(["Label","Observations"], list(report["labels"].items())))
        dist = [("Detection confidence",report["detection_confidence"]),
                ("Fresh ownership confidence",report["fresh_ownership_confidence"]),
                ("Displayed ownership score",report["displayed_ownership_score"]),
                ("Box center x",report["spatial"]["center_x"]),
                ("Box center y",report["spatial"]["center_y"]),
                ("Box area / image area",report["spatial"]["box_area_fraction"]),
                ("Center displacement / detector update",report["temporal"]["center_step_per_detector_update"]),
                ("Consecutive detection run (updates)",report["temporal"]["detection_run_length_updates"])]
        keys = ["n","mean","std","min","p10","p25","median","p75","p90","max"]
        sections.append(table(["Distribution"]+keys, [[name]+[d[k] for k in keys] for name,d in dist]))
        sections.append("<h3>Track details</h3>")
        keys = ["track_id","observations","first_s","last_s","span_s","within_span_coverage","missing_scheduled_frames","label_transitions","direct_self_other_flips","confidence_mean"]
        sections.append(table(keys, [[t[k] for k in keys] for t in report["tracks"]]))
    sections.append('<h2>Interpretation</h2><p>Track IDs are algorithmic tracks, not people. A low track count can mean stable tracking or incorrect merging; a high count can mean new hands or fragmentation. Gaps count scheduled updates missing between the first and last observation of the same ID; they exclude disappearance after the last observation and cannot detect a new ID assigned to the same hand. They are not a ground-truth dropout rate.</p><p>Label transitions and center displacement use only consecutive scheduled observations within the same ID. Displacement includes real hand and camera motion; it is not a pure jitter measure. Stable smoothed labels may hide raw classifier changes. Histograms are conditioned on accepted detections and the configured confidence threshold. Heatmaps use equal axes and a shared 0–0.2 fraction-per-cell color scale; darker blue indicates more observations.</p><p>Compare only matching clips, durations, sampling schedules and thresholds. Confidence scores from WiLoR and ownership YOLO have different meanings. Missing values appear as N/A; zero detections is different from zero confidence.</p>')
    document = '<!doctype html><html><head><meta charset="utf-8"><title>Hand prediction diagnostics</title><style>body{font:15px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#17212a}p{max-width:1000px;line-height:1.55}table{border-collapse:collapse;margin:20px 0;font-size:13px}td,th{padding:8px;border:1px solid #ccd4da;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f6}img{width:100%;height:auto}</style></head><body>'+''.join(sections)+'</body></html>'
    (output/"report.html").write_text(document)
    (output/"metrics.json").write_text(json.dumps({"description":"Descriptive diagnostics only; not accuracy.", "reports":reports}, indent=2, allow_nan=False))
    print(output.resolve()/"report.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+")
    parser.add_argument("--names", nargs="+", help="Display names, one per JSON file.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for HTML, JSON and PNG.")
    args = parser.parse_args()
    names = args.names or [p.parent.name for p in args.predictions]
    if len(names) != len(args.predictions):
        parser.error("Provide one name per input JSON.")
    if args.output.exists():
        parser.error("Choose a new output directory.")
    write_report(args.predictions, names, args.output)
