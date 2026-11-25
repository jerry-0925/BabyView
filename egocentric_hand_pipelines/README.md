# Egocentric hand tracking and ownership labels

Two ownership pipelines, a MediaPipe-only baseline, and a prediction report for first-person footage:

1. `mediapipe_hand_tracks.py` detects hands, assigns short-lived track IDs, and exports 21 landmarks per visible hand.
2. `fused_hand_tracks.py` detects and tracks hand boxes with YOLO, labels them as `self` (the camera wearer), `other` (someone else), or `unknown`, and adds MediaPipe landmarks when available. A MediaPipe miss does not suppress a YOLO detection.
3. `wilor_hand_tracks.py` uses the WiLoR MLX detector and 2D keypoints with the same YOLO ownership checkpoint. This Apple Silicon experiment has known keypoint-quality limitations.
4. `prediction_report.py` compares confidence, spatial spread, coverage, and temporal consistency from either ownership pipeline's JSON output.

The package deliberately does **not** identify individual partners. It answers only wearer versus non-wearer.

## Repository contents

```text
egocentric_hand_pipelines/
├── mediapipe_hand_tracks.py   # MediaPipe-only tracker
├── fused_hand_tracks.py       # YOLO hand tracks + optional MediaPipe landmarks
├── wilor_hand_tracks.py       # Experimental WiLoR 2D + YOLO ownership
├── prediction_report.py       # Offline HTML, PNG and JSON diagnostics
├── test_fused_hand_tracks.py  # Regression checks; no models needed
├── test_prediction_report.py
├── WILOR_EXPERIMENT.md        # Observed results and limitations
├── models/
│   └── self_other_yolo11s_75e.pt # Included self/other YOLO11s checkpoint
├── requirements.txt
├── requirements-wilor.txt
├── .gitignore
└── README.md
```

No videos, annotations, or generated results are included. The one included model is documented below; keep private video and annotations private, and acquire any additional assets under their own licenses.

## Setup

Python 3.10–3.12 is recommended. On Apple Silicon, the fusion pipeline can use PyTorch MPS through Ultralytics; on other machines, pass `--device cpu`.

Run the commands below from the cloned repository folder.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Run the small non-video checks after installation:

```sh
.venv/bin/python mediapipe_hand_tracks.py --self-test
.venv/bin/python fused_hand_tracks.py --self-test
```

The first run using landmarks downloads the official MediaPipe Hand Landmarker model into this folder as `hand_landmarker.task`. It is ignored by Git. Fusion with `--no-landmarks` does not import MediaPipe or download its model.

## 1. MediaPipe hand tracks

```sh
.venv/bin/python mediapipe_hand_tracks.py /path/to/clip.mp4 --output outputs/mediapipe
```

The output directory contains:

- `hand_tracks.mp4` — original video with 21-point hand skeletons and temporal track IDs.
- `hand_tracks.json` — every observation, including normalized `(x, y, z)` landmarks.
- `hand_tracks.csv` — one compact row per observed hand.

Useful options:

```sh
--max-hands 4     # maximum simultaneous hand detections
--stride 2        # process every second frame; improves speed but lowers temporal resolution
--model path/to/hand_landmarker.task
```

MediaPipe `handedness` is an appearance estimate, not an owner label. It can also need interpretation changes for mirrored video; do not infer wearer ownership from screen position or left/right handedness.

## 2. YOLO hand tracking with optional MediaPipe landmarks

The bundled `models/self_other_yolo11s_75e.pt` is a **two-class YOLO11s detection checkpoint** with class names exactly `self` and `other`. It was fine-tuned for 75 epochs (640-pixel inputs) from `yolo11s.pt` on the two-class EgoHands conversion used for this pilot: EgoHands “my left/right” boxes were mapped to `self`, and “your left/right” boxes to `other`. Use a held-out target-domain evaluation before treating its labels as data.

Its final validation metrics in that training run were box mAP50 `0.985`, box mAP50–95 `0.746`, precision `0.982`, and recall `0.953`. These are source-dataset validation metrics, not BabyView performance.

Run the included model directly:

```sh
.venv/bin/python fused_hand_tracks.py /path/to/clip.mp4 \
  --yolo-model models/self_other_yolo11s_75e.pt \
  --output outputs/fused
```

The bundled checkpoint is also the default when `--yolo-model` is omitted; its path is resolved relative to the script. To use a retrained model instead, replace the value of `--yolo-model`. It must have the class names `self` and `other`. Quote paths containing spaces.

For boxes and ownership without MediaPipe, add `--no-landmarks`. Use a new output directory to preserve older results. The script processes a video file sequentially; real-time speed depends on the machine and settings.

It writes:

- `fused_hand_tracks.mp4` — hand boxes colored green (`self`), orange (`other`), or gray (`unknown`), with skeletons when available. Missing skeletons are marked `box only`.
- `fused_hand_tracks.json` — schema version 2, run settings, normalized boxes, optional landmarks, ownership votes, and evidence age.
- `fused_hand_tracks.csv` — compact version of the same metadata.

### How fusion works

```text
every video frame:       YOLO → hand boxes + raw self/other evidence
box association:         motion + overlap → hand track ID
per detected track:      signed ownership votes → stable label
optional MediaPipe:      attach current landmarks to a matching box
output:                  box + owner label, with or without a skeleton
```

YOLO runs every frame by default. Box tracking uses one-to-one greedy association based on predicted motion, box overlap, and a size-limited center-distance gate. Ownership class is excluded from matching, so changing a raw class does not itself start a new track. Missing tracks retain their ID/history for up to `--max-track-age` source frames. A failed YOLO detection produces no row for that hand; retained tracks are available for reacquisition, not fabricated observations. This lightweight tracker can still swap IDs at hand crossings.

MediaPipe runs independently on the full frame. Its landmarks attach one-to-one to the current boxes; unmatched YOLO boxes are still drawn and exported. MediaPipe-only hands do not start tracks in this pipeline. The MediaPipe-only script remains available separately.

Ownership uses a signed vote: `+confidence` for self, `-confidence` for other. The evidence update is `e = (1-a)^gap * e + a * vote`, where `a` is smoothing and `gap` is elapsed frames divided by YOLO stride (at least 1). The exported self vote score is `(1+e)/2`. A weak self detection now moves evidence toward self rather than other. These scores are **not calibrated probabilities**.

A new track starts as unknown and receives a label when its vote score crosses the label threshold. A settled label persists through neutral evidence; changing it requires three consecutive updates whose accumulated score crosses the opposite threshold. Expired evidence returns unknown and resets on the next detection. Hysteresis reduces flicker but can delay correction of a wrong label.

With `--yolo-stride > 1`, frames between YOLO passes may show motion-predicted boxes from the last pass. They are marked `predicted` in the video and `observation_source=prediction` in the exports; they are not new detections or ownership votes.

### Export fields and migration from the old fusion

Schema version 2 replaces misleading probability names: `self_probability` becomes `self_vote_score`, and `agent_confidence` becomes `agent_score`. Update any downstream analysis accordingly; the numeric meaning has changed.

| Field | Meaning |
| --- | --- |
| `bbox_x1/y1/x2/y2` | Box coordinates normalized to image width/height. |
| `observation_source` | `yolo` for a current detection; `prediction` for a skipped frame. |
| `detection_confidence`, `yolo_label` | Raw score/class of the last YOLO detection; not an ownership probability. |
| `yolo_age_frames` | Source frames since that detection (0 for observed boxes). |
| `landmarks_available` | Whether a current MediaPipe hand was associated with this box. |
| `landmarks_normalized` | 21 points in JSON, or null when unavailable. No old skeleton is carried forward. |
| `handedness`, `handedness_confidence` | MediaPipe results, or null/empty when unavailable. |
| `self_vote_score` | Accumulated ownership evidence mapped to [0,1]; 0.5 is neutral. |
| `agent_score` | Support for the displayed label; null for unknown. May fall below 0.5 temporarily while a switch is pending. |

Filter to `observation_source=yolo` when counting actual hand detections, and additionally require `landmarks_available` for landmark analyses. Box-only hands are valid detection rows.

### Default operating point

The defaults intentionally favor coverage for exploratory review:

| Option | Default | Effect |
| --- | ---: | --- |
| `--yolo-stride` | 1 | Run YOLO every frame; use 2 or 3 to trade detection frequency for speed. |
| `--yolo-confidence` | 0.35 | Ignore weaker YOLO boxes. |
| `--match-iou` | 0.10 | Accept modest overlap between YOLO and landmark-derived boxes. |
| `--match-distance` | 0.12 | Alternative normalized-center gate. |
| `--smoothing` | 0.35 | Weight of a new signed YOLO ownership vote. |
| `--label-threshold` | 0.55 | Self/other entry and switch thresholds on the vote score. |
| `--switch-updates` | 3 | Consecutive opposite threshold crossings before a label switch. |
| `--max-label-age` | 45 | Expire ownership evidence after this many source frames. |
| `--max-track-age` | 15 | Retain a missing track for reacquisition; must be at least YOLO stride. |
| `--no-landmarks` | off | Disable MediaPipe entirely; retain boxes, tracks, and ownership. |

To make labels more conservative, raise `--label-threshold` (for example, `0.70`) and lower `--max-label-age`. To process only a brief test segment, add `--max-frames 90`.

```sh
.venv/bin/python fused_hand_tracks.py /path/to/clip.mp4 --yolo-model models/self_other_yolo11s_75e.pt \
  --label-threshold 0.70 --max-label-age 20 --max-frames 90
```

## Model development and evaluation

### Prediction confidence, spread, and stability reports

`prediction_report.py` reads fusion schema v2 and WiLoR experiment JSON exports.
It generates a self-contained HTML report, a PNG figure, and machine-readable
`metrics.json`. It needs no hand annotations and does not run inference again.
Matplotlib, installed with the existing YOLO dependencies, is used for plotting.

```sh
.venv/bin/python prediction_report.py outputs/fused/fused_hand_tracks.json --names "YOLO + MediaPipe" --output outputs/fused_report
.venv-wilor/bin/python prediction_report.py outputs/fused/fused_hand_tracks.json outputs/wilor_test/wilor_hand_tracks.json --names "YOLO + MediaPipe" "WiLoR + YOLO" --output outputs/comparison_report
python3 -m unittest test_prediction_report.py
```

Use a new output directory for each report. Open `report.html` in a browser.
Reports include:

- Confidence distributions: count, mean, population standard deviation, range,
  median, quartiles and 10th/90th percentiles. Detection scores and ownership
  vote scores are kept separate; unknown ownership is not converted to zero.
- Spatial spread: normalized box centers, box area fractions and a center heatmap.
- Output coverage: frames with detections, zero-inclusive hands per frame,
  self/other/unknown counts, landmark availability and fresh ownership matches.
- Temporal behavior: track spans, consecutive detection runs, missing scheduled
  updates within tracks, single-observation tracks, raw ownership flips,
  smoothed label transitions, and box-center displacement between adjacent updates.
- Per-track statistics and a timeline of detection counts.

Predicted/carried rows are excluded. For stride > 1, coverage and gaps use the
declared detector schedule. Track-gap fraction is missing updates divided by
observed plus missing updates inside each track's first-to-last span. It does
not include hands that never received a track or misses after the track ended.
Center displacement includes genuine hand/camera motion and is not pure jitter.
Raw flips require adjacent frames with fresh ownership predictions on both.

These are descriptive statistics, not precision, recall or accuracy. Confidence
values are model-specific and filtered by the run's detection threshold.
Compare equivalent clips and settings, and inspect the overlays alongside the report.

### Experimental WiLoR detector + YOLO ownership (Apple Silicon)

The first local experiment is complete: see [results and limitations](WILOR_EXPERIMENT.md).
Detector-only keypoints looked unreliable in inspected frames; treat this as an
experimental comparison, not a validated landmark replacement.

`wilor_hand_tracks.py` runs the [WiLoR MLX port](https://github.com/lyonsno/wilor-mlx)
detector and its 21 two-dimensional keypoints on every frame. The existing
75-epoch YOLO model runs independently and supplies ownership evidence through
one-to-one box matching (IoU >= 0.20). Unmatched hands remain unknown unless the
track has recent ownership evidence. Left/right handedness never determines ownership.

This experiment loads only the 107 MB detector, with strict weight validation;
it does not load WiLoR's 3D reconstruction model or MANO assets.
The third-party port and its keypoints still require visual evaluation.
Upstream [WiLoR model terms](https://github.com/rolpotamias/WiLoR#license)
also need to be considered; downloaded experiment weights are excluded from Git.

From the repository root, create an experiment environment:

```sh
python3 -m venv .venv-wilor
.venv-wilor/bin/python -m pip install -r requirements-wilor.txt
curl -L --fail https://huggingface.co/BasinShapers/wilor-mlx/resolve/main/hand-detector.safetensors -o models/hand-detector.safetensors
.venv-wilor/bin/python wilor_hand_tracks.py --self-test
.venv-wilor/bin/python wilor_hand_tracks.py "/path/to/clip.mp4" --output outputs/wilor_test --max-frames 900
```

The default processes the first 900 frames; use `--max-frames 0` for the whole
video. The output directory must be new. `--device cpu` changes the ownership
YOLO device; the WiLoR detector still requires Apple Silicon MLX.

Outputs are `wilor_hand_tracks.mp4`, JSON, CSV, and `summary.json`.
The `keypoints_2d_normalized` field contains 21 pairs of x/y coordinates, with
no invented depth. The upstream pipeline does not expose per-keypoint confidence.
Ownership scores remain uncalibrated votes. `ownership_matched` and
`ownership_age_frames` distinguish fresh matches from retained labels.

To create a side-by-side video, add
`--baseline-video "/path/to/fused_hand_tracks.mp4"`. It must represent the same
source starting at frame zero, with matching FPS and at least as many frames.
The resulting `comparison.mp4` shows the baseline on the left and WiLoR on the
right. Detection counts and processing FPS measure coverage and runtime, not
accuracy; neither model's outputs are ground truth.

BabyView annotations are not available for the current pilot. The changes above require no retraining and keep the existing 75-epoch EgoHands checkpoint. Local EgoCom clips can verify pipeline operation and reveal visible errors, but cannot establish BabyView accuracy. Smooth labels and higher output coverage do not establish correct ownership.

If target-domain annotations become available later:

1. Label each hand box as `self`, `other`, or `ignore`; retain `ignore` as excluded data rather than forcing a class.
2. Split train/validation/test by wearer, recording session, or child—not random frames. Neighboring video frames leak heavily.
3. Start with an ownership-labeled egocentric source dataset, then fine-tune on the target camera population and activities.
4. Review the fused video overlay and label frames with low YOLO confidence, poor MediaPipe–YOLO association, recent track starts, or frequent self/other flips.
5. Report detection precision/recall, self/other macro-F1, `other` precision, track-level label consistency, and the proportion retained as `unknown`.

The included checkpoint is derived from the public [EgoHands dataset](https://vision.soic.indiana.edu/projects/egohands/), whose use and attribution requirements should be reviewed before redistribution. Ultralytics/YOLO software and pretrained weights have separate licensing terms; review them before a commercial release.

## Limits and responsible use

- This is a hand-level ownership estimate, not person identification. It cannot distinguish two different partners.
- Occlusion, motion blur, skin-tone/lighting shifts, and domain shift can cause misses or wrong ownership labels.
- A lower threshold produces fewer `unknown` labels but more false labels. Always inspect overlays and validate on target-domain annotations before interpreting results.
- First-person video can be sensitive. Do not commit raw participant video, model outputs containing identifiable imagery, or annotations without appropriate consent and access controls.

## Publish with GitHub Desktop

This folder is a standalone Git repository. In GitHub Desktop:

1. Choose **File → Add Local Repository**, then select this `egocentric_hand_pipelines` folder.
2. Check that the repository contains both ownership methods, the MediaPipe-only baseline, report script, regression tests, documentation, requirements, `.gitignore`, and the included 75-epoch checkpoint.
3. Choose **Publish repository** to create its GitHub remote, selecting the desired account and visibility. If you already have a GitHub repository for this project, configure its URL under **Repository → Repository Settings → Remote** instead, and reconcile any existing history before pushing.

The prepared local commit uses branch `codex/yolo-first-fusion`. No GitHub remote is configured by this preparation, and nothing has been pushed. Videos, generated outputs, Python caches, local environment files, and macOS metadata are ignored.

Future changes can be committed and sent with **Push origin**. This project does not yet declare a license for its own code; the dependency and checkpoint licensing notes above still apply.
