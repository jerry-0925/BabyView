# Egocentric hand tracking and ownership labels

Two small video pipelines for first-person footage:

1. `mediapipe_hand_tracks.py` detects hands, assigns short-lived track IDs, and exports 21 landmarks per visible hand.
2. `fused_hand_tracks.py` detects and tracks hand boxes with YOLO, labels them as `self` (the camera wearer), `other` (someone else), or `unknown`, and adds MediaPipe landmarks when available. A MediaPipe miss does not suppress a YOLO detection.

The package deliberately does **not** identify individual partners. It answers only wearer versus non-wearer.

## Repository contents

```text
egocentric_hand_pipelines/
├── mediapipe_hand_tracks.py   # MediaPipe-only tracker
├── fused_hand_tracks.py       # YOLO hand tracks + optional MediaPipe landmarks
├── test_fused_hand_tracks.py  # Regression checks; no models needed
├── models/
│   └── self_other_yolo11s_75e.pt # Included self/other YOLO11s checkpoint
├── requirements.txt
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
2. Check that the repository contains the two pipelines, regression tests, README, requirements, `.gitignore`, and the included 75-epoch checkpoint.
3. Choose **Publish repository** to create its GitHub remote, selecting the desired account and visibility. If you already have a GitHub repository for this project, configure its URL under **Repository → Repository Settings → Remote** instead, and reconcile any existing history before pushing.

The prepared local commit uses branch `codex/yolo-first-fusion`. No GitHub remote is configured by this preparation, and nothing has been pushed. Videos, generated outputs, Python caches, local environment files, and macOS metadata are ignored.

Future changes can be committed and sent with **Push origin**. This project does not yet declare a license for its own code; the dependency and checkpoint licensing notes above still apply.
