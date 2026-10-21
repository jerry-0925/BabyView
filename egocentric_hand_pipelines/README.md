# Egocentric hand tracking and ownership labels

Two small video pipelines for first-person footage:

1. `mediapipe_hand_tracks.py` detects hands, assigns short-lived track IDs, and exports 21 landmarks per visible hand.
2. `fused_hand_tracks.py` uses those real-time MediaPipe tracks and periodically applies a separately trained YOLO classifier to label hands as `self` (the camera wearer), `other` (someone else), or `unknown`.

The package deliberately does **not** identify individual partners. It answers only wearer versus non-wearer.

## Repository contents

```text
egocentric_hand_pipelines/
├── mediapipe_hand_tracks.py   # MediaPipe-only tracker
├── fused_hand_tracks.py       # MediaPipe tracks + YOLO ownership evidence
├── models/
│   └── self_other_yolo11s_75e.pt # Included self/other YOLO11s checkpoint
├── requirements.txt
├── .gitignore
└── README.md
```

No videos, annotations, or generated results are included. The one included model is documented below; keep private video and annotations private, and acquire any additional assets under their own licenses.

## Setup

Python 3.10–3.12 is recommended. On Apple Silicon, the fusion pipeline can use PyTorch MPS through Ultralytics; on other machines, pass `--device cpu`.

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

The first real run downloads the official MediaPipe Hand Landmarker model into this folder as `hand_landmarker.task`. It is ignored by Git.

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

## 2. Fuse MediaPipe tracks with YOLO self/other labels

The bundled `models/self_other_yolo11s_75e.pt` is a **two-class YOLO11s detection checkpoint** with class names exactly `self` and `other`. It was fine-tuned for 75 epochs (640-pixel inputs) from `yolo11s.pt` on the two-class EgoHands conversion used for this pilot: EgoHands “my left/right” boxes were mapped to `self`, and “your left/right” boxes to `other`. Use a held-out target-domain evaluation before treating its labels as data.

Its final validation metrics in that training run were box mAP50 `0.985`, box mAP50–95 `0.746`, precision `0.982`, and recall `0.953`. These are source-dataset validation metrics, not BabyView performance.

Run the included model directly:

```sh
.venv/bin/python fused_hand_tracks.py /path/to/clip.mp4 \
  --yolo-model models/self_other_yolo11s_75e.pt \
  --output outputs/fused
```

To use a retrained model instead, replace the value of `--yolo-model`. It must have the class names `self` and `other`.

It writes:

- `fused_hand_tracks.mp4` — skeletons colored green (`self`), orange (`other`), or gray (`unknown`).
- `fused_hand_tracks.json` — landmarks plus `self_probability`, YOLO evidence age, and fused label.
- `fused_hand_tracks.csv` — compact version of the same metadata.

### How fusion works

```text
every video frame:       MediaPipe → landmarks + hand track ID
every N frames:          YOLO → self/other hand boxes and confidence
at YOLO frames:          match box ↔ landmark-derived hand box
per matched track:       smooth self probability over time
every output frame:      self / other / unknown
```

The match is one-to-one and accepts a pair when its bounding-box IoU or center distance passes a gate. A YOLO class score becomes a `self` probability (`self`: confidence; `other`: `1 - confidence`) and is exponentially smoothed for the matched MediaPipe track. `unknown` is emitted when there is no recent matched YOLO evidence or the probability remains near 0.5.

### Default operating point

The defaults intentionally favor coverage for exploratory review:

| Option | Default | Effect |
| --- | ---: | --- |
| `--yolo-stride` | 5 | Run YOLO at 6 Hz for 30 fps video; MediaPipe still runs every frame. |
| `--yolo-confidence` | 0.35 | Ignore weaker YOLO boxes. |
| `--match-iou` | 0.10 | Accept modest overlap between YOLO and landmark-derived boxes. |
| `--match-distance` | 0.12 | Alternative normalized-center gate. |
| `--smoothing` | 0.35 | Weight of a new YOLO observation in the track probability. |
| `--label-threshold` | 0.55 | `≥` is self and `≤ 1-threshold` is other. |
| `--max-label-age` | 45 | Keep a label for up to 45 frames after matched evidence. |

To make labels more conservative, raise `--label-threshold` (for example, `0.70`) and lower `--max-label-age`. To process only a brief test segment, add `--max-frames 90`.

```sh
.venv/bin/python fused_hand_tracks.py /path/to/clip.mp4 --yolo-model models/self_other_yolo11s_75e.pt \
  --label-threshold 0.70 --max-label-age 20 --max-frames 90
```

## Model development and evaluation

For a useful ownership classifier:

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

Create a new local repository in GitHub Desktop, copy this folder's contents into it, review the `.gitignore`, then commit and publish. Add a project-specific license and a citation file before public release if your institution requires them.
