# WiLoR detector and ownership experiment

## Scope

The first 900 frames (30 seconds at 30 fps) of the local EgoCom sample were
processed with wilor-mlx 0.3.1 and the existing 75-epoch self/other checkpoint.
The detector runs in MLX; ownership runs in PyTorch MPS. The baseline is the
previous 900-frame YOLO + MediaPipe output of the same source.

Only the detector was loaded. The full WiLoR pose/reconstruction network was
not loaded. Detector weights passed strict loading after handling its fixed
stride and DFL constants. Having keypoint-head weights does not establish that
their output is accurate.

## Results

| Measurement | Result |
| --- | ---: |
| Frames processed | 900 |
| WiLoR hand observations | 2,618 |
| Observations with a fresh ownership-box match | 1,005 |
| Observations labeled unknown | 1,095 |
| Processing time | 33.06 seconds |
| Processing throughput | 27.22 frames/second |
| Baseline hand observations | 1,155 |

Observations are hand/frame pairs, not distinct hands. Runtime includes the
processing loop and video writing, including first inference, but excludes
model initialization and final JSON/CSV serialization. Baseline timing was not
measured under the same conditions, so these numbers are not a speed comparison.
Ownership can persist briefly without a fresh match, explaining why matched
and unknown counts do not sum to the total.

## Visual findings

Three comparison frames were inspected: source frames 69, 482, and 750.
WiLoR detected visible hands missing from the baseline in those examples.
However, detector-only keypoints showed repeated grid-like patterns and did
not reliably trace finger joints. The cause has not been isolated between
upstream detector keypoint quality and the MLX conversion/decoding.

This experiment supports further investigation of the detector, but does not
justify replacing MediaPipe's landmarks with these detector-only keypoints.
The next diagnostic is an identical-frame comparison against the original
PyTorch WiLoR detector, followed by evaluating the separate pose model if
the detector's keypoint head is itself unsuitable.

More detections are not evidence of higher precision or recall. There were no
ground-truth hand or ownership annotations, and BabyView was not evaluated.

## Outputs and reproduction

Local outputs are in `outputs/wilor_egocom_30s/`:

- `comparison.mp4`: baseline on the left, WiLoR experiment on the right.
- `wilor_hand_tracks.mp4`: experiment overlay alone.
- `wilor_hand_tracks.json` and CSV: per-hand observations with 2D coordinates,
  raw ownership matches, vote labels, and evidence age.
- `summary.json`: machine-readable counts and timing.

The outputs and downloaded detector are ignored by Git. See the main README
for installation and execution. Use the same source and frame-zero baseline
to reproduce the side-by-side comparison.

Detector SHA-256:
`266890896e0ac1b46c39fe48035c09e90e41b36cd8d79c11e483234fafc5bdf3`

Ownership checkpoint SHA-256:
`06479f072da1dfdebc2eae70932f4e36806b290057af747d555dc446b2b1b164`

Association and coordinate regression checks passed, as did the eight existing
fusion tests. Video frame counts, CSV/JSON row counts, and 21-point output
shapes were checked. These validate implementation, not model accuracy.
