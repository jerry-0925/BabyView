"""Regression checks requiring no model downloads or ML dependencies."""
import unittest

if __package__:
    from .fused_hand_tracks import BoxTrack, Detection, assign_box_tracks, match_detections, observation, owner_label, update_owner
else:
    from fused_hand_tracks import BoxTrack, Detection, assign_box_tracks, match_detections, observation, owner_label, update_owner


class FusionTests(unittest.TestCase):
    def vote(self, state, label, confidence, frame, **overrides):
        options = dict(smoothing=0.35, threshold=0.55, max_age=45, stride=1, switch_updates=3)
        options.update(overrides)
        update_owner(state, label, confidence, frame, **options)

    def test_weak_scores_never_reverse_class(self):
        for label in ("self", "other"):
            for confidence in (0.35, 0.49, 0.9):
                state = {}
                self.vote(state, label, confidence, 0)
                self.assertEqual(owner_label(state, 0, 45)[0], label)
                if label == "self":
                    self.assertGreater(state["evidence"], 0)
                else:
                    self.assertLess(state["evidence"], 0)

    def test_weak_vote_has_less_influence(self):
        weak, strong = {}, {}
        self.vote(weak, "self", 0.35, 0)
        self.vote(strong, "self", 0.9, 0)
        self.assertLess(weak["evidence"], strong["evidence"])

    def test_switch_needs_sustained_opposition(self):
        state = {}
        self.vote(state, "self", 0.9, 0, smoothing=1)
        for frame in (1, 2):
            self.vote(state, "other", 0.9, frame, smoothing=1)
            self.assertEqual(state["label"], "self")
        self.vote(state, "other", 0.9, 3, smoothing=1)
        self.assertEqual(state["label"], "other")

    def test_expiry_and_new_evidence_reset(self):
        state = {}
        self.vote(state, "self", 0.9, 0)
        self.assertEqual(owner_label(state, 46, 45), ("unknown", None))
        self.vote(state, "other", 0.35, 46)
        self.assertEqual(state["label"], "other")

    def test_track_exists_without_mediapipe(self):
        tracks = {}
        ids, _ = assign_box_tracks([((0.1, 0.1, 0.3, 0.3), "self", 0.35)], tracks, 0, 1)
        self.assertEqual(ids, [1])
        track = tracks[1]
        self.vote(track.owner, track.label, track.confidence, 0)
        self.assertEqual(match_detections([], [(track.box, track.label, track.confidence)], 0.1, 0.12), {})
        row = observation(track, 1, None, 0, 30, 45)
        self.assertEqual(row["agent"], "self")
        self.assertFalse(row["landmarks_available"])
        self.assertIsNone(row["landmarks_normalized"])
        self.assertEqual(row["observation_source"], "yolo")

    def test_tracking_ignores_class_and_detection_order(self):
        first, second = (0.1, 0.1, 0.2, 0.2), (0.7, 0.7, 0.8, 0.8)
        tracks = {}
        ids, next_id = assign_box_tracks([(first, "self", 0.9), (second, "other", 0.9)], tracks, 0, 1)
        self.assertEqual(ids, [1, 2])
        ids, next_id = assign_box_tracks([(second, "self", 0.4), (first, "other", 0.4)], tracks, 1, next_id)
        self.assertEqual(ids, [2, 1])
        self.assertEqual(assign_box_tracks([], tracks, 2, next_id)[0], [])
        ids, next_id = assign_box_tracks([(first, "self", 0.9)], tracks, 5, next_id)
        self.assertEqual(ids, [1])
        ids, _ = assign_box_tracks([(first, "self", 0.9)], tracks, 21, next_id)
        self.assertEqual(ids, [3])
        self.assertFalse(tracks[3].owner)

    def test_landmarks_optional_and_one_to_one(self):
        detection = Detection([(0.4, 0.4, 0), (0.6, 0.6, 0)] * 10 + [(0.5, 0.5, 0)], "Left", 0.9)
        boxes = [((0.39, 0.39, 0.61, 0.61), "self", 0.9)] * 2
        matches = match_detections([detection], boxes, 0.1, 0.12)
        self.assertEqual(len(matches), 1)
        track = BoxTrack(boxes[0][0], 0, "self", 0.9)
        row = observation(track, 1, detection, 0, 30, 45)
        self.assertTrue(row["landmarks_available"])
        self.assertEqual(len(row["landmarks_normalized"]), 21)

    def test_skipped_frame_is_flagged_prediction(self):
        track = BoxTrack((0.7, 0.7, 0.9, 0.9), 0, "self", 0.9, velocity=(0.1, 0.1))
        self.vote(track.owner, "self", 0.9, 0)
        row = observation(track, 1, None, 4, 30, 2)
        self.assertEqual(row["observation_source"], "prediction")
        self.assertEqual(row["yolo_age_frames"], 4)
        self.assertEqual(row["agent"], "unknown")
        self.assertLessEqual(row["bbox_x2"], 1)
        self.assertAlmostEqual(row["bbox_x2"] - row["bbox_x1"], 0.2)


def run_tests():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FusionTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    run_tests()
