import unittest
from prediction_report import distribution, summarize


def row(frame, track=1, label="self", source="yolo"):
    return dict(frame=frame, track_id=track, agent=label, agent_score=.6,
                yolo_label=label,
                detection_confidence=.8, observation_source=source,
                bbox_x1=.1,bbox_y1=.2,bbox_x2=.3,bbox_y2=.4,landmarks_available=False)


class ReportTests(unittest.TestCase):
    def test_distribution(self):
        self.assertEqual(distribution([])["mean"], None)
        self.assertEqual(distribution([1,3])["median"],2)
        self.assertEqual(distribution([1,3])["std"],1)

    def test_zero_detections(self):
        r,_=summarize(dict(frames_processed=5,fps=30,rows=[]),"empty")
        self.assertEqual(r["frame_coverage"],0)
        self.assertEqual(r["hands_per_evaluated_frame"]["mean"],0)
        self.assertIsNone(r["detection_confidence"]["mean"])

    def test_gaps_transitions_and_skipped_frames(self):
        data=dict(frames_processed=8,fps=2,settings={"yolo_stride":2},
                  rows=[row(0),row(1,source="prediction"),row(2,label="other"),row(6,label="self")])
        r,_=summarize(data,"x")
        self.assertEqual(r["observations"],3)
        self.assertEqual(r["frame_coverage"],.75)
        self.assertEqual(r["prediction_only_rows_excluded"],1)
        self.assertEqual(r["temporal"]["within_track_missing_updates"],1)
        self.assertEqual(r["temporal"]["direct_self_other_flips"],1)
        self.assertEqual(r["temporal"]["raw_ownership_flips"],1)
        self.assertEqual(r["temporal"]["within_track_gap_fraction"],.25)
        self.assertEqual(r["temporal"]["detection_run_length_updates"]["max"],2)

    def test_wilor_unknown(self):
        item=dict(frame=0,track_id=1,agent="unknown",agent_score=None,
                  detector_confidence=.7,bbox_normalized=[0,0,1,1],
                  ownership_matched=False,ownership_raw_confidence=None,keypoints_2d_normalized=[[0,0]]*21)
        r,_=summarize(dict(summary=dict(frames=2,fps=30),detector_sha256="x",rows=[item]),"w")
        self.assertEqual(r["unknown_fraction"],1)
        self.assertEqual(r["fresh_ownership_fraction"],0)
        self.assertEqual(r["displayed_ownership_score"]["n"],0)

    def test_invalid_frame_and_duplicate(self):
        for rows in [[row(3)],[row(0),row(0)]]:
            with self.assertRaises(ValueError):
                summarize(dict(frames_processed=3,fps=30,rows=rows),"x")


if __name__=="__main__":
    unittest.main()
