import unittest
from shot_logic import confirm, associate


def point(t,x,y,r=8):
    return dict(time=t,x=x,y=y,radius=r,confidence=.8)


class ShotLogicTests(unittest.TestCase):
    def test_made_at_different_fps_and_sizes(self):
        for fps in (25,30,50,60):
            for scale in (.5,1,2):
                track=[point(i/fps,125*scale,(min(100,65+170*i/fps)+max(0,i/fps-35/170)*60)*scale,8*scale) for i in range(int(.85*fps))]
                self.assertIsNotNone(confirm(track,(100*scale,150*scale,100*scale)))

    def test_no_need_to_see_ascent(self):
        track=[point(i*.02,125,min(100,70+i*6)+max(0,i-5)*2) for i in range(25)]
        self.assertIsNotNone(confirm(track,(100,150,100)))

    def test_side_miss(self):
        track=[point(i*.02,155,70+i*6) for i in range(14)]
        self.assertIsNone(confirm(track,(100,150,100)))

    def test_rim_out_must_not_filter_bad_middle_points(self):
        ys=[70,78,86,94,102,108,97,112,122,135]
        self.assertIsNone(confirm([point(i*.02,125 if i<6 else 148,y) for i,y in enumerate(ys)],(100,150,100)))

    def test_rim_bounce_then_real_drop_can_score(self):
        ys=[70,78,86,94,102,108,90,76,70,78,86,94,101,104,107,110,113,116,119,122,125,128,131,134]
        self.assertIsNotNone(confirm([point(i*.02,125,y) for i,y in enumerate(ys)],(100,150,100)))

    def test_free_fall_in_projection_needs_net_evidence(self):
        self.assertIsNone(confirm([point(i*.02,125,70+i*6) for i in range(20)],(100,150,100)))

    def test_brief_net_occlusion_with_exit(self):
        track=[point(0,125,70),point(.02,125,80),point(.04,125,88),point(.22,125,114),point(.24,125,120),point(.30,125,134)]
        self.assertIsNotNone(confirm(track,(100,150,100)))

    def test_rim_loiter_then_half_second_occlusion(self):
        track=[point(0,125,91),point(.02,125,93),point(.04,125,95),point(.52,125,120),point(.54,125,128),point(.56,125,138)]
        self.assertIsNotNone(confirm(track,(100,150,100)))

    def test_occlusion_alone_is_not_a_goal(self):
        track=[point(i*.02,125,70+i*6) for i in range(7)]
        self.assertIsNone(confirm(track,(100,150,100)))

    def test_large_gap_not_bridged(self):
        track=[point(0,125,70),point(.02,125,80),point(.04,125,90),point(.74,125,120),point(.76,125,140)]
        self.assertIsNone(confirm(track,(100,150,100)))

    def test_upward_ball_not_goal(self):
        self.assertIsNone(confirm([point(i*.02,125,160-i*6) for i in range(20)],(100,150,100)))

    def test_one_detection_cannot_extend_two_tracks(self):
        tracks=[[point(0,124,90)],[point(0,127,90)]]
        result=associate(tracks,[point(.02,125,95)],.02,50)
        self.assertEqual(sorted(map(len,result)),[1,2])

    def test_oversize_net_box_cannot_start_a_track(self):
        result=associate([[point(0,125,90)]],[point(.02,125,95,30)],.02,50)
        self.assertEqual(len(result),1)


if __name__=='__main__': unittest.main()
