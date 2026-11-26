import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from check_aupro_toy import toy_aupro


class ToyPROTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((2,8,8),dtype=np.uint8)
        self.mask[0,1,1]=1
        self.mask[0,5,3:6]=1

    def test_known_curves(self):
        for limit in (.05,.3):
            for scores,expected in [(np.where(self.mask,.9,.1),1),
                                    (np.where(self.mask,.1,.9),0),
                                    (np.full(self.mask.shape,.5),limit/2)]:
                value,_=toy_aupro(scores,self.mask,limit)
                self.assertAlmostEqual(value,expected)

    def test_equal_region_weight_despite_different_areas(self):
        scores=np.zeros(self.mask.shape)
        scores[0,1,1]=1
        _,points=toy_aupro(scores,self.mask,.05)
        self.assertEqual(points[1],(0.,.5))

    def test_per_image_components_and_score_scale(self):
        mask=np.concatenate([self.mask[:1],self.mask[:1]])
        selected=np.zeros(mask.shape)
        selected[0,1,1]=1
        _,points=toy_aupro(selected,mask,.05)
        self.assertEqual(points[1],(0.,.25))
        scores=np.random.default_rng(3).random(mask.shape)
        before,_=toy_aupro(scores,mask,.05)
        after,_=toy_aupro(scores*7+3,mask,.05)
        self.assertAlmostEqual(before,after)

    def test_missing_class_is_unavailable(self):
        for mask in [np.zeros((1,3,3)),np.ones((1,3,3))]:
            with self.assertRaises(ValueError):toy_aupro(np.zeros(mask.shape),mask,.05)


if __name__=='__main__':unittest.main()
