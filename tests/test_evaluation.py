import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_handoff import calibrate, components, transform_mask, mask_record, locate
from audit_image_pilot import auc_rank


class EvaluationTests(unittest.TestCase):
    def test_calibration_ties_and_boundary(self):
        for alpha in (0, .01, .05, .1, .5):
            for scores in (list(range(41)), [1]*41, [1]*39+[2,2]):
                threshold,n = calibrate(scores,alpha)
                self.assertLessEqual(n, int(alpha*41))
                self.assertEqual(n, sum(s>=threshold for s in scores))
        threshold,n = calibrate(list(range(41)),.05)
        self.assertEqual(n,2)
        self.assertGreater(threshold,38)

    def test_invalid_calibration(self):
        for scores,alpha in [([], .05), ([1,float('nan')],.05), ([1],1), ([1],-.1)]:
            with self.assertRaises(ValueError):
                calibrate(scores,alpha)

    def test_auroc(self):
        self.assertEqual(auc_rank([0,0,1,1],[.1,.2,.8,.9]),1)
        self.assertEqual(auc_rank([0,1],[1,1]),.5)
        self.assertIsNone(auc_rank([0,0],[1,2]))

    def test_eight_connectivity(self):
        _,areas=components(np.eye(3,dtype=bool))
        self.assertEqual(areas.tolist(),[3])

    def test_crop_detects_lost_region(self):
        raw=np.zeros((1900,1400),dtype=np.uint8)
        raw[0:100,0:100]=255
        raw[800:1000,600:800]=255
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            path=Path(tmp)/'mask.png'
            Image.fromarray(raw).save(path)
            row,cropped=mask_record('synthetic',path)
            self.assertEqual(row['original_component_count'],2)
            self.assertEqual(row['components_lost_crop'],1)
            self.assertGreater(row['crop_pixels_lost'],0)
            self.assertTrue(cropped.any())
            self.assertEqual(cropped.shape,(224,224))

    def test_resize_matches_torchvision(self):
        try:
            from torchvision import transforms
        except ImportError:
            self.skipTest('optional torchvision not installed')
        raw=np.random.default_rng(8).choice([0,255],size=(1900,1400)).astype(np.uint8)
        image=Image.fromarray(raw)
        expected=transforms.Compose([transforms.Resize((256,256),interpolation=transforms.InterpolationMode.NEAREST),transforms.CenterCrop(224),transforms.ToTensor()])(image).numpy()[0]>.5
        np.testing.assert_array_equal(transform_mask(image),expected)

    def test_missing_or_ambiguous_assets_fail(self):
        with self.assertRaises(ValueError): locate([], '/a.npy')
        with self.assertRaises(ValueError): locate([Path('one/a.npy'),Path('two/a.npy')], '/a.npy')


if __name__=='__main__': unittest.main()
