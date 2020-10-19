import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

from vedadet.models.backbones.resnet import ResNet
from vedadet.models.necks.fpn import FPN


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)
        resnet = ResNet(depth=18).cuda()
        fpn = FPN(channels=[512, 256, 128]).cuda()
        imgs = torch.rand(4, 3, 448, 448).cuda()
        feats = resnet(imgs)
        feats = fpn(feats)
        for feat in feats:
            print(feat.shape)


if __name__ == '__main__':
    unittest.main()
