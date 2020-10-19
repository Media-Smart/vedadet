import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import unittest

import torch
import mmcv.parallel as parallel

from vedadet.data.datasets import build_dataset, build_dataloader
from vedadet.models.backbones.resnet import ResNet
from vedadet.models.necks.fpn import FPN
from vedadet.models.heads.head import Head
from vedadet.models.detectors.retinanet import RetinaNet

from vedadet.data.builder import build_dummy_dataloader
from vedadet.models.builder import build_dummy_retinanet


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)

        dataloader = build_dummy_dataloader()
        retinanet = build_dummy_retinanet()

        for batch in dataloader:
            batch, _ = parallel.scatter_kwargs(batch, {}, target_gpus=[0])
            batch = batch[0]
            imgs = batch['img']
            gt_labels = batch['gt_labels']
            gt_bboxes = batch['gt_bboxes']
            feats = retinanet(imgs)
            print(imgs.shape, gt_labels, gt_bboxes)
            for feat in zip(feats[0], feats[1]):
                print(feat[0].shape, feat[1].shape)
            break
        return


if __name__ == '__main__':
    unittest.main()
