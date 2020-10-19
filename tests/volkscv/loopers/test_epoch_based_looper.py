import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

import mmcv.parallel as parallel

from vedadet.data.builder import build_dummy_dataloader
from vedadet.bridge.generators.meshgrids.point_meshgrid import PointMeshGrid
from vedadet.bridge.assigners.fcos_assigner import FCOSAssigner
from vedadet.models.builder import build_dummy_detector
from volkscv.hooks.builder import build_dummy_hook_pool
from volkscv.loopers import EpochBasedLooper
from vedadet.criteria import FCOSCriterion
from vedadet.engines import AnchorFreeTrainEngine

from .models.backbones.resnet import ResNet
from .models.necks.fpn import FPN
from .models.dense_heads.fcos_head import FCOSHead


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)
        strides = [8, 16, 32, 64, 128]
        regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512,
                                                                        10000))
        device = torch.device('cuda:0')

        #dataloader = build_dummy_dataloader()
        detector = build_dummy_detector()
        detector = detector.cuda()
        detector.train()

        meshgrid = PointMeshGrid(strides)
        assigner = FCOSAssigner(regress_ranges, strides, 80)
        criterion = FCOSCriterion()

        with open(
                '/media/data/home/yichaoxiong/workspace/ection-2.2.1/data.pkl',
                'rb') as fd:
            import pickle
            batch = pickle.load(fd)

        imgs = batch['img']
        img_metas = batch['img_metas']
        gt_labels = batch['gt_labels']
        gt_bboxes = batch['gt_bboxes']
        workflow = [('train', 1)]
        dataloaders = {'train': [(imgs, img_metas, gt_labels, gt_bboxes)]}

        engine = AnchorFreeTrainEngine(detector, criterion, meshgrid, assigner)
        engines = {'train': engine}

        hook_pool = build_dummy_hook_pool()
        looper = EpochBasedLooper(workflow, dataloaders, engines, hook_pool)
        looper.start(1)


if __name__ == '__main__':
    unittest.main()
