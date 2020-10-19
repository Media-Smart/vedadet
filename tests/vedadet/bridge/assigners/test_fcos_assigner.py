import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

import mmcv.parallel as parallel

from vedadet.data.builder import build_dummy_dataloader
from vedadet.bridge.generators.meshgrids.point_meshgrid import PointMeshGrid
from vedadet.bridge.assigners.fcos_assigner import FCOSAssigner
from vedadet.models.builder import build_dummy_retinanet


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)
        strides = [4, 8, 16, 32, 64]
        regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512,
                                                                        10000))
        device = torch.device('cuda:0')

        dataloader = build_dummy_dataloader()
        retinanet = build_dummy_retinanet()

        meshgrid = PointMeshGrid(strides)
        assigner = FCOSAssigner(regress_ranges, strides, 80)

        for batch in dataloader:
            batch, _ = parallel.scatter_kwargs(batch, {}, target_gpus=[0])
            batch = batch[0]
            imgs = batch['img']
            gt_labels = batch['gt_labels']
            gt_bboxes = batch['gt_bboxes']
            feats = retinanet(imgs)
            featmap_sizes = [feat.shape[-2:] for feat in feats[0]]
            #print(featmap_sizes)
            anchor_mesh = meshgrid.get_anchor_mesh(featmap_sizes, torch.float,
                                                   device)
            #print(anchor_mesh)
            bbox_targets, labels = assigner.get_targets(
                anchor_mesh, gt_bboxes, gt_labels)
            for idx in range(len(bbox_targets)):
                print(labels[idx].shape, bbox_targets[idx].shape)
            break


if __name__ == '__main__':
    unittest.main()
