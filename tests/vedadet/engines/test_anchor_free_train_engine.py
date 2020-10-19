import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

import mmcv.parallel as parallel

from vedadet.data.builder import build_dummy_dataloader
from vedadet.bridge.generators.meshgrids.point_meshgrid import PointMeshGrid
from vedadet.bridge.assigners.fcos_assigner import FCOSAssigner
from vedadet.models.builder import build_dummy_detector
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

        engine = AnchorFreeTrainEngine(detector, criterion, meshgrid, assigner)

        with open(
                '/media/data/home/yichaoxiong/workspace/ection-2.2.1/data.pkl',
                'rb') as fd:
            import pickle
            bb = pickle.load(fd)
        with open(
                '/media/data/home/yichaoxiong/workspace/ection-2.2.1/backbone.pkl',
                'rb') as fd:
            import pickle
            backbone_ = pickle.load(fd)
        with open(
                '/media/data/home/yichaoxiong/workspace/ection-2.2.1/neck.pkl',
                'rb') as fd:
            import pickle
            neck_ = pickle.load(fd)

        #for batch in dataloader:
        #batch, _ = parallel.scatter_kwargs(batch, {}, target_gpus=[0])
        #batch = batch[0]
        for ii in range(1):  #batch in dataloader:
            batch = bb

            imgs = batch['img']
            img_metas = batch['img_metas']
            gt_labels = batch['gt_labels']
            gt_bboxes = batch['gt_bboxes']
            #feats = detector(imgs)
            losses = engine.run(imgs, img_metas, gt_labels, gt_bboxes)
            #featmap_sizes = [feat.shape[-2:] for feat in feats2_head[0]]
            #anchor_mesh = meshgrid.get_anchor_mesh(featmap_sizes, torch.float, device)
            #labels, bbox_targets = assigner.get_targets(anchor_mesh, gt_labels, gt_bboxes)
            #criterion = FCOSCriterion()

            #for idx in range(len(bbox_targets)):
            #    print(labels[idx].shape, bbox_targets[idx].shape)

            #losses = criterion(anchor_mesh, bbox_targets, labels, feats[0], feats[1])
            #losses = criterion(anchor_mesh, labels, bbox_targets, feats2_head)
            #print(losses)
            #losses = head.forward_train(feats2_neck, img_metas, gt_bboxes, gt_labels)
            #loss = losses['loss_bbox']
            #loss.backward()
            print(losses)
            #break


if __name__ == '__main__':
    unittest.main()
