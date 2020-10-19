import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

import mmcv.parallel as parallel

from vedadet.data.builder import build_dummy_dataloader
from vedadet.bridge.generators.meshgrids.point_meshgrid import PointMeshGrid
from vedadet.bridge.assigners.fcos_assigner import FCOSAssigner
from vedadet.criteria import FCOSCriterion

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
        #retinanet = build_dummy_retinanet()

        meshgrid = PointMeshGrid(strides)
        assigner = FCOSAssigner(regress_ranges, strides, 80)

        backbone = ResNet(depth=50,
                          num_stages=4,
                          out_indices=(0, 1, 2, 3),
                          frozen_stages=1,
                          norm_cfg=dict(type='BN', requires_grad=False),
                          norm_eval=True,
                          style='caffe')
        backbone.train()
        backbone.init_weights()
        backbone = backbone.cuda()

        neck = FPN(
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs=True,
            extra_convs_on_inputs=False,  # use P5
            num_outs=5,
            relu_before_extra_convs=True)
        neck.train()
        neck.init_weights()
        neck = neck.cuda()

        head = FCOSHead(num_classes=80,
                        in_channels=256,
                        stacked_convs=4,
                        feat_channels=256,
                        strides=strides,
                        norm_cfg=None,
                        loss_cls=dict(type='FocalLoss',
                                      use_sigmoid=True,
                                      gamma=2.0,
                                      alpha=0.25,
                                      loss_weight=1.0),
                        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                        loss_centerness=dict(type='CrossEntropyLoss',
                                             use_sigmoid=True,
                                             loss_weight=1.0))
        head.train()
        head.init_weights()
        head = head.cuda()

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
            #feats = retinanet(imgs)
            feats2_backbone = backbone(imgs)
            feats2_neck = neck(feats2_backbone)
            feats2_head = head(feats2_neck)
            #import pdb
            #pdb.set_trace()
            featmap_sizes = [feat.shape[-2:] for feat in feats2_head[0]]
            #print(featmap_sizes)
            anchor_mesh = meshgrid.get_anchor_mesh(featmap_sizes, torch.float,
                                                   device)
            #print(anchor_mesh)
            labels, bbox_targets = assigner.get_targets(
                anchor_mesh, gt_labels, gt_bboxes)
            criterion = FCOSCriterion()

            #for idx in range(len(bbox_targets)):
            #    print(labels[idx].shape, bbox_targets[idx].shape)

            #losses = criterion(anchor_mesh, bbox_targets, labels, feats[0], feats[1])
            losses = criterion(anchor_mesh, labels, bbox_targets, feats2_head)
            print(losses)
            #losses = head.forward_train(feats2_neck, img_metas, gt_bboxes, gt_labels)
            #loss = losses['loss_bbox']
            #loss.backward()
            #print(losses)
            #break


if __name__ == '__main__':
    unittest.main()
