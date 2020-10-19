import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

from vedadet.bridge import gen_bbox_base_anchors
from vedadet.bridge import BBoxAnchorMeshGrid


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)
        strides = [4, 8]
        base_anchors = gen_bbox_base_anchors([1, 2], [1, 2], strides)
        meshgrid = BBoxAnchorMeshGrid(strides, base_anchors)
        print(meshgrid._gen_anchor_mesh([[1, 1], [1, 1]], torch.float, 'cpu'))


if __name__ == '__main__':
    unittest.main()
