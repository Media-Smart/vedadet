import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

from vedadet.bridge.generators.meshgrids.point_meshgrid import PointMeshGrid


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)
        meshgrid = PointMeshGrid([8, 4])
        anchors = meshgrid.get_anchor_mesh([[2, 3], [4, 6]], torch.float,
                                           torch.device('cpu'))
        for anchor in anchors:
            print('anchor shape', anchor.shape)
            print('anchor content', anchor)
            print('-----------')


if __name__ == '__main__':
    unittest.main()
