import sys
sys.path.insert(0, '/media/data/home/yichaoxiong/workspace/vedadet')

import torch
import unittest

import mmcv.parallel as parallel
from mmcv.utils import Config
from vedadet.assembler import trainval


def main():
    cfg = Config.fromfile('/media/data/home/yichaoxiong/workspace/vedadet/configs/retinanet.py')
    trainval(cfg)


class SimpleTest(unittest.TestCase):
    def test_true(self):
        self.assertTrue(True)
        main()

if __name__ == '__main__':
    #unittest.main()
    main()

