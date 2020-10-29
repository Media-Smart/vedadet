from abc import ABCMeta, abstractmethod


class BaseMeshGrid(metaclass=ABCMeta):

    def __init__(self, strides):
        assert len(strides) > 0
        #        if isinstance(strides[0], (list, tuple)):
        #            assert len(strides) == 2
        #        else:
        #            strides = [_pair(stride) for stride in strides]
        self.strides = strides

    @abstractmethod
    def gen_anchor_mesh(self):
        pass
