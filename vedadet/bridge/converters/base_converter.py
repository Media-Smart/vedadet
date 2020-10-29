from abc import ABCMeta, abstractmethod


class BaseConverter(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_bboxes(self):
        pass
