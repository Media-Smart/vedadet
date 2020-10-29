import torch.nn as nn

from vedadet.models import build_detector


class BaseEngine(nn.Module):

    def __init__(self, detector):
        super().__init__()
        self.detector = build_detector(detector)
