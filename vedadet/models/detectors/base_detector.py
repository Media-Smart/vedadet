import torch.nn as nn


class BaseDetector(nn.Module):

    def __init__(self):
        super().__init__()
