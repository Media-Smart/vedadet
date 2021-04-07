# Copyright (c) Open-MMLab. All rights reserved.
from vedacore.misc import registry
from .base_hook import BaseHook


@registry.register_module('hook')
class DistSamplerSeedHook(BaseHook):
    """Data-loading sampler for distributed training.
    When distributed training, it is only useful in conjunction with
    :obj:`EpochBasedRunner`, while :obj:`IterBasedRunner` achieves the same
    purpose with :obj:`IterLoader`.
    """

    def before_train_epoch(self, looper):
        if hasattr(looper.train_dataloader.sampler, 'set_epoch'):
            # in case the data loader uses `SequentialSampler` in Pytorch
            looper.train_dataloader.sampler.set_epoch(looper.epoch)
        elif hasattr(looper.train_dataloader.batch_sampler.sampler, 'set_epoch'):
            # batch sampler in pytorch warps the sampler as its attributes.
            looper.train_dataloader.batch_sampler.sampler.set_epoch(looper.epoch)

    @property
    def modes(self):
        return ['train']
