from vedacore.misc import registry
from .base_hook import BaseHook


@registry.register_module('hook')
class WorkerInitHook(BaseHook):
    """Worker init for training.
    """

    def before_train_epoch(self, looper):
        worker_init_fn = looper.train_dataloader.worker_init_fn
        if worker_init_fn is not None and hasattr(worker_init_fn, 'set_epoch'):
            worker_init_fn.set_epoch(looper.epoch)

    @property
    def modes(self):
        return ['train']
