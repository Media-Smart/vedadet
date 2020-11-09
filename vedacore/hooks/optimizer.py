from torch.nn.utils import clip_grad

from vedacore.misc import registry
from .base_hook import BaseHook


@registry.register_module('hook')
class OptimizerHook(BaseHook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, looper):
        optimizer = looper.train_engine.optimizer
        results = looper.cur_train_results
        optimizer.zero_grad()
        results['loss'].backward()
        if self.grad_clip is not None:
            model = looper.train_engine.model
            # grad_norm = self.clip_grads(model.parameters())
            self.clip_grads(model.parameters())
        optimizer.step()

    @property
    def modes(self):
        return ['train']
