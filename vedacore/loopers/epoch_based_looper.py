from .base_looper import BaseLooper


class EpochBasedLooper(BaseLooper):

    def __init__(self, modes, dataloaders, engines, hook_pool, logger,
                 workdir):
        super().__init__(modes, dataloaders, engines, hook_pool, logger,
                         workdir)

    def epoch_loop(self, mode):
        self.mode = mode
        dataloader = self.dataloaders[mode]
        engine = self.engines[mode]
        for idx, data in enumerate(dataloader):
            self.hook_pool.fire(f'before_{mode}_iter', self)
            self.cur_results[mode] = engine(data)
            if mode == BaseLooper.TRAIN:
                self._iter += 1
            self._inner_iter = idx + 1
            self.hook_pool.fire(f'after_{mode}_iter', self)

    def start(self, max_epochs):
        self.hook_pool.fire('before_run', self)
        while self.epoch < max_epochs:
            for mode in self.modes:
                mode = mode.lower()
                self.hook_pool.fire(f'before_{mode}_epoch', self)
                self.epoch_loop(mode)
                if mode == BaseLooper.TRAIN:
                    self._epoch += 1
                self.hook_pool.fire(f'after_{mode}_epoch', self)
            if len(self.modes) == 1 and self.modes[0] == EpochBasedLooper.VAL:
                break
