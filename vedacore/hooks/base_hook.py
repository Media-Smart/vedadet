from abc import ABCMeta, abstractmethod


class BaseHook(metaclass=ABCMeta):

    def before_run(self, looper):
        pass

    def after_run(self, looper):
        pass

    def before_epoch(self, looper):
        pass

    def after_epoch(self, looper):
        pass

    def before_iter(self, looper):
        pass

    def after_iter(self, looper):
        pass

    def before_train_epoch(self, looper):
        self.before_epoch(looper)

    def before_val_epoch(self, looper):
        self.before_epoch(looper)

    def after_train_epoch(self, looper):
        self.after_epoch(looper)

    def after_val_epoch(self, looper):
        self.after_epoch(looper)

    def before_train_iter(self, looper):
        self.before_iter(looper)

    def before_val_iter(self, looper):
        self.before_iter(looper)

    def after_train_iter(self, looper):
        self.after_iter(looper)

    def after_val_iter(self, looper):
        self.after_iter(looper)

    def every_n_epochs(self, looper, n):
        return looper.epoch % n == 0 if n > 0 else False

    def every_n_inner_iters(self, looper, n):
        return looper.inner_iter % n == 0 if n > 0 else False

    def every_n_iters(self, looper, n):
        return looper.iter % n == 0 if n > 0 else False

    def end_of_epoch(self, looper):
        return looper.inner_iter + 1 == len(looper.data_loader)

    @property
    @abstractmethod
    def modes(self):
        pass
