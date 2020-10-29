# adapted from https://github.com/open-mmlab/mmcv or
# https://github.com/open-mmlab/mmdetection
from collections import OrderedDict

from vedacore.misc import registry
from .base_hook import BaseHook


@registry.register_module('hook')
class LoggerHook(BaseHook):

    def __init__(self, interval=1, by_epoch=True):
        self.interval = interval
        self.by_epoch = by_epoch

    def _train_log_info(self, log_dict, looper):
        if log_dict['mode'] == 'train':
            if isinstance(log_dict['lr'], dict):
                lr_str = []
                for k, val in log_dict['lr'].items():
                    lr_str.append(f'lr_{k}: {val:.4}')
                lr_str = ' '.join(lr_str)
            else:
                lr_str = f'lr: {log_dict["lr"]:.4}'

            # by epoch: Epoch [4][100/1000]
            # by iter:  Iter [100/100000]
            if self.by_epoch:
                log_str = f'Epoch [{log_dict["epoch"]}][{log_dict["iter"]}' \
                          f'/{len(looper.train_dataloader)}] {lr_str}, '
        log_items = []
        for name, val in log_dict.items():
            # TODO: resolve this hack
            # these items have been in log_str
            if name in ['mode', 'iter', 'lr', 'epoch']:
                continue
            if isinstance(val, float):
                val = f'{val:.4f}'
            log_items.append(f'{name}: {val}')
        log_str += ', '.join(log_items)

        looper.logger.info(log_str)

    def after_train_iter(self, looper):
        log_dict = OrderedDict()
        cur_lr = looper.current_lr()
        # only record lr of the first param group
        if isinstance(cur_lr, list):
            log_dict['lr'] = cur_lr[0]
        else:
            assert isinstance(cur_lr, dict)
            log_dict['lr'] = {}
            for k, lr_ in cur_lr.items():
                assert isinstance(lr_, list)
                log_dict['lr'].update({k: lr_[0]})

        log_dict['mode'] = looper.mode
        log_dict['epoch'] = looper.epoch + 1

        if self.by_epoch:
            log_dict['iter'] = looper.inner_iter
        else:
            log_dict['iter'] = looper.iter

        results = looper.cur_train_results
        log_dict = dict(log_dict, **results['log_vars'])

        if self.every_n_inner_iters(looper, self.interval):
            self._train_log_info(log_dict, looper)

    def _val_log_info(self, log_dict, looper):
        log_str = f'Validating ' \
                  f'[{log_dict["iter"]}/{len(looper.val_dataloader)}]'

        looper.logger.info(log_str)

    def after_val_iter(self, looper):
        if 'train' in looper.modes:
            return

        log_dict = OrderedDict()
        log_dict['mode'] = looper.mode
        log_dict['iter'] = looper.inner_iter

        if self.every_n_inner_iters(looper, self.interval):
            self._val_log_info(log_dict, looper)

    @property
    def modes(self):
        return ['train', 'val']
