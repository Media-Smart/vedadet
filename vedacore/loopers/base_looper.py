import os.path as osp
import torch
from abc import ABCMeta, abstractmethod

from vedacore.misc import (load_meta, load_optimizer, load_weights, save_meta,
                           save_optimizer, save_weights)
from vedacore.parallel import is_module_wrapper


class BaseLooper(metaclass=ABCMeta):

    TRAIN = 'train'
    VAL = 'val'
    MODES = (TRAIN, VAL)

    def __init__(self, modes, dataloaders, engines, hook_pool, logger,
                 workdir):
        for mode in modes:
            assert mode in BaseLooper.MODES
            assert mode in dataloaders
            assert mode in engines

        self.modes = modes
        self.dataloaders = dataloaders
        self.engines = engines
        self.cur_results = {BaseLooper.TRAIN: None, BaseLooper.VAL: None}
        self.his_results = {BaseLooper.TRAIN: None, BaseLooper.VAL: None}
        self.hook_pool = hook_pool
        self.logger = logger
        self.workdir = workdir

        self.meta = None
        self._iter = 0
        self._inner_iter = 0
        self._epoch = 0
        self._mode = None

    @abstractmethod
    def start(self):
        pass

    @property
    def train_engine(self):
        return self.engines[BaseLooper.TRAIN].module

    @property
    def val_engine(self):
        return self.engines[BaseLooper.VAL].module

    @property
    def train_dataloader(self):
        return self.dataloaders[BaseLooper.TRAIN]

    @property
    def val_dataloader(self):
        return self.dataloaders[BaseLooper.VAL]

    @property
    def train_dataset(self):
        return self.train_dataloader.dataset

    @property
    def val_dataset(self):
        return self.val_dataloader.dataset

    @property
    def cur_train_results(self):
        return self.cur_results[BaseLooper.TRAIN]

    @property
    def his_train_results(self):
        return self.his_results[BaseLooper.TRAIN]

    @property
    def cur_val_results(self):
        return self.cur_results[BaseLooper.VAL]

    @property
    def his_val_results(self):
        return self.his_results[BaseLooper.VAL]

    @his_val_results.setter
    def his_val_results(self, data):
        self.his_results[BaseLooper.VAL] = data

    @property
    def iter(self):
        return self._iter

    @property
    def inner_iter(self):
        return self._inner_iter

    @property
    def epoch(self):
        return self._epoch

    def load_weights(self,
                     filepath,
                     map_location='cpu',
                     strict=False,
                     prefix=None):
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()

            # map_location = lambda storage, loc: storage.cuda(device_id)
            def map_location(storage, loc):
                storage.cuda(device_id)

        self.logger.info('Loading weights from %s', filepath)
        # Wether to load train or val engine is OK
        # they share the same model
        for engine in self.engines.values():
            if is_module_wrapper(engine):
                engine = engine.module
            model = engine.model
            load_weights(model, filepath, map_location, strict, self.logger,
                         prefix)

    def load_optimizer(self, filepath, map_location='cpu'):
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()

            # map_location = lambda storage, loc: storage.cuda(device_id)
            def map_location(storage, loc):
                storage.cuda(device_id)

        self.logger.info('Loading optimizer from %s', filepath)
        # Wether to load train or val engine is OK
        # they share the same model
        engine = self.train_engine
        if is_module_wrapper(engine):
            engine = engine.module
        optimizer = engine.optimizer
        load_optimizer(optimizer, filepath, map_location)

    def load_meta(self, filepath, map_location='cpu'):
        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()

            # map_location = lambda storage, loc: storage.cuda(device_id)
            def map_location(storage, loc):
                storage.cuda(device_id)

        self.logger.info('Loading meta from %s', filepath)
        # Wether to load train or val engine is OK
        # they share the same model
        meta = load_meta(filepath, map_location)
        self._epoch = meta['epoch']
        self._iter = meta['iter']

    def save_snapshot(self,
                      out_dir,
                      filename_tmpl='epoch_{}',
                      save_optim_flag=True,
                      save_meta_flag=True,
                      meta=None):
        """Save the checkpoint.
        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
        """
        if meta is None:
            meta = dict(epoch=self.epoch, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch)
        prefix = osp.join(out_dir, filename)
        log_info = ['weights']
        if save_optim_flag:
            log_info.append('optimizer')
            filepath = '%s_optim.pth' % prefix
            engine = self.train_engine
            if is_module_wrapper(engine):
                engine = engine.module
            optimizer = engine.optimizer
            save_optimizer(optimizer, filepath)
        if save_meta_flag:
            log_info.append('meta')
            filepath = '%s_meta.pth' % prefix
            save_meta(meta, filepath)

        filepath = '%s_weights.pth' % prefix
        model = self.train_engine.model
        save_weights(model, filepath)
        self.logger.info('Saved %s at epoch %d, iter %d as %s' %
                         (', '.join(log_info), self.epoch, self.iter, prefix))

    def current_lr(self):
        """Get current learning rates.
        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
                param groups. If the runner has a dict of optimizers, this
                method will return a dict.
        """
        optimizer = self.train_engine.optimizer
        if isinstance(optimizer, torch.optim.Optimizer):
            lr = [group['lr'] for group in optimizer.param_groups]
        elif isinstance(optimizer, dict):
            lr = dict()
            for name, optim in optimizer.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
