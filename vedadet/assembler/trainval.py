import torch

from vedacore.hooks import HookPool
from vedacore.loopers import EpochBasedLooper
from vedacore.parallel import MMDataParallel, MMDistributedDataParallel
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine


def trainval(cfg, distributed, logger):

    for mode in cfg.modes:
        assert mode in ('train', 'val')

    dataloaders = dict()
    engines = dict()
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    if 'train' in cfg.modes:
        dataset = build_dataset(cfg.data.train)

        dataloaders['train'] = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=distributed,
            seed=cfg.get('seed', None))
        engine = build_engine(cfg.train_engine)

        if distributed:
            engine = MMDistributedDataParallel(
                engine.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            engine = MMDataParallel(
                engine.cuda(), device_ids=[torch.cuda.current_device()])

        engines['train'] = engine

    if 'val' in cfg.modes:
        dataset = build_dataset(cfg.data.val, dict(test_mode=True))

        dataloaders['val'] = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)

        engine = build_engine(cfg.val_engine)
        if distributed:
            engine = MMDistributedDataParallel(
                engine.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            engine = MMDataParallel(
                engine.cuda(), device_ids=[torch.cuda.current_device()])
        engines['val'] = engine

    hook_pool = HookPool(cfg.hooks, cfg.modes, logger)

    looper = EpochBasedLooper(cfg.modes, dataloaders, engines, hook_pool,
                              logger, cfg.workdir)

    if isinstance(looper, EpochBasedLooper):
        looper.hook_pool.register_hook(dict(typename='WorkerInitHook'))
        if distributed:
            looper.hook_pool.register_hook(
                dict(typename='DistSamplerSeedHook'))

    if 'weights' in cfg:
        looper.load_weights(**cfg.weights)
    if 'train' in cfg.modes:
        if 'optimizer' in cfg:
            looper.load_optimizer(**cfg.optimizer)
        if 'meta' in cfg:
            looper.load_meta(**cfg.meta)
    else:
        if 'optimizer' in cfg:
            logger.warning('optimizer is not needed in non-training mode')
        if 'meta' in cfg:
            logger.warning('meta is not needed in non-training mode')
    looper.start(cfg.max_epochs)
