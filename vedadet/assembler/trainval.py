import torch

from vedadet.datasets import build_dataset, build_dataloader
from vedadet.engines import build_engine
from vedacore.parallel import init_dist
from vedacore.hooks import HookPool
from vedacore.loopers import EpochBasedLooper
from vedacore.parallel import MMDistributedDataParallel
from vedacore.parallel import MMDataParallel


def trainval(cfg, launcher, logger):

    for mode in cfg.modes:
        assert mode in ('train', 'val')

    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    dataloaders = dict()
    engines = dict()
    find_unused_parameters = cfg.get('find_unused_parameters', False)
    if 'train' in cfg.modes:
        dataset = build_dataset(cfg.data.train)

        dataloaders['train'] = build_dataloader(dataset,
                                                cfg.data.samples_per_gpu,
                                                cfg.data.workers_per_gpu,
                                                dist=distributed)
        engine = build_engine(cfg.train_engine)

        if distributed:
            engine = MMDistributedDataParallel(
                engine.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            engine = MMDataParallel(engine.cuda(),
                                    device_ids=[torch.cuda.current_device()])

        engines['train'] = engine

    if 'val' in cfg.modes:
        dataset = build_dataset(cfg.data.val)

        dataloaders['val'] = build_dataloader(
            dataset,
            1,  #cfg.data.samples_per_gpu,
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
            engine = MMDataParallel(engine.cuda(),
                                    device_ids=[torch.cuda.current_device()])
        engines['val'] = engine

    hook_pool = HookPool(cfg.hooks)

    looper = EpochBasedLooper(cfg.modes, dataloaders, engines, hook_pool,
                              logger, cfg.workdir)
    if 'weights' in cfg:
        looper.load_weights(**cfg.weights)
    if 'optimizer' in cfg:
        looper.load_optimizer(**cfg.optimizer)
    if 'meta' in cfg:
        looper.load_meta(**cfg.meta)
    looper.start(cfg.max_epochs)
