import os.path as osp
import random
import shutil
import tempfile
import time
import torch
import torch.distributed as dist

import vedacore.fileio as fileio
from vedacore.misc import mkdir_or_exist, registry
from vedacore.parallel import get_dist_info
from .base_hook import BaseHook


def collect_results_cpu(result_part, size, tmpdir=None):
    # adapted from https://github.com/open-mmlab/mmcv or
    # https://github.com/open-mmlab/mmdetection
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    fileio.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(fileio.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


@registry.register_module('hook')
class EvalHook(BaseHook):

    def __init__(self, tmpdir=None):
        if tmpdir is None:
            timestamp = time.strftime('%Y%m%d%H%M%S', time.localtime())
            self.tmpdir = '%s_%03d' % (timestamp, random.randint(0, 999))
        else:
            self.tmpdir = tmpdir

    def after_val_iter(self, looper):
        cur_results = looper.cur_val_results
        if looper.his_val_results is None:
            looper.his_val_results = []
        looper.his_val_results.extend(cur_results)

    def after_val_epoch(self, looper):
        results = looper.his_val_results
        logger = looper.logger
        rank, world_size = get_dist_info()
        if world_size > 1:
            all_results = collect_results_cpu(results, len(looper.val_dataset),
                                              self.tmpdir)
        else:
            all_results = results
        if rank == 0:
            metric = looper.val_dataset.evaluate(all_results, logger=logger)
            for k, v in metric.items():
                if 'copypaste' in k:
                    logger.info(f'{k}: {v}')
        looper.his_val_results = None

    @property
    def modes(self):
        return ['val']
