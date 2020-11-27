import argparse
import os
import os.path as osp
import numpy as np
import torch

from vedacore.misc import Config, load_weights, ProgressBar, mkdir_or_exist
from vedacore.fileio import dump
from vedadet.datasets import build_dataloader, build_dataset
from vedadet.engines import build_engine
from vedacore.parallel import MMDataParallel


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--outdir', default='eval_dirs/tmp/tinaface/', help='directory where widerface txt will be saved')

    args = parser.parse_args()
    return args


def write_txt(save_folder, img_name, dets):

    save_name = osp.join(save_folder, img_name[:-4] + ".txt")
    dirname = osp.dirname(save_name)
    if not osp.isdir(dirname):
        os.makedirs(dirname)
    with open(save_name, "w") as fd:
        bboxs = dets[0]
        file_name = osp.basename(save_name)[:-4] + "\n"
        bboxs_num = str(len(bboxs)) + "\n"
        fd.write(file_name)
        fd.write(bboxs_num)
        for box in bboxs:
            x = int(box[0])
            y = int(box[1])
            w = int(box[2]) - int(box[0])
            h = int(box[3]) - int(box[1])
            confidence = str(box[4])
            line = f'{x} {y} {w} {h} {confidence}\n'
            fd.write(line)


def prepare(cfg, checkpoint):

    engine = build_engine(cfg.val_engine)
    load_weights(engine.model, checkpoint, map_location='cpu')

    device = torch.cuda.current_device()
    engine = MMDataParallel(
        engine.to(device), device_ids=[torch.cuda.current_device()])

    dataset = build_dataset(cfg.data.val, dict(test_mode=True))
    dataloader = build_dataloader(
        dataset,
        1,
        1,
        dist=False,
        shuffle=False)

    return engine, dataloader


def test(engine, data_loader, outdir):
    engine.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):

        filename = data['img_metas'][0].data[0][0]['filename'].split('/')[-2:]
        filename = osp.join(*filename)

        with torch.no_grad():
            result = engine(data)[0]

        if outdir is not None:
            write_txt(outdir, filename, result)
        results.append(result)
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    mkdir_or_exist(osp.abspath(args.outdir))

    engine, data_loader = prepare(cfg, args.checkpoint)

    results = test(engine, data_loader, args.outdir)

    data_loader.dataset.evaluate(results, 'mAP')

if __name__ == '__main__':
    main()
