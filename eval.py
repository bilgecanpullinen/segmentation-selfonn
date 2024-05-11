# System libs
import os
import argparse
from distutils.version import LooseVersion
from multiprocessing import Queue, Process
# Numerical libs
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.io import loadmat
# Our libs
from semseg.config import cfg
from semseg.dataset import ValDataset
from semseg.models import ModelBuilder, SegmentationModule
from semseg.utils import AverageMeter, colorEncode, setup_logger, parse_devices, accuracy, intersectionAndUnion
from semseg.lib.nn import user_scattered_collate, async_copy_to
from semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
import sys
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


import time
def get_confusion_matrix(pred_label, label, num_classes, ignore_index):
    """Intersection over Union
       Args:
           pred_label (np.ndarray): 2D predict map
           label (np.ndarray): label 2D label map
           num_classes (int): number of categories
           ignore_index (int): index ignore in evaluation
       """

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    n = num_classes
    inds = n * label + pred_label

    mat = np.bincount(inds, minlength=n**2).reshape(n, n)

    return mat

def ConfusionMatrix(pre_mask, gt_mask, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    gt_mask_mask = (gt_mask >= 0) & (gt_mask < num_classes)
    pre_mask_mask = (pre_mask >= 0) & (pre_mask < num_classes)
    mask_final = np.logical_and(gt_mask_mask,pre_mask_mask)
    label = num_classes * gt_mask[mask_final].astype("int") + pre_mask[mask_final].astype("int")
    count = np.bincount(label, minlength=num_classes ** 2)
    confusion_matrix += count.reshape(num_classes, num_classes)
    return confusion_matrix

conf_m = torch.zeros(150,150) # classes (object or no-object)
colors = loadmat('./data/color150.mat')['colors']


def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, cfg, gpu_id, result_queue):
    segmentation_module.eval()

    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']
        start = time.time()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu_id)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu_id)

                # forward pass
                scores_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + scores_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
            
        end = time.time()
        acc, pix = accuracy(pred, seg_label)
        intersection, union, area_pred, area_lab = intersectionAndUnion(pred, seg_label, cfg.DATASET.num_class)
        co = get_confusion_matrix(pred, seg_label, cfg.DATASET.num_class, ignore_index=-1)
        tp = np.sum(np.logical_and(pred==1,seg_label==1))
        tn = np.sum(np.logical_and(pred==0,seg_label==0))
        fp = np.sum(np.logical_and(pred==1,seg_label==0))
        fn = np.sum(np.logical_and(pred==0,seg_label==1))
        result_queue.put_nowait((acc, pix, intersection, union, area_pred, area_lab, co,tp,tn,fp,fn,end))

        # visualization
        if cfg.VAL.visualize:
            visualize_result(
                (batch_data['img_ori'], seg_label, batch_data['info']),
                pred,
                os.path.join(cfg.DIR, 'result')
            )


def worker(cfg, gpu_id, start_idx, end_idx, result_queue):
    torch.cuda.set_device(gpu_id)

    # Dataset and Loader
    dataset_val = ValDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_val,
        cfg.DATASET,
        start_idx=start_idx, end_idx=end_idx)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.VAL.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, cfg, gpu_id, result_queue)


def main(cfg, gpus):
    with open(cfg.DATASET.list_val, 'r') as f:
        lines = f.readlines()
        num_files = len(lines)

    num_files_per_gpu = math.ceil(num_files / len(gpus))

    pbar = tqdm(total=num_files)

    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    area_pred_meter = AverageMeter()
    area_lab_meter = AverageMeter()
    co_meter = AverageMeter()
    tp_meter = AverageMeter()
    tn_meter = AverageMeter()
    fp_meter = AverageMeter()
    fn_meter = AverageMeter()
    end_meter = AverageMeter()

    result_queue = Queue(500)
    procs = []
    for idx, gpu_id in enumerate(gpus):
        start_idx = idx * num_files_per_gpu
        end_idx = min(start_idx + num_files_per_gpu, num_files)
        proc = Process(target=worker, args=(cfg, gpu_id, start_idx, end_idx, result_queue))
        print('gpu:{}, start_idx:{}, end_idx:{}'.format(gpu_id, start_idx, end_idx))
        proc.start()
        procs.append(proc)

    # master fetches results
    processed_counter = 0
    while processed_counter < num_files:
        if result_queue.empty():
            continue
        (acc, pix, intersection, union, area_pred, area_lab, co, tp, tn, fp, fn,end) = result_queue.get()
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
        area_pred_meter.update(area_pred)
        area_lab_meter.update(area_lab)
        co_meter.update(co)
        tp_meter.update(tp)
        tn_meter.update(tn)
        fp_meter.update(fp)
        fn_meter.update(fn)
        end_meter.update(end)
        processed_counter += 1
        pbar.update(1)

    for p in procs:
        p.join()

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    dice = 2*intersection_meter.sum / (area_pred_meter.sum + area_lab_meter.sum + 1e-10)
    calc_dice = 2*iou / (iou+1)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))
    print('[Eval Summary]:')
    tp = tp_meter.average()
    tn= tn_meter.average()
    fp = fp_meter.average()
    fn = fn_meter.average()
    end = end_meter.average()
    print(tp,tn,fp,tn)
    plotti = co_meter.average()
    pl = pd.DataFrame(data=plotti.astype(float))
    pl.to_csv('outfile.csv', sep=' ', header=False, float_format='%.2f', index=False)
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, dice: {:.4f}, calc_dice: {:.4f}'
          .format(iou.mean(), acc_meter.average()*100, dice.mean(), calc_dice.mean(),end_meter.average()))

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        default="0-3",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.VAL.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.VAL.checkpoint)
    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    if not os.path.isdir(os.path.join(cfg.DIR, "result")):
        os.makedirs(os.path.join(cfg.DIR, "result"))

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]

    main(cfg, gpus)

