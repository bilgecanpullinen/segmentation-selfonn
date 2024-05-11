import os
import time
import random
import argparse
from distutils.version import LooseVersion
import torch
import torch.nn as nn
from mit_semseg.config import cfg
from mit_semseg.dataset import TrainDataset, ValidationDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import AverageMeter, parse_devices, setup_logger
from mit_semseg.lib.nn import UserScatteredDataParallel, user_scattered_collate, patch_replication_callback
from mit_semseg.models.fastonn.osl import *
from mit_semseg.models.fastonn.utils import *
from mit_semseg.models.fastonn import OpTier
from mit_semseg.models.fastonn import SelfONN2d, Trainer
from mit_semseg.models.fastonn.SelfONN import SelfONN2d
import sys


# train one epoch
def train(segmentation_module, iterator, optimizers, history, epoch, cfg, iterator_validation):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    area_pred_meter = AverageMeter()
    area_lab_meter = AverageMeter()
    val_acc_meter = AverageMeter()
    val_intersection_meter = AverageMeter()
    val_union_meter = AverageMeter()
    val_area_pred_meter = AverageMeter()
    val_area_lab_meter = AverageMeter()
    val_co_meter = AverageMeter()
    val_loss_meter = AverageMeter()
    val_time_meter = AverageMeter()
    ave_total_val_loss = AverageMeter()
    ave_val_acc = AverageMeter()
    segmentation_module.train(not cfg.TRAIN.fix_bn)
    best_loss = float('inf')
    # main loop
    tic = time.time()
    for i in range(cfg.TRAIN.epoch_iters):
        # load a batch of data
        batch_data = next(iterator)
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()
        for optimizer in optimizers:
            optimizer.zero_grad()
        # adjust learning rate
        adjust_learning_rate(optimizers, args)

        # forward pass
        loss, acc, intersection, union, area_pred, area_lab = segmentation_module(batch_data)

        loss = loss.mean()
        acc = acc.mean()
        #print(intersection)
        intersection = intersection.mean()
        union = union.mean()
        area_pred = area_pred.mean()
        area_lab = area_lab.mean()
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)
        intersection_meter.update(intersection.data.item())
        union_meter.update(union.data.item())
        area_pred_meter.update(area_pred.data.item())
        area_lab_meter.update(area_lab.data.item())
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        dice = 2*intersection_meter.sum / (area_pred_meter.sum + area_lab_meter.sum + 1e-10)
        #validate
        segmentation_module.eval()
        batch_dataa = next(iterator_validation)
        val_loss, val_acc, val_intersection, val_union, val_area_pred, val_area_lab = segmentation_module(batch_dataa)
        val_loss = val_loss.mean()
        val_acc = val_acc.mean()
        val_intersection = val_intersection.mean()
        val_union = val_union.mean()
        val_area_pred = val_area_pred.mean()
        val_area_lab = val_area_lab.mean()
        val_intersection_meter.update(val_intersection.data.item())
        val_union_meter.update(val_union.data.item())
        val_area_pred_meter.update(val_area_pred.data.item())
        val_area_lab_meter.update(val_area_lab.data.item())
        val_iou = val_intersection_meter.sum / (val_union_meter.sum + 1e-10)
        val_dice = 2*val_intersection_meter.sum / (val_area_pred_meter.sum + val_area_lab_meter.sum + 1e-10)
        if val_loss.data.item() < best_loss:
            best_loss = val_loss.data.item()
        ave_total_val_loss.update(val_loss.data.item())
        ave_val_acc.update(val_acc.data.item()*100)
        sys.stdout = open("selfonnet18-q3.txt", "a+")
        # calculate accuracy, and display
        if ((i) % cfg.TRAIN.disp_iter) == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_encoder: {:.6f}, lr_decoder: {:.6f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f},'
                  'Mean IoU: {:.8f}, dice: {:.10f},'
                  'Val_Accuracy: {:.12f}, Val_Loss: {:.14f},'
                  'Val_Mean_IoU: {:.16f}, Val_dice: {:.18f}'
                      .format(epoch, i, cfg.TRAIN.epoch_iters,
                          batch_time.average(), data_time.average(),
                          cfg.TRAIN.running_lr_encoder, cfg.TRAIN.running_lr_decoder,
                          ave_acc.average(), ave_total_loss.average(),
                          iou, dice,
                          ave_val_acc.average(), ave_total_val_loss.average(),
                          val_iou, val_dice))
            sys.stdout.close()
            fractional_epoch = epoch - 1 + 1. * i / cfg.TRAIN.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())
            history['val']['loss'].append(val_loss.data.item())
def checkpoint(nets, history, cfg, epoch, best_loss):
    #print('Saving checkpoints...')
    (net_encoder, net_decoder, crit) = nets
    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_encoder,
        '{}/encoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    torch.save(
        dict_decoder,
        '{}/decoder_epoch_{}.pth'.format(cfg.DIR, epoch))
    
    if best_loss is not None and history['val']['loss'][-1] < best_loss:
        best_loss = history['val']['loss'][-1]
        torch.save(
            dict_encoder,
            '{}/best_encoder.pth'.format(cfg.DIR))
        torch.save(
            dict_decoder,
            '{}/best_decoder.pth'.format(cfg.DIR))

    return best_loss

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, SelfONN2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers(nets, cfg):
    (net_encoder, net_decoder, crit) = nets
    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=cfg.TRAIN.lr_encoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=cfg.TRAIN.lr_decoder,
        momentum=cfg.TRAIN.beta1,
        weight_decay=cfg.TRAIN.weight_decay)
    return (optimizer_encoder, optimizer_decoder)

def adjust_learning_rate(optimizers, args):
    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = cfg.TRAIN.running_lr_decoder


def main(cfg, gpus):
    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder.lower(),
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder)

    crit = nn.NLLLoss(ignore_index=-1)

    if cfg.MODEL.arch_decoder.endswith('deepsup'):
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit, cfg.TRAIN.deep_sup_scale)
    else:
        segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_train = TrainDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_train,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=len(gpus),
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    print('1 Epoch = {} iters'.format(cfg.TRAIN.epoch_iters))
    # create loader iterator
    iterator_train = iter(loader_train)
    dataset_validation = ValidationDataset(
        cfg.DATASET.root_dataset,
        cfg.DATASET.list_validation,
        cfg.DATASET,
        batch_per_gpu=cfg.TRAIN.batch_size_per_gpu)

    loader_validation = torch.utils.data.DataLoader(
        dataset_validation,
        batch_size=len(gpus),
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=cfg.TRAIN.workers,
        drop_last=True,
        pin_memory=True)
    iterator_validation=iter(loader_validation)
    # load nets into gpu
    if len(gpus) > 1:
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=gpus)
        # For sync bn
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, cfg)

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}, 'val': {'epoch': [], 'loss': [], 'acc': []}}
    best_loss = None
    for epoch in range(cfg.TRAIN.start_epoch, cfg.TRAIN.num_epoch):
        train(segmentation_module, iterator_train, optimizers, history, epoch+1, cfg, iterator_validation)
        # checkpointing
        best_loss = checkpoint(nets, history, cfg, epoch+1, best_loss)
        torch.cuda.empty_cache()
    
    print('Training Done!')

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
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
        default="0,1",
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

    # Output directory
    if not os.path.isdir(cfg.DIR):
        os.makedirs(cfg.DIR)
    logger.info("Outputing checkpoints to: {}".format(cfg.DIR))
    with open(os.path.join(cfg.DIR, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))

    # Start from checkpoint
    if cfg.TRAIN.start_epoch > 0:
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
        assert os.path.exists(cfg.MODEL.weights_encoder) and \
            os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    cfg.TRAIN.batch_size = num_gpus * cfg.TRAIN.batch_size_per_gpu

    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder
    cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder

    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)

    main(cfg, gpus)
