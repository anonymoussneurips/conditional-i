import numpy as np
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from models import resnet18, resnet18_bank
from utils.tools import create_logger
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.display_results import show_performance, get_measures, print_measures, print_measures_with_std

parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Setup
parser.add_argument('--test_bs', type=int, default=256)
parser.add_argument('--num_to_avg', type=int, default=1, help='Average measures across num_to_avg runs.')
parser.add_argument('--validate', '-v', action='store_true', help='Evaluate performance on validation distributions.')
parser.add_argument('--method_name', '-m', type=str, default='cifar10_allconv_baseline', help='Method name.')
parser.add_argument('--bank_size', default=512, type=int, help='mmd-weight')
parser.add_argument('--code_dir', type=str, default=None, help='Folder to save checkpoints.')
# Loading details
parser.add_argument('--save', type=str, default=None, help='Checkpoint path to resume / test.')
parser.add_argument('--load', '-l', type=str, default=None, help='Checkpoint path to resume / test.')
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')


def main():
    args = parser.parse_args()

    if args.code_dir is None:
        args.code_dir = path.dirname(path.abspath(__file__))

    # create logger
    global logger
    logger = create_logger(
        path.join(args.code_dir, args.save, 'logs', 'test_corv.log'), 0)

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    test_transform = trn.Compose([trn.Resize(64), trn.CenterCrop(64),
                                  trn.ToTensor(), trn.Normalize(mean, std)])
    num_classes = 1000

    train_data = dset.ImageFolder(os.path.join(args.code_dir, 'data/in1k/in1k_train_64'), transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.test_bs, shuffle=False,
                                               num_workers=args.prefetch, pin_memory=True)
    test_data = dset.ImageFolder(os.path.join(args.code_dir, 'data/in1k/in1k_val_64'), transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                              num_workers=args.prefetch, pin_memory=True)

    # Create model
    logger.info("=> creating model '{}'".format(args.method_name))
    if 'r18_bank' in args.method_name:
        net = resnet18_bank(
            args.queue_len,
            num_classes=num_classes)
    elif 'r18' in args.method_name:
        net = resnet18(
            num_classes=num_classes)

    # Restore model
    if args.load:
        checkpoint = torch.load(path.join(args.code_dir, args.load))
        if 'state_dict' in checkpoint:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)
        logger.info("=> loaded checkpoint '{}'".format(args.load))

    # create csv
    os.makedirs(path.join(args.code_dir, args.save, 'test'), exist_ok=True)
    csv_dir = path.join(args.code_dir, args.save, 'test', 'corv.csv')
    with open(csv_dir, 'w') as f:
        f.write('data,top1,fpr95,auroc,aupr\n')
        f.close()

    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
        net.cuda()
        # torch.cuda.manual_seed(1)

    net.eval()

    cudnn.benchmark = True  # fire on all cylinders

    # /////////////// Detection Prelims ///////////////

    ood_num_examples = len(test_data) // 5

    concat = lambda x: np.concatenate(x, axis=0)
    to_np = lambda x: x.data.cpu().numpy()


    def get_ood_scores(loader, in_dist=False, in_cluster=None, in_test=False):
        _score = []
        _feat = []
        _target = []
        _right_score = []
        _wrong_score = []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                if batch_idx >= ood_num_examples // args.test_bs and in_test is False:
                    break

                data = data.cuda()

                output, feat = net(data, return_feat=True)
                _num_classes = output.size(-1)
                if in_cluster is None:
                    # sfm metric
                    smax = F.softmax(output, dim=1)
                    _score.append(-np.max(to_np(smax), axis=1))
                else:
                    # novel test metric: Eq. (11)
                    C, D = in_cluster.size()
                    N = feat.size(0)
                    scores = []
                    for i in range(N):
                        _feat = feat[i, :][None, None, :].repeat(C, 1, 1)
                        _in_cluster = in_cluster[:, :, None]
                        score = torch.bmm(_in_cluster, _feat).reshape(C, -1)  # (C, D*D)
                        score = score.square().sum(dim=-1, keepdim=True).sqrt()
                        scores.append(score)
                    scores = torch.cat(scores, dim=-1).permute(1, 0)
                    max_score, _ = scores.max(dim=1)
                    _score.append(-to_np(max_score))

                if in_dist:
                    smax_in = to_np(F.softmax(output[:, :num_classes], dim=1))
                    preds = np.argmax(smax_in, axis=1)
                    targets = target.numpy().squeeze()
                    right_indices = preds == targets
                    wrong_indices = np.invert(right_indices)

                    _feat.append(feat)
                    _target.append(target)
                    _right_score.append(-np.max(smax_in[right_indices, :num_classes], axis=1))
                    _wrong_score.append(-np.max(smax_in[wrong_indices, :num_classes], axis=1))

        if in_dist:
            _feat = torch.cat(_feat, dim=0)
            _target = torch.cat(_target, dim=0)
            unique_cls = _target.unique()
            cls_cluster = []
            for c in unique_cls:
                c_inds = (_target == c)
                cls_cluster.append(_feat[c_inds, :].mean(dim=0, keepdim=True))
            cls_cluster = torch.cat(cls_cluster, dim=0)
        if in_dist:
            return concat(_score).copy(), concat(_right_score).copy(), concat(_wrong_score).copy(), cls_cluster
        else:
            if in_test:
                return concat(_score).copy()
            else:
                return concat(_score)[:ood_num_examples].copy()

    incluster_pth = path.join(args.code_dir, args.save, 'feats/in_cluster.npy')
    if not path.exists(incluster_pth):
        _, _, _, in_cluster = get_ood_scores(train_loader, in_dist=True, in_test=True)
        in_cluster_np = in_cluster.cpu().numpy()
        os.makedirs(path.dirname(incluster_pth), exist_ok=True)
        np.save(incluster_pth, in_cluster_np)
    else:
        in_cluster = torch.from_numpy(np.load(incluster_pth)).cuda()
    in_score, right_score, wrong_score, _ = get_ood_scores(test_loader, in_dist=True, in_test=True)
    in_score = get_ood_scores(test_loader, in_cluster=in_cluster, in_test=True)

    num_right = len(right_score)
    num_wrong = len(wrong_score)
    top1_err = num_wrong / (num_wrong + num_right)
    logger.info('=> * Error Rate {:.2f}'.format(100 * top1_err))

    # /////////////// End Detection Prelims ///////////////

    logger.info('=> Using IN1K as typical data')

    # /////////////// Error Detection ///////////////

    logger.info('=> Error Detection')
    show_performance(wrong_score, right_score, method_name=args.method_name, logger=logger)

    # /////////////// OOD Detection ///////////////
    auroc_list, aupr_list, fpr_list = [], [], []

    def get_and_print_results(ood_loader, num_to_avg=args.num_to_avg, in_cluster=None):

        aurocs, auprs, fprs = [], [], []
        for _ in range(num_to_avg):
            out_score = get_ood_scores(ood_loader, in_cluster=in_cluster)
            measures = get_measures(out_score, in_score)
            aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

        auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)
        auroc_list.append(auroc); aupr_list.append(aupr); fpr_list.append(fpr)

        if num_to_avg >= 5:
            print_measures_with_std(aurocs, auprs, fprs, args.method_name, logger=logger)
        else:
            print_measures(auroc, aupr, fpr, args.method_name, logger=logger)
        return dict(fpr95=fpr, auroc=auroc, aupr=aupr)

    # /////////////// Textures ///////////////
    ood_data = dset.ImageFolder(root=os.path.join(args.code_dir, "data/dtd/images_64"),
                                transform=trn.Compose([trn.CenterCrop(64),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> Texture Detection')
    results = get_and_print_results(ood_loader, in_cluster=in_cluster)
    with open(csv_dir, 'a') as f:
        f.write(f'Texture,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// SVHN ///////////////
    ood_data = dset.ImageFolder(root=os.path.join(args.code_dir, "data/svhn/test_64"),
                                transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> SVHN Detection')
    results = get_and_print_results(ood_loader, in_cluster=in_cluster)
    with open(csv_dir, 'a') as f:
        f.write(f'SVHN,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// Places365 ///////////////
    ood_data = dset.ImageFolder(root=os.path.join(args.code_dir, "data/place/place365_test_64"),
                                transform=trn.Compose([trn.CenterCrop(64),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> Places365 Detection')
    results = get_and_print_results(ood_loader, in_cluster=in_cluster)
    with open(csv_dir, 'a') as f:
        f.write(f'Places365,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// LSUN ///////////////
    ood_data = dset.ImageFolder(root=os.path.join(args.code_dir, "data/lsun/test_64"),
                                transform=trn.Compose([trn.CenterCrop(64),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=args.test_bs, shuffle=True,
                                             num_workers=args.prefetch, pin_memory=True)

    logger.info('=> LSUN Detection')
    results = get_and_print_results(ood_loader, in_cluster=in_cluster)
    with open(csv_dir, 'a') as f:
        f.write(f'LSUN,{top1_err:02f},{results["fpr95"]:02f},{results["auroc"]:02f},{results["aupr"]:02f}\n')
        f.close()

    # /////////////// Mean Results ///////////////
    logger.info('=> Mean Test Results')
    print_measures(np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.method_name, logger=logger)
    with open(csv_dir, 'a') as f:
        f.write(f'Mean,{top1_err:02f},{np.mean(fpr_list):02f},{np.mean(auroc_list):02f},{np.mean(aupr_list):02f}\n')
        f.close()

if __name__ == '__main__':
    main()