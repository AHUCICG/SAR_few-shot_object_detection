# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------
import sys
from os import path

sys.path.append(path.split(sys.path[0])[0])

import xml.etree.ElementTree as ET
import os, sys
import pickle
import numpy as np
import argparse
from termcolor import colored
from utils import *
import pdb

classes = []
cfg = {}


def get_novels(root, id=None):
    if root.endswith('txt'):
        if id == 'None':
            return []
        with open(root, 'r') as f:
            novels = f.readlines()
        return novels[int(id)].strip().split(',')
    else:
        return root.split(',')


def filter(detlines, clsfile):
    # pdb.set_trace()
    with open(clsfile, 'r') as f:
        imgids = [l.split()[0] for l in f.readlines() if l.split()[1] == '1']
    dls = [dl for dl in detlines if dl[0] in imgids]

    # dls = [dl for dl in dls if float(dl[1]) > 0.05]
    return dls


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    objects = []

    if cfg['data'] == 'nwpu':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': classes[int(obj[4]) - 1],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    elif cfg['data'] == 'dior':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': obj[4],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    elif cfg['data'] == 'ship':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': obj[4],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    elif cfg['data'] == 'tank':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': obj[4],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    elif cfg['data'] == 'haisi':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': obj[4],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    elif cfg['data'] == 'plane':
        with open(filename, 'r') as f:
            objs = [x.strip().split(' ') for x in f.readlines()]

            for obj in objs:
                obj_struct = {'name': obj[4],
                              'bbox': [int(float(obj[0])),
                                       int(float(obj[1])),
                                       int(float(obj[2])),
                                       int(float(obj[3]))]}
                objects.append(obj_struct)
    else:
        raise RuntimeError('No dataset issued')

    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetpath,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images

    files = os.listdir(imagesetpath)
    imagenames = [x.strip('.png').strip('.jpg').strip('.tif') for x in files]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        det = [False] * len(R)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox, 'det': det}

    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    # pdb.set_trace()
    clsfile = path.join(path.dirname(imagesetpath), '{}_test.txt')
    clsfile = clsfile.format(classname)
    splitlines = [x.strip().split(' ') for x in lines]
    # print('before', len(splitlines))
    if args.single:
        print('before', len(splitlines))
        splitlines = filter(splitlines, clsfile)
        print('after', len(splitlines))
    # splitlines = bbox_filter(splitlines, conf=0.02)
    # print('after', len(splitlines))
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :] if len(BB) != 0 else BB
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def _do_python_eval(res_prefix, conf_path, novel=False, output_dir='output'):
    global cfg
    cfg = read_data_cfg(conf_path)
    # _devkit_path = '/data2/bykang/pytorch-yolo2/VOCdevkit'
    _devkit_path = os.path.split(cfg['valid'])[0]
    _year = '2007'
    dataset_name = cfg['data']
    global classes
    if dataset_name == 'nwpu':
        classes = ['airplane',
                   'ship',
                   'storage-tank',
                   'baseball-diamond',
                   'tennis-court',
                   'basketball-court',
                   'ground-track-field',
                   'harbor',
                   'bridge',
                   'vehicle']
    elif dataset_name == 'dior':
        classes = ['airplane',
                   'airport',
                   'baseballfield',
                   'basketballcourt',
                   'bridge',
                   'chimney',
                   'dam',
                   'Expressway-Service-area',
                   'Expressway-toll-station',
                   'golffield',
                   'groundtrackfield',
                   'harbor',
                   'overpass',
                   'ship',
                   'stadium',
                   'storagetank',
                   'tenniscourt',
                   'trainstation',
                   'vehicle',
                   'windmill']
    elif dataset_name == 'ship':
        classes = ['ship', 'aircraft-carrier']
    elif dataset_name == 'tank':
        classes = ['ship', '1030102', '10301030102']
    elif dataset_name == 'haisi':
        classes = ['ship','oil-tank','bridge','airplane']
    elif dataset_name == 'plane':
        classes = ['A220',
           'A330',
           'A320-321',
           'Boeing737-800',
           'Boeing787',
           'ARJ21',
           'other']
    else:
        raise RuntimeError('No dataset issued')
    _novel_file = cfg['novel']
    novelid = cfg['novelid']
    print('novelid: {}'.format(novelid))
    _novel_classes = get_novels(_novel_file, novelid)

    # _novel_classes = ('bird', 'bus', 'cow', 'motorbike', 'sofa')

    # filename = '/data/hongji/darknet/results/comp4_det_test_{:s}.txt'
    filename = res_prefix + '{:s}.txt'
    annopath1 = os.path.join(_devkit_path, 'evaluation', 'annotations')
    annopath = annopath1 + '/{:s}.txt'
    imagesetpath = os.path.join(_devkit_path, 'evaluation', 'images')
    cachedir = os.path.join(_devkit_path, 'annotations_cache')
    aps = []
    novel_aps = []
    base_aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(_year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(classes):

        rec, prec, ap = voc_eval(
            filename, annopath, imagesetpath, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        recmax=0
        precmax=0
        for j in range(len(rec)):
            if (rec[j]>=0.8) & (rec[j]<=0.95):
                if recmax<rec[j]:
                    recmax=rec[j]
        for j in range(len(prec)):
            if (prec[j] >= 0.8) & (prec[j] <= 0.95) :
                if prec[j]>precmax:
                    precmax = prec[j]
        aps += [ap]
        msg = 'AP for {} = {:.4f}'.format(cls, ap)
        msg1 = 'Recall for {} = {:.4f}'.format(cls, recmax)
        msg2 = 'Precision for {} = {:.4f}'.format(cls, precmax)
        # print(rec, prec)
        # msg = 'AP for {} = {:.4f}, recall: {:.4f}, precision: {:.4f}'.format(cls, ap, rec, prec)
        if novel and cls in _novel_classes:
            msg = colored(msg, 'green')
            novel_aps.append(ap)
        else:
            base_aps.append(ap)

        print(msg)
        print(msg1)
        print(msg2)
        # print(rec)
        # print(prec)

        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('~~~~~~~~')
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    if novel:
        print(colored('Mean Base AP = {:.4f}'.format(np.mean(base_aps)), 'green'))
        print(colored('Mean Novel AP = {:.4f}'.format(np.mean(novel_aps)), 'green'))
    # print('~~~~~~~~')
    # print('Results:')
    # # pdb.set_trace()
    # for ap in aps:
    #     print('{:.3f}'.format(ap))
    # print('{:.3f}'.format(np.mean(aps)))
    # print('~~~~~~~~')
    # print('')
    # s = ('{:.2f}\t' * 10).format((np.array(aps) * 100).tolist())
    # if novel:
    #     s += ('{:.2f}\t' * 3).format(np.mean(aps) * 100, np.mean(base_aps) * 100, np.mean(novel_aps) * 100)
    # # print(('{:.2f}\t'*20).format(*(np.array(aps) * 100).tolist()))
    # print(s)
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
    print('-- Thanks, The Management')
    print('--------------------------------------------------------------')


if __name__ == '__main__':
    # res_prefix = '/data/hongji/darknet/project/voc/results/comp4_det_test_'
    parser = argparse.ArgumentParser()
    parser.add_argument('res_prefix', type=str,default='E:/few_shot/torch_high/FSODM-master/results/fewyolov3_nwpu_novel0_neg1/ene000001/comp4_det_test_')
    parser.add_argument('conf_path', type=str,default='E:/few_shot/torch_high/FSODM-master/cfg/fewyolov3_nwpu.data')
    parser.add_argument('--novel', action='store_true')
    parser.add_argument('--single', action='store_true')
    args = parser.parse_args()
    args.novel = True
    print('prefix: {}'.format(args.res_prefix))
    print('config file path: {}'.format(args.conf_path))
    _do_python_eval(args.res_prefix, args.conf_path, novel=args.novel, output_dir='output')
