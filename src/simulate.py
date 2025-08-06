#!/usr/bin/env python3

import argparse
import ast
import time
import warnings
from functools import partial
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchio as tio
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import os
import re

import models
import push
from interpret import attr_methods, attribute
from utils import (FocalLoss, get_hashes, get_m_indexes, get_transform_aug, iad, lc, load_data,
                   load_subjs_batch, makedir, output_results, preload, preprocess, print_param,
                   process_iad, save_cvs)
from utils import *


def overlay_mask_on_image(image, mask, output_dir='output_slices', filename=''):
    image = image.squeeze(0)  # (H, W, D)
    mask = mask.squeeze(0)    # (H, W, D)
    H, W, D = image.shape
    os.makedirs(output_dir, exist_ok=True)
    # take slice
    slice_id = D // 2
    img_slice = image[:, :, slice_id].cpu().numpy()
    mask_slice = mask[:, :, slice_id].cpu().numpy()
    # plot
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    plt.figure(figsize=(5, 5))
    plt.imshow(img_slice, cmap='gray', interpolation='none')
    plt.imshow(mask_slice, cmap='jet', alpha=0.4, interpolation='none')  # heatmap 样式
    plt.axis('off')
    plt.tight_layout()
    # save
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()




def print_results(dataset, f_x, y, m_f_xes=None, lcs=None, n_prototypes=None, iads=None, splits=None):
    print(f"{dataset}", end='', flush=True)
    if splits is not None:
        accs = f_splits(splits, accuracy, f_x, y)
        bacs = f_splits(splits, balanced_accuracy, f_x, y)
        sens = f_splits(splits, sensitivity, f_x, y)
        spes = f_splits(splits, specificity, f_x, y)
        aucs = f_splits(splits, auroc, f_x, y)
        print(f" ACC: {accs.mean():.3f}±{accs.std():.3f},"
              f" BAC: {bacs.mean():.3f}±{bacs.std():.3f},"
              f" SEN: {sens.mean():.3f}±{sens.std():.3f},"
              f" SPE: {spes.mean():.3f}±{spes.std():.3f},"
              f" AUC: {aucs.mean():.3f}±{aucs.std():.3f}")
    else:
        print(f" ACC: {accuracy(f_x, y):.3f},"
              f" BAC: {balanced_accuracy(f_x, y):.3f},"
              f" SEN: {sensitivity(f_x, y):.3f},"
              f" SPE: {specificity(f_x, y):.3f},"
              f" AUC: {auroc(f_x, y):.3f}")
    if m_f_xes:
        maxlen_remaining = 0
        for remaining in m_f_xes:
            maxlen_remaining = max(len(remaining), maxlen_remaining)
        for remaining, m_f_x in m_f_xes.items():
            print(f"({remaining:>{maxlen_remaining}})", end='', flush=True)
            if splits is not None:
                accs = f_splits(splits, accuracy, m_f_x, y)
                bacs = f_splits(splits, balanced_accuracy, m_f_x, y)
                sens = f_splits(splits, sensitivity, m_f_x, y)
                spes = f_splits(splits, specificity, m_f_x, y)
                aucs = f_splits(splits, auroc, m_f_x, y)
                print(f" ACC: {accs.mean():.3f}±{accs.std():.3f},"
                      f" BAC: {bacs.mean():.3f}±{bacs.std():.3f},"
                      f" SEN: {sens.mean():.3f}±{sens.std():.3f},"
                      f" SPE: {spes.mean():.3f}±{spes.std():.3f},"
                      f" AUC: {aucs.mean():.3f}±{aucs.std():.3f}")
            else:
                print(f" ACC: {accuracy(m_f_x, y):.3f},"
                      f" BAC: {balanced_accuracy(m_f_x, y):.3f},"
                      f" SEN: {sensitivity(m_f_x, y):.3f},"
                      f" SPE: {specificity(m_f_x, y):.3f},"
                      f" AUC: {auroc(m_f_x, y):.3f}")
    if lcs:
        for remaining in lcs:
            print(f"({remaining:>{maxlen_remaining}})", end='', flush=True)
            maxlen_method, maxlen_metric = 0, 0
            for method, lcs_ in lcs[remaining].items():
                for metric, lcs__ in lcs_.items():
                    maxlen_method = max(len(method), maxlen_method)
                    maxlen_metric = max(len(metric), maxlen_metric)

            if splits is None:
                for method, lcs_ in lcs[remaining].items():
                    for metric, lcs__ in lcs_.items():
                        print(f"{method:>{maxlen_method}} {metric:<{maxlen_metric}}:"
                              f" {lcs__.mean(1).mean():.3f}±{lcs__.mean(1).std():.3f}")
            else:
                for method, lcs_ in lcs[remaining].items():
                    for metric, lcs__ in lcs_.items():
                        res = np.array(lcs__)
                        print(f"{method:>{maxlen_method}} {metric:<{maxlen_metric}}:"
                              f" {np.mean(res):.3f}±{np.std(res):.3f}", end='')
                    print()

def test(net, data_loader, grid=None):
    if not use_da:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    net.eval()
    f_x, m_f_xes, lcs, iads = [], {}, {}, {}
    m_lcs = {}
    if grid and grid.get('attrs'):
        methods = grid['attrs']
    elif 'MProtoNet' in model_name:
        methods = 'M'
    else:
        methods = 'G'
    with torch.no_grad():
        for b, subjs_batch in enumerate(data_loader):
            data, target, seg_map = load_subjs_batch(subjs_batch)
            data = data.to(device, non_blocking=True)
            target = target.argmax(1).to(device, non_blocking=True)
            seg_map = seg_map.to(device, non_blocking=True)
            f_x.append(F.softmax(net(data), dim=1).cpu().numpy())

            print("Missing Modalities:", end='', flush=True)
            m_indexes = get_m_indexes()
            for remaining, m_index in m_indexes.items():
                # 0. data preparation
                data_missing = data.clone()
                r_index = list(set(range(data.shape[1])) - set(m_index))
                data_missing[:, m_index] = 0 # data[:, r_index].mean(1, keepdim=True)
                
                # 1. foward: f_x
                if m_f_xes.get(remaining) is None:
                    m_f_xes[remaining] = []
                if hasattr(net, 'p_mode') and net.p_mode >= 6:
                    net.mamba_score = 1
                    modality_mask = torch.ones((data_missing.shape[0], 4)).to(device)
                    if 'T1' not in remaining:
                        modality_mask[:, 0] = 0
                    if 'T1CE' not in remaining:
                        modality_mask[:, 1] = 0
                    if 'T2' not in remaining:
                        modality_mask[:, 2] = 0
                    if 'FLAIR' not in remaining:
                        modality_mask[:, 3] = 0
                    m_f_xes[remaining].append(F.softmax(net(data_missing, modality_mask), dim=1).cpu().numpy())
                else:
                    m_f_xes[remaining].append(F.softmax(net(data_missing), dim=1).cpu().numpy())
                
                # 2. lcs
                for method_i in methods:
                    method = attr_methods[method_i]
                    print(f" {method}:", end='', flush=True)
                    if m_lcs.get(remaining) is None:
                        m_lcs[remaining] = {}
                    if not m_lcs[remaining].get(method):
                        m_lcs[remaining][method] = {f'({a}, Th=0.5) {m}': [] for a in ['WT'] for m in ['AP', 'DSC']}
                    tic = time.time()
                    attr = attribute(net, data_missing, target, device, method)
                    m_lcs[remaining][method]['(WT, Th=0.5) AP'].append(lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='AP'))
                    m_lcs[remaining][method]['(WT, Th=0.5) DSC'].append(lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='DSC'))
                    toc = time.time()    
            
            # full
            for method_i in methods:
                method = attr_methods[method_i]
                if m_lcs.get('T1, T1CE, T2, FLAIR') is None:
                    m_lcs['T1, T1CE, T2, FLAIR'] = {}
                if not m_lcs['T1, T1CE, T2, FLAIR'].get(method):
                    m_lcs['T1, T1CE, T2, FLAIR'][method] = {f'({a}, Th=0.5) {m}': [] for a in ['WT'] for m in ['AP', 'DSC']}
                attr = attribute(net, data, target, device, method)
                m_lcs['T1, T1CE, T2, FLAIR'][method]['(WT, Th=0.5) AP'].append(lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='AP'))
                m_lcs['T1, T1CE, T2, FLAIR'][method]['(WT, Th=0.5) DSC'].append(lc(attr, seg_map, annos=[1, 2, 4], threshold=0.5, metric='DSC'))

            print(f" {toc - tic:.6f}s,", end='', flush=True)
            print(" Finished.")

        
    for remaining, m_f_x in m_f_xes.items():
        m_f_xes[remaining] = np.vstack(m_f_x)
    for remaining, _ in m_lcs.items():
        for method, lcs_ in m_lcs[remaining].items():
            for metric, lcs__ in lcs_.items():
                m_lcs[remaining][method][metric] = np.vstack(lcs__)
    return np.vstack(f_x), m_f_xes, m_lcs, iads


def parse_arguments():
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, required=True, help="name of the model")
    parser.add_argument('-d', '--data-path', type=str,
                        default='/home/sunzhe/vhua/data/MICCAI_BraTS2020_TrainingData/',
                        help="path to the data files")
    parser.add_argument('-n', '--max-n-epoch', type=int, default=200,
                        help="maximum number of epochs to train on")
    parser.add_argument('-p', '--param-grid', type=str, default=None,
                        help="grid of hyper-parameters")
    parser.add_argument('-s', '--seed', type=int, default=0, help="random seed")
    parser.add_argument('--bc-opt', type=str,
                        choices={'Off', 'BCE', 'B2CE', 'CBCE', 'FL', 'BFL', 'B2FL', 'CBFL'},
                        default='BFL', help="balanced classification option")
    parser.add_argument('--op-opt', type=str, choices={'Adam', 'AdamW'}, default='AdamW',
                        help="optimizer option")
    parser.add_argument('--lr-opt', type=str,
                        choices={'Off', 'StepLR', 'CosALR', 'WUStepLR', 'WUCosALR'},
                        default='WUCosALR', help="learning rate scheduler option")
    parser.add_argument('--lr-n', type=int, default=0, help="learning rate scheduler number")
    parser.add_argument('--wu-n', type=int, default=0, help="number of warm-up epochs")
    parser.add_argument('--early-stop', type=int, choices={0, 1}, default=0,
                        help="whether to enable early stopping")
    parser.add_argument('--n-workers', type=int, default=8, help="number of workers in data loader")
    parser.add_argument('--n-threads', type=int, default=4, help="number of CPU threads")
    parser.add_argument('--preloaded', type=int, choices={0, 1, 2}, default=1,
                        help="whether to preprocess (1) and preload (2) the dataset")
    parser.add_argument('--augmented', type=int, choices={0, 1}, default=1,
                        help="whether to perform data augmentation during training")
    parser.add_argument('--aug-seq', type=str, default=None, help="data augmentation sequence")
    parser.add_argument('--use-cuda', type=int, choices={0, 1}, default=1,
                        help="whether to use CUDA if available")
    parser.add_argument('--use-amp', type=int, choices={0, 1}, default=1,
                        help="whether to use automatic mixed precision")
    parser.add_argument('--use-da', type=int, choices={0, 1}, default=0,
                        help="whether to use deterministic algorithms.")
    parser.add_argument('--gpus', type=str, default='0', help="index(es) of GPU(s)")
    parser.add_argument('--load-model', type=str, default=None,
                        help="whether to load the model files")
    parser.add_argument('--save-model', type=int, choices={0, 1}, default=0,
                        help="whether to save the best model")
    parser.add_argument('--mamba_dim', type=int, default=32, help="dimension of the mamba model")
    parser.add_argument('--align_mode', type=str, default='0.5', help="alignment schedule of mamba_model")

    return parser.parse_args()


if __name__ == '__main__':
    tic = time.time()
    # Parse command-line arguments
    args = parse_arguments()
    model_name, data_path = args.model_name, args.data_path
    max_n_epoch, param_grid, seed = args.max_n_epoch, args.param_grid, args.seed
    bc_opt, op_opt, lr_opt, lr_n, wu_n = args.bc_opt, args.op_opt, args.lr_opt, args.lr_n, args.wu_n
    early_stop, n_workers, n_threads = args.early_stop, args.n_workers, args.n_threads
    preloaded, augmented, aug_seq = args.preloaded, args.augmented, args.aug_seq
    use_cuda, use_amp, use_da, gpus = args.use_cuda, args.use_amp, args.use_da, args.gpus
    load_model, save_model = args.load_model, args.save_model
    if param_grid is not None:
        param_grid = ast.literal_eval(param_grid)
        for k, v in param_grid.items():
            if not isinstance(v, list):
                param_grid[k] = [v]
    else:
        param_grid = {'batch_size': [32], 'lr': [1e-3], 'wd': [1e-2], 'features': ['resnet152_ri'],
                      'n_layers': [6]}
    transform = [tio.ToCanonical(), tio.CropOrPad(target_shape=(192, 192, 144))]
    transform += [tio.Resample(target=(1.5, 1.5, 1.5))]
    transform += [tio.ZNormalization()]
    if augmented:
        if aug_seq is not None:
            transform_aug = get_transform_aug(aug_seq=aug_seq)
        else:
            transform_aug = get_transform_aug()
    else:
        transform_aug = []
    transform_train = tio.Compose(transform + transform_aug)
    transform = tio.Compose(transform)
    if '_pm' in model_name:
        p_mode = int(model_name[model_name.find('_pm') + 3])
        model_name = model_name.replace(f'_pm{p_mode}', '')
    gpu_ids = ast.literal_eval(f'[{gpus}]')
    device = torch.device(
        'cuda:' + str(gpu_ids[0]) if use_cuda and torch.cuda.is_available() else 'cpu')
    use_amp = use_amp == 1 and use_cuda == 1 and torch.cuda.is_available()
    if use_da:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    torch.set_num_threads(n_threads)
    opts_hash = get_hashes(args)
    warnings.filterwarnings('ignore', message="The epoch parameter in `scheduler.step\(\)` was not")
    # Load data
    x, y = load_data(data_path)
    if preloaded:
        np.random.seed(seed)
        torch.manual_seed(seed)
        dataset = tio.SubjectsDataset(list(x), transform=transform)
        if preloaded > 1:
            data_loader = DataLoader(dataset, batch_size=(n_workers + 1) // 2, num_workers=n_workers)
            x, y, seg = preload(data_loader)
            n_workers = 0
        else:
            data_loader = DataLoader(dataset, num_workers=n_workers)
            x = preprocess(data_loader)
            transform_train = tio.Compose(transform_aug) if augmented else None
            transform = None
        del dataset, data_loader
        toc = time.time()
        print(f"Elapsed time is {toc - tic:.6f} seconds.")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    cv_train = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed)
    # 5-fold CV
    cv_fold = cv.get_n_splits()
    cv_train_fold = cv_train.get_n_splits()
    splits = np.zeros(y.shape[0], dtype=int)
    param_grids = ParameterGrid(param_grid)
    f_x, m_f_xes, lcs, n_prototypes, iads = np.zeros(y.shape), {}, {}, None, {}
    for i, (I_train, I_test) in enumerate(cv.split(x, y.argmax(1))):
        print(f">>>>>>>> CV = {i + 1}:")
        splits[I_test] = i
        if len(param_grids) > 1 or early_stop:
            # TODO: 5-fold inner CV
            pass
        else:
            best_grid = param_grids[0]
        if early_stop:
            # TODO: 5-fold inner CV
            best_n_epoch = max_n_epoch
        else:
            best_n_epoch = max_n_epoch
        # Training and test
        np.random.seed(seed)
        torch.manual_seed(seed)
        if preloaded > 1:
            dataset_train = TensorDataset(x[I_train], y[I_train], seg[I_train])
            dataset_test = TensorDataset(x[I_test], y[I_test], seg[I_test])
            in_size = (4,) + x.shape[2:]
            out_size = y.shape[1]
        else:
            dataset_train = tio.SubjectsDataset(list(x[I_train]), transform=transform_train)
            if 'MProtoNet' in model_name:
                dataset_push = tio.SubjectsDataset(list(x[I_train]), transform=transform)
            dataset_test = tio.SubjectsDataset(list(x[I_test]), transform=transform)
            in_size = (4,) + dataset_train[0]['t1']['data'].shape[1:]
            out_size = dataset_train[0]['label'].shape[0]
        loader_train = DataLoader(dataset_train, batch_size=best_grid['batch_size'], shuffle=True, num_workers=n_workers, pin_memory=True, drop_last=True)
        loader_test = DataLoader(dataset_test, batch_size=(best_grid['batch_size'] + 1) // 2, num_workers=n_workers, pin_memory=True)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        kwargs = {'in_size': in_size, 'out_size': out_size}
        for arg in ['features', 'n_layers']:
            if best_grid.get(arg):
                kwargs[arg] = best_grid[arg]
        if 'MProtoNet' in model_name:
            if '_pm' in args.model_name:
                kwargs['p_mode'] = p_mode
            for arg in ['prototype_shape', 'f_dist', 'topk_p']:
                if best_grid.get(arg):
                    kwargs[arg] = best_grid[arg]
            if p_mode >= 6:
                kwargs['mamba_dim'] = args.mamba_dim
        net = getattr(models, model_name)(**kwargs).to(device)
        if use_cuda and torch.cuda.is_available() and len(gpu_ids) > 1:
            net = nn.DataParallel(net, device_ids=gpu_ids)
            is_parallel = True
        else:
            is_parallel = False
        if load_model is not None:
            if load_model.startswith(args.model_name):
                model_name_i = f'{load_model}_cv{i}'
                model_path_i = f'../results/models/{model_name_i}.pt'
            else:
                model_name_i = f'{load_model[load_model.find(args.model_name):]}_cv{i}'
                model_path_i = f'{load_model}_cv{i}.pt'
        else:
            model_name_i = f'{args.model_name}_{opts_hash}_cv{i}'
        print(f"Model: {model_name_i}\n{str(net)}")
        print_param(net, show_each=False)
        print(f"Hyper-parameters = {param_grid}")
        print(f"Best Hyper-parameters = {best_grid}")
        print(f"Best Number of Epoch = {best_n_epoch}")

        if is_parallel:
            net.module.load_state_dict(torch.load(model_path_i, map_location=device))
        else:
            net.load_state_dict(torch.load(model_path_i, map_location=device))
        del dataset_train, loader_train

        # test
        if 'MProtoNet' in model_name:
            method = 'M'
        else:
            method = 'G'
        method = attr_methods[method]
        with torch.no_grad():
            total_num_sample_missing = total_num_sample = 0
            for b, (subjs_batch, j) in enumerate(zip(loader_test, np.arange(1))):
                data, target, seg_map = load_subjs_batch(subjs_batch)
                data = data.to(device, non_blocking=True)
                target = target.argmax(1).to(device, non_blocking=True)
                seg_map = seg_map.to(device, non_blocking=True)
                print("Missing Modalities:", end='', flush=True)
                m_indexes = get_m_indexes()
                for remaining, m_index in m_indexes.items():
                    num_sample_missing = total_num_sample_missing
                    # 0. data preparation
                    data_missing = data.clone()
                    r_index = list(set(range(data.shape[1])) - set(m_index))
                    data_missing[:, m_index] = 0
                    # 1. maps    
                    attr = attribute(net, data_missing, target, device, method) # (B, 4, H, W, D)
                    # 2. merge
                    remaining = re.sub(r'[^a-zA-Z0-9_]', '_', remaining).replace('__', '_')
                    for k, (img, mask) in enumerate(zip(data, attr)):
                        overlay_mask_on_image(img[1], mask[1], 'missing_maps', f'cv{i}_{num_sample_missing}_{remaining}_{model_name}_t1ce.png')
                        if k == 0:
                            overlay_mask_on_image(img[1], seg_map[j][0], 'missing_maps', f'cv{i}_{num_sample_missing}_GT_{model_name}_t1ce.png')
                        num_sample_missing += 1
                total_num_sample_missing = num_sample_missing

                # full and GT
                # 1. maps
                attr = attribute(net, data, target, device, method) # (B, 4, H, W, D)
                # 2. merge
                for k, (img, mask) in enumerate(zip(data, attr)):
                    # full
                    overlay_mask_on_image(img[1], mask[1], 'missing_maps', f'cv{i}_{total_num_sample}_full_{model_name}_t1ce.png')
                    # GT
                    seg = seg_map[k][0]
                    seg[seg > 0] = 1
                    overlay_mask_on_image(img[1], seg, 'missing_maps', f'cv{i}_{total_num_sample}_GT_{model_name}_t1ce.png')
                    total_num_sample += 1
                    
