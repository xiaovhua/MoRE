# 1. Dataset
import nibabel
import torchio as tio
from torch.utils.data import DataLoader

from utils import load_data, preprocess

n_samples = 64
cv = 1
data_path=f'/home/sunzhe/vhua/data/MICCAI_BraTS2020_TrainingData/' 
model_path = f'/home/sunzhe/vhua/project/mprotonet/results/models/mamba/4_no_mambas/MProtoNet3D_pm6_7797d809_cv{cv}.pt'
prototype_path = f'/home/sunzhe/vhua/project/mprotonet/results/saved_imgs/MProtoNet3D_pm6_7797d809_cv{cv}/bb-raw_images.npy'
target_path = f'/home/sunzhe/vhua/project/mprotonet/results/saved_imgs/MProtoNet3D_pm6_7797d809_cv{cv}/bb-receptive_fieldNone.npy'

base_transform = tio.Compose(
    [
        tio.ToCanonical(),
        tio.CropOrPad(target_shape=(192, 192, 144)),
        tio.Resample(target=(1.5, 1.5, 1.5)),
        tio.ZNormalization()
    ]
)

# load
x, y = load_data(data_path=data_path)
# preprocess
dataset = tio.SubjectsDataset(list(x), transform=base_transform)
data_loader = DataLoader(dataset, num_workers=8)
x = preprocess(data_loader)
print(f'Finish Pre-processing Dataset from {data_path}')


# 2. Model
import torch
from models import MProtoNet3D, MProtoNet

# build
kwargs = {
    'in_size': (4, 128, 128, 96),
    'out_size': 2,
    'backbone': 'resnet152_ri',
    'prototype_shape': (30, 128, 1, 1, 1),
    'f_dist': 'cos',
    'topk_p': 1,
    'p_mode': 6,
    'mamba_dim': 64,
    'n_layers': 5,
}

net = MProtoNet3D(**kwargs)

# load
net.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
net.mamba_score = 1.
print(f'Finish Loading Model from {model_path}, #P={sum([p.numel() for p in net.parameters() if p.requires_grad])}')

# 3. Diagnosis & Attribution
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from utils import load_subjs_batch
from interpret import attribute

def diag_and_attr(net, data_loader, device):
    net = net.to(device)
    net.eval()
    net.alpha = 1.0
    f_xs, scores, attrs, ys = [], [], [], []
    if isinstance(net, MProtoNet):
        method = 'MProtoNet'
    else:
        method = 'GradCAM'
    with torch.no_grad():
        for b, subjs_batch in enumerate(tqdm(data_loader)):
            data, target, _ = load_subjs_batch(subjs_batch)
            data = data.to(device, non_blocking=True)
            target = target.argmax(1).to(device, non_blocking=True)
            # forward
            f_x, p_map, x, x_mamba, x_feat = net.conv_features(data)
            distances, p_map, _ = net.prototype_distances(f_x, p_map)
            distances = distances.flatten(1)
            prototype_activations = net.distance_2_similarity(distances)
            logits = net(data) # net.last_layer(prototype_activations)
            # append output
            scores.append(prototype_activations.cpu().numpy())
            f_xs.append(F.softmax(logits, dim=1).cpu().numpy())
            ys.extend(list(target.cpu().numpy()))
            # attribute
            attr = attribute(net, data, target, 0, method)
            attrs.append(attr.cpu().numpy()[:, 0])
    return np.vstack(f_xs), np.vstack(scores), np.array(ys), np.vstack(attrs)


def diag_and_attr_missing(net, data_loader, device):
    net = net.to(device)
    net.eval()
    f_xs, scores, attrs, ys = [], [], [], []
    if isinstance(net, MProtoNet):
        method = 'MProtoNet'
    else:
        method = 'GradCAM'
    with torch.no_grad():
        for b, subjs_batch in enumerate(tqdm(data_loader)):
            data, target, _ = load_subjs_batch(subjs_batch)
            data = data.to(device, non_blocking=True)
            # missing settings
            data_missing = data.clone()
            modality_mask = torch.ones((data_missing.shape[0], 4)).to(device)
            data_missing[:, 0] = 0  # drop t1
            data_missing[:, 2] = 0  # drop t2
            modality_mask[:, 0] = 0
            modality_mask[:, 2] = 0
            target = target.argmax(1).to(device, non_blocking=True)
            # forward
            f_x, p_map, x, x_mamba, x_feat = net.conv_features(data_missing, modality_mask)
            distances, p_map, _ = net.prototype_distances(f_x, p_map)
            distances = distances.flatten(1)
            prototype_activations = net.distance_2_similarity(distances)
            logits = net(data_missing, modality_mask) # logits = net.last_layer(prototype_activations)
            # append output
            scores.append(prototype_activations.cpu().numpy())
            f_xs.append(F.softmax(logits, dim=1).cpu().numpy())
            ys.extend(list(target.cpu().numpy()))
            # attribute
            attr = attribute(net, data_missing, target, 0, method, mask=modality_mask)
            attrs.append(attr.cpu().numpy()[:, 0])
    return np.vstack(f_xs), np.vstack(scores), np.array(ys), np.vstack(attrs)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# reload dataset
x_raw = list(x)
x = []
idx = []
for i, xx in enumerate(x_raw):
    if torch.all(xx['label'] == torch.tensor([1, 0])): # LGG
        x.append(xx)
        idx.append(i)
    if len(x) >= n_samples // 2:
        break
for i, xx in enumerate(x_raw):
    if torch.all(xx['label'] == torch.tensor([0, 1])): # HGG
        x.append(xx)
        idx.append(i)
    if len(x) >= n_samples:
        break
print(idx)

# import collections
# cnt = collections.Counter([xx['label'] for xx in x_raw])

dataset = tio.SubjectsDataset(list(x[:n_samples]))
data_loader = DataLoader(dataset, batch_size=8, num_workers=8, pin_memory=True, shuffle=False)
# forward
f_xs, scores, ys, attrs = diag_and_attr(net, data_loader, device)
f_xs_missing, scores_missing, ys_missing, attrs_missing = diag_and_attr_missing(net, data_loader, device)

# filter scores
prototype_filters = net.prototype_class_identity[:, ys].mT.cpu().numpy()
scores = prototype_filters * scores
scores_missing = prototype_filters * scores_missing
print('Forwarding Complete')

# 4. Visualize Localization
import os
import cv2

saved = True  # whether to save the resulting figures
save_path = './visualize/'
modalities = ['t1ce', 'flair']  # only use t1ce and flair as an example here

if saved:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

modality_ids = {'t1': 0, 't1ce': 1, 't2': 2, 'flair': 3}

gts = {}
combines = {}
for m in modalities:
    gts[m] = []
    combines[m] = []
reserved_id = []
raw_xs = []

for i, (raw_x, attr) in enumerate(zip(list(dataset), attrs)):
    # 4.1. load and slice
    # get raw MRI 'raw_x', attribution map 'attr' and the segmentation map (ground truth) 'seg'
    seg = np.array(raw_x['seg'])[0]
    raw_x = np.concatenate([raw_x['t1'], raw_x['t1ce'], raw_x['t2'], raw_x['flair']], axis=0)
    # only use the middle slice of depth (the third dimension in (H, W, D)) for visualization
    axial_median = attr.shape[-1] // 2
    x_median, attr_median, seg_median = raw_x[..., axial_median], attr[..., axial_median], seg[..., axial_median]
    # 4.2. calculate bounding box according to the segmentation map
    seg_median = np.array(seg_median >= 1, dtype=int)
    seg_median = np.uint8(255 * seg_median)
    seg_rot = cv2.rotate(seg_median, cv2.ROTATE_90_COUNTERCLOCKWISE)
    xx = np.sum(seg_rot, axis=0)
    yy = np.sum(seg_rot, axis=1)
    xx = np.nonzero(xx)[0]
    yy = np.nonzero(yy)[0]
    if len(xx) == 0:  # if the ground truth has a very small bounding box, then drop it
        continue
    x_start, x_end = xx[0], xx[-1]
    y_start, y_end = yy[0], yy[-1]
    # 4.3. reload segmentation map
    seg_median = cv2.applyColorMap(seg_median, cv2.COLORMAP_JET)
    cv2.imwrite('temp_seg.png', seg_median)  # temporary files
    seg_median = cv2.imread('temp_seg.png')
    # 4.4. calculate heatmap from attribution map
    attr_median = (attr_median - np.min(attr_median)) / (np.max(attr_median) - np.min(attr_median))
    heatmap = cv2.applyColorMap(np.uint8(255 * attr_median), cv2.COLORMAP_JET)
    # 4.5. visualize
    for m in modalities:
        # raw image
        m_id = modality_ids[m]
        x_i = x_median[m_id]
        x_i = (x_i - np.min(x_i)) / (np.max(x_i) - np.min(x_i))
        x_i = np.uint8(255 * x_i)
        cv2.imwrite('temp_raw.png', x_i)  # temporary files
        x_i = cv2.imread('temp_raw.png')

        # raw + attribution map + bounding box
        combine = cv2.rotate(x_i, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combine = cv2.addWeighted(heatmap, 0.4, x_i, 0.6, 0)
        combine = cv2.rotate(combine, cv2.ROTATE_90_COUNTERCLOCKWISE)
        combine = cv2.rectangle(combine, (x_start, y_start), (x_end, y_end), (153, 201, 245), 1)

        # raw + attribution map + crop
        crop = combine[y_start + 1:y_end, x_start + 1:x_end].copy()

        # raw + segmentation map (ground truth) + bounding box
        gt = cv2.addWeighted(seg_median, 0.4, x_i, 0.6, 0)
        gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
        gt = cv2.rectangle(gt, (x_start, y_start), (x_end, y_end), (153, 201, 245), 1)

        # raw + segmentation map (ground truth) + crop
        crop_gt = gt[y_start + 1:y_end, x_start + 1:x_end].copy()

        # save
        if saved:
            cv2.imwrite(save_path + f'{i}_{m}_raw.png', x_i)
            cv2.imwrite(save_path + f'{i}_{m}_combine.png', combine)
            cv2.imwrite(save_path + f'{i}_{m}_crop_combine.png', crop)
            cv2.imwrite(save_path + f'{i}_{m}_gt.png', gt)
            cv2.imwrite(save_path + f'{i}_{m}_crop_gt.png', crop_gt)

        combines[m].append(combine)
        gts[m].append(gt)

    reserved_id.append(i)

os.remove('temp_seg.png')
os.remove('temp_raw.png')

print(reserved_id)
f_xs_2 = f_xs[reserved_id]
ys_2 = ys[reserved_id]
f_xs_2_missing = f_xs_missing[reserved_id]
ys_2_missing = ys_missing[reserved_id]
print('Finish Processing for Visualization')


# 5. Show Localization
import matplotlib.pyplot as plt

img_size = 4
gap = 1

assert len(modalities) > 0 and len(gts[modalities[0]]) > 0, 'The list is empty. You do not have samples to plot.'

def cv2matplot(img):
    b, g, r = cv2.split(img)
    return cv2.merge((r, g, b))

n_columns = min(n_samples, len(gts[modalities[0]]))  # only plot 4 or less samples as an example
n_rows = len(modalities) * 2
figsize = (img_size * n_columns + gap * (n_columns - 1), img_size * n_rows + gap * (n_rows - 1))
text_gap = 1 / n_rows
fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize)

for c in range(n_columns):  # n_sample
    for r in range(n_rows):  # n_modalities * 2, for raw and attribution results
        modality = modalities[r // 2]
        if r % 2  == 0:
            axs[r, c].imshow(cv2matplot(gts[modality][c]))
            fig.text(-0.05, 1 - text_gap * (r + 0.5), 'raw_' + modality, ha='center', va='center', fontsize=5 * n_columns)
            axs[r, c].axis('off')
        else:
            axs[r, c].imshow(cv2matplot(combines[modality][c]))
            fig.text(-0.05, 1 - text_gap * (r + 0.5), modality, ha='center', va='center', fontsize=5 * n_columns)
            axs[r, c].axis('off')

# label
label_dict = {0: 'LGG', 1: 'HGG'}
print('-' * 100)
print('The diagnosis logits are: ', [(i, f) for i, f in zip(reserved_id, f_xs_2[:n_columns])])
print('The diagnosis logits with missing are: ', [(i, f) for i, f in zip(reserved_id, f_xs_2_missing[:n_columns])])
print('The classification labels are: ', [(i, f) for i, f in zip(reserved_id, ys_2[:n_columns])])
print('-' * 100)
print('The diagnosis results are: ', [label_dict[item] for item in np.argmax(f_xs_2, axis=-1)[:n_columns]])
print('The diagnosis results with missing are: ', [label_dict[item] for item in np.argmax(f_xs_2_missing, axis=-1)[:n_columns]])
print('The ground truth results are: ', [label_dict[item] for item in ys_2[:n_columns]])
print('-' * 100)

plt.tight_layout()
plt.show()

# 6. Visualize Attribution
# if you do not have these files, please uncomment the push_prototypes function in the load_model branch, and run evaluation with our checkpoint first

modality = 't1ce'
topk = 8
img_size = 4
gap = 1

# visualize the all prototype samples
prototype_input = torch.tensor(np.load(prototype_path)).float().to(device)
prototype_target = torch.tensor(np.load(target_path))[:, -1].long().to(device)

method = 'MProtoNet' if isinstance(net, MProtoNet) else 'GradCAM'
with torch.no_grad():
    prototype_attrs = attribute(net, prototype_input, prototype_target, 0, method).cpu().numpy()[:, 0]


# plot settings
n_rows = n_samples
n_columns = topk + 1
figsize = (img_size * n_columns + gap * (n_columns - 1), img_size * n_rows + gap * (n_rows - 1))
fig, axs = plt.subplots(n_rows, n_columns, figsize=figsize)
mid = {'t1': 0, 't1ce': 1, 't2': 2, 'flair': 3}[modality]

for r in range(n_rows):
    score = scores[r]
    score_missing = scores_missing[r]
    topk_index_score = sorted(list(enumerate(score)), key=lambda x: x[-1], reverse=True)[:topk]
    topk_index_score_missing = sorted(list(enumerate(score_missing)), key=lambda x: x[-1], reverse=True)[:topk]

    # raw input
    import nibabel
    import numpy as np
    raw_x = dataset[r]
    t1 = nibabel.Nifti1Image(raw_x['t1'].data[0].cpu().numpy(), affine=np.eye(4))
    nibabel.save(t1, f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_t1.nii.gz')
    t1ce = nibabel.Nifti1Image(raw_x['t1ce'].data[0].cpu().numpy(), affine=np.eye(4))
    nibabel.save(t1ce, f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_t1ce.nii.gz')
    t2 = nibabel.Nifti1Image(raw_x['t2'].data[0].cpu().numpy(), affine=np.eye(4))
    nibabel.save(t2, f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_t2.nii.gz')
    flair = nibabel.Nifti1Image(raw_x['flair'].data[0].cpu().numpy(), affine=np.eye(4))
    nibabel.save(flair, f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_flair.nii.gz')
    seg = nibabel.Nifti1Image(raw_x['seg'].data[0].cpu().numpy(), affine=np.eye(4))
    nibabel.save(seg, f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_seg.nii.gz')

    raw_x = np.concatenate([raw_x['t1'], raw_x['t1ce'], raw_x['t2'], raw_x['flair']], axis=0)

    # full maps
    attr = attrs[r]
    # combine
    axial_median = attr.shape[-1] // 2
    # image
    x_i = raw_x[..., axial_median][1]  # t1ce
    x_i = (x_i - np.min(x_i)) / (np.max(x_i) - np.min(x_i))
    x_i = np.uint8(255 * x_i)
    cv2.imwrite('temp_raw.png', x_i)  # temporary files
    x_i = cv2.imread('temp_raw.png')
    # heatmap
    attr_median = attr[..., axial_median]
    attr_median = (attr_median - np.min(attr_median)) / (np.max(attr_median) - np.min(attr_median))
    heatmap = cv2.applyColorMap(np.uint8(255 * attr_median), cv2.COLORMAP_JET)
    # combine
    combine = cv2.addWeighted(heatmap, 0.4, x_i, 0.6, 0)
    combine = cv2.rotate(combine, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_t1ce_full_maps.png', combine)
    os.remove('temp_raw.png')

    # full prototypes
    for t in range(topk):
        prototype_index, score = topk_index_score[t]
        x_p = prototype_input[prototype_index].cpu().numpy()
        attr = prototype_attrs[prototype_index]
        x_i = x_p[..., axial_median][1]  # t1ce
        x_i = (x_i - np.min(x_i)) / (np.max(x_i) - np.min(x_i))
        x_i = np.uint8(255 * x_i)
        cv2.imwrite('temp_raw.png', x_i)  # temporary files
        x_i = cv2.imread('temp_raw.png')
        attr_median = attr[..., axial_median]
        attr_median = (attr_median - np.min(attr_median)) / (np.max(attr_median) - np.min(attr_median))
        heatmap = cv2.applyColorMap(np.uint8(255 * attr_median), cv2.COLORMAP_JET)
        combine = cv2.addWeighted(heatmap, 0.4, x_i, 0.6, 0)
        cv2.imwrite(f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_prototype{t}.png', combine)
        os.remove('temp_raw.png')

    # missing maps
    attr_missing = attrs_missing[r]
    # image
    x_i_missing = raw_x[..., axial_median][1]  # t1ce
    x_i_missing = (x_i_missing - np.min(x_i_missing)) / (np.max(x_i_missing) - np.min(x_i_missing))
    x_i_missing = np.uint8(255 * x_i_missing)
    cv2.imwrite('temp_raw.png', x_i_missing)  # temporary files
    x_i_missing = cv2.imread('temp_raw.png')
    # heatmap
    attr_median_missing = attr_missing[..., axial_median]
    attr_median_missing = (attr_median_missing - np.min(attr_median_missing)) / (np.max(attr_median_missing) - np.min(attr_median_missing))
    heatmap_missing = cv2.applyColorMap(np.uint8(255 * attr_median_missing), cv2.COLORMAP_JET)
    # combine
    combine_missing = cv2.addWeighted(heatmap_missing, 0.4, x_i_missing, 0.6, 0)
    combine_missing = cv2.rotate(combine_missing, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_t1ce_missing_maps.png', combine_missing)
    os.remove('temp_raw.png')

    # missing prototypes
    for t in range(topk):
        prototype_index_missing, score_missing = topk_index_score_missing[t]
        x_p_missing = prototype_input[prototype_index_missing].cpu().numpy()
        attr_missing = prototype_attrs[prototype_index_missing]
        x_i_missing = x_p_missing[..., axial_median][1]  # t1ce
        x_i_missing = (x_i_missing - np.min(x_i_missing)) / (np.max(x_i_missing) - np.min(x_i_missing))
        x_i_missing = np.uint8(255 * x_i_missing)
        cv2.imwrite('temp_raw.png', x_i_missing)  # temporary files
        x_i_missing = cv2.imread('temp_raw.png')
        attr_median_missing = attr_missing[..., axial_median]
        attr_median_missing = (attr_median_missing - np.min(attr_median_missing)) / (np.max(attr_median_missing) - np.min(attr_median_missing))
        heatmap_missing = cv2.applyColorMap(np.uint8(255 * attr_median_missing), cv2.COLORMAP_JET)
        combine_missing = cv2.addWeighted(heatmap_missing, 0.4, x_i_missing, 0.6, 0)
        cv2.imwrite(f'/home/sunzhe/vhua/project/mprotonet/src/imgs/{r}_prototype{t}_missing.png', combine_missing)
        os.remove('temp_raw.png')


score_matrixs = scores[:n_samples]
for i, sm in enumerate(score_matrixs):
    max_v, min_v = max(sm), min([x for x in sm if x != 0])
    sm = list( (np.array(sm) - min_v) / (max_v - min_v) )
    print([i, sorted(sm, reverse=True)[:topk]])

print('*' * 100, len(score_matrixs))

score_matrixs = scores_missing[:n_samples]
for i, sm in enumerate(score_matrixs):
    max_v, min_v = max(sm), min([x for x in sm if x != 0])
    sm = list( (np.array(sm) - min_v) / (max_v - min_v) )
    print([i, sorted(sm, reverse=True)[:topk]])

