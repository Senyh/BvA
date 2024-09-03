import sys
import os
import random
import h5py
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset, Subset, random_split
from torchvision.transforms import Compose
import numpy as np
from skimage.transform import rescale
from scipy.ndimage import gaussian_filter, zoom
from copy import deepcopy
from builtins import range


class LADataset(Dataset):
    def __init__(self, image_path='', image_size=[80, 112, 112], stage='train', is_augmentation=False, labeled=False, percentage=0.1, nfc=False, ssl=False):
        super(LADataset, self).__init__()
        self.image_size = image_size
        self.sep = '\\' if sys.platform[:3] == 'win' else '/'
        self.stage = stage
        self.is_augmentation = is_augmentation
        self.image_path = image_path
        self.labeled = labeled
        self.nfc = nfc
        self.ssl = ssl
        if self.stage == 'train':
            with open(self.image_path + "/train.list", "r") as f1:
                patient_list = f1.readlines()
            patient_list = [item.replace("\n", "") for item in patient_list]
            if self.labeled:
                self.sample_list = patient_list[:int(len(patient_list)*percentage)]
            else:
                self.sample_list = patient_list[int(len(patient_list)*percentage):]
        else:
            with open(self.image_path + "/val.list", "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        self.pre_transform = self.pre_transform()
        self.augmentation = self.aug_transform()
        self.post_transform = self.post_transform()

    def __getitem__(self, item):
        if self.stage == 'train':
            image_name = self.sample_list[item]
            h5f = h5py.File(self.image_path + "/LA_h5/" + image_name + "/mri_norm2.h5", 'r')
            image = h5f['image'][:]   # W H D
            label = h5f['label'][:]   # W H D
            image = image.transpose(2, 1, 0)  # D H W
            label = label.transpose(2, 1, 0)  # D H W
            mid_idx = len(image) // 2
            if self.labeled:
                if self.ssl != True:
                    barely_label = torch.zeros(label.shape).fill_(-100).numpy()
                    barely_label[mid_idx] = label[mid_idx]
                    label = barely_label
                if self.nfc:
                    #######################################################################################
                    # Noise-Free Labeled Data Construction
                    slice_img = image[mid_idx]
                    slice_lbl = label[mid_idx]

                    # rescale
                    scale_factor = 0.8 + np.random.rand(1).item() * 0.4
                    slice_img = rescale(slice_img, scale_factor, order=3)
                    slice_lbl = rescale(slice_lbl, scale_factor, order=0)
                    slice_img = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0)
                    slice_lbl = torch.from_numpy(slice_lbl).unsqueeze(0).unsqueeze(0).float()
                    # divide and stack
                    r = 2
                    s = 8
                    h, w = slice_img.shape[2] // r, slice_img.shape[3] // r
                    with torch.no_grad():
                        unfold = torch.nn.Unfold(kernel_size=(h, w), stride=s)
                        slice_img_patches = unfold(slice_img).view(h, w, -1).permute(2, 0, 1).clone().numpy()
                        slice_lbl_patches = unfold(slice_lbl).view(h, w, -1).permute(2, 0, 1).clone().numpy().astype('uint8')
                    # reshape
                    xy_factor = r/scale_factor
                    image = zoom(slice_img_patches, (1, xy_factor, xy_factor), order=3)
                    label = zoom(slice_lbl_patches, (1, xy_factor, xy_factor), order=0)
                    #######################################################################################
            else:
                if self.nfc == True and torch.rand(1) < 0.5:
                    #######################################################################################
                    mid_idx = len(image) // 2
                    
                    slice_img = image[mid_idx]
                    slice_lbl = label[mid_idx]

                    # rescale
                    scale_factor = 0.8 + np.random.rand(1).item() * 0.4
                    slice_img = rescale(slice_img, scale_factor, order=3)
                    slice_lbl = rescale(slice_lbl, scale_factor, order=0)
                    slice_img = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0)
                    slice_lbl = torch.from_numpy(slice_lbl).unsqueeze(0).unsqueeze(0).float()
                    r = 2
                    s = 8
                    h, w = slice_img.shape[2] // r, slice_img.shape[3] // r
                    with torch.no_grad():
                        unfold = torch.nn.Unfold(kernel_size=(h, w), stride=s)
                        slice_img_patches = unfold(slice_img).view(h, w, -1).permute(2, 0, 1).clone().numpy()  # B x C x h x w x L
                        slice_lbl_patches = unfold(slice_lbl).view(h, w, -1).permute(2, 0, 1).clone().numpy().astype('uint8')  # B x C x h x w x L
                    xy_factor = r/scale_factor
                    image = zoom(slice_img_patches, (1, xy_factor, xy_factor), order=3)
                    label = zoom(slice_lbl_patches, (1, xy_factor, xy_factor), order=0)
                    #######################################################################################
            sample = {'image': image, 'label': label}
            sample = self.pre_transform(sample)
            sampleA1, sampleA2 = deepcopy(sample), deepcopy(sample)
            sampleA1, sampleA2 = self.augmentation(sampleA1), self.augmentation(sampleA2)
            sample = self.post_transform(sample)
            sampleA1 = self.post_transform(sampleA1)
            sampleA2 = self.post_transform(sampleA2)
            return sample['image'], sample['label'].unsqueeze(0), sampleA1['image'], sampleA2['image']
        else:
            image_name = self.sample_list[item]
            h5f = h5py.File(self.image_path + "/LA_h5/" + image_name + "/mri_norm2.h5", 'r')
            image = h5f['image'][:].transpose(2, 1, 0)
            label = h5f['label'][:].transpose(2, 1, 0)
            sample = {'image': image, 'label': label}
            sample = self.post_transform(sample)
            return sample['image'], sample['label'].unsqueeze(0)
        
        

    def __len__(self):
        return len(self.sample_list)

    def pre_transform(self):
        return Compose([
            RandomRotFlip(),
            RandomCrop(output_size=self.image_size)
        ])
    
    def aug_transform(self):
        return Compose([
            AugGama(p=0.5),
            GaussianBlur(p=0.8)
        ])

    def post_transform(self):
        return Compose([
            CenterCrop(output_size=self.image_size),
            ToTensor(),
        ])


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pd = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pw = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        (d, h, w) = image.shape

        d1 = int(round((d - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        w1 = int(round((w - self.output_size[2]) / 2.))

        label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pd = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pw = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)
            label = np.pad(label, [(pd, pd), (ph, ph), (pw, pw)], mode='constant', constant_values=0)

        (d, h, w) = image.shape
        d1 = np.random.randint(0, d - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        w1 = np.random.randint(0, w - self.output_size[2])

        label = label[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
        image = image[d1:d1 + self.output_size[0], h1:h1 + self.output_size[1], w1:w1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if torch.rand(1) < self.p:
            k = np.random.randint(0, 4)

            random_axes = tuple(random.sample([0, 1, 2], 2))
            image = np.rot90(image, k, axes=random_axes)
            label = np.rot90(label, k, axes=random_axes)

            axis = np.random.randint(0, 3)

            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
            return {'image': image, 'label': label}
        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1, p=0.8):
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if torch.rand(1) < self.p:
            noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
            noise = noise + self.mu
            image = image + noise
        return {'image': image, 'label': label}
    

class GaussianBlur(object):
    def __init__(self, sigma=(0.1, 2.), p=0.8):
        self.sigma = sigma
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        if torch.rand(1) < self.p:
            image = gaussian_filter(image, sigma=sigma)
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}


class Normalize(object):
    """Z-score Normalization."""

    def __call__(self, sample):
        image = sample['image']

        mean = torch.mean(image)
        std = torch.std(image)

        if std > 0:
            ret = (image - mean) / std
        else:
            ret = image * 0.
        return {'image': ret, 'label': sample['label']}

class AugContrast(object):
    def __init__(self, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p=0.8):
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, sample):
        data_sample, label = sample['image'], sample['label']
        if np.random.uniform() < self.p:
            if not self.per_channel:
                if callable(self.contrast_range):
                    factor = self.contrast_range()
                else:
                    if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                        factor = np.random.uniform(self.contrast_range[0], 1)
                    else:
                        factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])

                mn = data_sample.mean()
                if self.preserve_range:
                    minm = data_sample.min()
                    maxm = data_sample.max()

                data_sample = (data_sample - mn) * factor + mn

                if self.preserve_range:
                    data_sample[data_sample < minm] = minm
                    data_sample[data_sample > maxm] = maxm
            else:
                for c in range(data_sample.shape[0]):
                    if callable(self.contrast_range):
                        factor = self.contrast_range()
                    else:
                        if np.random.random() < 0.5 and self.contrast_range[0] < 1:
                            factor = np.random.uniform(self.contrast_range[0], 1)
                        else:
                            factor = np.random.uniform(max(self.contrast_range[0], 1), self.contrast_range[1])

                    mn = data_sample[c].mean()
                    if self.preserve_range:
                        minm = data_sample[c].min()
                        maxm = data_sample[c].max()

                    data_sample[c] = (data_sample[c] - mn) * factor + mn

                    if self.preserve_range:
                        data_sample[c][data_sample[c] < minm] = minm
                        data_sample[c][data_sample[c] > maxm] = maxm
        return {'image': data_sample, 'label': label}


class AugBrightnessMultiplicative(object):
    def __init__(self, multiplier_range=(0.75, 1.25), preserve_range=True, per_channel=False, p=0.2):
        self.multiplier_range = multiplier_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p = p

    def __call__(self, sample):
        data_sample, label = sample['image'], sample['label']
        if np.random.uniform() < self.p:
            multiplier = np.random.uniform(self.multiplier_range[0], self.multiplier_range[1])
            if not self.per_channel:
                data_sample *= multiplier
            else:
                for c in range(data_sample.shape[0]):
                    multiplier = np.random.uniform(self.multiplier_range[0], self.multiplier_range[1])
                    data_sample[c] *= multiplier
        return {'image': data_sample, 'label': label}


class AugGama(object):
    def __init__(self, gamma_range=(0.75, 1.25), invert_image=False, epsilon=1e-7, retain_stats=True, per_channel=False, p=0.2):
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.epsilon = epsilon
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.p = p
    def __call__(self, sample):
        data_sample, label = sample['image'], sample['label']
        if np.random.uniform() < self.p:
            if self.invert_image:
                data_sample = - data_sample

            if not self.per_channel:
                retain_stats_here = self.retain_stats() if callable(self.retain_stats) else self.retain_stats
                if retain_stats_here:
                    mn = data_sample.mean()
                    sd = data_sample.std()
                if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                    gamma = np.random.uniform(self.gamma_range[0], 1)
                else:
                    gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
                minm = data_sample.min()
                rnge = data_sample.max() - minm
                data_sample = np.power(((data_sample - minm) / float(rnge + self.epsilon)), gamma) * rnge + minm
                if retain_stats_here:
                    data_sample = data_sample - data_sample.mean()
                    data_sample = data_sample / (data_sample.std() + 1e-8) * sd
                    data_sample = data_sample + mn
            else:
                for c in range(data_sample.shape[0]):
                    retain_stats_here = self.retain_stats() if callable(self.retain_stats) else self.retain_stats
                    if retain_stats_here:
                        mn = data_sample[c].mean()
                        sd = data_sample[c].std()
                    if np.random.random() < 0.5 and self.gamma_range[0] < 1:
                        gamma = np.random.uniform(self.gamma_range[0], 1)
                    else:
                        gamma = np.random.uniform(max(self.gamma_range[0], 1), self.gamma_range[1])
                    minm = data_sample[c].min()
                    rnge = data_sample[c].max() - minm
                    data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + self.epsilon)), gamma) * float(rnge + self.epsilon) + minm
                    if retain_stats_here:
                        data_sample[c] = data_sample[c] - data_sample[c].mean()
                        data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                        data_sample[c] = data_sample[c] + mn
            if self.invert_image:
                data_sample = - data_sample
        return {'image': data_sample, 'label': label}

