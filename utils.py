# -*- coding: utf-8 -*-

import torch
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from PIL import Image
from torch.utils.data import Dataset


class RunningConfusionMatrix():
    """Running Confusion Matrix class that enables computation of confusion matrix
    on the go and has methods to compute such accuracy metrics as Mean Intersection over
    Union MIOU.
    
    Attributes
    ----------
    labels : list[int]
        List that contains int values that represent classes.
    overall_confusion_matrix : sklean.confusion_matrix object
        Container of the sum of all confusion matrices. Used to compute MIOU at the end.
    ignore_label : int
        A label representing parts that should be ignored during
        computation of metrics
        
    """

    def __init__(self, labels, ignore_label=255):

        self.labels = labels
        self.ignore_label = ignore_label
        self.overall_confusion_matrix = None

    def update_matrix(self, ground_truth, prediction):
        """Updates overall confusion matrix statistics.
        If you are working with 2D data, just .flatten() it before running this
        function.
        Parameters
        ----------
        groundtruth : array, shape = [n_samples]
            An array with groundtruth values
        prediction : array, shape = [n_samples]
            An array with predictions
        """

        # Mask-out value is ignored by default in the sklearn
        # read sources to see how that was handled
        # But sometimes all the elements in the groundtruth can
        # be equal to ignore value which will cause the crush
        # of scikit_learn.confusion_matrix(), this is why we check it here
        if (ground_truth == self.ignore_label).all():
            return

        current_confusion_matrix = confusion_matrix(y_true=ground_truth,
                                                    y_pred=prediction,
                                                    labels=self.labels)

        if self.overall_confusion_matrix is not None:

            self.overall_confusion_matrix += current_confusion_matrix
        else:

            self.overall_confusion_matrix = current_confusion_matrix

    def compute_current_mean_intersection_over_union(self):

        intersection = np.diag(self.overall_confusion_matrix)
        ground_truth_set = self.overall_confusion_matrix.sum(axis=1)
        predicted_set = self.overall_confusion_matrix.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        mean_intersection_over_union = np.mean(intersection_over_union)

        return mean_intersection_over_union


def read_images(root, training=True):
    txt_filename = root + "/ImageSets/Segmentation/" + ('train.txt' if training else 'val.txt')
    with open(txt_filename, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i + '.png') for i in images]
    return data, label


class VOCSegDataset(Dataset):
    def __init__(self, training, crop_size, transforms):
        self.training = training
        self.crop_size = crop_size
        self.transforms = transforms
        data_list, label_list = read_images(root="./VOCdevkit/VOC2012/", training=training)
        self.data_list = self._filter(data_list)
        self.label_list = self._filter(label_list)

        print('Reading ' + str(len(self.data_list)) + ' images')

        self.images = []
        self.labels = []

        for idx in range(len(self.data_list)):
            # self.images[idx] = cv2.imread(self.data_list[idx], cv2.COLOR_BGR2RGB)
            # self.images[idx] = Image.fromarray(self.images[idx])

            self.images.append(Image.open(self.data_list[idx]).copy())

            # label_mask = Image.open(self.label_list[idx])
            self.labels.append(Image.open(self.label_list[idx]).copy())

    def _filter(self, images):
        return [im for im in images if (Image.open(im).size[1] >= self.crop_size[0] and
                                        Image.open(im).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        #        img = self.data_list[idx]
        #        label = self.label_list[idx]
        #
        #        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
        #        label = cv2.imread(label, cv2.COLOR_BGR2RGB)

        # img = Image.open(self.data_list[idx])
        # label = Image.open(self.label_list[idx])

        img = self.images[idx]
        label = self.labels[idx]

        #        img = Image.open(img).convert("RGB")

        # print(img.shape)

        img, label = self.transforms(img, label, self.crop_size)
        # label[label==255] = -1
        #        print(img.shape) # (3, H, W)
        return img, label

    def __len__(self):
        return len(self.data_list)

def bilinear_kernel(in_channels, out_channels, kernel_size):
    # reference: https://zh.gluon.ai/chapter_computer-vision/fcn.html
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight)
