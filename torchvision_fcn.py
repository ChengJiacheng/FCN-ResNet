# -*- coding: utf-8 -*-

import os
import random
import cv2
import time
import PIL
import tqdm
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from PIL import Image

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

from utils import RunningConfusionMatrix, VOCSegDataset
import argparse
import pprint

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


from torch.autograd import Variable

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print("Pillow Version: ", PIL.PILLOW_VERSION)

#  datasets dir
voc_root = "./VOCdevkit/VOC2012/"
print(voc_root)


classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]
# 128 =64*2, 192 = 64*3
len(classes), len(colormap)
num_classes = len(classes)

# label1 = Image.open('./VOCdevkit/VOC2012/SegmentationClass/2007_003349.png')
# label1[label1 == 255] = 0

label1 = np.asarray(Image.open('./VOCdevkit/VOC2012/SegmentationClass/2007_003349.png')).copy()
label1[label1 == 255] = 0
label1 = Image.fromarray(label1.astype(np.uint8))

label1 = np.asarray(Image.open('./VOCdevkit/VOC2012/SegmentationClass/2007_000033.png')).copy()
label1[label1 == 255] = 0


# label1 = np.asarray(label1)
# label1 = image2label(label1)
#
#
# label2 = cv2.imread('./VOCdevkit/VOC2012/SegmentationClass/2007_000033.png', cv2.COLOR_BGR2RGB)
# label2 = cv2.imread('./VOCdevkit/VOC2012/SegmentationClass/2007_000033.png')
# label2 = image2label(label2)


def bilinear_kernel(in_channels, out_channels, kernel_size):
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


image = cv2.imread('./VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg', cv2.COLOR_BGR2RGB)

# x=Image.open('./VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
# x=np.array(x)
#
# img=cv2.imread('./VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
# img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#
# im_tfs = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# im_tfs(img)
#
# plt.figure()
# plt.imshow(x)
# plt.show()
#
#
# x=torch.from_numpy(x.astype('float32')).permute(2,0,1).unsqueeze(0)
# conv_trans=nn.ConvTranspose2d(3,3,4,2,1)
# conv_trans.weight.data=bilinear_kernel(3,3,4)
#
# y=conv_trans(Variable(x)).data.squeeze().permute(1,2,0).numpy()
#
# plt.figure()
# plt.imshow(y.astype('uint8'))
# print(y.shape)

num_classes = len(classes)


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


class ResnetFCN(nn.Module):
    def __init__(self, num_classes, backbone='resnet34'):
        super(ResnetFCN, self).__init__()

        if backbone == 'resnet34':
            pretrained_net = torchvision.models.resnet34(pretrained=True)
        elif backbone == 'resnet50':
            pretrained_net = torchvision.models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            pretrained_net = torchvision.models.resnet101(pretrained=True)

        if backbone in ['resnet34']:
            self.scores1 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            self.scores2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.scores3 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)
        elif backbone in ['resnet50', 'resnet101']:
            self.scores1 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            self.scores2 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.scores3 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4])  # 第一段
        self.stage2 = list(pretrained_net.children())[-4]  # 第二段
        self.stage3 = list(pretrained_net.children())[-3]  # 第三段

        del pretrained_net



        self.upsample_x32 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_x16 = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        self.upsample_x8 = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16,
                                              stride=8, padding=4, bias=False)
        
        if args.bilinear:
            self.upsample_x32 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.upsample_x16 = nn.Upsample(scale_factor=2, mode='bilinear')
            self.upsample_x8 = nn.Upsample(scale_factor=8, mode='bilinear')
        else:
                
            self.upsample_x8.weight.data = bilinear_kernel(num_classes, num_classes, 16)  # 使用双线性 kernel
            self.upsample_x16.weight.data = bilinear_kernel(num_classes, num_classes, 4)
            self.upsample_x32.weight.data = bilinear_kernel(num_classes, num_classes, 4)
    
            # self.upsample_x8.require_grad = False
            # self.upsample_x16.require_grad = False
            # self.upsample_x32.require_grad = False



    def forward(self, x):
        x = self.stage1(x)
        x_sampled_x8 = x  # 1/8

        x = self.stage2(x)
        x_sampled_x16 = x  # 1/16

        x = self.stage3(x)
        x_sampled_x32 = x  # 1/32

        s3 = self.scores1(x_sampled_x32)
        s3 = self.upsample_x32(s3)

        s2 = self.scores2(x_sampled_x16)
        s2 = s2 + s3

        s1 = self.scores3(x_sampled_x8)

        s2 = self.upsample_x16(s2)
        s = s1 + s2

        s = self.upsample_x8(s)
        return s


def img_transforms(image, mask, crop_size):
    # image = np.array(image)
    # mask = np.array(mask)
    # image, mask = rand_crop(image, mask, *crop_size)

    #    image = Image.fromarray(image)
    #    mask = Image.fromarray(mask)

    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    # print(i, j, h, w)
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)

    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image)

    mask = torch.from_numpy(np.asarray(mask))

    #    print(mask.shape)

    return image, mask.squeeze().long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bilinear', default=True)
    parser.add_argument('--gpu', default="0")
    parser.add_argument('--model', default='resnet101', choices=['resnet34', 'resnet50', 'resnet101'])
    parser.add_argument('--lr', type=float, default=1e-2)

    args = parser.parse_args()
    import platform

    if platform.system() == 'Windows':
        args.gpu = '0'
        args.num_workers = 0
        args.batch_size = 16

    pprint(vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # criterion = nn.NLLLoss2d()
    criterion = nn.NLLLoss(ignore_index=255)

    ## 2D loss example (used, for example, with image inputs)
    # N, C = 5, 4
    # loss = nn.NLLLoss()
    ## input is of size N x C x height x width
    # data = torch.randn(N, 16, 10, 10)
    # conv = nn.Conv2d(16, C, (3, 3))
    # m = nn.LogSoftmax(dim=1)
    ## each element in target has to have 0 <= value < C
    # target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    # output = loss(m(conv(data)), target)
    # output.backward()

    input_shape = (256, 256)

    voc_train = VOCSegDataset(True, input_shape, img_transforms)
    voc_test = VOCSegDataset(False, input_shape, img_transforms)
    train_data = DataLoader(voc_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_data = DataLoader(voc_test, batch_size=args.batch_size, num_workers=args.num_workers)

    dataloaders = {}
    dataloaders['train'] = train_data
    dataloaders['val'] = valid_data

    model = ResnetFCN(num_classes, args.model)
    model = model.cuda()

    # model = models.segmentation.fcn_resnet101(pretrained=True).eval()
    # model = nn.DataParallel(model)
    # model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0e-4)
    # IOUMetric = IOUMetric(num_classes)

    # 1/0

    for epoch in range(120):
        train_loss = 0
        eval_loss = 0

        start_time = time.time()

        for phase in ['train', 'val']:
            mIOU = RunningConfusionMatrix(labels=[i for i in range(num_classes)])

            for step, data in enumerate(dataloaders[phase]):
                model.train(phase == 'train')
                # model.train()

                im = (data[0].cuda())  # (bs, 3, H, W)
                #            print(im.shape)
                label = (data[1].cuda())
                # print(label.min())

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # out = model(im)['out']
                    out = model(im)

                    out = F.log_softmax(out, dim=1)  # (b, n, h, w)
                    loss = criterion(out, label)

                    label_pred = out.max(dim=1)[1].data.cpu().numpy()
                    label_true = label.data.cpu().numpy()

                    mIOU.update_matrix(ground_truth=label_true.flatten(), prediction=label_pred.flatten())

                if phase == 'train':
                    train_loss += loss.item()
                    # backward

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                elif phase == 'val':
                    eval_loss += loss.item()

                # if step%10 == 0:
                #     print(step, mIOU.compute_current_mean_intersection_over_union())

            if phase == 'train':
                # train_acc, train_acc_cls, _, train_mean_iu, train_fwavacc =  IOUMetric.evaluate()
                # print(train_acc, train_acc_cls, train_mean_iu)
                print(epoch, phase, mIOU.compute_current_mean_intersection_over_union())
            #                1/0
            elif phase == 'val':
                # t = IOUMetric.hist
                # val_acc, val_acc_cls, _, val_mean_iu, val_fwavacc =  IOUMetric.evaluate()
                # print(val_acc, val_acc_cls, val_mean_iu)
                print(epoch, phase, mIOU.compute_current_mean_intersection_over_union())
        #                1/0

        time_elapsed = time.time() - start_time

        #        epoch_str = ('Epoch {} takes {} seconds.\n Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f},\n Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
        #            epoch, time_elapsed, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
        #            eval_loss / len(valid_data), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))

        print('Epoch {} takes {} seconds.'.format(epoch, time_elapsed))
