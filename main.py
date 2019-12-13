import os
import random
import cv2
import time
import PIL
import torch, torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import argparse
import pprint
_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)
from torch.autograd import Variable

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
print("Pillow Version: ", PIL.PILLOW_VERSION)

#  datasets dir
voc_root="./VOCdevkit/VOC2012/"
print(voc_root)

def read_images(root=voc_root, training=True):
    txt_filename=root+"/ImageSets/Segmentation/"+('train.txt'if training else 'val.txt')
    with open(txt_filename,'r') as f:
        images=f.read().split()
    data=[os.path.join(root,'JPEGImages',i+'.jpg')for i in images]
    label=[os.path.join(root,'SegmentationClass',i+'.png')for i in images]
    return data,label

data,label=read_images()
# print(data,label)


classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']

# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]
#128 =64*2, 192 = 64*3
len(classes), len(colormap)
num_classes = len(classes)

cm2lbl=np.zeros(256**3)-1
for i, cm in enumerate(colormap):
    cm2lbl[(cm[0]*256 + cm[1])*256 + cm[2]] = i

def image2label(im):
    data = np.array(im,dtype='int32')
    idx = (data[:,:,0]*256 + data[:,:,1])*256 + data[:,:,2]
#    print(idx.shape)
    return np.array(cm2lbl[idx], dtype='int64')

#label1 = Image.open('./VOCdevkit/VOC2012/SegmentationClass/2007_003349.png')
#label1[label1 == 255] = 0

label1 = np.asarray(Image.open('./VOCdevkit/VOC2012/SegmentationClass/2007_003349.png')).copy()
label1[label1 == 255] = 0
label1 = Image.fromarray(label1.astype(np.uint8))

label1 = np.asarray(Image.open('./VOCdevkit/VOC2012/SegmentationClass/2007_000033.png')).copy()
label1[label1 == 255] = 0

#label1 = np.asarray(label1)
#label1 = image2label(label1)
#
#
#label2 = cv2.imread('./VOCdevkit/VOC2012/SegmentationClass/2007_000033.png', cv2.COLOR_BGR2RGB)
#label2 = cv2.imread('./VOCdevkit/VOC2012/SegmentationClass/2007_000033.png')
#label2 = image2label(label2)
#1/0
def img_transforms(image, mask, crop_size):
    # image = np.array(image)
    # mask = np.array(mask)
    # image, mask = rand_crop(image, mask, *crop_size)

#    image = Image.fromarray(image)
#    mask = Image.fromarray(mask)
    
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size = crop_size)
    # print(i, j, h, w)
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)


    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image)
    
#    mask = transforms.Compose([
#        transforms.ToTensor()
#    ])(mask)
    mask = torch.from_numpy(np.asarray(mask))
    
#    print(mask.shape)
    
    return image, mask.squeeze().long()


class VOCSegDataset(Dataset):
    def __init__(self, training, crop_size, transforms):
        self.training = training
        self.crop_size=crop_size
        self.transforms=transforms
        data_list, label_list = read_images(training=training)
        self.data_list=self._filter(data_list)
        self.label_list=self._filter(label_list)
        
        print('Reading '+str(len(self.data_list))+' images')
        
        self.images = {}
        self.labels = {}
        
        for idx in range(len(self.data_list)):
            self.images[idx] = cv2.imread(self.data_list[idx], cv2.COLOR_BGR2RGB)
            
            self.images[idx] = Image.fromarray(self.images[idx])

            label_mask = np.asarray(Image.open(self.label_list[idx])).copy()
            label_mask[ label_mask==255 ] = 0
#            print(label_mask.max())
            label_mask = Image.fromarray(label_mask)
            self.labels[idx] = label_mask
            
    def _filter(self,images):
        return [im for im in images if(Image.open(im).size[1] >= self.crop_size[0]-200 and
                                      Image.open(im).size[0] >= self.crop_size[1])-200]
    def __getitem__(self, idx):
#        img = self.data_list[idx]
#        label = self.label_list[idx]
#        
#        img = cv2.imread(img, cv2.COLOR_BGR2RGB)
#        label = cv2.imread(label, cv2.COLOR_BGR2RGB)
#

        img = self.images[idx]
        label = self.labels[idx]
        
#        img = Image.open(img).convert("RGB")
#        label = Image.open(label).convert("RGB")
        
#        img = Image.open(img)
#        label = Image.open(label)    
#        print(label)
        # print(img)

        # 1/0

        # print(img.shape)

        img, label = self.transforms(img, label, self.crop_size)
#        print(img.shape) # (3, H, W)
        return img, label
    
    def __len__(self):
        return len(self.data_list)

    




def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size+1) // 2
    if kernel_size % 2 == 1:
        center = factor-1
    else:
        center = factor-0.5
    og = np.ogrid[:kernel_size,:kernel_size]
    filt = (1-abs(og[0]-center)/factor)*(1-abs(og[1]-center)/factor)
    weight = np.zeros((in_channels,out_channels,kernel_size,kernel_size),dtype='float32')
    weight[range(in_channels),range(out_channels),:,:]=filt
    return torch.from_numpy(weight)


image = cv2.imread('./VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg', cv2.COLOR_BGR2RGB)

#x=Image.open('./VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
#x=np.array(x)
#
#img=cv2.imread('./VOCdevkit/VOC2012/JPEGImages/2007_005210.jpg')
#img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#
#im_tfs = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#])
#im_tfs(img)
#
#plt.figure()
#plt.imshow(x)
#plt.show()
#
#
#x=torch.from_numpy(x.astype('float32')).permute(2,0,1).unsqueeze(0)
#conv_trans=nn.ConvTranspose2d(3,3,4,2,1)
#conv_trans.weight.data=bilinear_kernel(3,3,4)
#
#y=conv_trans(Variable(x)).data.squeeze().permute(1,2,0).numpy()
#
#plt.figure()
#plt.imshow(y.astype('uint8'))
#print(y.shape)


num_classes = len(classes)

#1/0

class fcn(nn.Module):
    def __init__(self, num_classes):
        super(fcn, self).__init__()
        pretrained_net = torchvision.models.resnet34(pretrained=True)

        self.stage1 = nn.Sequential(*list(pretrained_net.children())[:-4]) # 第一段
        self.stage2 = list(pretrained_net.children())[-4] # 第二段
        self.stage3 = list(pretrained_net.children())[-3] # 第三段
        
        self.scores1 = nn.Conv2d(in_channels = 512, out_channels = num_classes, kernel_size = 1)
        self.scores2 = nn.Conv2d(in_channels = 256, out_channels = num_classes, kernel_size = 1)
        self.scores3 = nn.Conv2d(in_channels = 128, out_channels = num_classes, kernel_size = 1)
        
        # self.upsample_8x = nn.ConvTranspose2d(in_channels=num_classes, out_channels=num_classes, kernel_size=16, stride=8, padding=4, bias=False)
        # self.upsample_8x.weight.data = bilinear_kernel(num_classes, num_classes, 16) # 使用双线性 kernel
        #
        # self.upsample_4x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        # self.upsample_4x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel
        #
        # self.upsample_2x = nn.ConvTranspose2d(num_classes, num_classes, 4, 2, 1, bias=False)
        # self.upsample_2x.weight.data = bilinear_kernel(num_classes, num_classes, 4) # 使用双线性 kernel

        self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_x8 = nn.Upsample(scale_factor=8, mode='bilinear')

        
    def forward(self, x):
        x = self.stage1(x)
        s1 = x # 1/8
        # print(s1.shape)
        
        x = self.stage2(x)
        s2 = x # 1/16
        # print(s2.shape)
        
        x = self.stage3(x)
        s3 = x # 1/32
        # print(s3.shape)
        
        s3 = self.scores1(s3)
        # s3 = self.upsample_2x(s3)
        s3 = self.upsample_x2(s3)

        s2 = self.scores2(s2)
        s2 = s2 + s3
        
        s1 = self.scores3(s1)
        
        # s2 = self.upsample_4x(s2)
        s2 = self.upsample_x2(s2)
        s = s1 + s2

        # s = self.upsample_8x(s2)
        s = self.upsample_x8(s2)
        return s
    
def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
    return hist

def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc

# from sklearn.metrics import confusion_matrix

class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes)) # confusion matrix
        
    def reset(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes)) # confusion matrix
        
    def _fast_hist(self, label_pred, label_true):
#        mask = (label_true >= 0) & (label_true < self.num_classes)
#        hist = np.bincount(
#                self.num_classes * label_true[mask].astype(int) + label_pred[mask].astype(int), 
#                minlength = self.num_classes ** 2).reshape([self.num_classes, self.num_classes], order='C')
        temp = (self.num_classes) * label_true.astype(int) + label_pred.astype(int)
##        print(temp.shape)
#        print(label_true.astype(int).max())
#        print((self.num_classes), temp.min(), temp.max())
#        1/0
        hist = np.bincount(temp.flatten(), minlength = self.num_classes ** 2).reshape([self.num_classes, self.num_classes], order='C')
##        print(hist.shape)
#
#        hist= hist.reshape([self.num_classes, self.num_classes], order='C')
#        hist = confusion_matrix(y_true=label_true, y_pred=label_pred)
#        print(hist.shape)
#        print(np.diag(self.hist))
#        print(set(label_true))
        return hist
    
        
    def add_batch(self, predictions, gts):
#        for lp, lt in zip(predictions, gts):
#        print(predictions.flatten().shape, gts.flatten().shape)
        self.hist += self._fast_hist(predictions.flatten(), gts.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weight_normalization', default = False)
    parser.add_argument('--gpu', default="1")


    args = parser.parse_args()
    import platform
    if platform.system() == 'Windows':
        args.gpu = '0'
        args.num_workers = 0
        args.batch_size = 32

    pprint(vars(args))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    #criterion = nn.NLLLoss2d()
    criterion = nn.NLLLoss()

    ## 2D loss example (used, for example, with image inputs)
    #N, C = 5, 4
    #loss = nn.NLLLoss()
    ## input is of size N x C x height x width
    #data = torch.randn(N, 16, 10, 10)
    #conv = nn.Conv2d(16, C, (3, 3))
    #m = nn.LogSoftmax(dim=1)
    ## each element in target has to have 0 <= value < C
    #target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
    #output = loss(m(conv(data)), target)
    #output.backward()

    input_shape = (320, 480)

    voc_train = VOCSegDataset(True, input_shape, img_transforms)
    voc_test = VOCSegDataset(False, input_shape, img_transforms)
    train_data = DataLoader(voc_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_data = DataLoader(voc_test, batch_size=args.batch_size, num_workers=args.num_workers)

    dataloaders = {}
    dataloaders['train'] = train_data
    dataloaders['val'] = valid_data

    net=fcn(num_classes)
    net=net.cuda()

    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
    IOUMetric = IOUMetric(num_classes)
    fcn = models.segmentation.fcn_resnet101(pretrained=False).eval()


    for epoch in range(120):

        start_time = time.time()
        

        train_loss = 0
        train_acc = 0
        train_acc_cls = 0
        train_mean_iu = 0
        train_fwavacc = 0

        eval_loss = 0
        eval_acc = 0
        eval_acc_cls = 0
        eval_mean_iu = 0
        eval_fwavacc = 0

        prev_time = time.time()

        for phase in ['train', 'val']:
            
            IOUMetric.reset(num_classes)
            
            for data in dataloaders[phase]:
                net.train(phase == 'train')



                im = (data[0].cuda()) # (bs, 3, H, W)
    #            print(im.shape)
                label = (data[1].cuda())
        #         print(label.shape)
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    out = net(im)
            #         print(out.shape)
                    out = F.log_softmax(out, dim=1) # (b, n, h, w)
                    loss = criterion(out, label)

                    label_pred = out.max(dim=1)[1].data.cpu().numpy()
                    label_true = label.data.cpu().numpy()
#                    print(label_true.max())
                    
                    IOUMetric.add_batch(predictions=label_pred.flatten(), gts=label_true.flatten())

                if phase == 'train':
                    train_loss += loss.item()
                    # backward

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data
                    

#                    for lbt, lbp in zip(label_true, label_pred):
#                        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
#                        print(mean_iu)
#                        train_acc += acc
#                        train_acc_cls += acc_cls
#                        train_mean_iu += mean_iu
#                        train_fwavacc += fwavacc

                elif phase == 'val':

                    eval_loss += loss.item()

#                    for lbt, lbp in zip(label_true, label_pred):
#                        acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, num_classes)
#                        eval_acc += acc
#                        eval_acc_cls += acc_cls
#                        eval_mean_iu += mean_iu
#                        eval_fwavacc += fwavacc
                
            if phase == 'train':
                train_acc, train_acc_cls, _, train_mean_iu, train_fwavacc =  IOUMetric.evaluate()
                print(train_acc, train_acc_cls, train_mean_iu)
#                1/0
            elif phase == 'val':
                t = IOUMetric.hist
                val_acc, val_acc_cls, _, val_mean_iu, val_fwavacc =  IOUMetric.evaluate()
                print(val_acc, val_acc_cls, val_mean_iu)
#                1/0


        time_elapsed = time.time() - start_time

#        epoch_str = ('Epoch {} takes {} seconds.\n Train Loss: {:.5f}, Train Acc: {:.5f}, Train Mean IU: {:.5f},\n Valid Loss: {:.5f}, Valid Acc: {:.5f}, Valid Mean IU: {:.5f} '.format(
#            epoch, time_elapsed, train_loss / len(train_data), train_acc / len(voc_train), train_mean_iu / len(voc_train),
#            eval_loss / len(valid_data), eval_acc / len(voc_test), eval_mean_iu / len(voc_test)))

        print('Epoch {} takes {} seconds.\n '.format(epoch, time_elapsed))
        
#        print(epoch_str)