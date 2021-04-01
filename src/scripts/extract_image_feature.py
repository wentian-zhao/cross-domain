import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck, model_zoo, model_urls
from tqdm import tqdm
import skimage
import skimage.io
import numpy as np
import h5py
import scipy
import scipy.sparse


sparse_format = 'csr'


class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        fc = x.mean(3).mean(2).squeeze()
        att = F.adaptive_avg_pool2d(x, [att_size, att_size]).squeeze().permute(1, 2, 0)

        return fc, att


class ResNet(torchvision.models.resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__(block, layers, num_classes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        for i in range(2, 5):
            getattr(self, 'layer%d'%i)[0].conv1.stride = (2,2)
            getattr(self, 'layer%d'%i)[0].conv2.stride = (1,1)


def resnet101(pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def extract_image_feature(image_file_list, feature_files, sparse_feat_folder):
    print('total {} images'.format(len(image_file_list)))
    feat_file_avg = feature_files[0]
    feat_file_att = feature_files[1]
    for f in [feat_file_avg, feat_file_att]:
        if not os.path.exists(os.path.split(f)[0]):
            os.makedirs(os.path.split(f)[0])
    if not os.path.exists(sparse_feat_folder):
        os.makedirs(sparse_feat_folder)
    print('saving feature to: {}, {}'.format(feat_file_avg, feat_file_att))
    f_avg = h5py.File(feat_file_avg, 'a', swmr=True)
    f_att = h5py.File(feat_file_att, 'a', swmr=True)

    att_size = 14
    print('att_size = {}'.format(att_size))

    net = resnet101()
    net.load_state_dict(torch.load('../pretrained_model/resnet101.pth'))
    net = myResnet(net)
    net.cuda()
    net.eval()

    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    sparse_method = getattr(scipy.sparse, '{}_matrix'.format(sparse_format))

    for i, image_file in tqdm(enumerate(image_file_list), total=len(image_file_list), ncols=64):
        image_filename = os.path.split(image_file)[-1]
        # if image_filename in f_avg.keys() and image_filename in f_att.keys():
        #     print('continue')
        #     continue
        if os.path.exists(os.path.join(sparse_feat_folder, '{}.npz'.format(image_filename))):
            continue

        I = skimage.io.imread(image_file)
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose(2, 0, 1)).cuda()
        I = preprocess(I)

        with torch.no_grad():
            tmp_fc, tmp_att = net(I, att_size)
        
        tmp_fc = tmp_fc.detach().cpu().numpy()
        tmp_att = tmp_att.detach().cpu().numpy()

        _tmp_att = tmp_att.reshape(-1, tmp_att.shape[-1])
        b = sparse_method(_tmp_att)
        scipy.sparse.save_npz(os.path.join(sparse_feat_folder, '{}.npz'.format(image_filename)), b, compressed=False)

        # if image_filename not in f_avg.keys():
        #     dataset_fc = f_avg.create_dataset(image_filename, tmp_fc.shape, dtype=np.float32)
        #     dataset_fc[...] = tmp_fc
        # if image_filename not in f_att.keys():
        #     dataset_att = f_att.create_dataset(image_filename, tmp_att.shape, dtype=np.float32)
        #     dataset_att[...] = tmp_att


# mscoco
coco_image_path = ['/media/sdb1/caption_datasets/mscoco/train2014',
                   '/media/sdb1/caption_datasets/mscoco/val2014']
all_image = []
for p in coco_image_path:
    imgs = os.listdir(p)
    for i in imgs:
        all_image.append(os.path.join(p, i))
extract_image_feature(all_image, ['/media/sda3/image_feat/mscoco/coco_fc.h5',
                                  '/media/sda3/image_feat/mscoco/coco_att.h5'],
                                  '/media/sda3/image_feat/mscoco/coco_att_sparse')

# flickr30k
flickr30k_image_path = '/media/sdb1/caption_datasets/flickr30k/flickr30k_images/flickr30k_images'
all_image = [os.path.join(flickr30k_image_path, p) for p in os.listdir(flickr30k_image_path)]
extract_image_feature(all_image, ['/media/sda3/image_feat/flickr30k/flickr30k_fc.h5',
                                  '/media/sda3/image_feat/flickr30k/flickr30k_att.h5'],
                                  '/media/sda3/image_feat/flickr30k/flickr30k_att_sparse')

# # oxford102
# oxford102_image_path = '/media/sdb1/caption_datasets/oxford102/jpg'
# all_image = [os.path.join(oxford102_image_path, p) for p in os.listdir(oxford102_image_path)]
# extract_image_feature(all_image, ['/media/sda3/image_feat/oxford102/oxford102_fc.h5',
#                                   '/media/sda3/image_feat/oxford102/oxford102_att.h5'],
#                                   '/media/sda3/image_feat/oxford102/oxford102_att_sparse')

# # cub200
# all_image = []
# cub_image_path = r'/media/sdb2/work/caption_dataset/CUB_200_2011/images'
# for root, dirs, files in os.walk(cub_image_path):
#     for file in files:
#         if file.endswith('.jpg'):
#             all_image.append(os.path.join(root, file))
# extract_image_feature(all_image, ['/media/sda1/caption_features/cub200/cub200_fc.h5',
#                                   '/media/sda1/caption_features/cub200/cub200_att.h5'])
