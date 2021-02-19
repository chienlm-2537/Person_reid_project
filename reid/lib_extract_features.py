
import os

import scipy.io
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import cv2
from model import ft_net

name = 'ft_ResNet50'
which_epoch = 'last'

def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%which_epoch)
    network.load_state_dict(torch.load(save_path))
    return network


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature_image(model, image):
    features = torch.FloatTensor()
    n = 1
    ff = torch.FloatTensor(n, 512).zero_()
    img = fliplr(image)
    input_img = Variable(img)
    scale = 1
    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
    outputs = model(input_img)
    ff += outputs
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))

    features = torch.cat((features,ff.data.cpu()), 0)   
    return features

def model_init():
    nclasses = 751
    stride = 2
    model_structure = ft_net(nclasses, stride = stride)
    model = load_network(model_structure)
    model.classifier.classifier = nn.Sequential()
    model = model.eval()
    return model


tran = transforms.ToTensor()

def load_and_save_feature(image, model, frame_num):
    image = tran(image)
    image.unsqueeze_(0)
    ff = extract_feature_image(model, image)
    results = {"feature":ff.numpy()}
    scipy.io.savemat(str(frame_num)+".mat", results)
    

if __name__ == "__main__":
    model = model_init()
    image = cv2.imread("/home/le.minh.chien/Desktop/DATN/Person_reid_project/246.png")
    load_and_save_feature(image, model, 100)

    print("Done")
