import copy

import torch
from torch import nn
import torch.nn.functional as F

from torchvision.models.resnet import resnet50, Bottleneck

def make_model(args):
    return Base(args)

class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        num_classes = args.num_classes
        feats = args.feats
        resnet = resnet50(pretrained=True)

        #定义主干网络
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )
        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())
        self.p = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        #池化层
        if args.pool == 'max':
            pool2d = nn.MaxPool2d
        elif args.pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()
        self.pool_p = pool2d(kernel_size=(24, 8))

        #1x1卷积层，降维
        self.reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())
        self._init_reduction(self.reduction)

        #全连接层
        self.fc = nn.Linear(feats, num_classes)
        self._init_fc(self.fc)

    @staticmethod
    def _init_reduction(reduction):#初始化降维层
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):#初始化全连接层
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        '''
        :param x:
        :return: list
        :var predict:用来预测的特征向量，根据其形状修改train.py中extract_feature方法中:ff = torch.FloatTensor(inputs.size(0), predict.size[1]).zero_()
        :tips: pytorch tensor的形状[batch_size, channel_number, height, weight]
        '''


        x = self.backbone(x)
        p = self.p(x)
        #print(p.shape)

        zg_p = self.pool_p(p)
        #print(zg_p.shape)

        #fg_p = self.reduction(zg_p)
        #print(fg_p.shape)
        fg_p = self.reduction(zg_p).squeeze(dim=3).squeeze(dim=2)
        #print(zg_p.shape)

        l_p = self.fc(fg_p)
        #print(zg_p.shape)

        predict = torch.cat([fg_p], dim=1)
        #print(predict.shape)

        return predict, l_p #返回一个list，l_p使用softmax loss优化




