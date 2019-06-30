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

        # 定义主干网络
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

        # 池化层
        if args.pool == 'max':
            pool2d = nn.MaxPool2d
        elif args.pool == 'avg':
            pool2d = nn.AvgPool2d
        else:
            raise Exception()
        self.pool_zp = pool2d(kernel_size=(4, 8))

        # 1x1卷积层，降维
        reduction = nn.Sequential(nn.Conv2d(2048, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU())
        self._init_reduction(reduction)

        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)
        self.reduction_6 = copy.deepcopy(reduction)

        # 全连接层
        self.fc_1 = nn.Linear(feats, num_classes)
        self.fc_2 = nn.Linear(feats, num_classes)
        self.fc_3 = nn.Linear(feats, num_classes)
        self.fc_4 = nn.Linear(feats, num_classes)
        self.fc_5 = nn.Linear(feats, num_classes)
        self.fc_6 = nn.Linear(feats, num_classes)
        self._init_fc(self.fc_1)
        self._init_fc(self.fc_2)
        self._init_fc(self.fc_3)
        self._init_fc(self.fc_4)
        self._init_fc(self.fc_5)
        self._init_fc(self.fc_6)

    @staticmethod
    def _init_reduction(reduction):  # 初始化降维层
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):  # 初始化全连接层
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        nn.init.constant_(fc.bias, 0.)

    def forward(self, x):
        '''
        :param x:
        :return: list
        :tips: pytorch tensor的形状[batch_size, channel_number, height, weight]
        '''

        x = self.backbone(x)
        p = self.p(x)

        zp = self.pool_zp(p)
        z1_p = zp[:, :, 0:1, :]
        z2_p = zp[:, :, 1:2, :]
        z3_p = zp[:, :, 2:3, :]
        z4_p = zp[:, :, 3:4, :]
        z5_p = zp[:, :, 4:5, :]
        z6_p = zp[:, :, 5:6, :]

        f1_p = self.reduction_1(z1_p).squeeze(dim=3).squeeze(dim=2)
        f2_p = self.reduction_2(z2_p).squeeze(dim=3).squeeze(dim=2)
        f3_p = self.reduction_3(z3_p).squeeze(dim=3).squeeze(dim=2)
        f4_p = self.reduction_4(z4_p).squeeze(dim=3).squeeze(dim=2)
        f5_p = self.reduction_5(z5_p).squeeze(dim=3).squeeze(dim=2)
        f6_p = self.reduction_6(z6_p).squeeze(dim=3).squeeze(dim=2)

        l1_p = self.fc_1(f1_p)
        l2_p = self.fc_2(f2_p)
        l3_p = self.fc_3(f3_p)
        l4_p = self.fc_4(f4_p)
        l5_p = self.fc_5(f5_p)
        l6_p = self.fc_6(f6_p)

        predict = torch.cat([f1_p, f2_p, f3_p, f4_p, f5_p, f6_p], dim=1)  # 注意trainer.py中extract_feature方法的修改
                                                                          #ff = torch.FloatTensor(inputs.size(0), 1536).zero_()

        return predict, l1_p, l2_p, l3_p, l4_p, l5_p, l6_p  # 返回一个list，ln_p使用softmax loss优化




