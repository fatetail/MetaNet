#from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import pretrainedmodels
import torch
#from efficientnet_pytorch import EfficientNet

import torch.nn as nn
import pretrainedmodels
import torch
class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class TwoD_Model(nn.Module):
    def __init__(self, class_nums, **kwargs):
        super(TwoD_Model, self).__init__(**kwargs)
        #self.pretrained_net = EfficientNet.from_pretrained('efficientnet-b5')
        self.pretrained_net = pretrainedmodels.senet154(num_classes=1000, pretrained='imagenet')
        #new_last_linear = nn.Linear(self.pretrained_net.last_linear.in_features, 20)
        #self.pretrained_net.last_linear = new_last_linear
        self.fc1 = nn.Conv2d(17, 2048, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.fc3 = nn.Conv2d(2048, 7, kernel_size=1, padding=0)
        #new_last_linear = nn.Linear(self.pretrained_net._fc.in_features, class_nums)
        #self.pretrained_net._fc = new_last_linear

    def forward(self, x, y):
        x = self.pretrained_net.features(x)

        y = y.view(-1, 17, 1, 1)
        y = self.fc1(y)
        y = self.sigmoid(y)
        x = x*y
        x = self.avg_pool(x)
        x = self.fc3(x)
        x = x.view(x.size(0), -1)
        #print(x.size())
        #y = self.linear1(y)
        # x = torch.cat((x, y), dim=1)
        # x = self.linear2(x)
        return x


