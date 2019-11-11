import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

onset_net_cfg = {
    'cqt_pad_3': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (7, 2), 'pool2': (3, 1), 'fc1': 1050},
    'cqt_pad_4': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 1050},
    'cqt_pad_5': {'conv1': (25, 3), 'pool1': (3, 3), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 1050},
    'cqt_pad_7': {'conv1': (25, 7), 'pool1': (3, 3), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 1050},

    'fft_pad_3': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (7, 2), 'pool2': (3, 1), 'fc1': 2184},
    'fft_pad_4': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 2184},
    'fft_pad_5': {'conv1': (25, 3), 'pool1': (3, 3), 'conv2': (7, 3), 'pool2': (3, 1), 'fc1': 2184}}

note_net_cfg = {
    'cqt_pad_3': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (9, 2), 'pool2': (3, 1), 'fc1': 1536, 'fc2': 1024,
                  'fc3': 512},
    'cqt_pad_4': {'conv1': (25, 3), 'pool1': (3, 2), 'conv2': (9, 3), 'pool2': (3, 1), 'fc1': 1536, 'fc2': 1024,
                  'fc3': 512},
    'cqt_pad_5': {'conv1': (25, 3), 'pool1': (3, 3), 'conv2': (9, 3), 'pool2': (3, 1), 'fc1': 1536, 'fc2': 1024,
                  'fc3': 512},
    'cqt_pad_7': {'conv1': (25, 7), 'pool1': (3, 3), 'conv2': (9, 3), 'pool2': (3, 1), 'fc1': 1536, 'fc2': 1024,
                  'fc3': 512}
}


class onsetnet(nn.Module):
    """docstring for onsetnet"""

    def __init__(self,
                 pad_length=4,
                 spec_style='cqt',
                 dual_channel=False):
        super(onsetnet, self).__init__()
        nchannel = 2 if dual_channel else 1  # 是否双通道
        self.config = onset_net_cfg['{}_pad_{}'.format(spec_style, pad_length)]  # 选择卷积网络

        self.features = nn.Sequential(
            nn.Conv2d(nchannel, 21, kernel_size=self.config['conv1']),
            nn.BatchNorm2d(21),  # 归一化权重
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool1'], stride=self.config['pool1']),
            nn.Conv2d(21, 42, kernel_size=self.config['conv2']),
            nn.BatchNorm2d(42),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool2'], stride=self.config['pool2'])
        )
        self.fc1 = nn.Linear(self.config['fc1'], 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        #        print(x.size())

        x = x.view(-1, self.config['fc1'])
        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x) #py3.7中改为torch.sigmoid


class onsetconv3(nn.Module):
    """docstring for onsetnet"""

    def __init__(self,
                 pad_length=4,
                 spec_style='cqt',
                 dual_channel=False):
        super(onsetconv3, self).__init__()
        nchannel = 2 if dual_channel else 1
        self.conv1 = nn.Conv2d(
            nchannel, 32, kernel_size=(16, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(7, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(7, 1))
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(1088, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=(3, 2), stride=(3, 2))

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=(2, 2))

        x = x.view(-1, 1088)

        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.sigmoid(x)


class onsetconv1(nn.Module):
    """docstring for onsetnet"""

    def __init__(self,
                 pad_length=4,
                 spec_style='cqt',
                 dual_channel=False):
        super(onsetconv1, self).__init__()
        nchannel = 2 if dual_channel else 1

        self.conv1 = nn.Conv2d(
            nchannel, 32, kernel_size=(25, 3))
        self.bn1 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(2560, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.max_pool2d(x, kernel_size=(6, 3), stride=(6, 3))

        x = x.view(-1, 2560)

        x = F.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x)
        x = self.fc2(x)
        return F.sigmoid(x)


class notesnet(nn.Module):
    """docstring for notenet"""

    def __init__(self,
                 pad_length=4,
                 dual_channel=False):
        super(notesnet, self).__init__()
        nchannel = 2 if dual_channel else 1
        self.config = note_net_cfg['cqt_pad_' + str(pad_length)]

        self.features = nn.Sequential(
            nn.Conv2d(nchannel, 64, kernel_size=self.config['conv1']),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool1'],
                         stride=self.config['pool1']),
            nn.Conv2d(64, 64, kernel_size=self.config['conv2']),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.config['pool2'],
                         stride=self.config['pool2'])
        )

        self.fc1 = nn.Linear(self.config['fc1'], self.config['fc2'])
        self.fc2 = nn.Linear(self.config['fc2'], self.config['fc3'])
        self.fc3 = nn.Linear(self.config['fc3'], 88)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.config['fc1'])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x)
        x = self.fc3(x)
        return F.sigmoid(x)


def factory_net(net_style,
                pad_length,
                spec_style,
                dual_channel):
    name = net_style + 'net'
    return eval(name)(pad_length, spec_style, dual_channel)


if __name__ == '__main__':
    inputs_ = Variable(torch.randn(19, 1, 267, 9))
    net = onsetnet()

    output = net(inputs_)
    print(output.size(),output.data)