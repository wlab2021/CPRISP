# from utils.resnet import *
from math import log
import torch
import torch.nn as nn
from conv_layer import *
class DPCNN(nn.Module):

    def __init__(self, filter_num, number_of_layers):
        super(DPCNN, self).__init__()

        self.kernel_size_list = [1+x*2 for x in range(number_of_layers)]
        self.kernel_size_list = [5, 5, 5, 5, 5, 5]
        self.dilation_list = [1, 1, 1, 1, 1, 1]
        self.conv = Conv1d(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1, same_padding=False)
        self.conv1 = Conv1d(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1, same_padding=False)
        self.pooling = nn.MaxPool1d(kernel_size=(3, ), stride=2)
        self.padding_conv = nn.ConstantPad1d(((self.kernel_size_list[0]-1)//2), 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)

        self.DPCNNblocklist = nn.ModuleList(
            [DPCNNblock(filter_num, kernel_size=self.kernel_size_list[i],
                        dilation=self.dilation_list[i]) for i in range(len(self.kernel_size_list))]
        )
        self.classifier = nn.Linear(filter_num, 1)

    def forward(self, x):

        x = self.padding_conv(x)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        i = 0
        while x.size()[-1] > 2:
            x = self.DPCNNblocklist[i](x)
            i += 1

        x = x.squeeze(-1).squeeze(-1)

        logits = self.classifier(x)

        return logits

class multiscale(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(multiscale, self).__init__()

        self.conv0 = Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False)

        self.conv1 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False, bn=False),
            Conv1d(out_channel, out_channel, kernel_size=(3,), same_padding=True),
        )

        self.conv2 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(5,), same_padding=True),
        )

        self.conv3 = nn.Sequential(
            Conv1d(in_channel, out_channel, kernel_size=(1,), same_padding=False),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True),
            Conv1d(out_channel, out_channel, kernel_size=(7,), same_padding=True)
        )

    def forward(self, x):

        x0 = self.conv0(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = torch.cat([x0, x1, x2, x3], dim=1)
        return x4 + x

class HDRNet(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        base_channel = 8
        number_of_layers = int(log(101-k+1, 2))

        # self.conv0 = Conv1d(101, 128, kernel_size=(k,), stride=1, same_padding=False)
        self.conv0 = Conv1d(101, 128, kernel_size=(1,), stride=1)

        self.conv1 = Conv1d(512, 128, kernel_size=(1,), stride=1)

        self.multiscale_str = multiscale(128, 32)
        self.multiscale_bert = multiscale(128, 32)

    # def forward(self, data1, data2,data3,data4,data5):
    def forward(self, data1, data2,data3,data4):
    # def forward(self, data1):

        x0 = data1
        x1 = data2
        x2 = data3
        x3 = data4
        # x4 = data5

        x0 = self.multiscale_bert(x0)
        x1 = self.multiscale_bert(x1)
        x2 = self.multiscale_bert(x2)
        x3 = self.multiscale_bert(x3)
        # x4 = self.multiscale_bert(x4)

        # x = torch.cat([x0, x1,x2,x3,x4], dim=-1)
        x = torch.cat([x0, x1,x2,x3], dim=-1)
        # x = torch.cat([x0], dim=-1)

        return x
