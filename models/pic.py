import torch
import torch.nn as nn
import numpy as np

import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
# from attention import Seq_Transformer
import torch.nn.functional as F
# import warnings
# warnings.filterwarnings('ignore')


class PIC_no(nn.Module):

    def __init__(self, configs, device):
        super(PIC_no, self).__init__()
        self.num_channels = configs.final_out_channels  # configs.final_out_channels=128
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(configs.features_len)])  # configs.TC.hidden_dim=100, self.num_channels=128
        self.lsoftmax = nn.LogSoftmax(1)
        self.device = device

        self.projection_head_pic = nn.Sequential(

            nn.Conv1d(128, 1, kernel_size=1, stride=1, bias=False, padding=0),  # (447-7+4)/1+1=84
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            # nn.Linear(122, 100),  # Linear(100,64)
            # nn.Linear(117, 100),  # Linear(100,64)

            # nn.Linear(157, 100),
            # nn.Linear(162, 100),

            # nn.Linear(138, 100),
            # nn.Linear(143, 100),

            # nn.Linear(127, 100),
            # nn.Linear(132, 100),

            # nn.Linear(147, 100),
            nn.Linear(152, 100),

            # nn.Linear(84, 100),
            # nn.Linear(89, 100),

            # nn.Linear(9, 100),
            # nn.Linear(14, 100),

            # nn.Linear(24, 100),
            # nn.Linear(5, 100),

            # nn.Linear(30, 100),  # circena2vec
            # nn.Linear(35, 100),

            nn.ReLU(),
            # nn.Linear(100, 99),  # Linear(64,32)
            nn.Linear(100, 20),  # Linear(64,32)

            nn.ReLU(),

            nn.Linear(20, 2),
            # nn.Linear(20, 2),
            nn.Dropout(configs.dropout)
        )

    def forward(self, output): #, features_aug4, features5, features6
        c_t = output

        yt = self.projection_head_pic(c_t).squeeze(dim=1)

        return self.lsoftmax(yt)


class PIC_RBP(nn.Module):
    def __init__(self, configs, device):
        super(PIC_RBP, self).__init__()
        self.num_channels = configs.final_out_channels  # configs.final_out_channels=128
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in
                                 range(configs.features_len)])  # configs.TC.hidden_dim=100, self.num_channels=128
        self.lsoftmax = nn.LogSoftmax(1)
        self.device = device

        self.projection_head_pic = nn.Sequential(

            nn.Conv1d(128, 1, kernel_size=1, stride=1, bias=False, padding=0),  # (447-7+4)/1+1=84
            nn.ReLU(),
            nn.Dropout(configs.dropout),
            # nn.Linear(122, 100),  # Linear(100,64)
            # nn.Linear(117, 100),  # Linear(100,64)

            # nn.Linear(157, 100),
            # nn.Linear(162, 100),

            # nn.Linear(138, 100),
            # nn.Linear(143, 100),

            # nn.Linear(127, 100),
            # nn.Linear(132, 100),

            # nn.Linear(147, 100),
            nn.Linear(152, 100),

            # nn.Linear(84, 100),
            # nn.Linear(89, 100),

            # nn.Linear(9, 100),  # EIIP - ANF - CCN
            # nn.Linear(14, 100),

            # nn.Linear(24, 100),
            # nn.Linear(5, 100),

            # nn.Linear(30, 100),  # circena2vec
            # nn.Linear(35, 100),

            nn.ReLU(),
            # nn.Linear(100, 99),  # Linear(64,32)
            nn.Linear(100, 20),  # Linear(64,32)

            nn.ReLU(),

            nn.Linear(20, 2),
            # nn.Linear(20, 2),
            nn.Dropout(configs.dropout)
        )

    # def forward(self, features_aug1,RBP_aug, RBP_aug1, RBP_aug2, output,labels): #一个特征
    def forward(self, output):  # , features_aug4, features5, features6

        c_t = output

        yt = self.projection_head_pic(c_t).squeeze(dim=1)

        return self.lsoftmax(yt)
