import torch
import torch.nn as nn
import numpy as np

import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import torch.nn.functional as F
# import warnings
# warnings.filterwarnings('ignore')


class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels  # configs.final_out_channels=128
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(configs.features_len)])  # configs.TC.hidden_dim=100, self.num_channels=128
        self.lsoftmax = nn.LogSoftmax(1)
        self.device = device

        self.projection_head = nn.Sequential(

            nn.Conv1d(128, 1, kernel_size=1, stride=1, bias=False, padding=0),  # (447-7+4)/1+1=84
            nn.ReLU(),
            nn.Dropout(configs.dropout),


            # nn.Linear(147, 100),
            nn.Linear(152, 100),

            nn.ReLU(),
            nn.Linear(100, 20),  # Linear(64,32)

            nn.ReLU(),

            # nn.Linear(20, 4),
            nn.Linear(20, 2),
            nn.Dropout(configs.dropout)
        )

    def forward(self, features_aug1, features_aug2, features_aug3, features_aug4,RBP_aug, RBP_aug1, RBP_aug2, output,labels): #, features_aug4, features5, features6

        c_t = output

        a = features_aug1.size(-1) + features_aug2.size(-1) +features_aug3.size(-1) +features_aug4.size(-1)

        result = torch.zeros(64, 128, a + 5)

        zero_counter = 0
        one_counter = 0

        for idx, label in enumerate(labels):

            if label == 1:

                one_counter +=1
                result[idx, :, :a] = c_t[idx, :, :]
                result[idx, :, a:] = RBP_aug[0, :, :]
            else:
                result[idx, :, :a] = c_t[idx, :, :]
                if zero_counter % 2 == 0:
                    result[idx, :, a:] = RBP_aug1[0, :, :]
                else:
                    result[idx, :, a:] = RBP_aug2[0, :, :]

                zero_counter += 1

        result = result.to(self.device)

        # yt = self.projection_head(c_t).squeeze(dim=1)

        yt = self.projection_head(result).squeeze(dim=1)

        return self.lsoftmax(yt)

