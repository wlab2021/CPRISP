import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
# import sys
# sys.path.append('../data_preprocessing/CRBP')
from get_data_view import get_data
import os
import numpy as np
# from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset):
        super(Load_Dataset, self).__init__()
        # self.training_mode = training_mode

        X_data1 = dataset["samples1"]
        X_data2 = dataset["samples2"]
        X_data3 = dataset["samples3"]
        X_data4 = dataset["samples4"]
        X_data5 = dataset["samples5"]
        # X_data6 = dataset["samples6"]
        y_data = dataset["labels"]
        idxs = dataset["idx"]

        # else:
        self.x_data1 = X_data1
        self.x_data2 = X_data2
        self.x_data3 = X_data3
        self.x_data4 = X_data4
        self.x_data5 = X_data5
        # self.x_data6 = X_data6
        self.y_data = y_data.long()
        self.y_idxs = idxs.long()
        #     self.x_data = X_train
        #     self.y_data = y_train

        self.len = X_data1.shape[0]
        # if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            # self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        # if self.training_mode == "self_supervised":
        return self.x_data1[index], self.x_data2[index], self.x_data3[index], self.x_data4[index], self.x_data5[index], self.y_data[index], self.y_idxs[index]
        # return self.x_data1[index], self.x_data2[index], self.x_data3[index], self.x_data4[index],self.y_data[index] #, self.x_data4[index]
        # return self.x_data1[index], self.x_data2[index], self.x_data3[index],self.y_data[index] #, self.x_data4[index]

        # else:
        #     return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(protein, configs):
    try:
        train_dataset = torch.load("data/{}_train.pt".format(protein))
        # # valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
        test_dataset = torch.load("data/{}_test.pt".format(protein))
    except:
        print("{} dataset loading error, Regenerate!".format(protein))
        train_dataset, test_dataset = get_data(protein)


    train_dataset = Load_Dataset(train_dataset)
    # valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=True,
                                               num_workers=0)
    # valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=configs.batch_size,
    #                                            shuffle=False, drop_last=configs.drop_last,
    #                                            num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=True, drop_last=True,
                                              num_workers=0)

    return train_loader, test_loader
