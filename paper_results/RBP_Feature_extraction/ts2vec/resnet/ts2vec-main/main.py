from ts2vec import TS2Vec
import datautils
import os
import numpy as np
import pandas as pd
# Load the ECG200 dataset from UCR archive
# train_data, train_labels, test_data, test_labels = datautils.load_UCR('ECG200')
# train_df = pd.read_csv('D:\project\RBP\AGO1.tsv', sep='\t', header=None)
# train_df = pd.read_csv('D:\project\RBP\global_representation.tsv', sep='\t', header=None)
# train_df = np.loadtxt('F:\project\RBP\\resnet\global_representation.txt')  #proteinbert
train_df = np.loadtxt('F:\project\RBP\\resnet\global_representation_172.txt')  #proteinbert
train_array = np.array(train_df)
train =train_array[..., np.newaxis]
"""rpb数据处理"""

model = TS2Vec(
    input_dims=1,
    # device='cuda',
    device='cpu',
    # output_dims=320
    output_dims=5
)
# loss_log = model.fit(
#     train_data,
#     verbose=True
# )

# Compute timestamp-level representations for test set
# test_repr = model.encode(test_data)  # n_instances x n_timestamps x output_dims 转换
test_repr = model.encode(train)  # n_instances x n_timestamps x output_dims 转换

#3维数组保存
# with open('37RBP_37_512_320.txt', 'w') as outfile:
#     for slice_2d in test_repr:
#         np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')

with open('172RBP_172_512_5.txt', 'w') as outfile:
    for slice_2d in test_repr:
        np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')

test_repr2 = model.encode(train, encoding_window='full_series')  # n_instances x output_dims

# with open('37RBP_37_320.txt', 'w') as outfile:
#     for slice_2d in test_repr2:
#         np.savetxt(outfile, slice_2d, fmt = '%f', delimiter = ',')
# np.savetxt("37RBP_37_320.txt",test_repr2)
print(1)
