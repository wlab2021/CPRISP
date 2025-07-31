import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # from keras.backend.tensorflow_backend import set_session
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import scipy.io as sio
# import matlab.engine
import torch


from getSequence import *
from getSequenceAndStructure import *
# from BertDealEmbedding import *

from getCircRNA2Vec import *
from Pseudo_amino_dipeptide_structure import *
from sklearn.model_selection import train_test_split
from ANF_EIIP_CCN import *


def get_data(protein):


    Kmer, dataY = dealwithdata1(protein)  # X
    circrna2vec = dealwithCircRNA2Vec(protein)
    # Pseudo1 = Pseudo_amino_acid(protein)  # pseudo-amino acids 8 92,99(101-2),21
    Pseudo2 = Pseudo_dipeptide(protein)  # pseudo-dipeptide 892,400,10
    ANFAndEIIPAndCCN = dealwithANFAndEIIPAndCCN(protein)
    # trainX_PSTNPss_NCP = get_PSTNPss_NCP(protein)
    SequenceAndStructure = dealwithSequenceAndStructure(protein)
    # dataX4=dataX4.reshape(dataX4.shape[0],dataX4.shape[1],1)
    # Embedding1 = circRNABert(protein, 3)

    np.random.seed(4)
    indexes = np.random.choice(Kmer.shape[0], Kmer.shape[0], replace=False)
    # numpy.random.choice(a, size=None, replace=True, p=None)
    # Randomly select numbers from a (any ndarray is acceptable, but it must be one-dimensional) and form an array of a specified size.
    # replace: True indicates that identical numbers can be selected, while False indicates that identical numbers cannot be selected.

    # train_X1, test_X1,train_X2, test_X2,train_X3, test_X3,  train_X4, test_X4, train_Y, test_Y = train_test_split(dataX1[indexes],dataX2[indexes], dataX3[indexes],dataX4[indexes],dataY[indexes], test_size=0.2)

    # train and test dataset
    training_idx, test_idx = indexes[:round(((Kmer.shape[0])/10)*8)], indexes[round(((Kmer.shape[0])/10)*8):]  # 7:3
    X_train_1, X_test_1 = Kmer[training_idx, :, :], Kmer[test_idx, :, :]

    X_train_2, X_test_2 = ANFAndEIIPAndCCN[training_idx, :, :], ANFAndEIIPAndCCN[test_idx, :, :]

    X_train_3, X_test_3 = SequenceAndStructure[training_idx, :, :], SequenceAndStructure[test_idx, :, :]  # (892,101,24)

    X_train_4, X_test_4 = circrna2vec[training_idx, :, :], circrna2vec[test_idx, :, :]
    # X_train_4, X_test_4 = Pseudo1[training_idx, :, :], Pseudo1[test_idx, :, :]
    X_train_5, X_test_5 = Pseudo2[training_idx, :, :], Pseudo2[test_idx, :, :]
    y_train, y_test = dataY[training_idx], dataY[test_idx]

    # with open('WTAP_train_knf.txt', 'w') as outfile:
    #     for slice_2d in X_train_1:
    #         np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')
    # with open('WTAP_test_knf.txt', 'w') as outfile:
    #     for slice_2d in X_test_1:
    #         np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')
    # with open('WTAP_train_eiip.txt', 'w') as outfile:
    #     for slice_2d in X_train_2:
    #         np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')
    # with open('WTAP_test_eiip.txt', 'w') as outfile:
    #     for slice_2d in X_train_2:
    #         np.savetxt(outfile, slice_2d, fmt='%f', delimiter=',')

    train_dataset = dict()
    train_dataset["samples1"] = torch.from_numpy(X_train_1)
    train_dataset["samples2"] = torch.from_numpy(X_train_2)
    train_dataset["samples3"] = torch.from_numpy(X_train_3)
    train_dataset["samples4"] = torch.from_numpy(X_train_4)
    train_dataset["samples5"] = torch.from_numpy(X_train_5)
    train_dataset["labels"] = torch.from_numpy(y_train)
    train_dataset["idx"] = torch.from_numpy(training_idx)


    test_dataset = dict()
    test_dataset["samples1"] = torch.from_numpy(X_test_1)
    test_dataset["samples2"] = torch.from_numpy(X_test_2)
    test_dataset["samples3"] = torch.from_numpy(X_test_3)
    test_dataset["samples4"] = torch.from_numpy(X_test_4)
    test_dataset["samples5"] = torch.from_numpy(X_test_5)
    test_dataset["labels"] = torch.from_numpy(y_test)
    test_dataset["idx"] = torch.from_numpy(test_idx)

    torch.save(train_dataset,'data/{}_train.pt'.format(protein))
    torch.save(test_dataset,'data/{}_test.pt'.format(protein))

    return train_dataset, test_dataset
