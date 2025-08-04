import argparse
import numpy as np
# import keras.utils.np_utils as kutils
import pandas as pd
import torch
import torch.nn.functional as F


def ANF_NCP_EIIP_Onehot_Encoding(sampleSeq3DArr):
    CategoryLen = 1 + 3 + 1 + 4

    probMatr = np.zeros((len(sampleSeq3DArr), len(sampleSeq3DArr[0]), CategoryLen))

    sampleNo = 0
    for sequence in sampleSeq3DArr:
        AANo = 0
        sequenceStr = ''
        for AA in sequence:
            sequenceStr += AA
        for j in range(len(sequenceStr)):
            thisLetter = sequenceStr[j]
            if (thisLetter == "A"):
                probMatr[sampleNo][AANo][0] = 1
                probMatr[sampleNo][AANo][1] = 1
                probMatr[sampleNo][AANo][2] = 1
                probMatr[sampleNo][AANo][3] = 0.126
                probMatr[sampleNo][AANo][4] = 1      #A
                probMatr[sampleNo][AANo][5] = 0      #C
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 0
            elif (thisLetter == "C"):
                probMatr[sampleNo][AANo][0] = 0
                probMatr[sampleNo][AANo][1] = 1
                probMatr[sampleNo][AANo][2] = 0
                probMatr[sampleNo][AANo][3] = 0.134
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 1
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 0
            elif (thisLetter == "G"):
                probMatr[sampleNo][AANo][0] = 1
                probMatr[sampleNo][AANo][1] = 0
                probMatr[sampleNo][AANo][2] = 0
                probMatr[sampleNo][AANo][3] = 0.0806
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 0
                probMatr[sampleNo][AANo][6] = 1
                probMatr[sampleNo][AANo][7] = 0
            elif (thisLetter == "T"):
                probMatr[sampleNo][AANo][0] = 0
                probMatr[sampleNo][AANo][1] = 0
                probMatr[sampleNo][AANo][2] = 1
                probMatr[sampleNo][AANo][3] = 0.1335
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 0
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 1
            else:
                probMatr[sampleNo][AANo][0] = 0
                probMatr[sampleNo][AANo][1] = 0
                probMatr[sampleNo][AANo][2] = 0
                probMatr[sampleNo][AANo][3] = 0
                probMatr[sampleNo][AANo][4] = 0
                probMatr[sampleNo][AANo][5] = 0
                probMatr[sampleNo][AANo][6] = 0
                probMatr[sampleNo][AANo][7] = 0

            probMatr[sampleNo][AANo][8] = sequenceStr[0: j + 1].count(sequenceStr[j]) / float(j + 1)

            AANo += 1
        sampleNo += 1

    return probMatr


def convertRawToXY(rawDataFrameTrain, rawDataFrameTest, codingMode='ANF_NCP_EIIP_Onehot'):

    targetList = rawDataFrameTest[:, 0]
    targetList2 = []
    for ii in targetList:
        targetList2.append(int(ii))
    # targetArr = kutils.to_categorical(targetList2,2)

    targetList3 = torch.tensor(targetList2)
    gt_onthot = F.one_hot(targetList3, num_classes=2)
    gt_onthot2 = gt_onthot.numpy()
    gt_onthot3 = np.float32(gt_onthot2)
    sampleSeq3DArr = rawDataFrameTest[:, 1:]


    if codingMode == 'ANF_NCP_EIIP_Onehot':
        probMatr = ANF_NCP_EIIP_Onehot_Encoding(sampleSeq3DArr)


    # return probMatr, targetArr
    return probMatr, gt_onthot3


def read_fasta(fasta_file):
    try:
        fp = open(fasta_file)
    except IOError:
        print('cannot open ' + fasta_file + ', check if it exist!')
        exit()
    else:
        fp = open(fasta_file)
        lines = fp.readlines()

        fasta_dict = {}  # record seq for one id
        idlist = []  # record id list sorted
        gene_id = ""
        for line in lines:
            line = line.replace('\r', '')
            if line[0] == '>':
                if gene_id != "":
                    fasta_dict[gene_id] = seq.upper()
                    idlist.append(gene_id)
                seq = ""
                gene_id = line.strip('\n')  # line.split('|')[1] all in > need to be id
            else:
                seq += line.strip('\n')

        fasta_dict[gene_id] = seq.upper()  # last seq need to be record
        idlist.append(gene_id)

    return fasta_dict, idlist


def get_sequence_odd_fixed(fasta_dict, idlist, window=20, label=1):
    seq_list_2d = []
    id_list = []
    pos_list = []
    for id in idlist:  # for sort
        seq = fasta_dict[id]
        final_seq_list = [label] + [AA for AA in seq]

        id_list.append(id)
        pos_list.append(window)
        seq_list_2d.append(final_seq_list)

    df = pd.DataFrame(seq_list_2d)
    df2 = pd.DataFrame(id_list)
    df3 = pd.DataFrame(pos_list)

    return df, df2, df3


def analyseFixedPredict(fasta_file, window=20, label=1):
    fasta_dict, idlist = read_fasta(fasta_file)

    sequence, ids, poses = get_sequence_odd_fixed(fasta_dict, idlist, window, label)

    return sequence, ids, poses


def dealwithANFAndEIIPAndCCN(protein):
    seqpos_path='datasets/circRNA-RBP/' + protein + '/positive'
    seqneg_path ='datasets/circRNA-RBP/' + protein + '/negative'
    pos_data, pos_ids, pos_poses = analyseFixedPredict(seqpos_path, window=20, label=1)
    neg_data, neg_ids, neg_poses = analyseFixedPredict(seqneg_path, window=20, label=0)
    train_All2 = pd.concat([pos_data, neg_data])
    train_data = train_All2
    train_All = train_data
    trainX_ANF_NCP, trainY_ANF_NCP = convertRawToXY(train_All.values, train_data.values,
                                                    codingMode='ANF_NCP_EIIP_Onehot')
    # print(trainX_ANF_NCP)  ##892,101,9
    return trainX_ANF_NCP
