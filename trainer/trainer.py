import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# from models.loss import NTXentLoss
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

# from utils import _logger, set_requires_grad
coden_dict = {'AGO1': 0, 'AGO2': 1, 'AGO3': 2, 'ALKBH5': 3, 'AUF1': 4, 'C17ORF85': 5, 'C22ORF28': 6, 'CAPRIN1': 7,
              'DGCR8': 8, 'EIF4A3': 9, 'EWSR1': 10,
              'FMRP': 11, 'FOX2': 12, 'FUS': 13, 'FXR1': 14, 'FXR2': 15, 'HNRNPC': 16, 'HUR': 17, 'IGF2BP1': 18,
              'IGF2BP2': 19, 'IGF2BP3': 20,
              'LIN28A': 21, 'LIN28B': 22, 'METTL3': 23, 'MOV10': 24, 'PTB': 25, 'PUM2': 26, 'QKI': 27, 'SFRS1': 28,
              'TAF15': 29, 'TDP43': 30,
              'TIA1': 31, 'TIAL1': 32, 'TNRC6': 33, 'U2AF65': 34, 'WTAP': 35, 'ZC3H7B': 36,
              }


# coden_dict = {
#     "AARS": 0,"AATF": 1, "ABCF1": 2,"AGGF1": 3,"AGO": 4,"AKAP1": 5,"AKAP8L": 6,"ALKBH5": 7,"APOBEC3C": 8,"AQR": 9,"ATXN2": 10,"AUH": 11,"BCCIP": 12,
#     "BCLAF1": 13,"BUD13": 14,"C17ORF85": 15,"C22ORF28": 16,"CAPRIN1": 17,"CDC40": 18,"CPEB4": 19,"CPSF1": 20,"CPSF2": 21,
#     "CPSF3": 22,"CPSF4": 23,"CPSF6": 24,"CPSF7": 25,"CSTF2": 26,"CSTF2T": 27,"DDX3X": 28,"DDX6": 29,"DDX21": 30,
#     "DDX24": 31,"DDX42": 32,"DDX51": 33,"DDX52": 34,"DDX55": 35,"DDX59": 36,"DGCR8": 37,"DHX30": 38,
#     "DKC1": 39,"DROSHA": 40,"EFTUD2": 41,"EIF3D": 42,"EIF3G": 43,"EIF3H": 44,"EIF4A3": 45,"elF4Alll": 46,
#     "EIF4G2": 47,"ELAVL1": 48,"EWSR1": 49,"EXOSC5": 50,"FAM120A": 51,"FASTKD2": 52,"FBL": 53,
#     "FIP1L1": 54,"FKBP4": 55,"FMR1": 56,"FTO": 57,"FUS": 58,"FXR1": 59,"FXR2": 60,"G3BP1": 61,
#     "GEMIN5": 62,"GNL3": 63,"GPKOW": 64,"GRWD1": 65,"GTF2F1": 66,"HLTF": 67,"HNRNPA1": 68,
#     "HNRNPC": 69,"HNRNPD": 70,"HNRNPF": 71,"HNRNPK": 72,"HNRNPM": 73,"HNRNPU": 74,"HNRNPUL1": 75,
#     "IGF2BP1": 76,"IGF2BP2": 77, "IGF2BP3": 78,"ILF3": 79,"KHDRBS1": 80, "KHSRP": 81,
#     "LARP4": 82,"LARP7": 83,"LIN28A": 84,"LIN28B": 85,"LSM11": 86,"METAP2": 87,
#     "METTL3": 88,"METTL14": 89,"MOV10": 90,"MTPAP": 91,"NCBP2": 92,"NIP7": 93,"NIPBL": 94,
#     "NKRF": 95,"NOL12": 96,"NOLC1": 97,"NONO": 98, "NOP56": 99,"NOP58": 100,"NPM1": 101,
#     "NUDT21": 102,"PABPC4": 103,"PABPN1": 104,"PCBP1": 105,"PCBP2": 106,"PHF6": 107,"PPIG": 108,
#     "PRPF4": 109,"PRPF8": 110,"PTBP1": 111,"PTBP1PTBP2": 112,"PUM1": 113,"PUM2": 114,"PUS1": 115,"QKI": 116,
#     "RBFOX2": 117,"RBM15": 118,"RBM22": 119,"RBM27": 120,"RBPMS": 121,"RPS3": 122,"RPS11": 123,
#     "RTCB": 124,"SAFB2": 125,"SBDS": 126,"SDAD1": 127,"SERBP1": 128, "SF3A3": 129,"SF3B1": 130,"SF3B4": 131,
#     "SLBP": 132,"SLTM": 133,"SMNDC1": 134,"SND1": 135,"SRRM4": 136,"SRSF1": 137,"SRSF7": 138,
#     "SRSF9": 139,"SUB1": 140,"SUPV3L1": 141,"TAF15": 142,"TARDBP": 143,"TBRG4": 144,"TIA1": 145,
#     "TIAL1": 146,"TNRC6A": 147,"TRA2A": 148,"TROVE2": 149,"U2AF1": 150,"U2AF2": 151,"U2AF65": 152,"UCHL5": 153,
#     "UPF1": 154,"UTP3": 155,"UTP18": 156,"WDR3": 157,"WDR33": 158,"WDR43": 159,"WRN": 160,
#     "WTAP": 161, "XRCC6": 162,"XRN2": 163,"YBX3": 164,"YTHDF2": 165,"YWHAG": 166,
#     "ZC3H7B": 167,"ZC3H11A": 168,"ZNF622": 169,"ZNF800": 170,"ZRANB2": 171
# }

def Trainer(data_type, epochs, model, hdrnet, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl,
            test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()

    model_scheduler = torch.optim.lr_scheduler.StepLR(model_optimizer, 2, gamma=0.1, last_epoch=-1, verbose=False)
    scheduler = torch.optim.lr_scheduler.StepLR(temp_cont_optimizer, 2, gamma=0.1, last_epoch=-1, verbose=False)

    max_test_auc = 0
    max_test_acc = 0
    max_test_precision = 0
    max_test_recall = 0
    max_epoch = 0
    # typeF = "EIIP-ANF-CCN_withRBP"
    typeF = "Randomly generated RBP—structure_withRBP"
    for epoch in range(1, epochs + 1):
        # Train and validate
        train_loss, train_acc, train_auc, train_precision, train_recall = model_train(data_type, model, hdrnet,
                                                                                      temporal_contr_model,
                                                                                      model_optimizer,
                                                                                      temp_cont_optimizer,
                                                                                      criterion, train_dl, config,
                                                                                      device, training_mode, typeF)
        test_loss, test_acc, test_auc, test_precision, test_recall, _, _ = model_evaluate(data_type, model, hdrnet,
                                                                                          temporal_contr_model, test_dl,
                                                                                          device, training_mode, typeF)

        if max_test_auc <= test_auc:
            max_epoch = epoch
            max_test_acc = test_acc
            max_test_auc = test_auc
            max_test_precision = test_precision
            max_test_recall = test_recall

            os.makedirs(os.path.join(experiment_log_dir, "saved_models_best"), exist_ok=True)
            chkpoint = {'model_state_dict': model.state_dict(),
                        'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}

            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models_best", f'ckp_best.pt'))

        if training_mode != 'self_supervised':  # use scheduler in all other modes.
            # scheduler.step(test_loss)
            model_scheduler.step()
            scheduler.step()

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss : {train_loss:.4f}\t | \t Accuracy : {train_acc:2.4f} | \t AUC : {train_auc:2.4f} | \t Precision : {train_precision:2.4f} | \t Recall : {train_recall:2.4f}\n'
                     f'Test Loss : {test_loss:.4f}\t | \t Accuracy : {test_acc:2.4f} | \t AUC : {test_auc:2.4f} | \t Precision : {test_precision:2.4f} | \t Recall : {test_recall:2.4f}')

        if epoch == 5:  # or epoch == 3000: epoch == 50 or epoch == 100 or epoch == 150 or
            os.makedirs(os.path.join(experiment_log_dir, "saved_models" + str(epoch)), exist_ok=True)
            chkpoint = {'model_state_dict': model.state_dict(),
                        'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}

            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models" + str(epoch), f'ckp_last.pt'))
    with open("experiments_logs/result_all.txt", 'a', encoding='utf-8') as f:
        f.write("dataset:{}\n".format(data_type))
        f.write(
            f'epoch:{max_epoch}\t | \tmax_test_acc: {max_test_acc:2.4f} \t | \tmax_test_auc: {max_test_auc:2.4f} | \tmax_test_precision: {max_test_precision:2.4f} | \tmax_test_recall: {max_test_recall:2.4f}\n')

    logger.debug("\n################## Training is Done! #########################")


def Evaluator(data_type, epochs, model, hdrnet, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl,
              test_dl, device,
              logger, config, experiment_log_dir, training_mode):
    # Start evaluating
    logger.debug("Evaluating started ....")

    typeF = "Randomly generated RBP—structure_withRBP"
    checkpoint = torch.load(os.path.join(experiment_log_dir, "saved_models_best", f'ckp_best.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    temporal_contr_model.load_state_dict(checkpoint['temporal_contr_model_state_dict'])
    model.eval()
    temporal_contr_model.eval()

    test_loss, test_acc, test_auc, test_precision, test_recall, _, _ = model_evaluate(data_type, model, hdrnet,
                                                                                      temporal_contr_model, test_dl,
                                                                                      device, training_mode, typeF)

    logger.debug(
        f'Test Loss : {test_loss:.4f}\t | \t Accuracy : {test_acc:2.4f} | \t AUC : {test_auc:2.4f} | \t Precision : {test_precision:2.4f} | \t Recall : {test_recall:2.4f}')

    logger.debug("\n################## Evaluating is Done! #########################")


def model_train(data_type, model, hdrnet, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion,
                train_loader, config,
                device, training_mode, typeF):
    total_loss = []
    total_acc = []
    total_auc = []
    total_precision = []
    total_recall = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data1, data2, data3, data4, data5, labels, idxs) in enumerate(train_loader):

        data1, data2, data3, data4, data5 = data1.float().to(device), data2.float().to(device), data3.float().to(
            device), data4.float().to(device), data5.float().to(device)
        # data4, data5, data6 = data4.float().to(device), data5.float().to(device), data6.float().to(device)
        labels = labels.long().to(device)
        np.savetxt('labels.txt', labels.cpu())

        idxs = idxs.long().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        RBP = np.loadtxt('rbp_data/37RBP_37_512_5.txt', delimiter=',') \
            .reshape((37, 512, 5))

        RBP_PART = RBP[coden_dict[data_type]].reshape((1, 512, 5))  # WTAP

        index = coden_dict[data_type]
        available_indices = list(range(len(coden_dict)))
        available_indices.remove(index)

        RBP_1_index = random.choice(available_indices)
        available_indices.remove(RBP_1_index)

        RBP_2_index = random.choice(available_indices)

        RBP_1 = RBP[RBP_1_index].reshape((1, 512, 5))
        RBP_2 = RBP[RBP_2_index].reshape((1, 512, 5))

        RBP_ten = torch.from_numpy(RBP_PART)
        RBP_ten1 = torch.from_numpy(RBP_1)
        RBP_ten2 = torch.from_numpy(RBP_2)
        RBP_37 = torch.from_numpy(RBP)

        dataR = RBP_ten.float().to(device)
        data_RBP1 = RBP_ten1.float().to(device)
        data_RBP2 = RBP_ten2.float().to(device)
        data_RBP37 = RBP_37.float()

        data_RBP37 = data_RBP37.to(device)

        RBP_37 = model(data_RBP37, tag=4)
        RBP_37n = RBP_37.detach().cpu().numpy()

        features1 = model(data1, tag=1).to(device)
        features2 = model(data2, tag=2).to(device)

        features3 = model(data3, tag=3).to(device)
        features4 = model(data4, tag=3).to(device)

        output = hdrnet(features1, features2, features3, features4)
        prob = torch.sigmoid(output)

        RBP_f = model(dataR, tag=4)
        RBP_f1 = model(data_RBP1, tag=4)
        RBP_f2 = model(data_RBP2, tag=4)

        yt = temporal_contr_model(features1, features2, features3, features4, RBP_f, RBP_f1, RBP_f2, output, labels)
        ypos = -yt[:, 1]
        prob1 = torch.sigmoid(ypos)

        labels1 = labels.detach().cpu().numpy()
        prob = prob1.detach().cpu().numpy()
        idxs1 = idxs.detach().cpu().numpy()

        directory_path1 = "scan/" + format(data_type) + "/train"
        if not os.path.exists(directory_path1):
            os.makedirs(directory_path1)
        # # num =0
        file_path1 = os.path.join(directory_path1, format(data_type) + "_train_prob" + ".txt")

        with open(file_path1, 'a', encoding='utf-8') as f:
            for item in prob:
                f.write(str(item) + '\n')
        #
        #
        file_path2 = os.path.join(directory_path1, format(data_type) + "_train_lab" + ".txt")
        with open(file_path2, 'a', encoding='utf-8') as f:
            for item in labels1:
                f.write(str(item) + '\n')
        #
        file_path3 = os.path.join(directory_path1, format(data_type) + "_train_index" + ".txt")
        with open(file_path3, 'a', encoding='utf-8') as f:
            for item in idxs1:
                f.write(str(item) + '\n')

        if training_mode == "self_supervised":
            lambda1 = 0.7

            loss = 0

        else:  # supervised training or fine tuining
            #     predictions, features = output
            loss = criterion(yt, labels)
            # total_acc.append(labels.eq(yt.detach().argmax(dim=1)).float().mean())
            directory_path = "result/" + format(data_type) + "/train"

            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            file_path = os.path.join(directory_path, format(data_type) + "_train_" + typeF + ".txt")

            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f'pre:{yt.detach().cpu()[:, 1]}\n')
                f.write(f'labels:{labels}\n\n\n\n')

            # np.savetxt("yt.txt",yt.detach())

            auc = roc_auc_score(labels.cpu(), yt.detach().cpu()[:, 1])

            acc = accuracy_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
            precision = precision_score(labels.cpu(), yt.detach().cpu().argmax(dim=1), average='micro')
            recall = recall_score(labels.cpu(), yt.detach().cpu().argmax(dim=1), average='micro')

            total_acc.append(acc)
            total_precision.append(precision)
            total_recall.append(recall)
            total_auc.append(auc)

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
        total_auc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
        total_auc = torch.tensor(total_auc).mean()
        total_precision = torch.tensor(total_precision).mean()
        total_recall = torch.tensor(total_recall).mean()

    return total_loss, total_acc, total_auc, total_precision, total_recall


def model_evaluate(data_type, model, hdrnet, temporal_contr_model, test_dl, device, training_mode, typeF):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_precision = []
    total_recall = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data1, data2, data3, data4, data5, labels, idxs in test_dl:
            data1, data2, data3, data4, data5 = data1.float().to(device), data2.float().to(device), data3.float().to(
                device), data4.float().to(device), data5.float().to(device)
            # data4, data5, data6 = data4.float().to(device), data5.float().to(device), data6.float().to(device)
            labels = labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                # RBP = np.loadtxt('F:\project\CircSSNN-rbp\circrna-rbp\circrna-rbp\\37RBP_37_512_20.txt',
                #                  delimiter=',').reshape((37, 512, 20))

                RBP = np.loadtxt('rbp_data/37RBP_37_512_5.txt',
                                 delimiter=',').reshape((37, 512, 5))

                RBP_PART = RBP[coden_dict[data_type]].reshape((1, 512, 5))  # WTAP

                index = coden_dict[data_type]
                available_indices = list(range(len(coden_dict)))
                available_indices.remove(index)

                RBP_1_index = random.choice(available_indices)
                available_indices.remove(RBP_1_index)

                RBP_2_index = random.choice(available_indices)

                RBP_1 = RBP[RBP_1_index].reshape((1, 512, 5))
                RBP_2 = RBP[RBP_2_index].reshape((1, 512, 5))

                RBP_ten = torch.from_numpy(RBP_PART)
                RBP_ten1 = torch.from_numpy(RBP_1)
                RBP_ten2 = torch.from_numpy(RBP_2)

                dataR = RBP_ten.float().to(device)
                data_RBP1 = RBP_ten1.float().to(device)
                data_RBP2 = RBP_ten2.float().to(device)

                features1 = model(data1, tag=1)
                features2 = model(data2, tag=2)
                features3 = model(data3, tag=3)
                features4 = model(data4, tag=3)
                features5 = model(data5, tag=5)

                # output = hdrnet(features1, features2, features3,features4,features5)
                # output = hdrnet(features3)
                output = hdrnet(features1, features2, features3, features4)

                RBP_f = model(dataR, tag=4)
                RBP_f1 = model(data_RBP1, tag=4)
                RBP_f2 = model(data_RBP2, tag=4)

                yt = temporal_contr_model(features1, features2, features3, features4, RBP_f, RBP_f1, RBP_f2, output,
                                          labels)
                # yt = temporal_contr_model(features3,RBP_f, RBP_f1, RBP_f2,output,labels)

            # compute loss
            if training_mode != "self_supervised":
                # loss = (temp_cont_loss31 + temp_cont_loss32)
                loss = criterion(yt, labels)
                directory_path = "result/" + format(data_type) + "/test"

                if not os.path.exists(directory_path):
                    os.makedirs(directory_path)

                file_path = os.path.join(directory_path, format(data_type) + "_test_" + typeF + ".txt")

                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(f'pre:{yt.detach().cpu()[:, 1]}\n')
                    f.write(f'labels:{labels}\n\n\n\n')

                auc = roc_auc_score(labels.cpu(), yt.detach().cpu()[:, 1])
                acc = accuracy_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
                precision = precision_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))
                recall = recall_score(labels.cpu(), yt.detach().cpu().argmax(dim=1))

                total_loss.append(loss.item())
                total_acc.append(acc)
                total_precision.append(precision)
                total_recall.append(recall)
                total_auc.append(auc)

                # if training_mode != "self_supervised":
                pred = yt.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    total_auc = torch.tensor(total_auc).mean()  # average acc
    total_precision = torch.tensor(total_precision).mean()
    total_recall = torch.tensor(total_recall).mean()

    return total_loss, total_acc, total_auc, total_precision, total_recall, outs, trgs
