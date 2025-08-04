import torch
import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #!!
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
# from dataloader.dataloader import data_generator
from data_preprocessing.CRBP.dataloader import data_generator
from trainer.trainer import Trainer, model_evaluate, Evaluator
from models.TC import TC
from models.HDRNet import HDRNet
from utils import _calc_metrics, copy_Files
from models.model import base_Model
from config_files.circrna_configs import Config as Configs


class DotDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'")



def main(args, epochs):
    start_time = datetime.now()
    device = torch.device(args.device)
    experiment_description = args.experiment_description
    data_type = args.selected_dataset
    method = 'MISSM'
    training_mode = args.training_mode
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    os.makedirs(logs_save_dir, exist_ok=True)


    # exec(f'from config_files.circRNA_Configs import Config as Configs')
    configs = Configs()

    # ##### fix random seeds for reproducibility ########
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    #####################################################

    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # loop through domains
    counter = 0
    src_counter = 0


    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Dataset: {data_type}')
    logger.debug(f'Method:  {method}')
    # logger.debug(f'Mode:    {training_mode}')
    logger.debug("=" * 45)

    # Load datasets
    data_path = f"./data/circRNA-RBP/{data_type}"
    train_dl, test_dl = data_generator(data_type, configs)  #train_dataset, test_dataset

    # Load Model
    model = base_Model(configs).to(device)

    temporal_contr_model = TC(configs, device).to(device)

    hdrnet = HDRNet().to(device)


    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

    if args.isTrain:
        Trainer(data_type, epochs, model, hdrnet, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl,  test_dl, device, logger, configs, experiment_log_dir, training_mode)
    else:
        Evaluator(data_type, epochs, model, hdrnet, temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl,  test_dl, device, logger, configs, experiment_log_dir, training_mode)


    logger.debug(f"Training time is : {datetime.now()-start_time}")


if __name__ == "__main__":

    num = 1

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datasets',
        type=str,
        required=False,
        nargs='+',
        default=[
            'AGO1', 'AGO2', 'AGO3', 'WTAP'
        ]
    )
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--training_mode', default="train_linear", type=str,
                        help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
    parser.add_argument('--seed', default=123, type=int,
                        help='seed value')
    parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                        help='saving directory')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str,
                        help='cpu or cuda')
    parser.add_argument('--isTrain', action='store_true')

    args = parser.parse_args()

    for data_name in args.datasets:

    # for data_name in ['AGO1','AGO2','AGO3']:
    #     if data_name=='HUR':
    #         num=10
    #     for mode in ["train_linear"] * num:  # "self_supervised",
            ######################## Model parameters ########################

            # --experiment_description exp7 --run_description run_3 --seed 123 --training_mode train_linear --selected_dataset circRNA-RBP
            # data_name ='AGO2'#''
            # parser = argparse.ArgumentParser()
            # parser.add_argument('--selected_dataset', default=data_name, type=str,
            #                     help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
            # parser.add_argument('--training_mode', default=mode, type=str,
            #                     help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
            # parser.add_argument('--experiment_description', default=data_name + '_Exp1', type=str,
            #                     help='Experiment Description')
            # parser.add_argument('--run_description', default=data_name + 'feature_RBP', type=str,
            # # parser.add_argument('--run_description', default=data_name + 'feature', type=str,
            #                     help='Experiment Description')
            # parser.add_argument('--seed', default=123, type=int,
            #                     help='seed value')
            # parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
            #                     help='saving directory')
            # parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str,
            #                     help='cpu or cuda')
            # args = parser.parse_args()

        cfg = DotDict()
        cfg.selected_dataset = data_name
        cfg.training_mode = args.training_mode
        cfg.experiment_description = data_name + '_Exp1'
        cfg.run_description = data_name + 'feature_RBP'
        cfg.seed = args.seed
        cfg.logs_save_dir = args.logs_save_dir
        cfg.device = args.device
        cfg.isTrain = args.isTrain

        # if mode == 'self_supervised':
        main(cfg, epochs=args.epoch)
        # else:
        # main(args, epochs=10)
        # break
