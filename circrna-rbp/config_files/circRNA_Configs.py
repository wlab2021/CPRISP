class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 101
        self.kernel_size = 7
        self.stride = 1
        self.final_out_channels = 128 #

        self.num_classes = 2
        self.dropout = 0.3
        self.features_len = 22

        self.beta1 = 0.8
        self.beta2 = 0.99
        self.lr = 3e-3 #


        # data parameters
        self.drop_last = True

        self.batch_size =64


        self.TC = TC()


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10
