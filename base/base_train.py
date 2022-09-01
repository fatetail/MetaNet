import torch


class BaseTrain:
    def __init__(self, model, train_data, val_data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        if self.config.is_gpu:
            self.device = torch.device('cuda:0')
            torch.backends.cudnn.benchmark = True
            self.gpu_list = [i for i in range(0, self.config.gpu_cnt)]
        else:
            self.device = torch.device('cpu')
        self.model = model.to(self.device)
        torch.backends.cudnn.benchmark = True
        self.model = torch.nn.DataParallel(self.model, device_ids=self.gpu_list)

    def train(self):
        for cur_epoch in range(1, self.config.epochs+1):
            self.train_epoch()

    def train_epoch(self):
        raise NotImplementedError

    # def train_step(self):
    #     raise NotImplementedError
