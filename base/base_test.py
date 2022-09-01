import torch

class BaseTest:
    def __init__(self, model, test_data, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.test_data = test_data

    def train(self):
        self.test_epoch()

    def test_epoch(self):
        raise NotImplementedError