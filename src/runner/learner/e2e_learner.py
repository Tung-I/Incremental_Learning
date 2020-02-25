import torch
import random
import numpy as np
import logging
from tqdm import tqdm
from src.runner.base_learner import BaseLearner


class E2ELearner(BaseLearner):
    """
    """
    def __init__(self, device, optimizer, num_epochs, lr_scheduler, 
                 memory_size, temperature, net):
        super().__init__()


        self._device = device
        self._optimizer = optimizer
        self._num_epochs = num_epochs

        self._lr_scheduler = lr_scheduler

        self._k = memory_size

        self._temp = temperature
        self._net = net

        self._n_classes = 0
        self._examplars = {}
        self._old_model = []

        self._task_idxes = []

    def eval(self):
        self._net.eval()

    def train(self):
        self._net.train()










