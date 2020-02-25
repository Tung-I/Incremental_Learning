import torch
import logging
from tqdm import tqdm
import random
import numpy as np
import abc


class BaseLearner(abc.ABC):
    """The base class for all trainers.

    1. set_task_info
    2. before_task
    3. train_task
    4. after_task
    5. eval_task

    """

    def __init__(self, *args, **kwargs):
        pass

    def set_task_info(self, task, total_n_classes, increment, n_train_data, n_test_data,
                      n_tasks):

        logging.info("info setting")

        self._task = task
        self._task_size = increment
        self._total_n_classes = total_n_classes
        self._n_train_data = n_train_data
        self._n_test_data = n_test_data
        self._n_tasks = n_tasks

    def before_task(self, train_dataloader, valid_dataloader):
        logging.info("before task")
        self.eval()
        self._before_task(train_dataloader, valid_dataloader)

    def train_task(self, train_dataloader, valid_dataloader):
        logging.info("train task")
        self.train()
        self._train_task(train_dataloader, valid_dataloader)

    def after_task(self, inc_dataset):
        logging.info("after task")
        self.eval()
        self._after_task(inc_dataset)

    def eval_task(self, dataloader):
        logging.info("eval task")
        self.eval()
        return self._eval_task(dataloader)

    def get_memory(self):
        return None

    def eval(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def _before_task(self, dataloader):
        pass

    def _train_task(self, train_dataloader, valid_dataloader):
        raise NotImplementedError

    def _after_task(self, dataloader):
        pass

    def _eval_task(self, dataloader):
        raise NotImplementedError

    def _new_task_index(self):
        return self._task * self._task_size



