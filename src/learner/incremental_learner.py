import torch.nn.functional as F
import torch
import abc
import logging
import copy
import numpy as np
import src
import logging
from src.data.datasets import IncrementalDataset

LOGGER = logging.getLogger(__name__.split('.')[-1])

class IncrementalLearner(abc.ABC):
    """The base incremental learner.
    Methods should be implemented:
        _train_task()
        _eval_task()

    Optional:
        _before_task()
        _after_task()
    """

    def __init__(self, config, saved_dir, total_num_class, class_per_task, chosen_order,
                 device, net, loss_fns, loss_weights, metric_fns, valid_freq=1):
        """
        Args:
            data_dir (pathlib): path of data folders.
        """
        self.config = config
        self.saved_dir = saved_dir
        self.total_num_class = total_num_class
        self.class_per_task = class_per_task
        self.chosen_order = chosen_order

        self.class_order = self.get_class_order(chosen_order)
        self.total_num_task = self.total_num_class // class_per_task
        self.current_task = -1  # current step of task
        self.current_num_class = 0  # number of seen classes
        self.test_class_index = []

        self.train_kwargs = self.config.dataloader.kwargs.pop('train', {})
        self.valid_kwargs = self.config.dataloader.kwargs.pop('valid', {})
        self.test_kwargs = self.config.dataloader.kwargs.pop('test', {})

        self.device = device
        self.net = net
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights
        self.metric_fns = metric_fns

        self.optimizer = None
        self.lr_scheduler = None
        self.logger = None
        self.monitor = None
        
        self.num_epochs = self.config.trainer.get('num_epochs')
        self.valid_freq = valid_freq

        self.net_loaded_path = None
        # if the last learning stage was terminated
        load_or_not = self.config.get('load')
        if load_or_not is not None:
            self.net_loaded_path = self.config.load.get('loaded_path')
            task_complete = self.config.load.get('task_complete')
            for i in range(task_complete):
                self._before_task()
            LOGGER.info('Resume learning.')
            LOGGER.info(f'current task: {self.current_task + 1}, current classes: {self.current_num_class}')


    def before_task(self, train_loader, val_loader):
        LOGGER.info("Before Task:")
        self._before_task()

        LOGGER.info('Create the optimizer.')
        self.optimizer = _get_instance(torch.optim, self.config.optimizer, self.net.parameters())

        if 'lr_scheduler' in self.config:
            LOGGER.info('Create the learning rate scheduler.')
            self.lr_scheduler = _get_instance(torch.optim.lr_scheduler, self.config.lr_scheduler, self.optimizer)
        else:
            LOGGER.info('Not using the learning rate scheduler.')
            self.lr_scheduler = None

        logging.info('Create the logger.')
        self.config.logger.setdefault('kwargs', {}).update(log_dir=self.saved_dir / 'log', net=self.net)
        self.config.logger.setdefault('kwargs', {}).update(add_epoch=(self.current_task * self.num_epochs))
        self.logger = _get_instance(src.callbacks.loggers, self.config.logger)

        logging.info('Create the monitor.')
        self.config.monitor.setdefault('kwargs', {}).update(checkpoints_dir=self.saved_dir / 'checkpoints')
        self.config.monitor.setdefault('kwargs', {}).update(add_epoch=(self.current_task * self.num_epochs))
        self.monitor = _get_instance(src.callbacks.monitor, self.config.monitor)


    def train_task(self):
        LOGGER.info("Training:")
        self._train_task()

    def after_task(self, inc_dataset):
        LOGGER.info("After Task:")
        self._after_task(inc_dataset)

    def eval_task(self, data_loader):
        LOGGER.info("Evaluation:")
        return self._eval_task(data_loader)

    def _before_task(self, data_loader):
        pass

    def _train_task(self, train_loader, val_loader):
        raise NotImplementedError

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, data_loader):
        raise NotImplementedError

    def build_dataset(self):
        """ To build 3 different datasets and return a dict of these datasets
        """
        chosen_class_index = self.get_current_class_index(self.current_task*self.class_per_task, self.class_per_task)
        self.test_class_index += chosen_class_index
        self.test_class_index = list(set(self.test_class_index))

        LOGGER.info('Create the datasets.')

        self.config.dataset.setdefault('kwargs', {}).update(type_='Training')
        self.config.dataset.setdefault('kwargs', {}).update(chosen_index=chosen_class_index)
        train_dataset = _get_instance(src.data.datasets, self.config.dataset)

        self.config.dataset.setdefault('kwargs', {}).update(type_='Validation')
        self.config.dataset.setdefault('kwargs', {}).update(chosen_index=chosen_class_index)
        valid_dataset = _get_instance(src.data.datasets, self.config.dataset)

        self.config.dataset.setdefault('kwargs', {}).update(type_='Testing')
        self.config.dataset.setdefault('kwargs', {}).update(chosen_index=self.test_class_index)
        test_dataset = _get_instance(src.data.datasets, self.config.dataset)

        return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}

    def build_dataloader(self, dataset, dataset_type):
        """
        """
        dataset = dataset[dataset_type]

        LOGGER.info('Create the dataloaders.')

        if dataset_type == 'train':
            loader_kwargs = self.train_kwargs
        elif dataset_type =='valid':
            loader_kwargs = self.valid_kwargs
        else :
            loader_kwargs = self.test_kwargs

        config_dataloader = copy.deepcopy(self.config.dataloader)
        config_dataloader.kwargs.update(loader_kwargs)
        dataloader = _get_instance(src.data.dataloader, config_dataloader, dataset)
        return dataloader

    def get_current_class_index(self, _start, class_per_task):
        _end = _start + class_per_task
        return self.class_order[_start:_end]

    def get_class_order(self, order_num):
        """Return a list of class (int)
        """
        class_order = []
        class_order.append(
            [
            87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
            24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
            25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
            60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
            34, 55, 54, 26, 35, 39
            ]
        )

        if order_num >= len(class_order):
            raise ValueError("The order is not exist. Try to add order list or choose other orders.")
        elif max(class_order[order_num]) != (self.total_num_class - 1):
            raise ValueError("The number in the order does not match the number of class.")
        else:
            return class_order[order_num]


def _get_instance(module, config, *args):
    """
    Args:
        module (MyClass): The defined module (class).
        config (Box): The config to create the class instance.

    Returns:
        instance (MyClass): The defined class instance.
    """
    cls = getattr(module, config.name)
    return cls(*args, **config.get('kwargs', {}))
