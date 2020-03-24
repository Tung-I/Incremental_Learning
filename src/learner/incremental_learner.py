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
    """
    def __init__(self, config, saved_dir, total_num_class, class_per_task, chosen_order,
                 device, net, loss_fns, loss_weights, metric_fns):
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
        self.current_class_index = []
        self.current_class_label = []
        self.test_class_index = []
        self.test_class_label = []

        # save the config of dataloaders
        self.train_kwargs = self.config.dataloader.kwargs.pop('train', {})
        self.valid_kwargs = self.config.dataloader.kwargs.pop('valid', {})
        self.test_kwargs = self.config.dataloader.kwargs.pop('test', {})

        # objects which have been builded outside
        self.device = device
        self.net = net
        self.loss_fns = loss_fns
        self.loss_weights = loss_weights
        self.metric_fns = metric_fns

        # objects which should be rebuilded inside
        self.optimizer = None
        self.lr_scheduler = None
        self.logger = None
        self.monitor = None
        self.current_datasets = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.trainer = None

        # if the last learning stage was terminated
        self.net_loaded_path = None
        self.load_task()

        # for calculating distillation loss
        self.old_model = None

        # for trainer
        self.meta_data = {}

    def before_task(self):
        """
        """
        LOGGER.info("Before Task:")
        # update info
        self.current_task += 1
        self.current_num_class += self.class_per_task
        # expand the net (before creating optimizer)
        self.net.add_classes(self.class_per_task)
        # rebuild objects for current task
        self.optimizer = self.build_optimizer(self.config, self.net)
        self.lr_scheduler = self.build_scheduler(self.config, self.optimizer)
        _add_epoch = self.current_task * self.config.trainer.get('num_epochs')
        self.logger = self.build_logger(self.config, self.saved_dir, self.net, _add_epoch)
        self.monitor = self.build_monitor(self.config, self.saved_dir, _add_epoch)
        # rebuild the datasets, loaders
        self.current_datasets = self.build_dataset(self.config, self.current_task, self.class_per_task)
        self.train_loader = self.build_dataloader(self.config, self.current_datasets, 'train')
        self.valid_loader = self.build_dataloader(self.config, self.current_datasets, 'valid')
        self.test_loader = self.build_dataloader(self.config, self.current_datasets, 'test')
        # rebuild the trainer
        self.meta_data['old_model'] = self.old_model
        self.meta_data['class_per_task'] = self.class_per_task
        self.trainer = self.build_trainer(self.config, self.current_datasets, self.device, self.train_loader, self.valid_loader, self.net, 
                                        self.loss_fns, self.loss_weights, self.metric_fns, self.optimizer, self.lr_scheduler, self.logger, self.monitor,
                                        self.meta_data)
        # customized
        self._before_task()

    def train_task(self):
        LOGGER.info("Training:")
        self._train_task()
        LOGGER.info(f'End training of task{self.current_task + 1}.')

    def after_task(self):
        LOGGER.info("After Task:")
        self._after_task()

    def eval_task(self):
        LOGGER.info("Evaluation:")
        test_loader = self.build_dataloader(self.current_dataset, 'test')
        self._eval_task()

    def build_dataset(self, config, current_task, class_per_task):
        """ To build 3 different datasets and return a dict of these datasets
        """
        self.current_class_index, self.current_class_label = self.get_current_class_index_label(current_task*class_per_task, (current_task+1)*class_per_task)
        self.test_class_index, self.test_class_label = self.get_current_class_index_label(0, (current_task+1)*class_per_task)

        LOGGER.info('Create the datasets.')
        chosen_class_index = self.current_class_index
        chosen_class_label = self.current_class_label
        test_calss_index = self.test_class_index
        test_class_label = self.test_class_label

        config.dataset.setdefault('kwargs', {}).update(type_='Training')
        config.dataset.setdefault('kwargs', {}).update(chosen_index=chosen_class_index)
        config.dataset.setdefault('kwargs', {}).update(chosen_label=chosen_class_label)
        train_dataset = _get_instance(src.data.datasets, config.dataset)

        config.dataset.setdefault('kwargs', {}).update(type_='Validation')
        valid_dataset = _get_instance(src.data.datasets, config.dataset)

        config.dataset.setdefault('kwargs', {}).update(type_='Testing')
        config.dataset.setdefault('kwargs', {}).update(chosen_index=test_class_index)
        config.dataset.setdefault('kwargs', {}).update(chosen_label=test_class_label)
        test_dataset = _get_instance(src.data.datasets, config.dataset)

        return {'train':train_dataset, 'valid':valid_dataset, 'test':test_dataset}

    def build_dataloader(self, config, dataset, dataset_type):
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

        config_dataloader = copy.deepcopy(config.dataloader)
        config_dataloader.kwargs.update(loader_kwargs)
        dataloader = _get_instance(src.data.dataloader, config_dataloader, dataset)
        return dataloader

    def build_optimizer(self, config, net):
        LOGGER.info('Create the optimizer.')
        optimizer = _get_instance(torch.optim, config.optimizer, net.parameters())
        return optimizer

    def build_scheduler(self, config, optimizer):
        LOGGER.info('Create the lr scheduler.')
        if 'lr_scheduler' in config:
            LOGGER.info('Create the learning rate scheduler.')
            lr_scheduler = _get_instance(torch.optim.lr_scheduler, config.lr_scheduler, optimizer)
        else:
            LOGGER.info('Not using the learning rate scheduler.')
            lr_scheduler = None
        return lr_scheduler

    def build_logger(self, config, saved_dir, _net, _add_epoch):
        LOGGER.info('Create the logger.')
        config.logger.setdefault('kwargs', {}).update(log_dir=(saved_dir / 'log'), net=_net)
        config.logger.setdefault('kwargs', {}).update(add_epoch=_add_epoch)
        logger = _get_instance(src.callbacks.loggers, config.logger)
        return logger

    def build_monitor(self, config, saved_dir, _add_epoch):
        LOGGER.info('Create the monitor.')
        config.monitor.setdefault('kwargs', {}).update(checkpoints_dir=(saved_dir / 'checkpoints'))
        config.monitor.setdefault('kwargs', {}).update(add_epoch=add_epoch)
        monitor = _get_instance(src.callbacks.monitor, config.monitor)
        return monitor

    def build_trainer(self, config, dataset, device, train_laoder, valid_loader, net, loss_fns, 
                    loss_weights, metric_fns, optimizer, lr_scheduler, logger, monitor, meta_data):
        LOGGER.info(f'Create the trainer of the task.')

        kwargs = {
            'device': device,
            'train_dataloader': train_loader,
            'valid_dataloader': valid_loader,
            'net': net,
            'loss_fns': loss_fns,
            'loss_weights': loss_weights,
            'metric_fns': metric_fns,
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'logger': logger,
            'monitor': monitor,
            'meta_data': meta_data
        }
        config.trainer.kwargs.update(kwargs)
        trainer = _get_instance(src.runner.trainers, config.trainer)
        # if the parameters of net should be loaded
        if self.net_loaded_path is not None:
            logging.info(f'Load the previous checkpoint from "{self.net_loaded_path}".')
            trainer.load(Path(self.net_loaded_path))
            logging.info('Resume training.')
        return trainer

    def get_current_class_index_label(self, _start, _end):
        return self.class_order[_start:_end], list(range(_start, _end))

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

    def load_task(self):
        load_or_not = self.config.get('load')
        if load_or_not is not None:
            self.net_loaded_path = self.config.load.get('loaded_path')
            task_complete = self.config.load.get('task_complete')
            for i in range(task_complete):
                self._before_task()
            LOGGER.info('Resume learning.')
            LOGGER.info(f'current task: {self.current_task + 1}, current classes: {self.current_num_class}')

    def _before_task(self, data_loader):
        raise NotImplementedError

    def _train_task(self, train_loader, val_loader):
        raise NotImplementedError

    def _after_task(self, data_loader):
        raise NotImplementedError

    def _eval_task(self, data_loader):
        raise NotImplementedError

    def learn(self):
        LOGGER.info("Start incremental learning.")
        for i in range(self.total_num_task):
            LOGGER.info(f'Start task: {self.current_task + 2}')
            self.before_task()
            self.train_task()
            self.after_task()
            self.eval_task()
            LOGGER.info(f'End task: {self.current_task + 1}')


def _get_instance(module, config, *args):
    cls = getattr(module, config.name)
    return cls(*args, **config.get('kwargs', {}))
