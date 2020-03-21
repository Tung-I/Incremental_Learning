import logging
from torch.nn import functional as F
from tqdm import tqdm

from src.learner import IncrementalLearner

LOGGER = logging.getLogger(__name__.split('.')[-1])

class ICaRLLearner(IncrementalLearner):
    """The base incremental learner.
    Methods should be implemented:
        _train_task()
        _eval_task()

    Optional:
        _before_task()
        _after_task()
    """

    def __init__(self, exampler_size, **kwargs):
        super().__init__(**kwargs)
        self.exampler_size = exampler_size
        self.size_per_class = 0  # how many image per class in the exampler
        self.curent_datasets = None  # dict of training, validation, testing dataset

    def _before_task(self):
        # adding new task and rebuild the datasets
        self.current_task += 1
        self.current_datasets = self.build_dataset()
        # update the info for exampler
        self.current_num_class += self.calss_per_task
        self.size_per_class = self.exampler_size // self.current_num_class

    def _train_task(self):
        LOGGER.info(f'Create the trainer of task{self.current_task + 1}.')

        train_loader = self.build_dataloader(self.current_dataset, 'train')
        valid_loader = self.build_dataloader(self.current_dataset, 'valid')

        kwargs = {
            'device': self.device,
            'train_dataloader': train_loader,
            'valid_dataloader': valid_loader,
            'net': self.net,
            'loss_fns': self.loss_fns,
            'loss_weights': self.loss_weights,
            'metric_fns': self.metric_fns,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            'logger': self.logger,
            'monitor': self.monitor
        }
        self.config.trainer.kwargs.update(kwargs)
        trainer = _get_instance(src.runner.trainers, self.config.trainer)
        if self.net_loaded_path is None:
            logging.info('Start training.')
        else:  # if the parameters of net should be loaded
            logging.info(f'Load the previous checkpoint from "{self.net_loaded_path}".')
            trainer.load(Path(self.net_loaded_path))
            logging.info('Resume training.')
        trainer.train()

        LOGGER.info(f'End training of task{self.current_task + 1}.')

    def _after_task(self, data_loader):
        pass

    def _eval_task(self, data_loader):
        test_loader = self.build_dataloader(self.current_dataset, 'test')

