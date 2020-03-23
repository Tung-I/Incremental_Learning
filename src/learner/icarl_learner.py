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

    def _before_task(self):
        # update the info for exampler
        self.size_per_class = self.exampler_size // self.current_num_class

    def _train_task(self, trainer):
        trainer.train()

        LOGGER.info(f'End training of task{self.current_task + 1}.')

    def _after_task(self):
        pass

    def _eval_task(self):
        pass
        

