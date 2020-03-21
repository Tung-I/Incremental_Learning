from torch.nn import functional as F
from tqdm import tqdm

from src.learner import IncrementalLearner

class ICaRLLearner(IncrementalLearner):
    """The base incremental learner.
    Methods should be implemented:
        _train_task()
        _eval_task()

    Optional:
        _before_task()
        _after_task()
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)