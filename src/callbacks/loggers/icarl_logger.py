from torchvision.utils import make_grid

from src.callbacks.loggers import BaseLogger


class ICaRLLogger(BaseLogger):
    """The ACDC logger for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        epoch += self.add_epoch
        return
