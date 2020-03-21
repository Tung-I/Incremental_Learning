from torchvision.utils import make_grid

from src.callbacks.loggers import BaseLogger


class AcdcSegLogger(BaseLogger):
    """The ACDC logger for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_images(self, epoch, train_batch, train_output, valid_batch, valid_output):
        """Plot the visualization results.
        Args:
            epoch (int): The number of trained epochs.
            train_batch (dict): The training batch.
            train_output (torch.Tensor): The training output.
            valid_batch (dict): The validation batch.
            valid_output (torch.Tensor): The validation output.
        """
        """
        num_classes = train_output.size(1)
        train_img = make_grid(train_batch['input'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        train_label = make_grid(train_batch['target'].float(), nrow=1, normalize=True,
                                scale_each=True, range=(0, num_classes - 1), pad_value=1)
        train_pred = make_grid(train_output.argmax(dim=1, keepdim=True).float(), nrow=1, normalize=True,
                               scale_each=True, range=(0, num_classes - 1), pad_value=1)
        valid_img = make_grid(valid_batch['input'], nrow=1, normalize=True, scale_each=True, pad_value=1)
        valid_label = make_grid(valid_batch['target'].float(), nrow=1, normalize=True,
                                scale_each=True, range=(0, num_classes - 1), pad_value=1)
        valid_pred = make_grid(valid_output.argmax(dim=1, keepdim=True).float(), nrow=1, normalize=True,
                               scale_each=True, range=(0, num_classes - 1), pad_value=1)

        train_grid = torch.cat((train_img, train_label, train_pred), dim=-1)
        valid_grid = torch.cat((valid_img, valid_label, valid_pred), dim=-1)
        self.writer.add_image('train', train_grid)
        self.writer.add_image('valid', valid_grid)
        """
        pass
