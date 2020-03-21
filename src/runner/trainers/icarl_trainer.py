import torch.nn.functional as F

from src.runner.trainers import BaseTrainer


class ICaRLTrainer(BaseTrainer):
    """The ICaRL trainer for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = self.net(input)

        losses = [loss(output, target) for loss in self.loss_fns]
        metrics = [metric(output, target) for metric in self.metric_fns]

        loss = (torch.stack(losses) * self.loss_weights).sum()

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': output
        }

    def _valid_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = self.net(input)


        losses = [loss(output, target) for loss in self.loss_fns]
        metrics = [metric(output, target) for metric in self.metric_fns]

        loss = (torch.stack(losses) * self.loss_weights).sum()

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': output
        }