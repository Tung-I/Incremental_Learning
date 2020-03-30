import torch.nn.functional as F
import torch
from src.runner.trainers import BaseTrainer


class ICaRLTrainer(BaseTrainer):
    """The ICaRL trainer for the segmentation task.
    """

    def __init__(self, meta_data=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_data = meta_data

    def _train_step(self, batch):
        inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
        outputs = self.net(inputs)
        targets_onehot = torch.zeros_like(outputs).scatter_(1, targets, 1.) # (N, C+i)

        old_model = self.meta_data['old_model']
        class_per_task = self.meta_data['class_per_task']

        if old_model is None:
            losses = [loss(outputs, targets_onehot) for loss in self.loss_fns]
        else:
            old_targets = torch.sigmoid(old_model(inputs).detach()) # (N, C)
            new_targets = targets_onehot.clone()
            new_targets[..., :-class_per_task] = old_targets
            losses = [loss(outputs, new_targets) for loss in self.loss_fns]

        # metrics = [metric(outputs, targets) for metric in self.metric_fns]
        _losses = {}
        for i, loss_fn in enumerate(self.loss_fns):
            _losses[loss_fn.get_name()] = losses[i]

        loss = (torch.stack(losses) * self.loss_weights).sum()

        metrics = {metric.get_name():metric(outputs, targets) for metric in self.metric_fns}
        losses = _losses


        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': outputs
        }

    def _valid_step(self, batch):
        inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
        outputs = self.net(inputs)
        targets_onehot = torch.zeros_like(outputs).scatter_(1, targets, 1.) # (N, C+i)

        old_model = self.meta_data['old_model']
        class_per_task = self.meta_data['class_per_task']
        
        if old_model is None:
            losses = [loss(outputs, targets_onehot) for loss in self.loss_fns]
        else:
            old_targets = torch.sigmoid(old_model(inputs).detach()) # (N, C)
            new_targets = targets_onehot.clone()
            new_targets[..., :-class_per_task] = old_targets
            losses = [loss(outputs, new_targets) for loss in self.loss_fns]

        # metrics = [metric(outputs, targets) for metric in self.metric_fns]
        _losses = {}
        for i, loss_fn in enumerate(self.loss_fns):
            _losses[loss_fn.get_name()] = losses[i]

        loss = (torch.stack(losses) * self.loss_weights).sum()

        metrics = {metric.get_name():metric(outputs, targets) for metric in self.metric_fns}
        losses = _losses

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': outputs
        }