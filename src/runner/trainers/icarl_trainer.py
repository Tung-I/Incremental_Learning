import torch.nn.functional as F
import torch
from src.runner.trainers import BaseTrainer


class ICaRLTrainer(BaseTrainer):
    """The ICaRL trainer for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _train_step(self, batch, meta_data):
        inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
        outputs = self.net(inputs)
        # one hot encoding
        targets = torch.zeros_like(outputs).scatter_(1, targets, 1.) # (N, C)

        old_model = meta_data['old_model']
        class_per_task = meta_data['class_per_task']

        if old_model is None
            losses = [loss(outputs, targets) for loss in self.loss_fns]
        else
            old_targets = torch.sigmoid(old_model(inputs).detach())
            new_targets = targets.clone()
            new_targets[..., :-class_per_task] = old_targets
            losses = [loss(outputs, new_targets) for loss in self.loss_fns]

        metrics = [metric(outputs, targets) for metric in self.metric_fns]

        loss = (torch.stack(losses) * self.loss_weights).sum()

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': outputs
        }

    def _valid_step(self, batch, meta_data):
        inputs, targets = batch['inputs'].to(self.device), batch['targets'].to(self.device)
        outputs = self.net(inputs)
        # one hot encoding
        targets = torch.zeros_like(outputs).scatter_(1, targets, 1.) # (N, C)

        old_model = meta_data['old_model']
        class_per_task = meta_data['class_per_task']
        
        if old_model is None
            losses = [loss(outputs, targets) for loss in self.loss_fns]
        else
            old_targets = torch.sigmoid(old_model(inputs).detach())
            new_targets = targets.clone()
            new_targets[..., :-class_per_task] = old_targets
            losses = [loss(outputs, new_targets) for loss in self.loss_fns]
            
        metrics = [metric(outputs, targets) for metric in self.metric_fns]

        loss = (torch.stack(losses) * self.loss_weights).sum()

        return {
            'loss': loss,
            'losses': losses,
            'metrics': metrics,
            'outputs': outputs
        }