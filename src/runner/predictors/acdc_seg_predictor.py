import nibabel as nib
import torch.nn.functional as F

from src.runner.predictors import BasePredictor


class AcdcSegPredictor(BasePredictor):
    """The ACDC predictor for the segmentation task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.test_dataloader.batch_size != 1:
            raise ValueError(f'The testing batch size should be 1. Got {self.test_dataloader.batch_size}.')

        self.output_dir = self.saved_dir / 'prediction'
        if not self.output_dir.is_dir():
            self.output_dir.mkdir(parents=True)

    def _test_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target'].to(self.device)
        output = F.interpolate(self.net(input),
                               size=target.size()[2:],
                               mode='trilinear',
                               align_corners=False)

        cross_entropy_loss = self.loss_fns.cross_entropy_loss(output, target.squeeze(dim=1))
        dice_loss = self.loss_fns.dice_loss(F.softmax(output, dim=1), target)
        loss = (self.loss_weights.cross_entropy_loss * cross_entropy_loss +
                self.loss_weights.dice_loss * dice_loss)
        dice = self.metric_fns.dice(F.softmax(output, dim=1), target)

        (affine,), (name,) = batch['affine'], batch['name']
        pred = F.softmax(output, dim=1).argmax(dim=1).squeeze(dim=0).permute(1, 2, 0).contiguous()
        nib.save(nib.Nifti1Image(pred.cpu().numpy(), affine.numpy()), (self.output_dir / name).as_posix())
        return {
            'loss': loss,
            'losses': {
                'CrossEntropyLoss': cross_entropy_loss,
                'DiceLoss': dice_loss
            },
            'metrics': {
                'DiceRightVentricle': dice[1],
                'DiceMyocardium': dice[2],
                'DiceLeftVentricle': dice[3]
            }
        }

    """ For the testing dataset.
    def _test_step(self, batch):
        input, target = batch['input'].to(self.device), batch['target']
        output = F.interpolate(self.net(input),
                               size=target.size()[2:],
                               mode='trilinear',
                               align_corners=False)
        import torch
        loss = torch.tensor([1])

        (affine,), (name,) = batch['affine'], batch['name']
        pred = F.softmax(output, dim=1).argmax(dim=1).squeeze(dim=0).permute(1, 2, 0).contiguous()
        nib.save(nib.Nifti1Image(pred.cpu().numpy(), affine.numpy()), (self.output_dir / name).as_posix())
        return {
            'loss': loss
        }
    """
