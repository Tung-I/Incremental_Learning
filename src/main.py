import argparse
import copy
import logging
import random
import torch
from box import Box
from pathlib import Path
from shutil import copyfile

import src


def main(args):
    logging.info(f'Load the config from "{args.config_path}".')
    config = Box.from_yaml(filename=args.config_path)
    saved_dir = Path(config.main.saved_dir)
    if not saved_dir.is_dir():
        saved_dir.mkdir(parents=True)

    logging.info(f'Save the config to "{saved_dir}".')
    copyfile(args.config_path, saved_dir / 'config.yaml')

    if not args.test:

        random_seed = config.main.get('random_seed')
        if random_seed is None:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        else:
            logging.info('Make the experiment results deterministic.')
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logging.info('Create the device.')
        if 'cuda' in config.trainer.kwargs.device and not torch.cuda.is_available():
            raise ValueError("The cuda is not available. Please set the device to 'cpu'.")
        device = torch.device(config.trainer.kwargs.device)

        logging.info('Create the network architecture.')
        config.net.setdefault('kwargs', {}).update(device=device)
        net = _get_instance(src.model.nets, config.net).to(device)

        logging.info('Create the loss functions and corresponding weights.')
        loss_fns, loss_weights = [], []
        defaulted_loss_fns = [loss_fn for loss_fn in dir(torch.nn) if 'Loss' in loss_fn]
        for config_loss in config.losses:
            if config_loss.name in defaulted_loss_fns:
                loss_fn = _get_instance(torch.nn, config_loss)
            else:
                loss_fn = _get_instance(src.model.losses, config_loss)
            loss_fns.append(loss_fn)
            loss_weights.append(config_loss.weight)

        logging.info('Create the metric functions.')
        metric_fns = [_get_instance(src.model.metrics, config_metric) for config_metric in config.metrics]

        logging.info('Create the learner.')
        kwargs = {
            'config': config,
            'saved_dir': saved_dir,
            'device': device,
            'net': net,
            'loss_fns': loss_fns,
            'loss_weights': loss_weights,
            'metric_fns': metric_fns,
        }
        config.learner.kwargs.update(kwargs)
        learner = _get_instance(src.learner, config.learner)

        learner.learn()


class Base:
    """The Base class for easy debugging.
    """
    def __getattr__(self, name):
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                             f"Its attributes: {list(self.__dict__.keys())}.")


def _parse_args():
    parser = argparse.ArgumentParser(description="The main pipeline script.")
    parser.add_argument('config_path', type=Path, help='The path of the config file.')
    parser.add_argument('--test', action='store_true',
                        help='Perform testing if specified; otherwise perform training.')
    args = parser.parse_args()
    return args


def _get_instance(module, config, *args):
    """
    Args:
        module (MyClass): The defined module (class).
        config (Box): The config to create the class instance.

    Returns:
        instance (MyClass): The defined class instance.
    """
    cls = getattr(module, config.name)
    return cls(*args, **config.get('kwargs', {}))


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s | %(name)-16s | %(levelname)-8s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    args = _parse_args()
    main(args)
