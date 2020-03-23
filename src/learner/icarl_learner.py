import logging
import copy
from torch.nn import functional as F
from tqdm import tqdm

from src.learner import IncrementalLearner

LOGGER = logging.getLogger(__name__.split('.')[-1])

EPSILON = 1e-8

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

        self._examplars = {}
        self._means = None
        self._herding_matrix = []


    def _before_task(self):
        # update the info for exampler
        self.size_per_class = self.exampler_size // self.current_num_class

    def _train_task(self, trainer):
        self.trainer.train()

    def _after_task(self):
        self.build_examplars(self.current_dataset)
        self.old_model = self.net.copy().freeze()

    def _eval_task(self):
        ypred, ytrue = compute_accuracy(self._network, data_loader, self._class_means)
        
    def build_examplars(self):
        self._data_memory, self._targets_memory = [], []
        self._class_means = np.zeros((self.total_num_class, self.net.features_dim()))

        config = copy.deepcopy(self.config)

        for class_idx in range(self.total_num_class):
            # dataset
            config.dataset.setdefault('kwargs', {}).update(type_='Training')
            config.dataset.setdefault('kwargs', {}).update(chosen_index=[self.class_order[class_idx]])
            config.dataset.setdefault('kwargs', {}).update(chosen_label=[class_idx])
            _dataset = _get_instance(src.data.datasets, config.dataset)
            # loader
            config.dataloader.setdefault('kwargs', {}).update(batch_size=1)
            _loader = _get_instance(src.data.dataloader, config_dataloader, _dataset)

            features, targets, inputs = extract_features(self.net, _loader)
            #####
            features_flipped = copy.deepcopy(features)
            #####
            if class_idx >= (self.total_num_class - self.class_per_task):
                self._herding_matrix.append(select_examplars(features, self.size_per_class))

            examplar_mean, alph = compute_examplar_mean(features, features_flipped, self._herding_matrix[class_idx], self.size_per_class)
            self._data_memory.append(inputs[np.where(alph == 1)[0]])
            self._targets_memory.append(targets[np.where(alph == 1)[0]])

            self._class_means[class_idx, :] = examplar_mean

        self._data_memory = np.concatenate(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)

    def extract_features(model, loader):
        targets, features, inputs = [], [], []
        loader_iterator = iter(loader)
        for i in range(len(loader)):
            batch = next(loader_iterator)
            _inputs, _targets = batch['inputs'].to(self.device), batch['targets']
            _targets = _targets.numpy()
            _features = model.extract(_inputs).detach().cpu().numpy()

            features.append(_features)
            targets.append(_targets)
            inputs.append(_inputs.detach().cpu().numpy())
        return np.concatenate(features), np.concatenate(targets), np.concatenate(inputs)

    def select_examplars(features, nb_max):
        D = features.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)
        mu = np.mean(D, axis=1)
        herding_matrix = np.zeros((features.shape[0],))

        w_t = mu
        iter_herding, iter_herding_eff = 0, 0

        while not (
            np.sum(herding_matrix != 0) == min(nb_max, features.shape[0])
        ) and iter_herding_eff < 1000:
            tmp_t = np.dot(w_t, D)
            ind_max = np.argmax(tmp_t)
            iter_herding_eff += 1
            if herding_matrix[ind_max] == 0:
                herding_matrix[ind_max] = 1 + iter_herding
                iter_herding += 1

            w_t = w_t + mu - D[:, ind_max]

        return herding_matrix

    def compute_examplar_mean(feat_norm, feat_flip, herding_mat, nb_max):
        D = feat_norm.T
        D = D / (np.linalg.norm(D, axis=0) + EPSILON)

        D2 = feat_flip.T
        D2 = D2 / (np.linalg.norm(D2, axis=0) + EPSILON)

        alph = herding_mat
        alph = (alph > 0) * (alph < nb_max + 1) * 1.

        alph_mean = alph / np.sum(alph)

        mean = (np.dot(D, alph_mean) + np.dot(D2, alph_mean)) / 2
        mean /= np.linalg.norm(mean)

        return mean, alph


def _get_instance(module, config, *args):
    cls = getattr(module, config.name)
    return cls(*args, **config.get('kwargs', {}))
