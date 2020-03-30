import logging
import copy
import json
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
from scipy.spatial.distance import cdist

from src.learner import IncrementalLearner
import src

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
        self._class_means = None
        self._herding_matrix = []


    def _before_task(self):
        # update the info for examplar
        self.size_per_class = self.exampler_size // self.current_num_class

    def _train_task(self):
        self.trainer.train()

    def _after_task(self):
        self.build_examplars()
        self.old_model = self.net.copy().freeze()

    def _eval_task(self):
        ypred, ytrue = self.compute_ypred_ytrue(self.net, self.test_loader, self._class_means)
        # all_acc = self.compute_accuracy(ypred, ytrue)
        ypred = np.expand_dims(ypred, axis=1)
        output_path = '/home/tony/Desktop/' + str(self.current_task) +'_pred.npy' 
        np.save(output_path, ypred)
        output_path = '/home/tony/Desktop/' + str(self.current_task) +'_target.npy' 
        np.save(output_path, ytrue)
        # with open(output_path, 'w') as output_json_file:
        #     json.dump(to_save, output_json_file)
        
    def build_examplars(self):
        LOGGER.info('Build examplar')
        self._data_memory, self._targets_memory = [], []
        self._class_means = np.zeros((self.current_num_class, self.net.features_dim))

        config = copy.deepcopy(self.config)

        for class_idx in tqdm(range(self.current_num_class)):
            # dataset
            config.dataset.setdefault('kwargs', {}).update(type_='Training')
            config.dataset.setdefault('kwargs', {}).update(chosen_index=[self.class_order[class_idx]])
            config.dataset.setdefault('kwargs', {}).update(chosen_label=[class_idx])
            _dataset = _get_instance(src.data.datasets, config.dataset)
            # loader
            config.dataloader.setdefault('kwargs', {}).update(batch_size=1)
            _loader = _get_instance(src.data.dataloader, config.dataloader, _dataset)

            features, targets, inputs = self.extract_features(self.net, _loader)
            #####
            features_flipped = copy.deepcopy(features)
            #####
            if class_idx >= (self.current_num_class - self.class_per_task):
                self._herding_matrix.append(self.select_examplars(features, self.size_per_class))
            # print(f'{self.current_num_class}, {self.class_per_task}')
            # print(len(self._herding_matrix))
            examplar_mean, alph = self.compute_examplar_mean(features, features_flipped, self._herding_matrix[class_idx], self.size_per_class)
            self._data_memory.append(inputs[np.where(alph == 1)[0]])
            self._targets_memory.append(targets[np.where(alph == 1)[0]])

            self._class_means[class_idx, :] = examplar_mean

        self._data_memory = np.concatenate(self._data_memory)
        self._targets_memory = np.concatenate(self._targets_memory)

    def extract_features(self, model, loader):
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

    def select_examplars(self, features, nb_max):
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

    def compute_examplar_mean(self, feat_norm, feat_flip, herding_mat, nb_max):
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

    def compute_ypred_ytrue(self, model, loader, class_means):
        features, targets_, _ = self.extract_features(model, loader)

        targets = np.zeros((targets_.shape[0], 100), np.float32)
        targets[range(len(targets_)), targets_.astype('int32')] = 1.
        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T

        # Compute score for iCaRL
        sqd = cdist(class_means, features, 'sqeuclidean')
        score_icarl = (-sqd).T

        return np.argsort(score_icarl, axis=1)[:, -1], targets_

    def compute_accuracy(self, ypred, ytrue):
        task_size = self.class_per_task
        all_acc = {}

        all_acc["total"] = round((ypred == ytrue).sum() / len(ytrue), 3)

        for class_id in range(0, np.max(ytrue), task_size):
            idxes = np.where(
                    np.logical_and(ytrue >= class_id, ytrue < class_id + task_size)
            )[0]

            label = "{}-{}".format(
                    str(class_id).rjust(2, "0"),
                    str(class_id + task_size - 1).rjust(2, "0")
            )
            all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

        return all_acc


def _get_instance(module, config, *args):
    cls = getattr(module, config.name)
    return cls(*args, **config.get('kwargs', {}))
