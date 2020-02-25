import torch
from torch import nn
from backbone import densenet, resnet


class E2ENet(nn.Module):

    def __init__(self, backbone_type=None, bias=True, init_type="kaiming", use_multi_fc=False, device=None):
        super(E2ENet, self).__init__()

        self.convnet = get_convnet(backbone_type, nf=64, zero_init_residual=True)
        self.bias = bias
        self.init_type = init_type
        self.use_multi_fc = use_multi_fc

        self.classifier = None

        self.n_classes = 0
        self.device = device

        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("No classifier.")

        features = self.convnet(x)

        if self.use_multi_fc:
            logits = []
            for clf_name in self.classifier:
                logits.append(self.__getattr__(clf_name)(features))
            logits = torch.cat(logits, 1)
        else:
            logits = self.classifier(features)

        return logits

    def features_dim(self):
        return self.convnet.out_dim

    def extract(self, x):
        return self.convnet(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        if self.use_multi_fc:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.classifier is None:
            self.classifier = []

        new_classifier = self._gen_classifier(n_classes)
        name = "_clf_{}".format(len(self.classifier))
        self.__setattr__(name, new_classifier)
        self.classifier.append(name)

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.n_classes + n_classes)

        if self.classifier is not None:
            classifier.weight.data[:self.n_classes] = weight
            if self.bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, n_classes):
        classifier = nn.Linear(self.convnet.out_dim, n_classes, bias=self.bias).to(self.device)
        if self.init == "kaiming":
            nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
        if self.bias:
            nn.init.constant_(classifier.bias, 0.)

        return classifier
