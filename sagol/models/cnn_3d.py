import math
from random import shuffle

import logbook
import torch
import torch.nn as nn

DEFAULT_BATCH_SIZE = 32

REQUIRED_PARAMS = ['kernel_size', 'filters', 'max_pool_size', 'max_pool_every', 'batch_norm_every', 'hidden_dimensions',
                   'out_dimension', 'hidden_dimensions_of_regressor', 'learning_rate', 'eps', 'beta_1', 'beta_2',
                   'weight_decay', 'amsgrad', 'loss_function', 'batch_size', 'epochs', 'early_stopping']

logger = logbook.Logger(__name__)


# This PyTorch based convolutional network can be used for regression or 0-1 classification
# Architecture of the network: [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear -> (Linear -> ReLU)*K -> Linear
class ConvNet(nn.Module):

    # in_size: size of input images, e.g. (85, 101, 65)

    # out_dim: output dim of fully connected layers, e.g. 10

    # contrast_dim: contrast dim, which is added to the out_dim before regression/classification part of the network, e.g. 31

    # kernel_size: f, every conv layer uses fxfxf filter

    # filters: a list of length N, containing the number of filters in each conv layer

    # max_pool_size: S, each max pool layer uses SxSxS filter

    # max_pool_every: P, the number of conv layers before each max-pool layer

    # batch_norm_every: B, the number of conv layers before each batch-norm layer

    # hidden_dims: a list of length M, containing the hidden dimensions of each linear layer in the fully connected part of the
    # network (other than the output layer)

    # hidden_dims_regressor: if the network is a regressor, this K-length list contains the hidden dimensions of each linear layer
    # in the regression part of the network (other than the output layer)

    # hidden_dims_classifier: if the network is a 0-1 classifier, this K-length list contains the hidden dimensions of each linear
    # layer in the classification part of the network (other than the output layer)
    def __init__(self, out_dim, kernel_size, filters, max_pool_size, max_pool_every, batch_norm_every, hidden_dims,
                 contrast_dim, in_size=(85, 101, 65), hidden_dims_regressor=None, hidden_dims_classifier=None):
        super().__init__()
        self.in_size = in_size
        self.kernel_size = kernel_size
        self.out_dim = out_dim
        self.filters = filters
        self.max_pool_size = max_pool_size
        self.max_pool_every = max_pool_every
        self.batch_norm_every = batch_norm_every
        self.hidden_dims = hidden_dims
        self.hidden_dims_regressor = hidden_dims_regressor
        self.hidden_dims_classifier = hidden_dims_classifier
        self.contrast_dim = contrast_dim

        self.feature_extractor = self.make_feature_extractor()
        self.fully_connected = self.make_fully_connected()

        if not ((hidden_dims_regressor is None) ^ (hidden_dims_classifier is None)):
            raise RuntimeError(
                'Network must be exactly one of the two: a regressor or a classifier! Exactly one of the '
                'fields: hidden_dims_regressor and hidden_dims_classifier has to be None!')

        if hidden_dims_regressor is not None:
            self.is_reg = True
            self.regressor = self.make_regressor()
        if hidden_dims_classifier is not None:
            self.is_reg = False
            self.classifier = self.make_classifier()

    def make_feature_extractor(self):
        in_h, in_w, in_d = tuple(self.in_size)
        layers = []
        new_h, new_w, new_d = in_h, in_w, in_d
        last_filter = 1
        for i in range(len(self.filters)):
            layers.append(nn.Conv3d(last_filter, self.filters[i], kernel_size=self.kernel_size, stride=1,
                                    padding=(self.kernel_size - 1) // 2))
            last_filter = self.filters[i]
            layers.append(nn.ReLU())
            if (i + 1) % self.max_pool_every == 0:
                layers.append(nn.MaxPool3d(kernel_size=self.max_pool_size))
                new_h, new_w, new_d = math.ceil(new_h // self.max_pool_size), \
                                      math.ceil(new_w // self.max_pool_size), \
                                      math.ceil(new_d // self.max_pool_size)
            if (i + 1) % self.batch_norm_every == 0:
                layers.append(nn.BatchNorm3d(last_filter))
        self.out_h = new_h
        self.out_w = new_w
        self.out_d = new_d
        seq = nn.Sequential(*layers)
        return seq

    def make_fully_connected(self):
        layers = []
        last_d = self.out_h * self.out_w * self.out_d * self.filters[-1]
        for d in self.hidden_dims:
            layers.append(nn.Linear(last_d, d))
            last_d = d
            layers.append(nn.ReLU())
        layers.append(nn.Linear(last_d, self.out_dim))
        layers.append(nn.ReLU())
        seq = nn.Sequential(*layers)
        return seq

    def make_regressor(self):
        layers = []
        last_d = self.out_dim + self.contrast_dim
        for d in self.hidden_dims_regressor:
            layers.append(nn.Linear(last_d, d))
            last_d = d
            layers.append(nn.ReLU())
        layers.append(nn.Linear(last_d, 1))

        seq = nn.Sequential(*layers)
        return seq

    def make_classifier(self):
        layers = []
        last_d = self.out_dim + self.contrast_dim + 1
        for d in self.hidden_dims_classifier:
            layers.append(nn.Linear(last_d, d))
            last_d = d
            layers.append(nn.ReLU())
        layers.append(nn.Linear(last_d, 1))
        layers.append(nn.Sigmoid())

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # assert x dim
        x_0 = torch.cat(tuple(item[0].unsqueeze(0).unsqueeze(0) for item in x))
        x_1 = torch.cat(tuple(item[1].unsqueeze(0).unsqueeze(0) for item in x))
        out = self.feature_extractor(x_0)
        out = self.fully_connected(out.view(out.size(0), -1))

        if self.is_reg:
            reg_input = torch.cat((out.unsqueeze(1), x_1), axis=2)
            out_score = self.regressor(reg_input).reshape(-1)
        else:
            x_2 = torch.cat(tuple(item[2].unsqueeze(0).unsqueeze(0) for item in x))
            classifier_input = torch.cat((out.unsqueeze(0), x_1, x_2))
            out_score = self.classifier(classifier_input).reshape(-1)
        return out_score

    def fit(self, X, y, optimizer, loss_func, epochs, batch_size, early_stopping=-1, print_loss_every=-1):
        if optimizer is not None:
            self.optimizer = optimizer
        train_loss = []
        epochs_not_improved = 0
        epochs_completed = 0
        min_loss = math.inf
        if early_stopping <= 0:
            early_stopping = epochs
        if print_loss_every <= 0:
            print_loss_every = epochs + 1

        for epoch in range(epochs):
            total_loss = 0
            X_shuffled, y_shuffled = get_shuffled_X_y(X, y)
            num_batches = 0
            for i in range(0, len(X_shuffled), batch_size):
                num_batches += 1
                finish_idx = i + batch_size if i + batch_size <= len(X_shuffled) else len(X_shuffled)
                X_batch = X_shuffled[i:finish_idx]
                y_batch = y_shuffled[i:finish_idx]
                self.optimizer.zero_grad()
                predictions = self(X_batch)
                y_true = torch.Tensor(y_batch)
                loss = loss_func(predictions, y_true)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            loss = total_loss / num_batches

            epochs_completed += 1
            if (epoch + 1) % print_loss_every == 0:
                print("epoch " + str(epoch + 1) + ' loss: ' + str(loss))
            train_loss.append(loss)

            # Check early stopping
            if loss < min_loss:
                min_loss = loss
                epochs_not_improved = 0
            else:
                epochs_not_improved += 1
                if epochs_not_improved == early_stopping:
                    break
        return train_loss, epochs_completed

    def predict(self, X, batch_size=-1):
        with torch.no_grad():
            if batch_size > 0:
                ten = torch.Tensor()
                for i in range(0, len(X), batch_size):
                    finish_idx = i + batch_size if i + batch_size <= len(X) else len(X)
                    ten = torch.cat(
                        (ten, self(X[i:finish_idx]) if self.is_reg else torch.Tensor(
                            [0 if y < 0.5 else 1 for y in self(X)])))
                return ten
            else:
                return self(X) if self.is_reg else torch.Tensor([0 if y < 0.5 else 1 for y in self(X)])

    def score(self, X, y, batch_size=-1):
        y = torch.Tensor(y)
        if self.is_reg:
            y_pred = self.predict(X, batch_size)
            u = ((y - y_pred) ** 2).sum().item()
            v = ((y - y.mean()) ** 2).sum().item()
            return 1 - u / v
        return torch.mean([1 if dif == 0 else 0 for dif in (self.predict(X, batch_size) - y)]).item()


def train_cnn_3d(X_train, y_train, **kwargs):
    if not all(param in kwargs for param in REQUIRED_PARAMS):
        logger.info(f'Not all required params for cnn were passed. Required params: {REQUIRED_PARAMS}')
        return None, None

    # We supply these params on our own, so these will always exist.
    in_size = kwargs['in_size']
    contrast_dim = kwargs['contrast_dim']

    # User supplied params.
    kernel_size = kwargs['kernel_size']
    filters = kwargs['filters']
    max_pool_size = kwargs['max_pool_size']
    max_pool_every = kwargs['max_pool_every']
    batch_norm_every = kwargs['batch_norm_every']
    hidden_dims = kwargs['hidden_dimensions']
    out_dim = kwargs['out_dimension']
    hidden_dims_regressor = kwargs['hidden_dimensions_of_regressor']
    lr = kwargs['learning_rate']
    eps = kwargs['eps']
    beta_1 = kwargs['beta_1']
    beta_2 = kwargs['beta_2']
    weight_decay = kwargs['weight_decay']
    amsgrad = kwargs['amsgrad']
    loss_func = kwargs['loss_function']
    batch_size = kwargs['batch_size']
    epochs = kwargs['epochs']
    early_stopping = kwargs['early_stopping']

    cnn = ConvNet(out_dim=out_dim, kernel_size=kernel_size, filters=filters, max_pool_size=max_pool_size,
                  max_pool_every=max_pool_every, batch_norm_every=batch_norm_every, hidden_dims=hidden_dims,
                  contrast_dim=contrast_dim, in_size=in_size, hidden_dims_regressor=hidden_dims_regressor,
                  hidden_dims_classifier=None)
    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=lr, betas=(beta_1, beta_2), eps=eps,
                                 weight_decay=weight_decay, amsgrad=amsgrad)
    train_loss, kwargs['epochs_completed'] = cnn.fit(X=X_train, y=y_train, optimizer=optimizer, loss_func=loss_func,
                                                     epochs=epochs,
                                                     batch_size=batch_size, early_stopping=early_stopping,
                                                     print_loss_every=-1)
    del kwargs['in_size']
    del kwargs['contrast_dim']
    return cnn, kwargs


def get_shuffled_X_y(X, y):
    combined = list(zip(X, y))
    shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)
    return list(X_shuffled), list(y_shuffled)
