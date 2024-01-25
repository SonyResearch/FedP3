import torch
import copy
import math
from torch import nn, autograd
import numpy as np
from torch.utils.data import DataLoader
from utils.utils_dataset import Dataset
from models.Nets_VIB import KL_between_normals
import itertools
import torch.nn.functional as F
from typing import Any, Callable, Mapping, Sequence, Tuple

max_norm = 10


def add(target, source):
    for name in target:
        target[name].data += source[name].data.clone()


def add_mome(target, source, beta_):
    for name in target:
        target[name].data = (beta_ * target[name].data + source[name].data.clone())


def add_mome2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()


def add_mome3(target, source1, source2, source3, beta_1, beta_2, beta_3):
    for name in target:
        target[name].data = beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone() + beta_3 * \
                            source3[name].data.clone()


def add_2(target, source1, source2, beta_1, beta_2):
    for name in target:
        target[name].data += beta_1 * source1[name].data.clone() + beta_2 * source2[name].data.clone()


def scale(target, scaling):
    for name in target:
        target[name].data = scaling * target[name].data.clone()


def scale_ts(target, source, scaling):
    for name in target:
        target[name].data = scaling * source[name].data.clone()


def subtract(target, source):
    for name in target:
        target[name].data -= source[name].data.clone()


def subtract_(target, minuend, subtrahend):
    for name in target:
        target[name].data = minuend[name].data.clone() - subtrahend[name].data.clone()


def average(target, sources):
    for name in target:
        target[name].data = torch.mean(torch.stack([source[name].data for source in sources]), dim=0).clone()

def layer_wise_average(target, sources):
    # sources: {layer_key1: [client1, client2, ...], layer_key2: [client1, client2, ...]}
    for name in target:
        nums = len(sources[name])
        # sums = np.mean()
        target[name].data = torch.mean(torch.stack([mclient.W[name].data for mclient in sources[name]]), dim=0).clone()
        # target[name].data = torch.mean(torch.stack([source.W[name].data for source in sources]), dim=0).clone()

def weighted_average(target, sources, weights):
    for name in target:
        summ = torch.sum(weights)
        n = len(sources)
        modify = [weight / summ * n for weight in weights]
        target[name].data = torch.mean(torch.stack([m * source[name].data for source, m in zip(sources, modify)]),
                                       dim=0).clone()


def average_subset(target, sources, layers):
    """
    Aggregate only a subset of layers.

    :param target: The target dictionary where the aggregated weights will be stored.
    :param sources: A list of dictionaries, each representing a client's weights.
    :param layers: A list of layer names to be aggregated.
    """
    for name in layers:
        relevant_sources = [source[name].data for source in sources if name in source]
        if relevant_sources:  # Only update if there are any sources for this layer
            target[name].data = torch.mean(torch.stack(relevant_sources), dim=0).clone()

def weighted_mean(target, sources, layer_contributions):
    """
    Aggregate layers using weighted mean based on the number of layers each client contributes.

    :param target: The target dictionary where the weighted aggregated weights will be stored.
    :param sources: A list of tuples, each representing (client's weights, number of layers contributed).
    :param layer_contributions: A dictionary that counts the total contributions for each layer.
    """
    for layer_name in target.keys():
        # Initialize the aggregated layer as zero
        target[layer_name].data.zero_()
        total_contributions = layer_contributions[layer_name]

        # Accumulate weighted sum for the layer
        for source, num_layers in sources:
            if layer_name in source:
                weight = num_layers / total_contributions
                target[layer_name].data += source[layer_name].data * weight

def computer_norm(source1, source2):
    diff_norm = 0

    for name in source1:
        diff_source = source1[name].data.clone() - source2[name].data.clone()
        diff_norm += torch.pow(torch.norm(diff_source), 2)

    return (torch.pow(diff_norm, 0.5))


def majority_vote(target, sources, lr):
    for name in target:
        threshs = torch.stack([torch.max(source[name].data) for source in sources])
        mask = torch.stack([source[name].data.sign() for source in sources]).sum(dim=0).sign()
        target[name].data = (lr * mask).clone()


def get_mdl_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name, param in exp_mdl.named_parameters():
            n_par += len(param.data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def get_other_params(model_list, n_par=None):
    if n_par == None:
        exp_mdl = model_list[0]
        n_par = 0
        for name in exp_mdl:
            n_par += len(exp_mdl[name].data.reshape(-1))

    param_mat = np.zeros((len(model_list), n_par)).astype('float32')
    for i, mdl in enumerate(model_list):
        idx = 0
        for name in mdl:
            temp = mdl[name].data.cpu().numpy().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)
    return np.copy(param_mat)


def product_of_experts_two(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr  # Standard Deviation

    poe_var = torch.sqrt(torch.div((sigma_q ** 2 * sigma_p ** 2), (sigma_q ** 2 + sigma_p ** 2 + 1e-32)))

    poe_u = torch.div((mu_p * sigma_q ** 2 + mu_q * sigma_p ** 2), (sigma_q ** 2 + sigma_p ** 2 + 1e-32))

    return poe_u, poe_var


def product_of_experts(q_distr_set):
    mu_q_set, sigma_q_set = q_distr_set
    tmp1 = 1.0
    for i in range(len(mu_q_set)):
        tmp1 = tmp1 + (1.0 / (sigma_q_set[i] ** 2))
    poe_var = torch.sqrt(1.0 / tmp1)
    tmp2 = 0.0
    for i in range(len(mu_q_set)):
        tmp2 = tmp2 + torch.div(mu_q_set[i], sigma_q_set[i] ** 2)
    poe_u = torch.div(tmp2, tmp1)
    return poe_u, poe_var


'''
### DEFINE NETWORK-RELATED FUNCTIONS
def product_of_experts_two(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr    #Standard Deviation
    tmp1 = (1.0 / (sigma_q**2 + 1e-8)) + (1.0 / (sigma_p**2 + 1e-8))
    poe_var = torch.sqrt(1.0 / tmp1)
    tmp2 = torch.div(mu_q, sigma_q**2) + torch.div(mu_p, sigma_p**2)
    poe_u = torch.div(tmp2, tmp1)
    return poe_u, poe_var


def product_of_experts(q_distr_set):
    mu_q_set, sigma_q_set = q_distr_set
    tmp1 = 1.0
    for i in range(len(mu_q_set)):
        tmp1 = tmp1 + (1.0 / (sigma_q_set[i] ** 2 + 1e-8))
    poe_var = torch.sqrt(1.0 / tmp1)
    tmp2 = 0.0
    for i in range(len(mu_q_set)):
        tmp2 = tmp2 + torch.div(mu_q_set[i], sigma_q_set[i]**2)
    poe_u = torch.div(tmp2, tmp1)
    return poe_u, poe_var



def product_of_experts_copy(mask_, mu_set_, logvar_set_):
    tmp = 1.
    for m in range(len(mu_set_)):
        tmp += torch.reshape(mask_[:, m], [-1, 1]) * torch.div(1., torch.exp(logvar_set_[m]))
    poe_var = torch.div(1., tmp)
    poe_logvar = torch.log(poe_var)
    tmp = 0.
    for m in range(len(mu_set_)):
        tmp += torch.reshape(mask_[:, m], [-1, 1]) * torch.div(1., torch.exp(logvar_set_[m])) * mu_set_[m]
    poe_mu = poe_var * tmp
    return poe_mu, poe_logvar
'''


class DistributedTrainingDevice(object):
    '''
  A distributed training device (Client or Server)
  data : a pytorch dataset consisting datapoints (x,y)
  model : a pytorch neural net f mapping x -> f(x)=y_
  hyperparameters : a python dict containing all hyperparameters
  '''

    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()


class Client(DistributedTrainingDevice):

    def __init__(self, model, args, trn_x, trn_y, tst_x, tst_y, n_cls, dataset_name, alpha=0, id_num=0):
        super().__init__(model, args)

        self.trn_gen = DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                  batch_size=self.args.local_bs, shuffle=True)

        self.tst_x = tst_x
        self.tst_y = tst_y
        self.n_cls = n_cls
        self.id = id_num
        self.local_epoch = int(np.ceil(trn_x.shape[0] / self.args.local_bs))
        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}

        self.state_params_diff = 0.0
        self.train_loss = 0.0
        self.n_par = get_mdl_params([self.model]).shape[1]
        self.optimal_W = {name: value for name, value in self.model.named_parameters()}
        self.alpha = alpha
        self.skip_comm = False

        def h_initialization():
            self.h = copy.deepcopy(self.W)
            for name in self.W:
                self.h[name].data = 0 * self.W[name].data
            return self.h

        self.h = h_initialization()
        self.prob = args.prob

    def convex_combination(self, global_model, alpha=0.3):
        for name in self.W:
            self.W[name].data = alpha * self.optimal_W[name].data + (1 - alpha) * global_model[name].data

    def sparsity(self, weights, name, ratio=0.9):
        """
        This function takes a weight tensor and a ratio, and zeroes out all but the
        top `ratio` percentage of absolute values in the weights. It will not apply
        sparsity to bias terms.

        :param weights: A torch.Tensor of weights from a neural network layer.
        :param name: A string indicating the name of the layer the weights belong to.
        :param ratio: A float indicating the percentage of weights to keep.

        :return: A torch.Tensor of the same shape as weights, with only the top `ratio`
                 percentage of absolute values retained for weights, untouched for biases.
        """
        # Check if the tensor is a bias, if so, return it as-is without sparsification
        if 'bias' in name:
            print(f"{name} is a bias term and will not be sparsified.")
            return weights

        # Flatten the weight tensor to 1D
        flat_weights = weights.view(-1)

        # Calculate the number of weights to retain
        num_weights_to_retain = int(flat_weights.size(0) * ratio)

        # Use torch.topk to get the indices of the weights to retain
        _, indices_to_retain = torch.topk(flat_weights.abs(), k=num_weights_to_retain, largest=True)

        # Create a mask that will zero out all other weights
        mask = torch.zeros_like(flat_weights, dtype=torch.bool)
        mask[indices_to_retain] = True

        # Apply the mask to the flat weights and reshape to the original dimensions
        sparsified_weights = torch.zeros_like(flat_weights)
        sparsified_weights[mask] = flat_weights[mask]
        sparsified_weights = sparsified_weights.view_as(weights)

        # Optionally, log the layer name and the ratio of nonzero elements
        nonzero_ratio = torch.nonzero(sparsified_weights).size(0) / flat_weights.size(0)
        print(f"Sparsified {name}: retained {nonzero_ratio:.2%} of the weights")

        return sparsified_weights

    def synchronize_with_server(self, server, w_glob_keys, global_prune=1.0, w_global_keys_op=None):
        if self.args.method not in ['fedavg', 'ditto']:
            # Ensure w_global_keys_op is a set for faster look-up
            w_global_keys_op_set = set(w_global_keys_op) if w_global_keys_op is not None else set()

            for name in w_glob_keys:
                # Apply sparsity only to layers not in w_global_keys_op
                if name not in w_global_keys_op_set:
                    self.W[name].data = self.sparsity(server.W[name].data.clone(), name, global_prune)
                else:
                    # If the layer is in w_global_keys_op, simply copy it without applying sparsity
                    self.W[name].data = server.W[name].data.clone()
        else:
            self.model = copy.deepcopy(server.model)
            # self.W = {name: value for name, value in self.model.named_parameters()}
            self.W = {name: self.sparsity(value, name, global_prune) for name, value in self.model.named_parameters()}

    def compute_bias(self):
        if self.args.method == 'ditto':
            cld_mdl_param = torch.tensor(get_mdl_params([self.model], self.n_par)[0], dtype=torch.float32,
                                         device=self.args.device)
            self.state_params_diff = self.args.mu * (-cld_mdl_param)

    def train_cnn(self, w_glob_keys, server, last, skip_comm, compute_local_flag):

        self.model.train()

        if self.args.method == 'ditto':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay + self.args.mu)
        else:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                        weight_decay=self.args.weigh_delay)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1)

        local_eps = self.args.local_ep
        if last:
            if self.args.method == 'fedavg' or self.args.method == 'ditto':
                local_eps = self.args.last_local_ep
                if 'CIFAR100' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                elif 'CIFAR10' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                elif 'EMNIST' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1]]
                elif 'FMNIST' in self.args.dataset:
                    w_glob_keys = [self.model.weight_keys[i] for i in [0, 1, 2, 3]]
                w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))

            elif 'maml' in self.args.method:
                local_eps = self.args.last_local_ep / 2
                w_glob_keys = []
            else:
                local_eps = max(self.args.last_local_ep, self.args.local_ep - self.args.local_rep_ep)
        # all other methods update all parameters simultaneously
        else:
            for name, param in self.model.named_parameters():
                param.requires_grad = True

        # train and update
        epoch_loss = []

        if self.args.method == 'FedCR':

            self.dir_Z_u = torch.zeros(self.n_cls, 1, self.args.dimZ, dtype=torch.float32, device=self.args.device)
            self.dir_Z_sigma = torch.ones(self.n_cls, 1, self.args.dimZ, dtype=torch.float32, device=self.args.device)

            for iter in range(local_eps):

                if last:
                    for name, param in self.model.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                loss_by_epoch = []
                accuracy_by_epoch = []

                trn_gen_iter = self.trn_gen.__iter__()
                batch_loss = []

                for i in range(self.local_epoch):
                    dir_g_Z_u = torch.zeros(1, self.args.dimZ, dtype=torch.float32,
                                            device=self.args.device)
                    dir_g_Z_sigma = torch.ones(1, self.args.dimZ, dtype=torch.float32,
                                               device=self.args.device)

                    images, labels = trn_gen_iter.__next__()
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    labels = labels.reshape(-1).long()

                    batch_size = images.size()[0]

                    # prior_Z_distr_standard = torch.zeros(batch_size, self.args.dimZ).to(self.args.device), torch.ones(batch_size, self.args.dimZ).to(self.args.device)

                    for cls in range(len(labels)):
                        if cls == 0:
                            dir_g_Z_u = server.dir_global_Z_u[labels[cls]].clone().detach()
                            dir_g_Z_sigma = server.dir_global_Z_sigma[labels[cls]].clone().detach()
                        else:
                            dir_g_Z_u = torch.cat((dir_g_Z_u, server.dir_global_Z_u[labels[cls]].clone().detach()),
                                                  0).clone().detach()
                            dir_g_Z_sigma = torch.cat(
                                (dir_g_Z_sigma, server.dir_global_Z_sigma[labels[cls]].clone().detach()),
                                0).clone().detach()

                    prior_Z_distr = dir_g_Z_u, dir_g_Z_sigma

                    encoder_Z_distr, decoder_logits = self.model(images, self.args.num_avg_train)

                    decoder_logits_mean = torch.mean(decoder_logits, dim=0)

                    loss = nn.CrossEntropyLoss(reduction='none')
                    decoder_logits = decoder_logits.permute(1, 2, 0)
                    cross_entropy_loss = loss(decoder_logits, labels[:, None].expand(-1, self.args.num_avg_train))

                    # estimate E_{eps in N(0, 1)} [log q(y | z)]
                    cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)

                    I_ZX_bound = torch.mean(KL_between_normals(prior_Z_distr, encoder_Z_distr))
                    minusI_ZY_bound = torch.mean(cross_entropy_loss_montecarlo, dim=0)
                    total_loss = torch.mean(minusI_ZY_bound + self.args.beta * I_ZX_bound)

                    prediction = torch.max(decoder_logits_mean, dim=1)[1]
                    accuracy = torch.mean((prediction == labels).float())

                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                    optimizer.step()

                    loss_by_epoch.append(total_loss.item())
                    accuracy_by_epoch.append(accuracy.item())

                    if iter == local_eps - 1:
                        for cls in range(len(labels)):
                            if self.dir_Z_u[labels[cls]].equal(
                                    torch.zeros(1, self.args.dimZ, dtype=torch.float32, device=self.args.device)) and \
                                    self.dir_Z_sigma[labels[cls]].equal(
                                            torch.ones(1, self.args.dimZ, dtype=torch.float32,
                                                       device=self.args.device)):
                                self.dir_Z_u[labels[cls]], self.dir_Z_sigma[labels[cls]] = encoder_Z_distr[0][
                                    cls].clone().detach(), encoder_Z_distr[1][cls].clone().detach()
                            else:
                                q_distr = self.dir_Z_u[labels[cls]], self.dir_Z_sigma[labels[cls]]
                                encoder_Z_distr_cls = encoder_Z_distr[0][cls].clone().detach(), encoder_Z_distr[1][
                                    cls].clone().detach()
                                self.dir_Z_u[labels[cls]], self.dir_Z_sigma[labels[cls]] = product_of_experts_two(
                                    q_distr, encoder_Z_distr_cls)

                scheduler.step()
                epoch_loss.append(sum(loss_by_epoch) / len(loss_by_epoch))

        elif (self.args.method == 'Scafflix' and compute_local_flag) or (self.args.method == 'FLIX' and compute_local_flag):
            print(f'Accessed FLAG ######################################')
            for iter in range(local_eps):
                print(f'{iter}')
                for name, param in self.model.named_parameters():
                    param.requires_grad = True

                trn_gen_iter = self.trn_gen.__iter__()
                batch_loss = []

                for i in range(self.local_epoch):
                    images, labels = trn_gen_iter.__next__()
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    optimizer.zero_grad()
                    log_probs = self.model(images)
                    loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())

                    local_par_list = None
                    for param in self.model.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            # Initially nothing to concatenate
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    loss_algo = torch.sum(local_par_list * self.state_params_diff)

                    if self.args.method == 'ditto':
                        loss = loss_f_i + loss_algo
                    else:
                        loss = loss_f_i

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                    optimizer.step()
                    batch_loss.append(loss.item())

                scheduler.step()
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        elif self.args.method == 'FLIX':
            for iter in range(local_eps):
                for name, param in self.model.named_parameters():
                    param.requires_grad = True

                trn_gen_iter = self.trn_gen.__iter__()
                batch_loss = []

                for i in range(self.local_epoch):
                    images, labels = trn_gen_iter.__next__()
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    optimizer.zero_grad()
                    # convex combination
                    self.convex_combination(server.W, alpha=self.alpha)
                    log_probs = self.model(images)
                    loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())
                    loss = loss_f_i

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                    optimizer.step()
                    batch_loss.append(loss.item())

                scheduler.step()
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

            return sum(epoch_loss) / len(epoch_loss)

        elif self.args.method == 'Scafflix':
            for name, param in self.model.named_parameters():
                param.requires_grad = True

            trn_gen_iter = self.trn_gen.__iter__()
            batch_loss = []

            debug_flag = True
            for i in range(self.local_epoch):
                images, labels = trn_gen_iter.__next__()
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                optimizer.zero_grad()
                self.convex_combination(server.W, alpha=self.alpha)
                log_probs = self.model(images)
                loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())
                loss = loss_f_i
                loss.backward()
                for name, param in self.model.named_parameters():
                    # print(self.W[name].data[:5], self.args.lr, param.grad) # torch.Size([4, 1, 5, 5])
                    if debug_flag:
                        print(self.W[name].data[0, 0, 0, :4], param.grad[0, 0, 0, :4])
                        debug_flag = False
                    self.W[name].data = self.W[name].data - self.args.lr * (param.grad - self.h[name].data)
                    if not skip_comm:
                        self.h[name].data = self.h[name].data + self.args.prob / self.args.lr * (
                                    server.W[name].data - self.W[name].data)

                batch_loss.append(loss.item())

            scheduler.step()
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        else:
            for iter in range(local_eps):

                head_eps = local_eps - self.args.local_rep_ep
                # for FedRep, first do local epochs for the head
                if (iter < head_eps and self.args.method == 'fedrep') or last:
                    for name, param in self.model.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True

                # then do local epochs for the representation
                elif (iter == head_eps and self.args.method == 'fedrep') and not last:
                    for name, param in self.model.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                # all other methods update all parameters simultaneously
                elif self.args.method != 'fedrep':
                    for name, param in self.model.named_parameters():
                        param.requires_grad = True

                trn_gen_iter = self.trn_gen.__iter__()
                batch_loss = []

                for i in range(self.local_epoch):

                    images, labels = trn_gen_iter.__next__()
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    optimizer.zero_grad()
                    log_probs = self.model(images)
                    loss_f_i = self.loss_func(log_probs, labels.reshape(-1).long())

                    local_par_list = None
                    for param in self.model.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            # Initially nothing to concatenate
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    loss_algo = torch.sum(local_par_list * self.state_params_diff)

                    if self.args.method == 'ditto':
                        loss = loss_f_i + loss_algo
                    else:
                        loss = loss_f_i

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=max_norm)
                    optimizer.step()
                    batch_loss.append(loss.item())

                scheduler.step()
                epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return sum(epoch_loss) / len(epoch_loss)

    def compute_weight_update(self, w_glob_keys, server, last=False, skip_comm=False, compute_local_flag=False):

        # Training mode
        self.model.train()

        self.skip_comm = skip_comm

        # W = SGD(W, D)
        self.train_loss = self.train_cnn(w_glob_keys, server, last, self.skip_comm, compute_local_flag)

    @torch.no_grad()
    def evaluate_FedVIB(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        I_ZX_bound_by_epoch_test = []
        I_ZY_bound_by_epoch_test = []
        loss_by_epoch_test = []
        accuracy_by_epoch_test = []

        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            target = target.reshape(-1).long()
            batch_size = data.size()[0]
            prior_Z_distr = torch.zeros(batch_size, self.args.dimZ).to(self.args.device), torch.ones(batch_size,
                                                                                                     self.args.dimZ).to(
                self.args.device)
            encoder_Z_distr, decoder_logits = self.model(data, self.args.num_avg)

            decoder_logits_mean = torch.mean(decoder_logits, dim=0)
            loss = nn.CrossEntropyLoss(reduction='none')
            decoder_logits = decoder_logits.permute(1, 2, 0)
            cross_entropy_loss = loss(decoder_logits, target[:, None].expand(-1, self.args.num_avg))

            cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)

            I_ZX_bound_test = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))
            minusI_ZY_bound_test = torch.mean(cross_entropy_loss_montecarlo, dim=0)
            total_loss_test = torch.mean(minusI_ZY_bound_test + self.args.beta * I_ZX_bound_test)

            prediction = torch.max(decoder_logits_mean, dim=1)[1]
            accuracy_test = torch.mean((prediction == target).float())

            I_ZX_bound_by_epoch_test.append(I_ZX_bound_test.item())
            I_ZY_bound_by_epoch_test.append(minusI_ZY_bound_test.item())

            loss_by_epoch_test.append(total_loss_test.item())
            accuracy_by_epoch_test.append(accuracy_test.item())

        I_ZX = np.mean(I_ZX_bound_by_epoch_test)
        I_ZY = np.mean(I_ZY_bound_by_epoch_test)
        loss_test = np.mean(loss_by_epoch_test)
        accuracy_test = np.mean(accuracy_by_epoch_test)
        accuracy_test = 100.00 * accuracy_test
        return accuracy_test, loss_test

    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''

        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss

    @torch.no_grad()
    def evaluate_FLIX(self, data_x, data_y, dataset_name, server):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            self.convex_combination(server.W, alpha=self.alpha)
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''

        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss


'''
------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------
'''


class Server(DistributedTrainingDevice):

    def __init__(self, model, args, n_cls):
        super().__init__(model, args)

        # Parameters
        self.W = {name: value for name, value in self.model.named_parameters()}
        self.local_epoch = 0
        self.n_cls = n_cls
        if self.args.method == 'FedCR':
            self.dir_global_Z_u = torch.zeros(self.n_cls, 1, self.args.dimZ, dtype=torch.float32,
                                              device=self.args.device)
            self.dir_global_Z_sigma = torch.ones(self.n_cls, 1, self.args.dimZ, dtype=torch.float32,
                                                 device=self.args.device)

    def find_client_layers(self, clients, w_glob_keys_ops):
        client_layers = {}
        for name in self.W:
            for client in clients:
                # if name in client.W.keys():
                # print(w_glob_keys_ops[client].keys())
                print(w_glob_keys_ops[client.id]) # ['fc2.weight', 'fc2.bias']
                if name in w_glob_keys_ops[client.id]:
                    if name not in client_layers:
                        client_layers[name] = [client]
                    else:
                        client_layers[name].append(client)
        return client_layers

    # def aggregate_weight_updates(self, clients, iter,
    #                              client_layers, aggregation="mean"):
    #
    #     # Warning: Note that K is different for unbalanced dataset
    #     self.local_epoch = clients[0].local_epoch
    #     # dW = aggregate(dW_i, i=1,..,n)
    #     if aggregation == "mean":  # default aggregation for standard FL
    #         average(target=self.W, sources=[client.W for client in clients])
    #     else:
    #         if aggregation == "pmean": # personalized mean average
    #             layer_wise_average(target=self.W, sources=client_layers)
    def aggregate_weight_updates(self, clients, iter, aggregation="mean"):

        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch
        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":
            average(target=self.W, sources=[client.W for client in clients])

    def aggregate_weight_updates_subset_single(self, clients, iter, w_glob_keys_ops, aggregation="mean"):
        """
        Aggregate weight updates for a subset of layers from each client.

        :param clients: A list of client objects.
        :param iter: Current iteration (unused in this function but kept for compatibility).
        :param aggregation: A string indicating the aggregation method.
        """
        # Warning: Note that K is different for unbalanced dataset
        self.local_epoch = clients[0].local_epoch

        # dW = aggregate(dW_i, i=1,..,n)
        if aggregation == "mean":  # default aggregation for standard FL
            for client in clients:
                # Use the global list w_glob_keys_ops to determine the layers to aggregate for this client
                layers_to_aggregate = w_glob_keys_ops[client.id]
                average_subset(target=self.W, sources=[client.W], layers=layers_to_aggregate)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def aggregate_weight_updates_subset(self, clients, iter, w_glob_keys_ops, aggregation="mean"):
        """
        Aggregate weight updates for a subset of layers from each client using specified aggregation.

        :param clients: A list of client objects.
        :param iter: Current iteration (unused in this function but kept for compatibility).
        :param aggregation: A string indicating the aggregation method ('mean' or 'weighted_mean').
        """
        self.local_epoch = clients[0].local_epoch

        # Calculate the total number of contributions for each layer if using weighted_mean
        layer_contributions = None
        if aggregation == "weighted_mean":
            layer_contributions = {layer: 0 for layer in self.W.keys()}
            for client in clients:
                layers_to_aggregate = w_glob_keys_ops[client.id]
                for layer in layers_to_aggregate:
                    layer_contributions[layer] += 1

        # Aggregate layers based on the specified strategy
        for layer_name in self.W.keys():
            layer_sources = []
            for client in clients:
                if layer_name in w_glob_keys_ops[client.id]:
                    layer_sources.append(client.W[layer_name].data)

            # Stack the layer sources and compute mean or weighted mean
            if layer_sources:
                stacked_sources = torch.stack(layer_sources)
                if aggregation == "mean":
                    self.W[layer_name].data = torch.mean(stacked_sources, dim=0).clone()
                elif aggregation == "weighted_mean":
                    weights = torch.tensor([len(w_glob_keys_ops[client.id]) for client in clients if
                                            layer_name in w_glob_keys_ops[client.id]], dtype=torch.float32)
                    weights /= layer_contributions[layer_name]
                    self.W[layer_name].data = torch.sum(stacked_sources * weights[:, None, None, None],
                                                        dim=0).clone()
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation}")

    def global_POE(self, clients):

        dir_global_Z_u_copy = copy.deepcopy(self.dir_global_Z_u)
        dir_global_Z_sigma_copy = copy.deepcopy(self.dir_global_Z_sigma)

        for cls in range(self.n_cls):
            clients_all_Z_u = True
            clients_all_Z_sigma = True

            for i in range(len(clients)):

                if clients[i].dir_Z_u[cls].equal(
                        torch.zeros(1, self.args.dimZ, dtype=torch.float32, device=self.args.device)) and \
                        clients[i].dir_Z_sigma[cls].equal(
                                torch.ones(1, self.args.dimZ, dtype=torch.float32, device=self.args.device)):
                    pass
                elif isinstance(clients_all_Z_u, bool):
                    clients_all_Z_u = clients[i].dir_Z_u[cls].clone().detach()
                    clients_all_Z_sigma = clients[i].dir_Z_sigma[cls].clone().detach()
                else:
                    clients_all_Z_u = torch.cat((clients_all_Z_u, clients[i].dir_Z_u[cls].clone().detach()),
                                                0).clone().detach()
                    clients_all_Z_sigma = torch.cat((clients_all_Z_sigma, clients[i].dir_Z_sigma[cls].clone().detach()),
                                                    0).clone().detach()

            if not isinstance(clients_all_Z_u, bool):
                clients_all_Z = clients_all_Z_u, clients_all_Z_sigma
                dir_global_Z_u_copy[cls], dir_global_Z_sigma_copy[cls] = product_of_experts(clients_all_Z)

                self.dir_global_Z_u[cls] = (1 - self.args.beta2) * self.dir_global_Z_u[cls] + self.args.beta2 * \
                                           dir_global_Z_u_copy[cls]
                self.dir_global_Z_sigma[cls] = (1 - self.args.beta2) * self.dir_global_Z_sigma[cls] + self.args.beta2 * \
                                               dir_global_Z_sigma_copy[cls]

    @torch.no_grad()
    def evaluate_FedVIB(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        I_ZX_bound_by_epoch_test = []
        I_ZY_bound_by_epoch_test = []
        loss_by_epoch_test = []
        accuracy_by_epoch_test = []

        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            batch_size = data.size()[0]
            prior_Z_distr = torch.zeros(batch_size, self.args.dimZ).to(self.args.device), torch.ones(batch_size,
                                                                                                     self.args.dimZ).to(
                self.args.device)
            encoder_Z_distr, decoder_logits = self.model(data, self.args.num_avg)

            decoder_logits_mean = torch.mean(decoder_logits, dim=0)
            loss = nn.CrossEntropyLoss(reduction='none')
            decoder_logits = decoder_logits.permute(1, 2, 0)
            cross_entropy_loss = loss(decoder_logits, target[:, None].expand(-1, self.args.num_avg))

            cross_entropy_loss_montecarlo = torch.mean(cross_entropy_loss, dim=-1)

            I_ZX_bound_test = torch.mean(KL_between_normals(encoder_Z_distr, prior_Z_distr))
            minusI_ZY_bound_test = torch.mean(cross_entropy_loss_montecarlo, dim=0)
            total_loss_test = torch.mean(minusI_ZY_bound_test + self.args.beta * I_ZX_bound_test)

            prediction = torch.max(decoder_logits_mean, dim=1)[1]
            accuracy_test = torch.mean((prediction == target).float())

            I_ZX_bound_by_epoch_test.append(I_ZX_bound_test.item())
            I_ZY_bound_by_epoch_test.append(minusI_ZY_bound_test.item())

            loss_by_epoch_test.append(total_loss_test.item())
            accuracy_by_epoch_test.append(accuracy_test.item())

        I_ZX = np.mean(I_ZX_bound_by_epoch_test)
        I_ZY = np.mean(I_ZY_bound_by_epoch_test)
        loss_test = np.mean(loss_by_epoch_test)
        accuracy_test = np.mean(accuracy_by_epoch_test)
        accuracy_test = 100.00 * accuracy_test
        return accuracy_test, loss_test

    @torch.no_grad()
    def evaluate(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''

        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss

    @torch.no_grad()
    # TODO
    def evaluate_FLIX(self, data_x, data_y, dataset_name):
        self.model.eval()
        # testing
        test_loss = 0
        acc_overall = 0
        n_tst = data_x.shape[0]
        tst_gen = DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), batch_size=self.args.bs, shuffle=False)
        tst_gen_iter = tst_gen.__iter__()
        for i in range(int(np.ceil(n_tst / self.args.bs))):
            data, target = tst_gen_iter.__next__()
            data, target = data.to(self.args.device), target.to(self.args.device)
            log_probs = self.model(data)
            # sum up batch loss
            test_loss += nn.CrossEntropyLoss(reduction='sum')(log_probs, target.reshape(-1).long()).item()
            # get the index of the max log-probability
            log_probs = log_probs.cpu().detach().numpy()
            log_probs = np.argmax(log_probs, axis=1).reshape(-1)
            target = target.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(log_probs == target)
            acc_overall += batch_correct
            '''
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
            '''

        test_loss /= n_tst
        accuracy = 100.00 * acc_overall / n_tst
        return accuracy, test_loss
