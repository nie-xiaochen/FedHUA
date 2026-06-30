import numpy as np
import torch
import random

def evaluation(args, global_model, test_dl, best_acc, round):
    acc = compute_acc(global_model, test_dl)
    if best_acc < acc:
        best_acc = acc
    print('>> Round {} | Current Acc: {:.5f}, Best Acc: {:.5f}'.format(round, acc, best_acc))
    print('-'*80)
    return acc, best_acc


def compute_acc(net, test_data_loader):
    """Compute top-1 accuracy on CUDA when available, otherwise on CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    top1, total = 0, 0
    with torch.no_grad():
        loaders = test_data_loader if isinstance(test_data_loader, list) else [test_data_loader]
        for test_dl in loaders:
            for x, target in test_dl:
                x = x.to(device)
                target = target.to(device, dtype=torch.int64)
                predictions = net(x).argmax(dim=1)
                top1 += (predictions == target).sum().item()
                total += target.numel()
    net.to('cpu')
    return top1 / float(total)

def compute_local_test_accuracy(model, dataloader, data_distribution):

    model.eval()

    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(dataloader):
            x, target = x.to(device), target.to(device=device, dtype=torch.int64)
            out = model(x)
            _, pred_label = torch.max(out.data, 1)
            correct_filter = (pred_label == target.data)
            generalized_total += x.data.size()[0]
            generalized_correct += correct_filter.sum().item()
            for i, true_label in enumerate(target.data):
                total_label_num[true_label] += 1
                if correct_filter[i]:
                    correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()
    
    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total





def evaluate_global_model(args, nets_this_round, global_model, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list):
    for net_id, _ in nets_this_round.items():
        if net_id in benign_client_list:
            val_local_dl = val_local_dls[net_id]
            data_distribution = data_distributions[net_id]

            val_acc = compute_acc(global_model, val_local_dl)
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(global_model, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} | Personalized Test Acc: {:.5f} | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


import torch.nn.functional as F

# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss

# multilabel classification KL divergence
def mlc_KLDiv(s_out, t_out, T=1):
    loss = 0
    # add sigmoid to make the loss within range (0,1)
    s_out = torch.sigmoid(s_out) / T
    t_out = torch.sigmoid(t_out) / T
    ones = torch.ones(s_out.shape[0],1).cuda()

    for i in range(17):
        s = torch.cat((s_out[:,i].unsqueeze(1),ones-s_out[:,i].unsqueeze(1)),dim=1)
        t = torch.cat((t_out[:,i].unsqueeze(1),ones-t_out[:,i].unsqueeze(1)),dim=1)
        loss += torch.nn.functional.kl_div(s, t, reduction='batchmean') * (T * T)
    return loss