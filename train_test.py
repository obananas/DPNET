""" Training and testing of the model
"""
import os
from re import I
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from model import DPNET
from utils import gen_adj_mat_tensor, gen_test_adj_mat_tensor, cal_adj_mat_parameter

cuda = True if torch.cuda.is_available() else False


def prepare_trte_data(data_folder):
    num_view = 3
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view + 1):
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))

    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_all_list, idx_dict, labels


def gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, adj_parameter):
    adj_metric = "cosine"  # cosine distance
    adj_train_list = []
    adj_test_list = []
    for i in range(len(data_tr_list)):
        adj_parameter_adaptive = cal_adj_mat_parameter(adj_parameter, data_tr_list[i], adj_metric)
        adj_train_list.append(gen_adj_mat_tensor(data_tr_list[i], adj_parameter_adaptive, adj_metric))
        adj_test_list.append(gen_test_adj_mat_tensor(data_trte_list[i], trte_idx, adj_parameter_adaptive, adj_metric))

    return adj_train_list, adj_test_list


def train_epoch(data_list, adj_tr_list, label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss, _ = model(data_list, adj_tr_list, label)
    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()


def test_epoch(data_list, adj_trte_list, te_idx, model):
    model.eval()
    with torch.no_grad():
        logit = model.infer(data_list, adj_trte_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob[te_idx, :]


def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def train(data_folder, modelpath, testonly):
    test_inverval = 50
    if 'ROSMAP' in data_folder:
        hidden_dim = [300]
        num_epoch = 2000
        lr = 5e-5
        step_size = 500
        num_class = 2

    data_tr_list, data_trte_list, trte_idx, labels_trte = prepare_trte_data(data_folder)
    adj_tr_list, adj_trte_list = gen_trte_adj_mat(data_tr_list, data_trte_list, trte_idx, 10)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    labels_tr_tensor = labels_tr_tensor.cuda()
    dim_list = [x.shape[1] for x in data_tr_list]
    model = DPNET(dim_list, hidden_dim, num_class, dropout=0.5)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)
    if testonly:
        load_checkpoint(model, os.path.join(modelpath, data_folder, 'checkpoint.pt'))
        te_prob = test_epoch(data_trte_list, adj_trte_list, trte_idx["te"], model)
        if num_class == 2:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])))
        else:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1 weighted: {:.5f}".format(
                f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            print("Test F1 macro: {:.5f}".format(
                f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
    else:
        print("\nTraining...")
        for epoch in range(num_epoch + 1):
            train_epoch(data_tr_list, adj_tr_list, labels_tr_tensor, model, optimizer)
            scheduler.step()
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_trte_list, adj_trte_list, trte_idx["te"], model)
                print("\nTest: Epoch {:d}".format(epoch))
                if num_class == 2:
                    print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])))
                else:
                    print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1 weighted: {:.5f}".format(
                        f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                    print("Test F1 macro: {:.5f}".format(
                        f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
        save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder))
