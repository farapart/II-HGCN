import argparse
import os
from functools import reduce

import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
import scipy.sparse as sp
from utils import cal_single_domain_matrix, cal_single_domain_matrix_np, calculate_D_matrix, hitRatio, MRR, \
    data_sparse_transfer, do_dataset_sparse, remove_test_and_vali, EarlyStopping
from tqdm import tqdm

import torch
# from hy_model import SingleDomain, II_HGCN
from models import SingleDomain, II_HGCN
from torch import optim
import time
from torch import nn
import random
import torch
from scipy.linalg import fractional_matrix_power
from torch.autograd import Variable
import heapq
from sklearn.metrics import ndcg_score
import torch.nn.functional as F
import pandas as pd
from Dataset_CDAE import Dataset
from numba import njit, prange
import numba

single_k_list = [1, 3, 4, 5]

method_name = 'my'
numba.config.NUMBA_DEFAULT_NUM_THREADS = 4
numba.config.NUMBA_NUM_THREADS = 4

k_list = [5, 10]
best_ndcg_a = {k: 0 for k in k_list}
best_hit_ratio_a = {k: 0 for k in k_list}
best_ndcg_b = {k: 0 for k in k_list}
best_hit_ratio_b = {k: 0 for k in k_list}
best_mrr_a = {k: 0 for k in k_list}
best_mrr_b = {k: 0 for k in k_list}
this_ndcg_a = {k: 0 for k in k_list}
this_ndcg_b = {k: 0 for k in k_list}
all_results = [[], [], [], []]
train_loss_list = []
test_loss_list = []

SPLIT = "=" * 50

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True, help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--k-single', type=int, default=2, help='K value of hypergraph for single domain')
parser.add_argument('--k-dual', type=int, default=5, help='K value of hypergraph for dual domain')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight-decay', type=float, default=0.00001, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dataset', type=str, default="amazon3", help="Training in which dataset")
parser.add_argument('--batch-size', type=int, default=256, help="The batch size of training")
parser.add_argument('--single-layer-num', type=int, default=2, help="The conv layer num of single domain")
parser.add_argument('--dual-layer-num', type=int, default=1, help="The conv layer num of dual domain")
parser.add_argument('--t_percent', type=float, default=1.0, help='target percent')
parser.add_argument('--s_percent', type=float, default=1.0, help='source percent')
parser.add_argument('--pos-weight', type=float, default=1.0, help='weight for positive samples')
parser.add_argument('--embedding-size', type=int, default=128, help='embedding size')
parser.add_argument('--neg-frequency', type=int, default=5, help='negative sample choice frequency')
parser.add_argument('--if-sparse', type=bool, default=False, help='if doing sparse experiment')
parser.add_argument('--sparse-ratio', type=int, default=10, help='the sparse ratio of our experiment')
parser.add_argument('--log', type=str, default='logs/{}'.format(method_name), help='log directory')
parser.add_argument('--cuda-index', type=int, default=0, help='train in which GPU')
parser.add_argument('--model-structure', type=str, default="normal", help='model structure')
parser.add_argument('--intra-type', type=str, default="", help="the construction principle of intra-domain hypergraph")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(f"cuda:{args.cuda_index}" if args.cuda else "cpu")
print(f"Train in {device}")


@njit(fastmath=True, parallel=True)
def neg_sample_numba():
    user_neg_sample_a = {}
    user_neg_sample_b = {}
    for i in prange(len(user_list)):
        user_neg_sample_a[user_list[i]] = np.random.choice(user_neg_sample_movie_dict[user_list[i]],
                                                           neg_sample_length_movie[user_list[i]], replace=False)
    for i in prange(len(user_list)):
        user_neg_sample_b[user_list[i]] = np.random.choice(user_neg_sample_book_dict[user_list[i]],
                                                           neg_sample_length_book[user_list[i]], replace=False)
    return user_neg_sample_a, user_neg_sample_b


def do_neg_sample():
    t = time.time()
    user_neg_sample_a, user_neg_sample_b = neg_sample_numba()
    print(f"Sample succeed. Time = {time.time() - t}")
    return user_neg_sample_a, user_neg_sample_b


def neg_sample():
    t = time.time()
    user_neg_sample_a = {}
    user_neg_sample_b = {}
    for user_name in tqdm(dataset.peo2item_movie.keys()):
        user_neg_sample_a[user_name] = np.random.choice(user_neg_sample_movie_dict[user_name],
                                                        neg_sample_length_movie[user_name], replace=False)
        user_neg_sample_b[user_name] = np.random.choice(user_neg_sample_book_dict[user_name],
                                                        neg_sample_length_book[user_name], replace=False)
        # user_neg_sample_a[user_name] = np.random.choice(user_neg_sample_movie_dict[user_name],
        #                                                 len(user_neg_sample_movie_dict[user_name]), replace=False)
        # user_neg_sample_b[user_name] = np.random.choice(user_neg_sample_book_dict[user_name],
        #                                                 len(user_neg_sample_book_dict[user_name]), replace=False)
    print(f"Sample succeed. Time = {time.time() - t}")
    return user_neg_sample_a, user_neg_sample_b


def load_batch_train_sample(users):
    users = np.array(users).squeeze()
    user_list_movie = []
    user_list_book = []
    # user_list_pos_movie = []
    # user_list_pos_book = []
    # user_list_neg_movie = []
    # user_list_neg_book = []
    pos_result_movie = []
    neg_result_movie = []
    pos_result_book = []
    neg_result_book = []
    for user in users:
        # 202106007修改，恢复注释
        user_list_movie.extend([user for _ in range(len(dataset.peo2item_movie[user]))])
        user_list_book.extend([user for _ in range(len(dataset.peo2item_book[user]))])
        # user_list_pos_movie.extend([user for _ in range(len(dataset.peo2item_movie[user]))])
        # user_list_pos_book.extend([user for _ in range(len(dataset.peo2item_book[user]))])
        # user_list_neg_movie.extend([user for _ in range(len(user_neg_sample_movie[user]))])
        # user_list_neg_book.extend([user for _ in range(len(user_neg_sample_book[user]))])
        pos_result_movie.extend(dataset.peo2item_movie[user])
        pos_result_book.extend(dataset.peo2item_book[user])
        neg_result_movie.extend(user_neg_sample_movie[user])
        neg_result_book.extend(user_neg_sample_book[user])
    # 202106007修改，恢复注释
    return user_list_movie, user_list_book, pos_result_movie, neg_result_movie, pos_result_book, neg_result_book
    # return user_list_pos_movie, user_list_pos_book, user_list_neg_movie, user_list_neg_book, \
    #        pos_result_movie, neg_result_movie, pos_result_book, neg_result_book


def load_batch_vali_sample():
    user_list_movie = []
    user_list_book = []
    item_sample_movie = []
    item_sample_book = []
    true_label_movie = []
    true_label_book = []
    neg_99 = [0 for _ in range(99)]
    for user in dataset.peo2item_movie.keys():
        user_list_movie.extend([user for _ in range(100)])
        user_list_book.extend([user for _ in range(100)])
        item_sample_movie.append(dataset.movie_vali[user])
        item_sample_movie.extend(dataset.movie_nega[user])
        item_sample_book.append(dataset.book_vali[user])
        item_sample_book.extend(dataset.book_nega[user])
        true_label_movie.append(1)
        true_label_movie.extend(neg_99)
        true_label_book.append(1)
        true_label_book.extend(neg_99)
    return user_list_movie, user_list_book, item_sample_movie, item_sample_book, true_label_movie, true_label_book


def load_batch_test_sample():
    user_list_movie = []
    user_list_book = []
    item_sample_movie = []
    item_sample_book = []
    true_label_movie = []
    true_label_book = []
    neg_99 = [0 for _ in range(99)]
    for user in dataset.peo2item_movie.keys():
        user_list_movie.extend([user for _ in range(100)])
        user_list_book.extend([user for _ in range(100)])
        item_sample_movie.append(dataset.movie_test[user])
        item_sample_movie.extend(dataset.movie_nega[user])
        item_sample_book.append(dataset.book_test[user])
        item_sample_book.extend(dataset.book_nega[user])
        true_label_movie.append(1)
        true_label_movie.extend(neg_99)
        true_label_book.append(1)
        true_label_book.extend(neg_99)
    return user_list_movie, user_list_book, item_sample_movie, item_sample_book, true_label_movie, true_label_book


def train(epoch):
    print(SPLIT)
    print(f"epoch: {epoch}")
    hy_model.train()
    loss_list = []
    loss_a_list = []
    loss_b_list = []
    for batch_idx, data in tqdm(enumerate(train_loader)):
        data = data.reshape([-1])
        # 202106007修改，恢复注释
        user_list_movie, user_list_book, pos_sample_movie, neg_sample_movie, pos_sample_book, neg_sample_book \
            = load_batch_train_sample(data)
        # user_list_pos_movie, user_list_pos_book, user_list_neg_movie, user_list_neg_book, \
        # pos_sample_movie, neg_sample_movie, pos_sample_book, neg_sample_book \
        #     = load_batch_train_sample(data)
        # if args.cuda:
        data = data.to(device)
        # 202106007修改，恢复注释
        user_list_movie = torch.LongTensor(user_list_movie).to(device)
        user_list_book = torch.LongTensor(user_list_book).to(device)
        # user_list_pos_movie = torch.LongTensor(user_list_pos_movie).to(device)
        # user_list_pos_book = torch.LongTensor(user_list_pos_book).to(device)
        # user_list_neg_movie = torch.LongTensor(user_list_neg_movie).to(device)
        # user_list_neg_book = torch.LongTensor(user_list_neg_book).to(device)
        pos_sample_movie = torch.LongTensor(pos_sample_movie).to(device)
        neg_sample_movie = torch.LongTensor(neg_sample_movie).to(device)
        pos_sample_book = torch.LongTensor(pos_sample_book).to(device)
        neg_sample_book = torch.LongTensor(neg_sample_book).to(device)

        hy_model.train()
        optimizer.zero_grad()
        pos_score_a, pos_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, pos_sample_movie,
                                                    conv_bu, conv_bi, user_list_book, pos_sample_book, HyGCN_a, HyGCN_b)
        neg_score_a, neg_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, neg_sample_movie,
                                                    conv_bu, conv_bi, user_list_book, neg_sample_book, HyGCN_a, HyGCN_b)

        predict_label_a = torch.cat((pos_score_a, neg_score_a))
        true_label_a = torch.cat((torch.ones_like(pos_score_a), torch.zeros_like(neg_score_a)))
        predict_label_b = torch.cat((pos_score_b, neg_score_b))
        true_label_b = torch.cat((torch.ones_like(pos_score_b), torch.zeros_like(neg_score_b)))

        loss_a = loss_func(predict_label_a, true_label_a)
        loss_b = loss_func(predict_label_b, true_label_b)
        loss_a_list.append(loss_a.item())
        loss_b_list.append(loss_b.item())
        loss = torch.add(loss_a, loss_b)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
    print(f"Loss = {np.mean(loss_list)}, "
          f"Loss in domain a = {np.mean(loss_a_list)}, "
          f"Loss in domain b = {np.mean(loss_b_list)}")
    train_loss_list.append([np.mean(loss_list)])


@torch.no_grad()
def test(best_ndcg_a, best_hit_ratio_a, best_ndcg_b, best_hit_ratio_b, best_mrr_a, best_mrr_b, all_results: list,
         this_ndcg_a, this_ndcg_b):
    user_list_movie, user_list_book, item_sample_movie, item_sample_book, true_label_movie, true_label_book \
        = load_batch_vali_sample()
    # if args.cuda:
    user_list_movie = torch.LongTensor(user_list_movie).to(device)
    user_list_book = torch.LongTensor(user_list_book).to(device)
    item_sample_movie = torch.LongTensor(item_sample_movie).to(device)
    item_sample_book = torch.LongTensor(item_sample_book).to(device)
    true_label_movie = torch.FloatTensor(true_label_movie).to(device)
    true_label_book = torch.FloatTensor(true_label_book).to(device)
    pred_score_a, pred_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, item_sample_movie,
                                                  conv_bu, conv_bi, user_list_book, item_sample_book, HyGCN_a, HyGCN_b)
    true_label_a = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in dataset.movie_test.keys()])
    true_label_b = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in dataset.book_test.keys()])
    pred_label_a = pred_score_a.cpu().numpy().reshape(-1, 100)
    pred_label_b = pred_score_b.cpu().numpy().reshape(-1, 100)
    print(SPLIT)
    print("Begin Test!!!")
    loss_a = loss_func(pred_score_a, true_label_movie)
    loss_b = loss_func(pred_score_b, true_label_book)
    loss = torch.add(loss_a, loss_b)
    print(f"Test loss = {loss}, Domain A loss = {loss_a}, Domain B loss = {loss_b}")
    test_loss_list.append([loss.item()])
    print(f"NDCG@K for Domain A:")
    for k in k_list:
        ndcg_a = round(ndcg_score(y_true=true_label_a, y_score=pred_label_a, k=k), 4)
        this_ndcg_a[k] = ndcg_a
        if ndcg_a > best_ndcg_a[k]:
            best_ndcg_a[k] = ndcg_a
            torch.save(hy_model.state_dict(), os.path.join(log, f'best_ndcg_a_{k}.pkl'))
        print(f"k:{k}, ndcg = {ndcg_a}", end="\t\t")
    print()
    print(f"NDCG@K for Domain B:")
    for k in k_list:
        ndcg_b = round(ndcg_score(y_true=true_label_b, y_score=pred_label_b, k=k), 4)
        this_ndcg_b[k] = ndcg_b
        if ndcg_b > best_ndcg_b[k]:
            best_ndcg_b[k] = ndcg_b
            torch.save(hy_model.state_dict(), os.path.join(log, f'best_ndcg_b_{k}.pkl'))
        print(f"k:{k}, ndcg = {ndcg_b}", end="\t\t")
    print()
    print(f"MRR@K for Domain A:")
    for k in k_list:
        mrr_a = round(MRR(pred_score_a.reshape(-1, 100), k=k), 4)
        if mrr_a > best_mrr_a[k]:
            best_mrr_a[k] = mrr_a
            torch.save(hy_model.state_dict(), os.path.join(log, f'best_mrr_a_{k}.pkl'))
        print(f"k:{k}, mrr = {mrr_a}", end="\t\t")
    print()
    print(f"MRR@K for Domain B:")
    for k in k_list:
        mrr_b = round(MRR(pred_score_b.reshape(-1, 100), k=k), 4)
        if mrr_b > best_mrr_b[k]:
            best_mrr_b[k] = mrr_b
            torch.save(hy_model.state_dict(), os.path.join(log, f'best_mrr_b_{k}.pkl'))
        print(f"k:{k}, mrr = {mrr_b}", end="\t\t")
    print()
    print(f"HitRatio@K for Domain A: ")
    for k in k_list:
        hit_a = round(hitRatio(pred_score_a.reshape(-1, 100), k), 4)
        if hit_a > best_hit_ratio_a[k]:
            best_hit_ratio_a[k] = hit_a
            torch.save(hy_model.state_dict(), os.path.join(log, f'best_hit_a_{k}.pkl'))
        print(f"k:{k}, hitRatio = {hit_a}", end="\t")
    print()
    print(f"HitRatio@K for Domain B: ")
    for k in k_list:
        hit_b = round(hitRatio(pred_score_b.reshape(-1, 100), k), 4)
        if hit_b > best_hit_ratio_b[k]:
            best_hit_ratio_b[k] = hit_b
            torch.save(hy_model.state_dict(), os.path.join(log, f'best_hit_b_{k}.pkl'))
        print(f"k:{k}, hitRatio = {hit_b}", end="\t")
    print()
    print(f"Best Results")
    print(f"Domain A:")
    for k in k_list:
        print(f"K:{k}, ndcg = {best_ndcg_a[k]}, hitRatio = {best_hit_ratio_a[k]}, mrr = {best_mrr_a[k]}")
    print(f"Domain B:")
    for k in k_list:
        print(f"K:{k}, ndcg = {best_ndcg_b[k]}, hitRatio = {best_hit_ratio_b[k]}, mrr = {best_mrr_b[k]}")
    print(SPLIT)


@torch.no_grad()
def test_best_result():
    user_list_movie, user_list_book, item_sample_movie, item_sample_book, true_label_movie, true_label_book \
        = load_batch_test_sample()
    # if args.cuda:
    user_list_movie = torch.LongTensor(user_list_movie).to(device)
    user_list_book = torch.LongTensor(user_list_book).to(device)
    item_sample_movie = torch.LongTensor(item_sample_movie).to(device)
    item_sample_book = torch.LongTensor(item_sample_book).to(device)
    true_label_movie = torch.FloatTensor(true_label_movie).to(device)
    true_label_book = torch.FloatTensor(true_label_book).to(device)
    true_label_a = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in dataset.movie_test.keys()])
    true_label_b = np.array([np.concatenate((np.ones(1), np.zeros(99))) for _ in dataset.book_test.keys()])
    for k in k_list:
        # ndcg a
        hy_model.load_state_dict(torch.load(os.path.join(log, f'best_ndcg_a_{k}.pkl')))
        pred_score_a, pred_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, item_sample_movie,
                                                      conv_bu, conv_bi, user_list_book, item_sample_book, HyGCN_a,
                                                      HyGCN_b)
        pred_label_a = pred_score_a.cpu().numpy().reshape(-1, 100)
        ndcg_a = round(ndcg_score(y_true=true_label_a, y_score=pred_label_a, k=k), 4)
        # ndcg b
        hy_model.load_state_dict(torch.load(os.path.join(log, f'best_ndcg_b_{k}.pkl')))
        pred_score_a, pred_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, item_sample_movie,
                                                      conv_bu, conv_bi, user_list_book, item_sample_book, HyGCN_a,
                                                      HyGCN_b)
        pred_label_b = pred_score_b.cpu().numpy().reshape(-1, 100)
        ndcg_b = round(ndcg_score(y_true=true_label_b, y_score=pred_label_b, k=k), 4)
        # hit a
        hy_model.load_state_dict(torch.load(os.path.join(log, f'best_hit_a_{k}.pkl')))
        pred_score_a, pred_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, item_sample_movie,
                                                      conv_bu, conv_bi, user_list_book, item_sample_book, HyGCN_a,
                                                      HyGCN_b)
        hit_a = round(hitRatio(pred_score_a.reshape(-1, 100), k), 4)
        # hit b
        hy_model.load_state_dict(torch.load(os.path.join(log, f'best_hit_b_{k}.pkl')))
        pred_score_a, pred_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, item_sample_movie,
                                                      conv_bu, conv_bi, user_list_book, item_sample_book, HyGCN_a,
                                                      HyGCN_b)
        hit_b = round(hitRatio(pred_score_b.reshape(-1, 100), k), 4)
        # mrr a
        hy_model.load_state_dict(torch.load(os.path.join(log, f'best_mrr_a_{k}.pkl')))
        pred_score_a, pred_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, item_sample_movie,
                                                      conv_bu, conv_bi, user_list_book, item_sample_book, HyGCN_a,
                                                      HyGCN_b)
        mrr_a = round(MRR(pred_score_a.reshape(-1, 100), k=k), 4)
        # mrr b
        hy_model.load_state_dict(torch.load(os.path.join(log, f'best_mrr_b_{k}.pkl')))
        pred_score_a, pred_score_b = hy_model.forward(conv_au, conv_ai, user_list_movie, item_sample_movie,
                                                      conv_bu, conv_bi, user_list_book, item_sample_book, HyGCN_a,
                                                      HyGCN_b)
        mrr_b = round(MRR(pred_score_b.reshape(-1, 100), k=k), 4)
        print('Test TopK:{} ---> movie: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}, book: hr:{:.4f},ndcg:{:.4f},mrr:{:.4f}'
              .format(k, hit_a, ndcg_a, mrr_a, hit_b, ndcg_b, mrr_b))


print(f"Train in {args.dataset}, epoch = {args.epochs}")
print(args)

log = os.path.join(args.log,
                   '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.dataset, args.embedding_size, args.epochs, args.lr,
                                                                args.weight_decay, args.k_single, args.single_layer_num,
                                                                args.dual_layer_num,
                                                                args.k_dual, args.if_sparse,
                                                                args.sparse_ratio,
                                                                args.intra_type))
if os.path.isdir(log):
    print("%s already exist. are you sure to override? Ok, I'll wait for 5 seconds. Ctrl-C to abort." % log)
    time.sleep(5)
    os.system('rm -rf %s/' % log)

os.makedirs(log)
print("made the log directory", log)

dataset = Dataset(args.batch_size, args.dataset)
NUM_USER = dataset.num_user
NUM_MOVIE = dataset.num_movie
NUM_BOOK = dataset.num_book

print(f'Preparing the training data from {args.dataset}......')
# prepare data for domain A
row, col = dataset.get_part_train_indices('movie', args.s_percent)
values = np.ones(row.shape[0])
Ha = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray()

row, col = dataset.get_part_train_indices('book', args.t_percent)
values = np.ones(row.shape[0])
Hb = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray()
print('Preparing the training data over......')
print('Load processed con begin......')

remove_test_and_vali(dataset)

if not args.if_sparse:
    print(f"Training in no sparse dataset")
    load_dict = os.path.join('processed_data', args.dataset)
else:
    print(f"Training in sparse dataset, sparse ratio = {args.sparse_ratio}")
    load_dict = os.path.join('processed_data', args.dataset, f"sparse_{args.sparse_ratio}")

if args.if_sparse:
    print(f"Do dataset sparse")
    Ha = np.load(os.path.join(load_dict, "Ha_sparse.npz"))["arr_0"]
    Hb = np.load(os.path.join(load_dict, "Hb_sparse.npz"))["arr_0"]
    do_dataset_sparse(dataset, Ha, Hb)
    print(f"Dataset sparse succeed")
# conv_au = np.load(os.path.join('processed_data', args.dataset, "conv_au" + f"_{args.k_single}.npz"))["arr_0"]
# conv_ai = np.load(os.path.join('processed_data', args.dataset, "conv_ai" + f"_{args.k_single}.npz"))["arr_0"]
# single_back = f"_{args.k_single}_05.npz"
single_back = f"_{args.k_single}" + args.intra_type + ".npz"
print(f"back = {single_back}")
conv_au = np.load(os.path.join(load_dict, "conv_au" + single_back))["arr_0"]
conv_ai = np.load(os.path.join(load_dict, "conv_ai" + single_back))["arr_0"]
conv_bu = np.load(os.path.join(load_dict, "conv_bu" + single_back))["arr_0"]
conv_bi = np.load(os.path.join(load_dict, "conv_bi" + single_back))["arr_0"]
HyGCN_a = np.load(os.path.join(load_dict, "Ha_GCN" + f"_{args.k_dual}.npz"))["arr_0"]
HyGCN_b = np.load(os.path.join(load_dict, "Hb_GCN" + f"_{args.k_dual}.npz"))["arr_0"]
print('Load processed con over......')
user_id = np.arange(NUM_USER).reshape([NUM_USER, 1])

train_loader = torch.utils.data.DataLoader(torch.from_numpy(user_id),
                                           batch_size=args.batch_size,
                                           shuffle=True)
save_loader = torch.utils.data.DataLoader(torch.from_numpy(user_id),
                                          batch_size=args.batch_size,
                                          shuffle=False)

hy_model = II_HGCN(embedding_size=args.embedding_size, user_num_a=NUM_USER, item_num_a=NUM_MOVIE,
                   user_num_b=NUM_USER, item_num_b=NUM_BOOK, device=device, single_layer_num=args.single_layer_num,
                   dual_layer_num=args.dual_layer_num, model_structure=args.model_structure)
hy_model.to(device)
conv_au = torch.FloatTensor(conv_au).to(device)
conv_ai = torch.FloatTensor(conv_ai).to(device)
conv_bu = torch.FloatTensor(conv_bu).to(device)
conv_bi = torch.FloatTensor(conv_bi).to(device)
HyGCN_a = torch.FloatTensor(HyGCN_a).to(device)
HyGCN_b = torch.FloatTensor(HyGCN_b).to(device)

# optimizer = optim.Adam(hy_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer = optim.Adam(hy_model.parameters(), lr=args.lr)
loss_func = F.binary_cross_entropy
# loss_func = F.binary_cross_entropy_with_logits
early_stop = EarlyStopping()

print("Do neg sample prepare")
user_list = list(dataset.peo2item_movie.keys())
user_neg_sample_movie_dict = {}
user_neg_sample_book_dict = {}
neg_sample_length_movie = {}
neg_sample_length_book = {}
for user in tqdm(dataset.peo2item_movie.keys()):
    # 202106007修改，恢复注释
    user_neg_sample_movie_dict[user] = list(
        set(dataset.movie_set) - set(dataset.peo2item_movie[user]) - set(dataset.movie_nega[user]))
    user_neg_sample_book_dict[user] = list(
        set(dataset.book_set) - set(dataset.peo2item_book[user]) - set(dataset.book_nega[user]))
    # user_neg_sample_movie_dict[user] = list(
    #     set(dataset.movie_set) - set(dataset.peo2item_movie[user]))
    # user_neg_sample_book_dict[user] = list(
    #     set(dataset.book_set) - set(dataset.peo2item_book[user]))
    neg_sample_length_movie[user] = len(dataset.peo2item_movie[user])
    neg_sample_length_book[user] = len(dataset.peo2item_book[user])
print("Neg sample prepare succeed")

# 202106007修改，恢复注释
user_neg_sample_movie = {}
user_neg_sample_book = {}

# user_neg_sample_movie, user_neg_sample_book = neg_sample()

for epoch in range(args.epochs):
    # 202106007修改，恢复注释
    if epoch < 20:
        if epoch % 10 == 0:
            print("Do negative sampling")
            user_neg_sample_movie, user_neg_sample_book = neg_sample()
    elif epoch < 40:
        if epoch % 5 == 0:
            print("Do negative sampling")
            user_neg_sample_movie, user_neg_sample_book = neg_sample()
    else:
        print("Do negative sampling")
        user_neg_sample_movie, user_neg_sample_book = neg_sample()
    train(epoch)
    test(best_ndcg_a, best_hit_ratio_a, best_ndcg_b, best_hit_ratio_b, best_mrr_a, best_mrr_b, all_results, this_ndcg_a,
         this_ndcg_b)
    if epoch > 40 and not early_stop.update(epoch, this_ndcg_a[5], this_ndcg_b[5], this_ndcg_a[10], this_ndcg_b[10]):
        print(f"Best epoch get, epoch = {epoch}")
        print(SPLIT)
        break
    if (epoch + 1) % 20 == 0:
        test_best_result()
test_best_result()
print(f"Train loss list: ")
print(train_loss_list)
print(f"Test loss list: ")
print(test_loss_list)

print(args)
# process_data()
