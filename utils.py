import multiprocessing
import os
from functools import reduce

from scipy.linalg import fractional_matrix_power
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csc_matrix, csr_matrix
from Dataset_CDAE import Dataset
from numba import njit, prange
import numba
import time

numba.config.NUMBA_DEFAULT_NUM_THREADS = 8
numba.config.NUMBA_NUM_THREADS = 8
import multiprocessing

# import ray
# ray.shutdown()
# ray.init()

single_k_list = [2]
# single_k_list = [2]
single_top_k_ratio = 0.5

ML_ID = 0
ML_TITLE = 1
ML_GEN = 2
NETFLIX_ID = 0
NETFLIX_TITLE = 2
NETFLIX_YEAR = 1

batch_size = 256
dataset_string = "amazon3"
sparse_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]


def remove_test_and_vali(dataset):
    for user in tqdm(dataset.peo2item_movie.keys()):
        if dataset.movie_test[user] in dataset.peo2item_movie[user]:
            dataset.peo2item_movie[user].remove(dataset.movie_test[user])
        if dataset.movie_vali[user] in dataset.peo2item_movie[user]:
            dataset.peo2item_movie[user].remove(dataset.movie_vali[user])
        if dataset.book_test[user] in dataset.peo2item_book[user]:
            dataset.peo2item_book[user].remove(dataset.book_test[user])
        if dataset.book_vali[user] in dataset.peo2item_book[user]:
            dataset.peo2item_book[user].remove(dataset.book_vali[user])


def do_dataset_sparse(dataset, Ha, Hb):
    for k, v in tqdm(dataset.peo2item_movie.items()):
        for item in v:
            if Ha[k][item] == 0:
                dataset.peo2item_movie[k].remove(item)
    for k, v in tqdm(dataset.peo2item_book.items()):
        for item in v:
            if Hb[k][item] == 0:
                dataset.peo2item_book[k].remove(item)


def my_process_bitwise_and(a, b, i, j):
    return i, j, np.count_nonzero(np.bitwise_and(a, b))


@njit(fastmath=True, parallel=True)
def H_fuse_paral(Ha, Hb, H_result):
    for i in prange(H_result.shape[0]):
        for j in range(H_result.shape[1]):
            H_result[i][j] = np.sum(np.bitwise_and(Ha[:, i], Hb[:, j]))
    return H_result


def process_sparse_process():
    dataset = Dataset(batch_size, dataset_string)
    NUM_USER = dataset.num_user
    NUM_MOVIE = dataset.num_movie
    NUM_BOOK = dataset.num_book

    print('Cal begin')
    # prepare data for domain A
    row, col = dataset.get_part_train_indices('movie', 1)
    values = np.ones(row.shape[0])
    Ha = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray()

    # prepare  data fot domain B
    row, col = dataset.get_part_train_indices('book', 1)
    values = np.ones(row.shape[0])
    Hb = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray()
    for ratio in tqdm(sparse_ratio_list):
        print(f"Sparse for {dataset_string} begin")
        result_dict = os.path.join('processed_data', dataset_string, f"sparse_{int(ratio * 100)}")
        if not os.path.exists(result_dict):
            os.mkdir(result_dict)
        print(f"out dict: {result_dict}")
        if ratio != 0.5:
            Ha_sparse = data_sparse_transfer(Ha, ratio)
            Hb_sparse = data_sparse_transfer(Hb, ratio)

            np.savez(os.path.join(result_dict, "Ha_sparse.npz"), Ha_sparse)
            np.savez(os.path.join(result_dict, "Hb_sparse.npz"), Hb_sparse)
            print("Save sparse matrix succeed！ ")

            # cal conv and conu for sparse dataset:
            print(f"Cal matrixs begin for k = 2......")
            cal_single_domain_matrix_np(Ha_sparse, 2, "cuda:1",
                                        os.path.join(result_dict, "conv_au" + f"_{2}.npz"),
                                        os.path.join(result_dict, "conv_ai" + f"_{2}.npz"))
            cal_single_domain_matrix_np(Hb_sparse, 2, "cuda:1",
                                        os.path.join(result_dict, "conv_bu" + f"_{2}.npz"),
                                        os.path.join(result_dict, "conv_bi" + f"_{2}.npz"))
            print(f"Cal matrixs for k = 2 succeed......")
            print("===================================================================")
            print(f"Cal Ha_fuse and Hb_fuse begin! ")

            # cal Ha_fuse and Hb_fuse for sparse dataset
            H_fuse_a = np.empty((NUM_MOVIE, NUM_BOOK))
            H_fuse_b = np.empty((NUM_BOOK, NUM_MOVIE))
            Ha_sparse = Ha_sparse.astype(np.int32)
            Hb_sparse = Hb_sparse.astype(np.int32)
            print(f"Ha_fuse begin!")
            t = time.time()
            H_fuse_a = H_fuse_paral(Ha_sparse, Hb_sparse, H_fuse_a)
            print(f"Ha_fuse succeed, time = {time.time() - t}")
            print(f"Hb fuse begin!")
            t = time.time()
            H_fuse_b = H_fuse_paral(Hb_sparse, Ha_sparse, H_fuse_b)
            print(f"Hb_fuse succeed, time = {time.time() - t}")
            # for i in prange(NUM_MOVIE):
            #     for j in range(NUM_BOOK):
            #         H_fuse_a[i][j] = np.count_nonzero(np.bitwise_and(Ha_sparse[:, i], Hb_sparse[:, j]))
            # for i in prange(NUM_BOOK):
            #     for j in range(NUM_MOVIE):
            #         H_fuse_b[i][j] = np.count_nonzero(np.bitwise_and(Hb_sparse[:, i], Ha_sparse[:, j]))

            print("Cal succeed")
            np.savez(os.path.join(result_dict, "Ha_fuse.npz"), H_fuse_a)
            np.savez(os.path.join(result_dict, "Hb_fuse.npz"), H_fuse_b)
        else:
            Ha_sparse = np.load(os.path.join(result_dict, "Ha_sparse.npz"))["arr_0"]
            Hb_sparse = np.load(os.path.join(result_dict, "Hb_sparse.npz"))["arr_0"]
            H_fuse_a = np.empty((NUM_MOVIE, NUM_BOOK))
            H_fuse_b = np.empty((NUM_BOOK, NUM_MOVIE))
            Ha_sparse = Ha_sparse.astype(np.int32)
            Hb_sparse = Hb_sparse.astype(np.int32)
            print(f"Hb fuse begin!")
            t = time.time()
            H_fuse_b = H_fuse_paral(Hb_sparse, Ha_sparse, H_fuse_b)
            print(f"Hb_fuse succeed, time = {time.time() - t}")
            print("Cal succeed")
            np.savez(os.path.join(result_dict, "Hb_fuse.npz"), H_fuse_b)

        print("Save succeed")
        print("===================================================================")

        # cal top k H_fuse
        print(f"Cal topk dual domain matrix, k = {10}")
        print(f"Cal Ha begin")
        dual_domain_gcn_prepare(torch.FloatTensor(H_fuse_a), 10, result_dict, "Ha_fuse")
        print(f"Cal Hb begin")
        dual_domain_gcn_prepare(torch.FloatTensor(H_fuse_b), 10, result_dict, "Hb_fuse")
        print(f"Cal topk for k = {10} succeed!")
        print("===================================================================")

        # cal GCN params for top k H_fuse
        print(f"Cal topk dual domain matrix for GCN, k = {10}")
        print(f"Cal Ha begin")
        Ha_fuse = np.load(os.path.join(result_dict, "Ha_fuse" + f"_{10}.npz"))["arr_0"]
        cal_dual_GCN_params(Ha_fuse, 10, result_dict, "Ha_GCN")
        print(f"Cal Hb begin")
        Hb_fuse = np.load(os.path.join(result_dict, "Hb_fuse" + f"_{10}.npz"))["arr_0"]
        cal_dual_GCN_params(Hb_fuse, 10, result_dict, "Hb_GCN")
        print(f"Cal topk for GCN for k = {10} succeed!")
        print(f"Sparse for {dataset_string} succeed!")
        print("===================================================================")


def process_dual_domain_matrix():
    print(f"Cal dual matrix in {dataset_string}")
    dataset = Dataset(batch_size, dataset_string)
    NUM_USER = dataset.num_user
    NUM_MOVIE = dataset.num_movie
    NUM_BOOK = dataset.num_book

    print('Cal begin')
    # prepare data for domain A
    row, col = dataset.get_part_train_indices('movie', 1)
    values = np.ones(row.shape[0])
    Ha = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray().astype(np.int32)

    # prepare  data fot domain B
    row, col = dataset.get_part_train_indices('book', 1)
    values = np.ones(row.shape[0])
    Hb = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray().astype(np.int32)

    H_fuse_a = np.empty((NUM_MOVIE, NUM_BOOK))
    H_fuse_b = np.empty((NUM_BOOK, NUM_MOVIE))

    print(f"Ha_fuse begin!")
    t = time.time()
    H_fuse_a = H_fuse_paral(Ha, Hb, H_fuse_a)
    print(f"Ha_fuse succeed, time = {time.time() - t}")
    print(f"Hb fuse begin!")
    t = time.time()
    H_fuse_b = H_fuse_paral(Hb, Ha, H_fuse_b)
    print(f"Hb_fuse succeed, time = {time.time() - t}")

    # for i in tqdm(prange(NUM_MOVIE)):
    #     for j in range(NUM_BOOK):
    #         H_fuse_a[i][j] = np.count_nonzero(np.bitwise_and(Ha[:, i], Hb[:, j]))
    # for i in tqdm(prange(NUM_BOOK)):
    #     for j in range(NUM_MOVIE):
    #         H_fuse_b[i][j] = np.count_nonzero(np.bitwise_and(Hb[:, i], Ha[:, j]))

    print("Cal succeed")
    np.savez(os.path.join('processed_data', dataset_string, "Ha_fuse.npz"), H_fuse_a)
    np.savez(os.path.join('processed_data', dataset_string, "Hb_fuse.npz"), H_fuse_b)
    print("Save succeed")


def process_data():
    print(f"Process data in {dataset_string}")
    dataset = Dataset(batch_size, dataset_string)
    NUM_USER = dataset.num_user
    NUM_MOVIE = dataset.num_movie
    NUM_BOOK = dataset.num_book

    print('Preparing the training data......')
    # prepare data for domain A
    row, col = dataset.get_part_train_indices('movie', 1)
    values = np.ones(row.shape[0])
    Ha = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_MOVIE)).toarray()
    # Ha = Ha + np.full_like(Ha, 1e-10)

    # prepare  data fot domain B
    row, col = dataset.get_part_train_indices('book', 1)
    values = np.ones(row.shape[0])
    Hb = csr_matrix((values, (row, col)), shape=(NUM_USER, NUM_BOOK)).toarray()
    # Hb = Hb + np.full_like(Hb, 1e-10)
    # Hb = csc_matrix(Hb)
    # Hb = torch.FloatTensor(Hb)

    print('Preparing the training data over......')

    for k in single_k_list:
        print(f"Cal matrixs begin for k = {k}......")
        back = f"_{k}_top05.npz"
        print(f"back = {back}")
        cal_single_domain_matrix_np(Ha, k, "cuda:1",
                                    os.path.join('processed_data', dataset_string, "conv_au" + back),
                                    os.path.join('processed_data', dataset_string, "conv_ai" + back))
        cal_single_domain_matrix_np(Hb, k, "cuda:1",
                                    os.path.join('processed_data', dataset_string, "conv_bu" + back),
                                    os.path.join('processed_data', dataset_string, "conv_bi" + back))

        print(f"Cal matrixs for k = {k} succeed......")


def get_movie_data_ml():
    ml_df = [[], [], []]
    with open("Data/ml-25m/movies.csv") as f:
        lines = f.readlines()
        for line in lines[1:]:
            record = line.strip().split(",", 2)
            ml_df[ML_ID].append(record[ML_ID].strip())
            ml_df[ML_TITLE].append(record[ML_TITLE].strip())
            ml_df[ML_GEN].append(record[ML_GEN].strip())
    # for i in range(len(ml_df[ML_TITLE])):
    #     ml_df[ML_TITLE][i] = ml_df[ML_TITLE][i].strip().rsplit(" ", 1)[0]
    return ml_df


def get_movie_data_netflix():
    netflix_df = [[], [], []]
    with open("Data/netflix/movie_titles.csv") as f:
        lines = f.readlines()
        for line in lines:
            record = line.strip().split(",", 2)
            netflix_df[NETFLIX_ID].append(record[NETFLIX_ID].strip())
            netflix_df[NETFLIX_YEAR].append(record[NETFLIX_YEAR].strip())
            netflix_df[NETFLIX_TITLE].append(record[NETFLIX_TITLE].strip() + " (" + record[NETFLIX_YEAR].strip() + ")")
    return netflix_df


def get_common_item_id():
    movie_data_ml = get_movie_data_ml()
    movie_data_netflix = get_movie_data_netflix()
    common_title_list = list(set(movie_data_ml[ML_TITLE]).intersection(set(movie_data_netflix[NETFLIX_TITLE])))
    result_dict = {}
    for i in tqdm(range(len(common_title_list))):
        title = common_title_list[i]
        if title in movie_data_ml[ML_TITLE] and title in movie_data_netflix[NETFLIX_TITLE]:
            result_dict[title] = (movie_data_ml[ML_ID][movie_data_ml[ML_TITLE].index(title)],
                                  movie_data_netflix[NETFLIX_ID][movie_data_netflix[NETFLIX_TITLE].index(title)])
    with open("Data/common_movie.csv", "w") as f:
        for key, value in result_dict.items():
            f.write(key + "," + str(value[0]) + "," + str(value[1]) + "\n")


def get_rating_record(filename, device):
    df = pd.read_csv(filename)
    user_list = list(df["user_id"].drop_duplicates())
    item_list = list(df["movie_id"].drop_duplicates())
    user2id = {}
    id2user = {}
    item2id = {}
    id2item = {}
    for i in range(len(user_list)):
        user2id[user_list[i]] = i
        id2user[i] = user_list[i]
    for i in range(len(item_list)):
        item2id[item_list[i]] = i
        id2item[i] = item_list[i]
    record = torch.Tensor(np.array(df)).to(device)


def get_one_netflix_file(common_movie: list, filename):
    movie_id = 0
    data = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            if line.endswith(":"):
                movie_id = int(line.strip().split(":")[0])
            else:
                if movie_id in common_movie:
                    rating_record = line.strip().split(",")
                    user_id = int(rating_record[0])
                    rating = int(rating_record[1])
                    data.append([user_id, movie_id, rating])
    return data


def get_rating_data(file: str, device, outfile):
    data = []
    common = []
    out_header = ["user_id", "movie_id", "rating"]
    with open("Data/common_movie.csv", "r") as f:
        reader = csv.reader(f)
        for line in reader:
            common.append([line[0], int(line[1]), int(line[2])])
    if file == "ml":
        common_movie = [record[1] for record in common]
        with open("Data/ml-25m/ratings.csv", "r") as f:
            reader = csv.reader(f)
            next(reader)
            for line in tqdm(reader):
                movie_id = int(line[1])
                if movie_id in common_movie:
                    data.append([int(line[0]), movie_id, float(line[2])])
        pd.DataFrame(data=data).to_csv(outfile, index=False, header=out_header)
    if file == "netflix":
        common_movie = [record[2] for record in common]
        file_dict = "Data/netflix/"
        file_list = ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]
        data = []
        for file in file_list:
            data.extend(get_one_netflix_file(common_movie, file_dict + file))
        pd.DataFrame(data=data).to_csv(outfile, index=False, header=out_header)


def calculate_D_matrix(H):
    # v_degree = H.sum(axis=1)
    # e_degree = H.sum(axis=0)
    v_degree = np.array(H.sum(axis=1).reshape(1, H.sum(axis=1).shape[0])).squeeze()
    e_degree = np.array(H.sum(axis=0)).squeeze()
    # return torch.diag(v_degree), torch.diag(e_degree)
    return sp.diags(v_degree), sp.diags(e_degree)


def calculate_D_matrix_np(H):
    v_degree = H.sum(axis=1)
    e_degree = H.sum(axis=0)
    # return torch.diag(v_degree), torch.diag(e_degree)
    return np.diag(v_degree).astype(np.float32), np.diag(e_degree).astype(np.float32)


def cal_single_domain_matrix_np(single_H, k: int, device, out_file_u, out_file_i):
    single_Hu = [single_H]
    single_Hi = [single_H.T]
    H_for_user = np.matmul(single_H.T, single_H)
    H_for_item = np.matmul(single_H, single_H.T)
    Hu_k = H_for_user
    Hi_k = H_for_item
    print(f"Calculate Hu_k succeed")
    one_u = np.ones_like(Hu_k)
    one_i = np.ones_like(Hi_k)
    for i in tqdm(range(1, k)):
        # single_Hu.append(np.matmul(single_H, np.minimum(Hu_k, one_u)))
        # single_Hi.append(np.matmul(single_H.T, np.minimum(Hi_k, one_i)))

        Hu_add = np.matmul(single_H, np.minimum(Hu_k, one_u))
        Hi_add = np.matmul(single_H.T, np.minimum(Hi_k, one_i))
        Hud_full = np.full(Hu_add.shape, single_H.sum(axis=1).reshape((-1, 1)))
        Hid_full = np.full(Hi_add.shape, single_H.sum(axis=0).reshape((-1, 1)))
        Hu_column = Hu_add * Hud_full
        Hi_column = Hi_add * Hid_full
        Hu_column_sum = sorted(np.sum(Hu_column, axis=0))
        Hi_column_sum = sorted(np.sum(Hi_column, axis=0))

        Hu_flag = Hu_column_sum[int(len(Hu_column_sum) * single_top_k_ratio)]
        Hi_flag = Hi_column_sum[int(len(Hi_column_sum) * single_top_k_ratio)]
        for j in tqdm(range(len(Hu_column_sum))):
            if Hu_column_sum[j] < Hu_flag:
                Hu_add[:, j] = 0
        for j in tqdm(range(len(Hi_column_sum))):
            if Hi_column_sum[j] < Hi_flag:
                Hi_add[:, j] = 0
        single_Hu.append(Hu_add)
        single_Hi.append(Hi_add)

        print(f"Append single H succeed")
        Hu_k = np.matmul(Hu_k, H_for_user)
        Hi_k = np.matmul(Hi_k, H_for_item)
    single_Hu = np.concatenate(single_Hu, axis=1)
    single_Hi = np.concatenate(single_Hi, axis=1)
    print(f"Concatenate succeed")
    Duv, Due = calculate_D_matrix_np(single_Hu)
    Div, Die = calculate_D_matrix_np(single_Hi)
    print(f"Calculate D matrix succeed")
    Duv = my_power_D(Duv, -0.5)
    Div = my_power_D(Div, -0.5)
    Due = my_power_D(Due, -1)
    Die = my_power_D(Die, -1)
    # Duv = fractional_matrix_power(Duv, -0.5)
    # Div = fractional_matrix_power(Div, -0.5)
    # Due = fractional_matrix_power(Due, -1)
    # Die = fractional_matrix_power(Die, -1)
    print(f"Calculate power succeed")
    conv_u = reduce(np.matmul, [Duv, single_Hu, Due, single_Hu.T, Duv])
    conv_i = reduce(np.matmul, [Div, single_Hi, Die, single_Hi.T, Div])
    print(f"matmul process succeed")
    np.savez(out_file_u, conv_u)
    np.savez(out_file_i, conv_i)
    print(f"save succeed")


# @ray.remote(num_cpus=10)
def cal_single_domain_matrix(single_H: csc_matrix, k: int, device, out_file_u, out_file_i):
    # single_Hu = [single_H]
    # single_Hi = [single_H.T]
    # H_for_user = np.matmul(single_H.T, single_H)
    # H_for_item = np.matmul(single_H, single_H.T)
    single_Hu = [single_H]
    single_Hi = [single_H.transpose()]
    H_for_user = single_H.transpose().__matmul__(single_H)
    H_for_item = single_H.__matmul__(single_H.transpose())
    # H_for_user = torch.mm(single_H.T, single_H)
    # H_for_item = torch.mm(single_H, single_H.T)
    Hu_k = H_for_user
    Hi_k = H_for_item
    print(f"Calculate Hu_k succeed")
    Hu_k.data[:] = 1
    Hi_k.data[:] = 1
    print(f"Transfer to 1 succeed")
    # one_u = np.ones_like(Hu_k)
    # one_i = np.ones_like(Hi_k)
    # one_u = torch.ones(size=Hu_k.shape).to(device)
    # one_i = torch.ones(size=Hi_k.shape).to(device)
    # 到时候尝试一下不用（0，1）的结果
    for i in tqdm(range(1, k)):
        # single_Hu.append(torch.mm(single_H, torch.minimum(Hu_k, one_u)))
        # single_Hi.append(torch.mm(single_H.T, torch.minimum(Hi_k, one_i)))
        # Hu_k = torch.mm(Hu_k, H_for_user)
        # Hi_k = torch.mm(Hi_k, H_for_item)
        single_Hu.append(single_H.__matmul__(Hu_k))
        single_Hi.append(single_H.transpose().__matmul__(Hi_k))
        print(f"Append single H succeed")
        # single_Hu.append(np.matmul(single_H, np.minimum(Hu_k, one_u)))
        # single_Hi.append(np.matmul(single_H.T, np.minimum(Hi_k, one_i)))
        # Hu_k = np.matmul(Hu_k, H_for_user)
        # Hi_k = np.matmul(Hi_k, H_for_item)

        # 如果k>2，需要加上以下两行
        # Hu_k = Hu_k.__matmul__(H_for_user)
        # Hi_k = Hi_k.__matmul__(H_for_item)

    # single_Hu = torch.cat(single_Hu, dim=1)
    # single_Hi = torch.cat(single_Hi, dim=1)
    # single_Hu = np.concatenate(single_Hu, axis=1)
    # single_Hi = np.concatenate(single_Hi, axis=1)
    single_Hu = sp.hstack(single_Hu)
    single_Hi = sp.hstack(single_Hi)
    print(f"Concatenate succeed")
    Duv, Due = calculate_D_matrix(single_Hu)
    Div, Die = calculate_D_matrix(single_Hi)
    print(f"Calculate D matrix succeed")
    # 在求-0.5次方的时候对角线元素必须都有值，否则求出的结果所有元素会都为nan
    # Duv = torch.FloatTensor(fractional_matrix_power(Duv.cpu(), -0.5)).to(device)
    # Div = torch.FloatTensor(fractional_matrix_power(Div.cpu(), -0.5)).to(device)
    # Due = torch.FloatTensor(fractional_matrix_power(Due.cpu(), -1)).to(device)
    # Die = torch.FloatTensor(fractional_matrix_power(Die.cpu(), -1)).to(device)
    # Duv = fractional_matrix_power(Duv, -0.5)
    # Div = fractional_matrix_power(Div, -0.5)
    # Due = fractional_matrix_power(Due, -1)
    # Die = fractional_matrix_power(Die, -1)
    Duv = csc_matrix.power(Duv, -0.5)
    Div = csc_matrix.power(Div, -0.5)
    Due = csc_matrix.power(Due, -1)
    Die = csc_matrix.power(Die, -1)
    print(f"Calculate power succeed")
    # conv_u = reduce(np.matmul, [Duv, single_Hu, Due, single_Hu.T, Duv])
    # conv_i = reduce(np.matmul, [Div, single_Hi, Die, single_Hi.T, Div])
    conv_u = reduce(csc_matrix.__matmul__, [Duv, single_Hu, Due, single_Hu.T, Duv])
    conv_i = reduce(csc_matrix.__matmul__, [Div, single_Hi, Die, single_Hi.T, Div])
    print(conv_u.toarray())
    print(conv_i.toarray())
    print(f"matmul process succeed")
    # np.savez(out_file_u, conv_u)
    # np.savez(out_file_i, conv_i)
    sp.save_npz(out_file_u, conv_u)
    sp.save_npz(out_file_i, conv_i)
    print(f"save succeed")
    # return single_Hu, single_Hi, Duv, Due, Div, Die


def hitRatio(pred_label, k):
    hit = 0
    values, indices = torch.topk(pred_label, k)
    indices = indices.cpu().numpy()
    for i in range(pred_label.shape[0]):
        if 0 in indices[i]:
            hit += 1
    return hit / pred_label.shape[0]


def MRR(pred_label, k):
    mrr = 0
    values, indices = torch.topk(pred_label, k)
    indices = indices.cpu().numpy()
    for i in range(pred_label.shape[0]):
        for j in range(len(indices[i])):
            if indices[i][j] == 0:
                mrr += 1 / (j + 1)
                break
    return mrr / pred_label.shape[0]


def dual_domain_gcn_prepare(H, k, out_file_dict, out_file_name):
    out_file_name = os.path.join(out_file_dict, out_file_name + f"_{k}.npz")
    print(f"Out file path : {out_file_name}")
    values, indices = torch.topk(H, k)
    # values = values.cpu().numpy()
    # indices = values.cpu().numpy()
    values = values.flatten()
    row = []
    col = []
    for i in tqdm(range(len(indices))):
        for index in indices[i]:
            row.append(i)
            col.append(index.data)
    H_result = csr_matrix((values, (row, col)), shape=(H.shape)).toarray()
    np.savez(out_file_name, H_result)


def cal_dual_GCN_params(H, k, out_file_dict, out_file_name):
    out_file_name = os.path.join(out_file_dict, out_file_name + f"_{k}.npz")
    print(f"Out file path : {out_file_name}")
    Dv, De = calculate_D_matrix_np(H)
    Dv21 = my_power_D(Dv, -0.5)
    De21 = my_power_D(De, -0.5)
    result = reduce(np.matmul, [Dv21, H, De21])
    np.savez(out_file_name, result)
    print("Save succeed")


def my_power_D(H, pow):
    result = np.zeros_like(H)
    for i in tqdm(range(H.shape[0])):
        if H[i][i] != 0:
            result[i][i] = H[i][i] ** pow
    return result


def cal_all_dual_GCN_params():
    for k in range(0, 11):
        print(f"Cal topk dual domain matrix for GCN, k = {k}")
        print(f"Cal Ha begin")
        Ha_fuse = np.load(os.path.join('processed_data', dataset_string, "Ha_fuse" + f"_{k}.npz"))["arr_0"]
        cal_dual_GCN_params(Ha_fuse, k, os.path.join('processed_data', dataset_string), "Ha_GCN")
        print(f"Cal Hb begin")
        Hb_fuse = np.load(os.path.join('processed_data', dataset_string, "Hb_fuse" + f"_{k}.npz"))["arr_0"]
        cal_dual_GCN_params(Hb_fuse, k, os.path.join('processed_data', dataset_string), "Hb_GCN")
        print(f"Cal topk for GCN for k = {k} succeed!")
        print("===================================================================")


def data_sparse_transfer(H, sparse_ratio):
    num_interaction = np.sum(H == 1)
    original_sparse_ratio = num_interaction / (H.shape[0] * H.shape[1])
    drop_interaction_num = int(num_interaction * sparse_ratio)
    interaction_indices = np.argwhere(H == 1)
    drop_indices = np.random.choice(range(interaction_indices.shape[0]), drop_interaction_num, replace=False)
    H_sparse = H.copy()
    for index in drop_indices:
        x = interaction_indices[index][0]
        y = interaction_indices[index][1]
        H_sparse[x][y] = 0
    print(f"Original sparsity: {original_sparse_ratio}, "
          f"Now sparsity: {np.sum(H_sparse == 1) / (H_sparse.shape[0] * H_sparse.shape[1])}")
    return H_sparse


class EarlyStopping:
    def __init__(self) -> None:
        self.best_epoch = -1
        self.best_ndcg_a_5 = -1
        self.best_ndcg_b_5 = -1
        self.best_ndcg_a_10 = -1
        self.best_ndcg_b_10 = -1
        self.patience = 10

    def update(self, epoch: int, test_ndcg_a_5, test_ndcg_b_5, test_ndcg_a_10, test_ndcg_b_10):
        if test_ndcg_a_5 > self.best_ndcg_a_5 or test_ndcg_b_5 > self.best_ndcg_b_5 \
                or test_ndcg_a_10 > self.best_ndcg_a_10 or test_ndcg_b_10 > self.best_ndcg_b_10:
            self.best_ndcg_a_5 = max(self.best_ndcg_a_5, test_ndcg_a_5)
            self.best_ndcg_b_5 = max(self.best_ndcg_b_5, test_ndcg_b_5)
            self.best_ndcg_a_10 = max(self.best_ndcg_a_10, test_ndcg_a_10)
            self.best_ndcg_b_10 = max(self.best_ndcg_b_10, test_ndcg_b_10)
            self.best_epoch = epoch
            return 1
        else:
            if epoch - self.best_epoch > self.patience:
                return 0
            else:
                return 1

def cal_rating(Pu, Pi, user_sample, item_sample, device):
    return torch.sum(Pu[user_sample] * Pi[item_sample], dim=1)


if __name__ == '__main__':
    # step 1
    process_data()
    # step 2
    process_dual_domain_matrix()
    # step 3
    Ha_fuse = torch.FloatTensor(np.load(os.path.join('processed_data', dataset_string, "Ha_fuse.npz"))["arr_0"])
    Hb_fuse = torch.FloatTensor(np.load(os.path.join('processed_data', dataset_string, "Hb_fuse.npz"))["arr_0"])
    for k in range(0, 11):
        print(f"Cal topk dual domain matrix, k = {k}")
        print(f"Cal Ha begin")
        dual_domain_gcn_prepare(Ha_fuse, k, os.path.join('processed_data', dataset_string), "Ha_fuse")
        print(f"Cal Hb begin")
        dual_domain_gcn_prepare(Hb_fuse, k, os.path.join('processed_data', dataset_string), "Hb_fuse")
        print(f"Cal topk for k = {k} succeed!")
        print("===================================================================")
    # step 4
    cal_all_dual_GCN_params()
