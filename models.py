from functools import reduce

import torch.nn as nn
import torch
from utils import cal_rating


class SingleDomain(nn.Module):
    def __init__(self, embedding_size: int, user_num: int, item_num: int, device, layer_num: int,
                 model_structure: str):
        super(SingleDomain, self).__init__()
        self.user_emb = nn.Embedding(user_num, embedding_size)
        self.item_emb = nn.Embedding(item_num, embedding_size)
        self.user_index_tensor = torch.LongTensor(list(range(user_num))).to(device)
        self.item_index_tensor = torch.LongTensor(list(range(item_num))).to(device)
        self.user_num = user_num
        self.item_num = item_num
        self.device = device
        self.layer_num = layer_num
        self.model_structure = model_structure
        self.linears = nn.ModuleList()
        for _ in range(layer_num):
            self.linears.append(nn.Linear(embedding_size, embedding_size))
        self.predict_layer_u = nn.Sequential(
            nn.Linear(embedding_size * (layer_num + 1), embedding_size),
            nn.ReLU()
        )
        self.predict_layer_i = nn.Sequential(
            nn.Linear(embedding_size * (layer_num + 1), embedding_size),
            nn.ReLU()
        )
        self.predict_test_layer = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.Softmax()
        )

    # def forward(self, conv_u, conv_i, user_sample: torch.LongTensor, item_sample: torch.LongTensor):
    #     Eu = [self.user_emb(self.user_index_tensor)]
    #     Ei = [self.item_emb(self.item_index_tensor)]
    #     for i in range(self.layer_num):
    #         Mu = reduce(torch.mm, [conv_u, Eu[-1]]) + Eu[-1]
    #         Mi = reduce(torch.mm, [conv_i, Ei[-1]]) + Ei[-1]
    #         Eu.append(self.linears[i](Mu))
    #         Ei.append(self.linears[i](Mi))
    #     Eu = torch.cat(Eu, dim=1)
    #     Ei = torch.cat(Ei, dim=1)
    #     # Pu = self.predict_layer_u(Eu)
    #     # Pi = self.predict_layer_i(Ei)
    #     Pu = Eu
    #     Pi = Ei
    #     if self.model_structure == "single":
    #         score = torch.cosine_similarity(Pu[user_sample], Pi[item_sample], dim=1)
    #         return torch.clamp(score, 0, 1)
    #         # return cal_rating(Pu, Pi, user_sample, item_sample, self.device)
    #     else:
    #         return Pu, Pi

    def forward(self):
        Eu = [self.user_emb(self.user_index_tensor)]
        Ei = [self.item_emb(self.item_index_tensor)]
        return Eu, Ei


class II_HGCN(nn.Module):
    def __init__(self, embedding_size: int, user_num_a, item_num_a, user_num_b, item_num_b, device, single_layer_num,
                 dual_layer_num, model_structure: str):
        super(II_HGCN, self).__init__()
        print(f"Train in {model_structure} hypergraph!")
        self.embedding_size = embedding_size
        self.user_num_a = user_num_a
        self.item_num_a = item_num_a
        self.user_num_b = user_num_b
        self.item_num_b = item_num_b
        self.common_user_num = user_num_a
        self.device = device
        self.single_layer_num = single_layer_num
        self.dual_layer_num = dual_layer_num
        self.model_structure = model_structure
        self.domain_a_model = SingleDomain(embedding_size, user_num_a, item_num_a, device, single_layer_num,
                                           self.model_structure)
        self.domain_b_model = SingleDomain(embedding_size, user_num_b, item_num_b, device, single_layer_num,
                                           self.model_structure)
        self.param_size = self.single_layer_num + 1
        self.ac_func = nn.ReLU()
        # self.element_wise_a_user_W = nn.Parameter(torch.empty(size=(self.common_user_num,
        #                                                             embedding_size * self.param_size)),
        #                                           requires_grad=True)
        # nn.init.xavier_uniform_(self.element_wise_a_user_W.data, gain=1)
        # self.element_wise_b_user_W = nn.Parameter(torch.empty(size=(self.common_user_num,
        #                                                             embedding_size * self.param_size)),
        #                                           requires_grad=True)
        # nn.init.xavier_uniform_(self.element_wise_b_user_W.data, gain=1)
        # self.element_wise_a_item_W = nn.Parameter(torch.empty(size=(self.item_num_a,
        #                                                             embedding_size * self.param_size)),
        #                                           requires_grad=True)
        # nn.init.xavier_uniform_(self.element_wise_a_item_W.data, gain=1)
        # self.element_wise_b_item_W = nn.Parameter(torch.empty(size=(self.item_num_b,
        #                                                             embedding_size * self.param_size)),
        #                                           requires_grad=True)
        # nn.init.xavier_uniform_(self.element_wise_b_item_W.data, gain=1)
        self.linears_a = nn.ModuleList()
        for _ in range(self.single_layer_num):
            self.linears_a.append(nn.Linear(embedding_size, embedding_size))
        self.linears_b = nn.ModuleList()
        for _ in range(self.single_layer_num):
            self.linears_b.append(nn.Linear(embedding_size, embedding_size))
        self.element_wise_a_user_W = nn.ParameterList()
        self.element_wise_b_user_W = nn.ParameterList()
        self.element_wise_a_item_W = nn.ParameterList()
        self.element_wise_b_item_W = nn.ParameterList()
        for i in range(self.single_layer_num):
            self.element_wise_a_user_W.append(nn.Parameter(torch.empty(size=(self.common_user_num, embedding_size)),
                                                           requires_grad=True))
            self.element_wise_a_item_W.append(nn.Parameter(torch.empty(size=(self.item_num_a, embedding_size)),
                                                           requires_grad=True))
            self.element_wise_b_user_W.append(nn.Parameter(torch.empty(size=(self.common_user_num, embedding_size)),
                                                           requires_grad=True))
            self.element_wise_b_item_W.append(nn.Parameter(torch.empty(size=(self.item_num_b, embedding_size)),
                                                           requires_grad=True))
            nn.init.xavier_uniform_(self.element_wise_a_item_W[-1].data, gain=1)
            nn.init.xavier_uniform_(self.element_wise_b_item_W[-1].data, gain=1)
            nn.init.xavier_uniform_(self.element_wise_a_user_W[-1].data, gain=1)
            nn.init.xavier_uniform_(self.element_wise_b_user_W[-1].data, gain=1)

    def forward(self, conv_au, conv_ai, user_sample_a: torch.LongTensor, item_sample_a: torch.LongTensor,
                conv_bu, conv_bi, user_sample_b: torch.LongTensor, item_sample_b: torch.LongTensor,
                HyGCN_a, HyGCN_b):
        Eua, Eia = self.domain_a_model()
        Eub, Eib = self.domain_b_model()
        for i in range(self.single_layer_num):
            Mua = Eua[-1]
            Mia = Eia[-1]
            Mub = Eub[-1]
            Mib = Eib[-1]
            if i != 0:
                Mua = reduce(torch.mm, [conv_au, Mua])
                Mia = reduce(torch.mm, [conv_ai, Mia])
                Mub = reduce(torch.mm, [conv_bu, Mub])
                Mib = reduce(torch.mm, [conv_bi, Mib])
                Mua = self.ac_func(self.linears_a[i](Mua) + Mua)
                Mia = self.ac_func(self.linears_a[i](Mia) + Mia)
                Mub = self.ac_func(self.linears_b[i](Mub) + Mub)
                Mib = self.ac_func(self.linears_b[i](Mib) + Mib)
            if self.model_structure == "inter-user" or self.model_structure == "normal":
                Mua_inter = torch.add(torch.mul(Mua, self.element_wise_a_user_W[i]),
                                      torch.mul(Mub, 1 - self.element_wise_a_user_W[i]))
                Mub_inter = torch.add(torch.mul(Mub, self.element_wise_b_user_W[i]),
                                      torch.mul(Mua, 1 - self.element_wise_b_user_W[i]))
                Eua.append(Mua_inter)
                Eub.append(Mub_inter)
            if self.model_structure == "inter-item" or self.model_structure == "normal":
                Mia_from_b = self.ac_func(torch.matmul(HyGCN_a, Mib))
                Mib_from_a = self.ac_func(torch.matmul(HyGCN_b, Mia))
                Mia_inter = torch.add(torch.mul(Mia, self.element_wise_a_item_W[i]),
                                      torch.mul(Mia_from_b, 1 - self.element_wise_a_item_W[i]))
                Mib_inter = torch.add(torch.mul(Mib, self.element_wise_b_item_W[i]),
                                      torch.mul(Mib_from_a, 1 - self.element_wise_b_item_W[i]))
                Eia.append(Mia_inter)
                Eib.append(Mib_inter)
            if self.model_structure == "single" or self.model_structure == "inter-user":
                Eia.append(Mia)
                Eib.append(Mib)
            if self.model_structure == "single" or self.model_structure == "inter-item":
                Eua.append(Mua)
                Eub.append(Mub)
        Pua = torch.cat(Eua, dim=1)
        Pia = torch.cat(Eia, dim=1)
        Pub = torch.cat(Eub, dim=1)
        Pib = torch.cat(Eib, dim=1)
        score_a = torch.cosine_similarity(Pua[user_sample_a], Pia[item_sample_a], dim=1)
        score_b = torch.cosine_similarity(Pub[user_sample_b], Pib[item_sample_b], dim=1)
        return torch.clamp(score_a, 0, 1), torch.clamp(score_b, 0, 1)

        # return cal_rating(Pua, Pia, user_sample_a, item_sample_a, self.device), \
        #        cal_rating(Pub, Pib, user_sample_b, item_sample_b, self.device)

# def forward(self, conv_au, conv_ai, user_sample_a: torch.LongTensor, item_sample_a: torch.LongTensor,
#             conv_bu, conv_bi, user_sample_b: torch.LongTensor, item_sample_b: torch.LongTensor,
#             HyGCN_a, HyGCN_b):
#     if self.model_structure == "single":
#         score_a = self.domain_a_model(conv_au, conv_ai, user_sample_a, item_sample_a)
#         score_b = self.domain_b_model(conv_bu, conv_bi, user_sample_b, item_sample_b)
#         return score_a, score_b
#     else:
#         Pua, Pia = self.domain_a_model(conv_au, conv_ai, user_sample_a, item_sample_a)
#         Pub, Pib = self.domain_b_model(conv_bu, conv_bi, user_sample_b, item_sample_b)
#         if self.model_structure == "inter-user" or self.model_structure == "normal":
#             Pua = torch.add(torch.mul(Pua, self.element_wise_a_user_W),
#                             torch.mul(Pub, 1 - self.element_wise_a_user_W))
#             Pub = torch.add(torch.mul(Pub, self.element_wise_b_user_W),
#                             torch.mul(Pua, 1 - self.element_wise_b_user_W))
#         if self.model_structure == "inter-item" or self.model_structure == "normal":
#             for _ in range(self.dual_layer_num):
#                 Pia_from_b = torch.matmul(HyGCN_a, Pib)
#                 Pib_from_a = torch.matmul(HyGCN_b, Pia)
#                 Pia = torch.add(torch.mul(Pia, self.element_wise_a_item_W),
#                                 torch.mul(Pia_from_b, 1 - self.element_wise_a_item_W))
#                 Pib = torch.add(torch.mul(Pib, self.element_wise_b_item_W),
#                                 torch.mul(Pib_from_a, 1 - self.element_wise_b_item_W))
#
#         score_a = torch.cosine_similarity(Pua[user_sample_a], Pia[item_sample_a], dim=1)
#         score_b = torch.cosine_similarity(Pub[user_sample_b], Pib[item_sample_b], dim=1)
#         return torch.clamp(score_a, 0, 1), torch.clamp(score_b, 0, 1)
#         # return cal_rating(Pua, Pia, user_sample_a, item_sample_a, self.device), \
#         #        cal_rating(Pub, Pib, user_sample_b, item_sample_b, self.device)
