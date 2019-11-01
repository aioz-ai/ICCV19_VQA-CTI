"""
Compact Trilinear Interaction
This code is written by Huy Tran.
"""
import torch
import torch.nn as nn
from src.fc import FCNet
import src.Tensor as Tensor
class TCNet(nn.Module):
    def __init__(self, v_dim, q_dim, a_dim, h_dim, h_out, rank, glimpse, act='ReLU', dropout=[.2, .5], k=1):
        super(TCNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.a_dim = a_dim
        self.h_out = h_out
        self.rank  = rank
        self.h_dim = h_dim*k
        self.hv_dim = int(h_dim/rank)
        self.hq_dim = int(h_dim/rank)
        self.ha_dim = int(h_dim/rank)


        self.v_tucker = FCNet([v_dim, self.h_dim], act=act, dropout=dropout[1])
        self.q_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])
        if self.h_dim < 1024:
            self.a_tucker = FCNet([a_dim, self.h_dim], act=act, dropout=dropout[0])
            self.v_net = nn.ModuleList([FCNet([self.h_dim, self.hv_dim], act=act, dropout=dropout[1]) for _ in range(rank)])
            self.q_net = nn.ModuleList([FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0]) for _ in range(rank)])
            self.a_net = nn.ModuleList([FCNet([self.h_dim, self.ha_dim], act=act, dropout=dropout[0]) for _ in range(rank)])

            if h_out > 1:
                self.ho_dim = int(h_out / rank)
                h_out = self.ho_dim

            self.T_g = nn.Parameter(torch.Tensor(1, rank, self.hv_dim, self.hq_dim, self.ha_dim, glimpse, h_out).normal_())
        self.dropout = nn.Dropout(dropout[1])


    def forward(self, v, q, a):
        f_emb = 0
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        a_tucker = self.a_tucker(a)
        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)
            a_ = self.a_net[r](a_tucker)
            f_emb = Tensor.ModeProduct(self.T_g[:, r, :, :, :, :, :], v_, q_, a_, None) + f_emb

        return f_emb.squeeze(4)

    def forward_with_weights(self, v, q ,a ,w):
        v_ = self.v_tucker(v).transpose(2, 1) #b x d x v
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3) #b x d x q x 1
        a_ = self.a_tucker(a).transpose(2, 1).unsqueeze(3) #b x d x a

        logits = torch.einsum('bdv,bvqa,bdqi,bdaj->bdij',[v_,w,q_,a_])
        logits = logits.squeeze(3).squeeze(2)
        return logits
