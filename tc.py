import torch
import torch.nn as nn
from fc import FCNet
import Tensor
import time
from numba import cuda

@cuda.jit()
def numba_loop(v, v_net, rank):
    v_ = [v_net[r](v) for r in range(rank)]
    return v_

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
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        a_tucker = self.a_tucker(a)

        v_ = [self.v_net[r](v_tucker) for r in range(self.rank)]
        v_ = torch.stack(v_, 1)
        q_ = [self.q_net[r](q_tucker) for r in range(self.rank)]
        q_ = torch.stack(q_, 1)
        a_ = [self.a_net[r](a_tucker) for r in range(self.rank)]
        a_ = torch.stack(a_, 1)
        f_emb = torch.einsum('brijkg,briv,brjq,brka->brvqag',
                             [self.T_g.squeeze(6).expand(v_.size(0), self.rank, self.hv_dim, self.hq_dim, self.ha_dim, 2),
                              v_.transpose(3, 2), q_.transpose(3, 2), a_.transpose(3, 2)])
        f_emb = f_emb.sum(1)


        # for r in range(self.rank):
        #     v_ = self.v_net[r](v_tucker)
        #     q_ = self.q_net[r](q_tucker)
        #     a_ = self.a_net[r](a_tucker)
        #     f_emb = Tensor.ModeProduct(self.T_g[:, r, :, :, :, :, :], v_, q_, a_, None) + f_emb
        return f_emb.squeeze(4)

    def forward_with_weights(self, v, q ,a ,w):
        v_ = self.v_tucker(v).transpose(2, 1)  # b x d x v
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3)  # b x d x q x 1
        a_ = self.a_tucker(a).transpose(2, 1).unsqueeze(3)  # b x d x a
        #logits = torch.matmul(torch.matmul(q_, torch.matmul(v_, w.view(-1, 1, v_dim, q_dim*a_dim)).view(-1, self.h_dim, q_dim, a_dim))\
        #                      , a_)
        logits = torch.einsum('bdv,bvqa,bdqi,bdaj->bdij', [v_, w, q_, a_])
        logits = logits.squeeze(3).squeeze(2)
        return logits


class PDBCNet(nn.Module):
    def __init__(self, v_dim, q_dim, h_dim, h_out, rank, glimpse, act='ReLU', dropout=[.2, .1], k=1):
        super(PDBCNet, self).__init__()

        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_out = h_out
        self.rank  = rank
        self.h_dim = h_dim*k
        self.hv_dim = int(h_dim/rank)
        self.hq_dim = int(h_dim/rank)
        self.ha_dim = int(h_dim/rank)
        self.glimpse = glimpse

        self.v_tucker = FCNet([v_dim, self.h_dim], act=act, dropout=dropout[0])
        self.q_tucker = FCNet([q_dim, self.h_dim], act=act, dropout=dropout[0])
        if self.h_dim < 2048:
            self.v_net = nn.ModuleList([FCNet([self.h_dim, self.hv_dim], act=act, dropout=dropout[0]) for _ in range(rank)])
            self.q_net = nn.ModuleList([FCNet([self.h_dim, self.hq_dim], act=act, dropout=dropout[0]) for _ in range(rank)])

            if h_out > 1:
                self.ho_dim = int(h_out / rank)
                h_out = self.ho_dim

            self.T_g = nn.Parameter(torch.Tensor(1, rank, self.hv_dim, self.hq_dim, glimpse, h_out).normal_())
        self.dropout = nn.Dropout(dropout[1])


    def forward(self, v, q):
        f_emb = 0
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)

            f_emb = Tensor.ModeProduct(self.T_g[:, r, :, :, :, :], v_, q_, None, None, 2) + f_emb

        f_emb = f_emb.transpose(3,1).transpose(3,2)
        return f_emb.squeeze(4)

    def forward_with_weights(self, v, q, w):
        fz_emb = [0]*self.rank
        v_tucker = self.v_tucker(v)
        q_tucker = self.q_tucker(q)
        for r in range(self.rank):
            v_ = self.v_net[r](v_tucker)
            q_ = self.q_net[r](q_tucker)

            fz_emb[r] = Tensor.ModeProduct(self.T_g[:, r, :, :, :, :], v_, q_, None, None, 2)
        fz_emb = torch.cat(fz_emb, dim=4)
        logits = torch.mul(w.unsqueeze(3), fz_emb.squeeze(3))
        return logits.sum(1).sum(1)

    def forward_with_weights_none_tensor(self, v, q, w):
        v_ = self.v_tucker(v).transpose(2, 1).unsqueeze(2)
        q_ = self.q_tucker(q).transpose(2, 1).unsqueeze(3)

        logits = torch.matmul(torch.matmul(v_, w.unsqueeze(1)), q_)  # b x d x 1 x 1
        logits = logits.squeeze(3).squeeze(2)
        return logits


if __name__=='__main__':
    torch.cuda.set_device(1)
    v = torch.randn(32, 50, 2048).cuda()
    q = torch.randn(32, 12, 1024).cuda()
    a = torch.randn(32, 3129, 1024).cuda()

    #tri_att = attention.TriAttention(2048, 1024, 1024, h_dim =160, h_out=1, rank=4).cuda()
    tri_emb = TCNet(2048, 1024, 1024, h_dim=160, h_out=1, rank=8).cuda()

    #att, logits = tri_att(v, q, a)
    emb = tri_emb(v, q, a)
    emb = tri_emb.forward_with_weights(v, q, a, None)
