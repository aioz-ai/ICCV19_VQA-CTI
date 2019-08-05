"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import torch
import torch.nn as nn
from utils import tfidf_loading
from attention import BiAttention, TriAttention, PDBiAttention, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet, FCSTL
from bc import BCNet
from tc import TCNet, PDBCNet
from counting import Counter
import time


class BanModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, wa_emb, ans_emb, v_att, b_net, va_att, tva_net,\
                 q_prj, a_prj, c_prj, classifier, combining_layer, counter, op, glimpse):
        super(BanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.wa_emb = wa_emb
        self.ans_emb = ans_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.a_prj = nn.ModuleList(a_prj)
        self.va_att = va_att
        self.tva_net = nn.ModuleList(tva_net)
        if counter is not None:
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter

    def forward(self, v, b, q, ans):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        wa_emb = self.wa_emb(ans)
        ans_emb = self.ans_emb.forward_all(wa_emb)

        boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.glimpse
        va_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q
        va_att, va_logits = self.va_att.forward_all(v, ans_emb)
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :]) # b x l x h
            va_emb[g] = self.tva_net[g].forward_with_weights(v, ans_emb, va_att[:, g, :, :])
            if self.counter is not None:
                atten, _ = logits[:,g,:,:].max(2)
                embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            ans_emb = self.a_prj[g](va_emb[g].unsqueeze(1)) + ans_emb
            if self.counter is not None:
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        q_emb = q_emb.sum(1) + ans_emb.sum(1)
        logits = self.classifier(q_emb)
        return logits, att


class PDBanModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, intermediate_layer,\
                 combining_layer, counter, op, glimpse):
        super(PDBanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q
        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights_none_tensor(v, q_emb, att[:,g,:,:]) # b x l x h
            if self.counter is not None:
                atten, _ = logits[:,g,:,:].max(2)
                embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            if self.counter is not None:
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))
        return logits, att


class StackedAttentionModel(nn.Module):
    def __init__(self, w_emb, q_emb, wa_emb, a_emb, v_att, va_att, classifier):
        super(StackedAttentionModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.wa_emb = wa_emb
        self.a_emb = a_emb
        self.va_att = va_att
        self.v_att = v_att
        self.classifier = classifier

    def forward(self, v, q, a):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        wa_emb = self.wa_emb(a)
        a_emb = self.a_emb(wa_emb)

        vq_logits = self.v_att(v, q_emb)
        va_logits = self.va_att(v, a_emb)
        logits = vq_logits + va_logits
        logits = self.classifier(logits)
        return logits


class TanModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, wa_emb, ans_emb, v_att, t_net, q_prj, a_prj, classifier, op, glimpse):
        super(TanModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.wa_emb = wa_emb
        self.ans_emb = ans_emb
        self.v_att = v_att
        self.t_net = nn.ModuleList(t_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.a_prj = nn.ModuleList(a_prj)
        self.classifier = classifier

    def forward(self, v, b, q, ans):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]

        wa_emb = self.wa_emb(ans)
        ans_emb = self.ans_emb.forward_all(wa_emb)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att(v, q_emb, ans_emb)  # b x v x q x a
        for g in range(self.glimpse):
            b_emb[g] = self.t_net[g].forward_with_weights_none_tensor(v, q_emb, ans_emb, att[:, :, :, :, g])
            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            ans_emb = self.a_prj[g](b_emb[g].unsqueeze(1)) + ans_emb

        q_emb = q_emb.sum(1) + ans_emb.sum(1)
        logits = self.classifier(q_emb)
        return logits, att


class STLModel(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, wa_emb, a_emb, v_att, va_att, v_emb, ques_emb,\
                 ans_emb, QI_emb, classifier, op, glimpse):
        super(STLModel, self).__init__()
        self.dataset = dataset
        self.op = op
        self.glimpse = glimpse
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.wa_emb = wa_emb
        self.a_emb = a_emb
        self.v_att = v_att
        self.va_att = va_att
        self.v_emb = v_emb
        self.ques_emb = ques_emb
        self.ans_emb = ans_emb
        self.QI_emb = QI_emb
        self.classifier = classifier

    def forward(self, v, q, ans):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]

        wa_emb = self.wa_emb(ans)
        ans_emb = self.a_emb.forward_all(wa_emb)

        vq_att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q
        va_att, va_logits = self.va_att.forward_all(v, ans_emb)

        vq_att = torch.max(vq_att, 3)[0]
        va_att = torch.max(va_att, 3)[0]

        qa_att = vq_att + va_att
        qa_att = torch.div(qa_att, torch.max(qa_att, 2)[0].unsqueeze(2))

        for g in range(self.glimpse):
            v_emb = self.v_emb(v)
            v_emb = torch.matmul(v_emb.transpose(2, 1), qa_att[:, g, :].unsqueeze(2)).squeeze(2)
            state_que = torch.div(q_emb.sum(1), q_emb.size(1))
            state_ans = torch.div(ans_emb.sum(1), ans_emb.size(1))
            Q_emb = self.ques_emb(state_que)
            A_emb = self.ans_emb(state_ans)

            QI = torch.mul(Q_emb, v_emb)
            QI_emb = self.QI_emb(QI)

            QIA = torch.mul(QI_emb, A_emb)

        logits = self.classifier(QIA)
        return logits


def build_ban(args, dataset):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)
    wa_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    ans_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data')
        wa_emb = tfidf_loading(args.tfidf, wa_emb, args)
    v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    va_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    b_net = []
    tva_net = []
    a_prj = []
    q_prj = []
    c_prj = []
    objects = 10 # minimum number of boxes
    for i in range(args.gamma):
        b_net.append(BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1))
        tva_net.append(BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1))
        q_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        a_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, args.num_hid], 'ReLU', .0))
    classifier = (SimpleClassifier(args.num_hid, args.num_hid * 2, 2, args))
    combining_layer = nn.Linear(args.gamma, 1, False)
    if args.use_counter:
        counter = Counter(objects)
    else:
        counter = None
    return BanModel(dataset, w_emb, q_emb, wa_emb, ans_emb, v_att, b_net, va_att, tva_net, q_prj, a_prj, c_prj, \
                    classifier, combining_layer, counter, args.op, args.gamma)


def build_pdban(args, dataset):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data')
    v_att = PDBiAttention(dataset.v_dim, args.num_hid, args.h_mm, 1, args.rank, args.gamma)
    b_net = []
    q_prj = []
    c_prj = []
    intermediate_layer = []
    objects = 10 # minimum number of boxes
    for i in range(args.gamma):
        b_net.append(PDBCNet(dataset.v_dim, args.num_hid, args.h_mm, args.h_out, args.rank, 1, k=2))
        q_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, args.num_hid], 'ReLU', .0))
        intermediate_layer.append(FCNet([args.h_out, args.num_hid], act='ReLU', dropout=.2))
    classifier = (SimpleClassifier(args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args))


    combining_layer = nn.Linear(args.gamma, 1, False)

    if args.use_counter:
        counter = Counter(objects)
    else:
        counter = None
    return PDBanModel(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, intermediate_layer,\
                      combining_layer, counter, args.op, args.gamma)


def build_tan(args, dataset):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)

    wa_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    ans_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data')
        wa_emb = tfidf_loading(args.tfidf, wa_emb, args)

    v_att = TriAttention(dataset.v_dim, args.num_hid, args.num_hid, args.h_mm, 1, args.rank, args.gamma, args.k, \
                         dropout=[.2, .5])
    t_net = []
    q_prj = []
    a_prj = []
    for i in range(args.gamma):
        t_net.append(TCNet(dataset.v_dim, args.num_hid, args.num_hid, args.h_mm, args.h_out, args.rank, 1, \
                           dropout=[.2, .5], k=2))
        q_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        a_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
    classifier = SimpleClassifier(args.num_hid, args.num_hid*2, 2, args)
    return TanModel(dataset, w_emb, q_emb, wa_emb, ans_emb, v_att, t_net, q_prj, a_prj, \
                    classifier, args.op, args.gamma)


def build_stacked_attention(args, dataset):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0)

    wa_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    a_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0)

    v_att = StackedAttention(args.num_stacks, dataset.v_dim, args.num_hid, args.num_hid, dataset.num_ans_candidates,
                             args.dropout)

    va_att = StackedAttention(args.num_stacks, dataset.v_dim, args.num_hid, args.num_hid, dataset.num_ans_candidates,
                             args.dropout)
    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data')
        wa_emb = tfidf_loading(args.tfidf, wa_emb, args)

    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, 2, args)
    return StackedAttentionModel(w_emb, q_emb, wa_emb, a_emb, v_att, va_att, classifier)

def build_STL(args, dataset):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)

    wa_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    a_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)

    v_emb = FCSTL([dataset.v_dim, args.num_hid], act='Tanh')

    ques_emb = FCSTL([args.num_hid, args.num_hid], act='Tanh')
    ans_emb = FCSTL([args.num_hid, args.num_hid], act='Tanh')
    QI_emb = FCSTL([args.num_hid, args.num_hid], act='Tanh')



    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args, 'data')
        wa_emb = tfidf_loading(args.tfidf, wa_emb, args)
    v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    va_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    classifier = SimpleClassifier(args.num_hid, 2 * args.num_hid, 2, args)

    return STLModel(dataset, w_emb, q_emb, wa_emb, a_emb, v_att, va_att, v_emb, ques_emb, ans_emb, QI_emb, classifier, \
                    args.op, args.gamma)


