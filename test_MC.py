"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_MC import Dictionary, V7WDataset
import base_model_MC as base_model
import utils
import time
from train_MC import evaluate

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='ban', choices=['ban', 'stacked_attention', 'cti'],
                        help='the model we use')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=str, default=12)

    # BAN
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout', help='Dropout of rate of final classifier')

    parser.add_argument('--use_counter', action='store_true', default=False, help='use counter module')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'swish'],
                        help='The activation to use for final classifier')

    parser.add_argument('--question_len', default=12, type=int, metavar='N', help='maximum length of input question')
    parser.add_argument('--v_dropout', default=0.5, type=float, metavar='v_dropout', help='Dropout of rate of linearing visual embedding')

    # Tan
    parser.add_argument('--rank', default=32, type=int, help='number of rank decomposition')
    parser.add_argument('--h_out', default=1, type=int)
    parser.add_argument('--h_mm', default=512, type=int)
    parser.add_argument('--k', default=1, type=int)

    # Train with TDIUC
    parser.add_argument('--use_TDIUC', action='store_true', default=False, help='Using TDIUC dataset to train')
    parser.add_argument('--TDIUC_dir', type=str, help='TDIUC dir')

    # distributed
    parser.add_argument('--local_rank', type=int)

    # v7w
    parser.add_argument('--use_feature', default='bottom', type=str, help='use bottom-up feature or grid feature')
    # san
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')


    args = parser.parse_args()
    return args


def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)


def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]


def compute_score_mc(logits, labels):
    prob_preds = torch.softmax(logits, 1)
    result = [torch.max(prob_preds[idx*4:idx*4+4,0], 0)[1]+(idx*4) for idx in range(int(logits.size(0)/4))]
    idx = torch.stack(result)
    scores = labels[:, 0].gather(0, idx)
    return scores


def get_logits(args, model, dataloader, device):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    idx = 0
    score = 0
    bar = progressbar.ProgressBar(maxval=N)
    bar.start()
    with torch.no_grad():
        for v, b, q, a, ans_mc, ans_gt in iter(dataloader):
            bar.update(idx)
            batch_size = v.size(0)
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            ans_mc = ans_mc.to(device)

            v = v.unsqueeze(1).expand(v.size(0), 4, v.size(1), v.size(2)).contiguous().view(v.size(0) * 4, v.size(1),
                                                                                            v.size(2))
            q = q.unsqueeze(1).expand(q.size(0), 4, q.size(1)).contiguous().view(q.size(0) * 4, q.size(1))
            ans_mc = ans_mc.view(ans_mc.size(0) * ans_mc.size(1), ans_mc.size(2))
            a = a.view(ans_mc.size(0), 1)
            labels = torch.cat([a, 1 - a], 1)
            labels = labels.to(device)
            if args.model == "san":
                    logits = model(v, q, ans_mc)
            if args.model == "ban":
                    feats, att = model(v, b, q, ans_mc)
                    logits = feats
            if args.model == "cti":
                    logits, att = model(v, b, q, ans_mc)

            score += compute_score_mc(logits, labels).sum()
            idx += batch_size

    bar.update(idx)
    score = score.float() * 100.0 / len(dataloader.dataset)
    print(score)


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    torch.cuda.set_device(args.gpu)
    dictionary = Dictionary.load_from_file('data_v7w/dictionary.pkl')
    eval_dset = V7WDataset(args.split, args, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size #* n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(args, eval_dset)
    # model = nn.DataParallel(model)

    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    def process(args, model, eval_loader):

        model_path = args.input+'/model_epoch%s.pth' % (args.epoch)
        print('loading %s' % model_path)
        print(torch.cuda.current_device())
        model_data = torch.load(model_path)

        # Comment because not using multi-gpu or distributed
        # model = nn.DataParallel(model).cuda()
        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)
        score, _ = evaluate(model, eval_loader, args)
        print('\teval score: %.2f' % (100 * score))

    process(args, model, eval_loader)
