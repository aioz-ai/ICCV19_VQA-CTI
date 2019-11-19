"""
This code is modified from jnhwkim's repository.
https://github.com/jnhwkim/ban-vqa
"""
import argparse
import json
import progressbar
import torch
from torch.utils.data import DataLoader
from src.FFOE.dataset import Dictionary, VQAFeatureDataset, TDIUCFeatureDataset
import src.FFOE.base_model as base_model
import src.utils as utils
import numpy as np
import _pickle as pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--ensemble', action='store_true', default=False)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='ban', choices=['ban', 'san', 'cti'],
                        help='the model we use')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=str, default=12)
    parser.add_argument('--use_counter', action='store_true', default=False, help='use counter module')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'swish'], help='The activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout', help='Dropout of rate of final classifier')
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N', help='number of maximum bounding boxes for K-adaptive')

    # Train with TDIUC
    parser.add_argument('--use_TDIUC', action='store_true', default=False, help='Using TDIUC dataset to train')
    parser.add_argument('--TDIUC_dir', type=str, help='TDIUC dir')

    # Distillation
    parser.add_argument('--distillation', default=False, action='store_true', help='use KD loss')
    parser.add_argument('--T', default=1.5, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)

    # Visualize attention map
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


def compute_score_with_logits(logits, labels, ans_emb):
    ans_emb = ans_emb.unsqueeze(0).expand(logits.size(0), ans_emb.size(0), ans_emb.size(1))
    distance = torch.norm(logits.unsqueeze(1) - ans_emb, 2, 2)
    logits = torch.min(distance, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def get_logits(args, model, dataloader, device):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N)
    bar.start()
    with torch.no_grad():
        for v, b, q, _, ans, i, _ in iter(dataloader):
            bar.update(idx)
            batch_size = v.size(0)
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            ans = ans.to(device)
            if args.model == "stacked_attention":
                logits = model(v, q)

            elif args.model == "ban":
                logits, _ = model(v, b, q, None)

            elif args.model == 'cti':
                logits = model(v, q, ans)

            pred[idx:idx+batch_size, :].copy_(logits.data)
            qIds[idx:idx+batch_size].copy_(i)
            idx += batch_size
            if args.debug:
                print(get_question(q.data[0], dataloader))
                print(get_answer(logits.data[0], dataloader))

    bar.update(idx)
    return pred, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = int(qIds[i])
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results


def make_json_with_logits(logits, qIds):
    utils.assert_eq(logits.size(0), len(qIds))
    results = {}
    for i in range(logits.size(0)):
        results[int(qIds[i])] = np.float16(logits[i].detach().numpy())
    return results


if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    torch.cuda.set_device(args.gpu)
    if args.use_TDIUC:
        dictionary = Dictionary.load_from_file('data_TDIUC/dictionary.pkl')
        eval_dset = TDIUCFeatureDataset(args.split, args, dictionary, dataroot='data_TDIUC', adaptive=True)

    else:
        dictionary = Dictionary.load_from_file('data_vqa/dictionary.pkl')
        eval_dset = VQAFeatureDataset(args.split, args, dictionary, dataroot='data_vqa', adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(args, eval_dset)
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

        logits, qIds = get_logits(args, model, eval_loader, device)
        results = make_json(logits, qIds, eval_loader)
        # results = make_json_with_logits(logits, qIds)
        model_label = '%s%s%d_%s' % (args.model, args.op, args.num_hid, args.label)
        if args.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % args.index)

        utils.create_dir(args.output)
        model_label += 'epoch%s' % args.epoch
        # out_file = args.output + '/' + args.input.split('/')[-1] + '.json'
        # with open(args.output + '/%s_%s.pkl' % (args.split, model_label), 'wb') as f:
        #     pickle.dump(results, f, protocol=2)
        with open(args.output+'/%s_%s.json' % (args.split, model_label), 'w') as f:
            json.dump(results, f)
        if args.model == 'cti':
            results = make_json_with_logits(logits, qIds)
            with open('results/%s_%s_logits.pkl' % (args.model, args.split), 'wb') as f:
                pickle.dump(results, f)
    process(args, model, eval_loader)
