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
from torch.autograd import Variable
from dataset import Dictionary, VQAFeatureDataset
import base_model
import utils
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1, help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--ensemble', type=bool, default=False)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='ban', choices=['ban', 'relational', 'stacked_attention', \
                                                                     'tan'],
                        help='the model we use')
    parser.add_argument('--op', type=str, default='c')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--gamma', type=int, default=2)
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--input', type=str, default='saved_models/ban')
    parser.add_argument('--output', type=str, default='results')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--logits', type=bool, default=False)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--epoch', type=str, default=12)
    parser.add_argument('--weight_init', type=str, default='none', choices=['none', 'kaiming_normal'])
    parser.add_argument('--word_init', type=str, default='glove', choices=['glove', 'bert'])

    # BAN
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout', help='Dropout of rate of final classifier')

    parser.add_argument('--use_counter', action='store_true', default=False, help='use counter module')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'swish'],
                        help='The activation to use for final classifier')

    parser.add_argument('--question_len', default=12, type=int, metavar='N', help='maximum length of input question')
    parser.add_argument('--v_dropout', default=0.5, type=float, metavar='v_dropout', help='Dropout of rate of linearing visual embedding')


    # Use MoD features
    parser.add_argument('--use_MoD', action='store_true', default=False, help='Using MoD features')
    parser.add_argument('--MoD_dir', type=str, help='MoD features dir')
    parser.add_argument('--distillation', default=False, action='store_true', help='use KD loss')

    # Train with TDIUC
    parser.add_argument('--use_TDIUC', action='store_true', default=False, help='Using TDIUC dataset to train')
    parser.add_argument('--TDIUC_dir', type=str, help='TDIUC dir')

    # Tan
    parser.add_argument('--rank', default=8, type=int, help='number of rank decomposition')
    parser.add_argument('--h_out', default=40, type=int)
    parser.add_argument('--h_mm', default=160, type=int)
    parser.add_argument('--answer_file', default='ans_embedding.pkl', type=str, help='Answer embedding file')
    parser.add_argument('--k', default=1, type=int)
    parser.add_argument('--distance_ratio', default=0.1, type=float)
    parser.add_argument('--obj_semantic', default='end2end', choices=['end2end', 'pretrained'], \
                        help='Strategy train object semantic end2end or from pre-trained bert embedding')

    # distributed
    parser.add_argument('--local_rank', type=int)

    # distillation
    parser.add_argument('--T', default=1.5, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)

    # v7w
    parser.add_argument('--use_feature', default='bottom', type=str, help='use bottom-up feature or grid feature')

    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Visualize attention map
    parser.add_argument('--visualize_att_map', action = 'store_true', default=False, help='Visualize attention map')


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

def get_answer_and_score(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()], float(_m.item())

def compute_score_with_logits(logits, labels, ans_emb):
    ans_emb = ans_emb.unsqueeze(0).expand(logits.size(0), ans_emb.size(0), ans_emb.size(1))
    distance = torch.norm(logits.unsqueeze(1) - ans_emb, 2, 2)
    logits = torch.min(distance, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
def compute_score_mc(logits, labels):
    prob_preds = torch.softmax(logits, 1)
    result = [torch.max(prob_preds[idx*4:idx*4+4,0], 0)[1]+(idx*4) for idx in range(int(logits.size(0)/4))]
    idx = torch.stack(result)
    scores = labels[:,0].gather(0, idx)
    return scores
def create_onehot_centroid(num_centroid):
    centroids = [indice for indice in range(num_centroid)]
    centroids = torch.Tensor(centroids).unsqueeze(1).long()
    onehot_centroid = torch.zeros(num_centroid,num_centroid)
    onehot_centroid.scatter_(1,centroids,1)
    return  onehot_centroid

def visualize_3D_map(logits, boxes, id, qes, ans, q_id, cl_pallete):
    #fig = plt.figure(figsize=(7, 5))
    fig = plt.figure(figsize=(7, 15))
    #create 3D subplot
    ax = fig.add_subplot(122, projection='3d')

    # reference for cmap. note cmap and c are different!
    # http://matplotlib.org/examples/color/colormaps_reference.html
    h1 = []
    h2 = []
    h3 = []
    for i in range(0, 6):
        for j in range(0, len(qes)):
            for k in range(0, len(ans)):
                h1.append(i+1)
                h2.append(j+1)
                h3.append(k+1)

    # colors = cm.hsv(the_fourth_dimension/max(the_fourth_dimension))
    attention = logits.sum(2).sum(1)

    logits = logits.detach().cpu().numpy()
    logits = logits.flatten()
    print('Color pallete: {}'.format(cl_pallete))
    colmap = cm.ScalarMappable(cmap=getattr(cm, cl_pallete))
    colmap.set_array(logits)

    ax.scatter(h1, h2, h3, marker='s', s=180, c=logits, cmap=cl_pallete)
    cb = fig.colorbar(colmap)

    ax.set_xlabel('Boxes')
    ax.set_ylabel('Question')
    #ax.set_zlabel('Answer')

    ax.set_yticklabels(qes)
    # ax.set_zticklabels(ans)

    img_dir = '/media/data-aioz/VQA/Visual7W/test/v7w_' + str(int(id)) + '.jpg'
    img = cv2.imread(img_dir)
    h, w, c = np.shape(img)

    overlay = img.copy()
    output = img.copy()
    tmp = img.copy()

    cv2.rectangle(img, (0, 0), (w, h), (255, 255, 255), -1)
    cv2.addWeighted(img, 0.75, output, 0.25, 0, img)
    x1_max, x2_max, y1_max, y2_max = 0, 0, 0, 0
    max_val, _ = attention.max(0)
    max_idx = 0
    for idx, b in enumerate(boxes[0][:6]):
        x_min = int(b[0] * w)
        y_min = int(b[1] * h)
        x_max = int(b[2] * w)
        y_max = int(b[3] * h)
        alpha = round(float((3*attention[idx]) /(max_val*4)), 2)

        #save locations of the highest attetion bbox
        if (alpha == 0.75):
            x1_max, y1_max, x2_max, y2_max = x_min, y_min, x_max, y_max
            max_idx = idx

        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.75-alpha, output, 0.25+alpha, 0, output)

        img[y_min:y_max, x_min:x_max] = output[y_min:y_max, x_min:x_max]

        overlay = tmp.copy()
        output = tmp.copy()
    # Re-draw the box which has the highest attention
    cv2.rectangle(overlay, (x1_max, y1_max), (x2_max, y2_max), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0, output, 1, 0, output)

    img[y1_max:y2_max, x1_max:x2_max] = output[y1_max:y2_max, x1_max:x2_max]
    cv2.rectangle(img, (x1_max, y1_max), (x2_max, y2_max), (0, 0, 255), 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    ax = fig.add_subplot(121)
    ax.imshow(img)
    plt.axis('off')
    # set color for each bbox
    color = ['yellow', 'b', 'g', 'red', 'cyan', 'purple']
    color[3] = color[max_idx]
    for idx, b in enumerate(boxes[0]):
        x = int(b[0] * w)
        y = int(b[1] * h)
        w_box = int(b[4] * w)
        h_box = int(b[5] * h)
        if idx == max_idx:
            rect = patches.Rectangle((x, y), w_box, h_box, linewidth=2, edgecolor='red', facecolor='none')
            plt.text(x, y, str(idx + 1), fontsize=12, bbox=dict(facecolor='red', alpha=.5))
        else:
            rect = patches.Rectangle((x, y), w_box, h_box, linewidth=1, edgecolor=color[idx], facecolor='none')
            plt.text(x, y, str(idx + 1), fontsize=12, bbox=dict(facecolor=color[idx], alpha=.5))
        ax.add_patch(rect)
        if idx > 4:
            break
    plt.show()
    #fig.savefig('tan_1/v7w_%d.pdf' % q_id)

def get_logits(args, model, dataloader, device):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    score = 0
    bar = progressbar.ProgressBar(maxval=N)
    bar.start()
    q_ids = [23944, 25581, 30727, 36146, 44863, 59883, 60782, 65463, 69852]
    with torch.no_grad():
        for v, b, q, a, im_id, i, ans_mc, ans_gt, mc, gt in iter(dataloader):
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
            start = time.time()
            if args.model == "stacked_attention":
                    logits = model(v, q, ans_mc)
                
            if args.model == "ban":
                    feats, att = model(v, b, q, ans_mc)
                    logits = feats#model.classify(feats.sum(1))
            if args.model == "tan":
                    logits, att = model(v, b, q, ans_mc)
            print('Time process per sample: {}'.format(time.time() - start))
            prob_preds = torch.softmax(logits, 1)
            idx = torch.max(prob_preds[:,0], 0)[1]
            #ans = mc[idx]
            ques = get_question(q.data[0], dataloader)
            ques = ques.split('_')[0].split()
            #ans_vs = ans[0].split()
            len_qes = len(ques)
            #pred[idx:idx+batch_size,:].copy_(logits.data)
            #compute_score_with_logits(feats, a, ans_emb)
            score += compute_score_mc(logits, labels).sum()
            #qIds[idx:idx+batch_size].copy_(i)

            if args.debug:
                if len(gt[0].split()) < 7 and len(gt[0].split()) > 3 and ans==gt:
                    att = att.sum(4)[idx, :6, :len_qes, :len(ans_vs)]
                    print(get_question(q.data[0], dataloader))
                    print('MC answer: {}'.format(mc))
                    print('Answer: {}'.format(ans))
                    print('GT: {}'.format(gt))
                    print('Score: {}'.format(score))
                    print('Q_ids: {}'.format(i.item()))
                    visualize_3D_map(att, b, im_id, ques, ans_vs, i.item(), 'CMRmap')
                #print(get_answer(logits.data[0], dataloader))

    bar.update(idx)
    score = score.float() / len(dataloader.dataset)
    print(score)
    return pred#, qIds


def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = int(qIds[i])
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

def make_json_with_score(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = int(qIds[i])
        result['answer'], result['score'] = get_answer_and_score(logits[i], dataloader)
        results.append(result)
    return results

if __name__ == '__main__':
    args = parse_args()

    torch.backends.cudnn.benchmark = True

    # dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    # eval_dset = VQAFeatureDataset(args.split,args, dictionary, adaptive=True)
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    torch.cuda.set_device(args.gpu)
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    eval_dset = VQAFeatureDataset(args.split, args, dictionary, adaptive=True)

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

        logits, qIds = get_logits(args, model, eval_loader, device)
        results = make_json(logits, qIds, eval_loader)

        model_label = '%s%s%d_%s' % (args.model, args.op, args.num_hid, args.label)
        if args.logits:
            utils.create_dir('logits/'+model_label)
            torch.save(logits, 'logits/'+model_label+'/logits%d.pth' % args.index)

        utils.create_dir(args.output)
        model_label += '_epoch%s' % args.epoch

        if args.ensemble is True:
            pass
        else:
            with open(args.output+'/%s_%s.json'% (args.split, model_label), 'w') as f:
                json.dump(results, f)

    process(args, model, eval_loader)
