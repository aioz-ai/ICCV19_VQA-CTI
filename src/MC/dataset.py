"""
This code is modified from jnhwkim's repository.
https://github.com/jnhwkim/ban-vqa
"""
from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import src.utils as utils
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import torch
from torch.utils.data import Dataset
import itertools

COUNTING_ONLY = False


def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('.','')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer, label, ans_gt, ans_mc):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'label'      : label,
        'ans_gt'      : ans_gt,
        'ans_mc'      : ans_mc}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans, ans_candidates):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    question_path = os.path.join(
        dataroot, 'v7w_%s_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    entries = []
    for question in questions:
        img_id = question['image_id']
        ans_mc = ans_candidates[str(question['question_id'])]['mc']
        ans_gt = ans_candidates[str(question['question_id'])]['ans_gt']
        label = ans_candidates[str(question['question_id'])]['label']

        entries.append(_create_entry(img_id2val[img_id], question, None, label, ans_gt, ans_mc))

    return entries


def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['id']==vgv_id:
            return v['coco_id']
    return None


class V7WDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data_v7w', max_boxes=100, question_len=14, adaptive=False):
        super(V7WDataset, self).__init__()
        assert name in ['train', 'val', 'test']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        answer_candidate_path = os.path.join(dataroot, 'answer_%s.json' % name)

        self.answer_candidates = json.load(open(answer_candidate_path))
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.adaptive = adaptive
        self.max_box = max_boxes
        self.question_len = question_len
        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))
        
        h5_path = os.path.join(dataroot, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))

        if args.use_feature == 'grid':
            self.adaptive = False
            self.img_id2idx = cPickle.load(open(os.path.join(dataroot, 'v7w/%s_imgid2idx.pkl'%name),'rb'))
            h5_path = os.path.join(dataroot, 'v7w/%s.hdf5' % name)

        print('loading features from h5 file')
        with h5py.File(h5_path, 'r') as hf:
            if args.use_feature == 'grid':
                self.features = np.array(hf.get('image_features'))
                self.spatials = np.zeros(np.shape(self.features))
            else:
                self.features = np.array(hf.get('image_features'))
                self.spatials = np.array(hf.get('spatial_features'))
                if self.adaptive:
                    self.pos_boxes = np.array(hf.get('pos_boxes'))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans, self.answer_candidates)
        self.tokenize(self.question_len)
        self.ans_tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def ans_tokenize(self, max_length=6):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['ans_gt'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['ans_gt_token'] = tokens
            ans_mc = []
            for ans in entry['ans_mc']:
                token = self.dictionary.tokenize(ans, False)
                token = token[:max_length]
                if len(token) < max_length:
                    # Note here we pad in front of the sentence
                    padding = [self.dictionary.padding_idx] * (max_length - len(token))
                    token = token + padding
                utils.assert_eq(len(token), max_length)
                ans_mc.append(token)
            entry['ans_mc_token'] = ans_mc

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            ans_gt_token = torch.from_numpy(np.array(entry['ans_gt_token']))
            entry['ans_gt_token'] = ans_gt_token

            ans_mc_token = torch.from_numpy(np.array(entry['ans_mc_token']))
            entry['ans_mc_token'] = ans_mc_token

            label = torch.from_numpy(np.float32(entry['label']))
            entry['label'] = label

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        if not self.adaptive:
            features = self.features[entry['image']]
            spatials = self.spatials[entry['image']]
        else:
            features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
            spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]

            features = features[:self.max_box]
            spatials = spatials[:self.max_box]

        question = entry['q_token']
        question_id = entry['question_id']

        ans_gt = entry['ans_gt_token']
        ans_mc = entry['ans_mc_token']
        label = entry['label']

        return features, spatials, question, label, ans_mc, ans_gt

    def __len__(self):
        return len(self.entries)


def tfidf_from_questions(names, dictionary, dataroot='data_v7w', target=['vqa']):
    inds = [[], []] # rows, cols for uncoalesce sparse matrix
    df = dict()
    N = len(dictionary)

    def populate(inds, df, text):
        tokens = dictionary.tokenize(text, True)
        for t in tokens:
            df[t] = df.get(t, 0) + 1
        combin = list(itertools.combinations(tokens, 2))
        for c in combin:
            if c[0] < N:
                inds[0].append(c[0]); inds[1].append(c[1])
            if c[1] < N:
                inds[0].append(c[1]); inds[1].append(c[0])

    if 'vqa' in target: # VQA 2.0
        for name in names:
            assert name in ['train', 'val', 'test']
            question_path = os.path.join(
                dataroot, 'v7w_%s_questions.json' % name)
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

            ans_path = os.path.join(dataroot, 'answer_%s.json' % name)
            answers = json.load(open(ans_path))
            for ans in answers.values():
                ans_mc = ans['mc']
                for a in ans_mc:
                    populate(inds, df, a)

    if 'vg' in target: # Visual Genome
        question_path = os.path.join(dataroot, 'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])


    # TF-IDF 
    vals = [1] * len(inds[1])
    for idx, col in enumerate(inds[1]):
        assert df[col] >= 1, 'document frequency should be greater than zero!'
        vals[col] /= df[col]

    # Make stochastic matrix
    def normalize(inds, vals):
        z = dict()
        for row, val in zip(inds[0], vals):
            z[row] = z.get(row, 0) + val
        for idx, row in enumerate(inds[0]):
            vals[idx] /= z[row]
        return vals

    vals = normalize(inds, vals)

    tfidf = torch.sparse.FloatTensor(torch.LongTensor(inds), torch.FloatTensor(vals))
    tfidf = tfidf.coalesce()

    # Latent word embeddings
    emb_dim = 300
    glove_file = 'data/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = utils.create_glove_embedding_init(dictionary.idx2word[N:], glove_file)
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights
