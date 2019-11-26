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
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
import torch
from torch.utils.data import Dataset
import tools.compute_softscore
import itertools

COUNTING_ONLY = False

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering

def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False


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
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
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


def _create_entry(img, question, answer, ans, teacher_logit):
    if None!=answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id' : question['question_id'],
        'image_id'    : question['image_id'],
        'image'       : img,
        'question'    : question['question'],
        'answer'      : answer,
        'ans'         : ans,
        'teacher_logit': teacher_logit}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans, teacher_logits):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % \
        (name + '2014' if 'test'!=name[:4] else name))
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    if 'test' !=name[:4]:  # train, val
        answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
        answers = cPickle.load(open(answer_path, 'rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        utils.assert_eq(len(questions), len(answers))
        entries = []
        questions = questions[:10000]
        answers = answers[:10000]
        for question, answer in zip(questions, answers):
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            if answer['scores']:
                idx_ans = np.argmax(answer['scores'])
                ans = answer['labels'][idx_ans]
                ans = label2ans[ans]
            else:
                ans = ''
            if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
                entries.append(_create_entry(img_id2val[img_id], question, answer, ans, \
                               teacher_logits[question['question_id']] if len(teacher_logits)>0 else None))

    else: # test2015
        entries = []
        for question in questions:
            img_id = question['image_id']
            if not COUNTING_ONLY or is_howmany(question['question'], None, None):
                entries.append(_create_entry(img_id2val[img_id], question, None , None))

    return entries


def _load_TDIUC_dataset(dataroot, name, img_id2val, label2ans, teacher_logits):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(
        dataroot, 'TDIUC_%s_questions.json' % name)
    questions = sorted(json.load(open(question_path))['questions'],
                       key=lambda x: x['question_id'])
    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['question_id'])

    utils.assert_eq(len(questions), len(answers))
    entries = []
    # questions = questions[:10000]
    # answers = answers[:10000]
    for question, answer in zip(questions, answers):
        utils.assert_eq(question['question_id'], answer['question_id'])
        utils.assert_eq(question['image_id'], answer['image_id'])
        img_id = question['image_id']

        if not COUNTING_ONLY or is_howmany(question['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], question, answer, \
                           teacher_logits[question['question_id']] if len(teacher_logits) > 0 else None))


    return entries


def _load_visualgenome(dataroot, name, img_id2val, label2ans, adaptive=True):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    question_path = os.path.join(dataroot, 'question_answers.json')
    image_data_path = os.path.join(dataroot, 'image_data.json')
    ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
    cache_path = os.path.join(dataroot, 'cache', 'vg_%s%s_target.pkl' % (name, '_adaptive' if adaptive else ''))

    if os.path.isfile(cache_path):
        entries = cPickle.load(open(cache_path, 'rb'))
    else:
        entries = []
        ans2label = cPickle.load(open(ans2label_path, 'rb'))
        vgq = json.load(open(question_path, 'r'))
        _vgv = json.load(open(image_data_path, 'r')) #108,077
        vgv = {}
        for _v in _vgv: 
            if None != _v['coco_id']:
                vgv[_v['image_id']] = _v['coco_id']
        counts = [0, 0, 0, 0] # used image, used question, total question, out-of-split
        for vg in vgq:
            coco_id = vgv.get(vg['id'], None)
            if None != coco_id:
                counts[0] += 1
                img_idx = img_id2val.get(coco_id, None)
                if None == img_idx:
                    counts[3] += 1
                for q in vg['qas']:
                    counts[2] += 1
                    _answer = tools.compute_softscore.preprocess_answer(q['answer'])
                    label = ans2label.get(_answer, None)
                    if None != label and None != img_idx:
                        counts[1] += 1
                        answer = {
                            'labels': [label],
                            'scores': [1.]}
                        entry = {
                            'question_id' : q['qa_id'],
                            'image_id'    : coco_id,
                            'image'       : img_idx,
                            'question'    : q['question'],
                            'answer'      : answer}
                        if not COUNTING_ONLY or is_howmany(q['question'], answer, label2ans):
                            entries.append(entry)

        print('Loading VisualGenome %s' % name)
        print('\tUsed COCO images: %d/%d (%.4f)' % \
            (counts[0], len(_vgv), counts[0]/len(_vgv)))
        print('\tOut-of-split COCO images: %d/%d (%.4f)' % \
            (counts[3], counts[0], counts[3]/counts[0]))
        print('\tUsed VG questions: %d/%d (%.4f)' % \
            (counts[1], counts[2], counts[1]/counts[2]))
        with open(cache_path, 'wb') as f:
            cPickle.dump(entries, open(cache_path, 'wb'))

    return entries


def _find_coco_id(vgv, vgv_id):
    for v in vgv:
        if v['id']==vgv_id:
            return v['coco_id']
    return None


class VQAFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data_vqa', max_boxes=100, question_len=14, adaptive=False):
        super(VQAFeatureDataset, self).__init__()
        assert name in ['train', 'val', 'test2015']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.teacher_logits = []
        if args.distillation:
            teacher_logits_path = os.path.join(dataroot, '%s_teacher_logits.pkl' % name)
            self.teacher_logits = cPickle.load(open(teacher_logits_path, 'rb'))

        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.max_boxes = max_boxes
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))
        
        h5_path = os.path.join(dataroot, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))

        print('loading features from h5 file')    
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans, self.teacher_logits)
        self.tokenize(question_len)
        self.ans_token()
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

    def ans_token(self, max_length=3):
        """Tokenizes the answer.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['ans'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['ans_token'] = tokens

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question
            ans = torch.from_numpy(np.array(entry['ans_token']))
            entry['ans_token'] = ans

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

        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        features = features[:self.max_boxes]
        spatials = spatials[:self.max_boxes]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        ans = entry['ans_token']

        if None!=answer:
            try:
                teacher_logit = torch.from_numpy(np.float32(entry['teacher_logit']))
            except:
                teacher_logit = 0
            labels = answer['labels']
            scores = answer['scores']
            target = torch.zeros(self.num_ans_candidates)
            if labels is not None:
                target.scatter_(0, labels, scores)
            return features, spatials, question, target, ans, question_id, teacher_logit
        else:
            return features, spatials, question, 0, question_id, 0

    def __len__(self):
        return len(self.entries)


class TDIUCFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data_TDIUC', max_boxes=100, question_len=14, adaptive=True):
        super(TDIUCFeatureDataset, self).__init__()
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.teacher_logits = []
        if args.distillation and name is 'train':
            teacher_logits_path = os.path.join(dataroot, '%s_teacher_logits.pkl' % name)
            self.teacher_logits = cPickle.load(open(teacher_logits_path, 'rb'))

        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.max_boxes = max_boxes
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
            open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        h5_path = os.path.join(dataroot, '%s%s.hdf5' % (name, '' if self.adaptive else '36'))

        print('loading features from h5 file')
        with h5py.File(h5_path, 'r') as hf:
            self.features = np.array(hf.get('image_features'))
            self.spatials = np.array(hf.get('spatial_features'))
            if self.adaptive:
                self.pos_boxes = np.array(hf.get('pos_boxes'))
        self.entries = _load_TDIUC_dataset(dataroot, name, self.img_id2idx, self.label2ans, self.teacher_logits)
        self.tokenize(question_len)
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

    def tensorize(self):
        self.features = torch.from_numpy(self.features)
        self.spatials = torch.from_numpy(self.spatials)

        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None != answer:
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

        features = self.features[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]
        spatials = self.spatials[self.pos_boxes[entry['image']][0]:self.pos_boxes[entry['image']][1], :]

        features = features[:self.max_boxes]
        spatials = spatials[:self.max_boxes]

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']

        try:
            teacher_logit = torch.from_numpy(np.float32(entry['teacher_logit']))
        except:
            teacher_logit = 0
        labels = answer['labels']
        scores = answer['scores']
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return features, spatials, question, target, question_id, teacher_logit

    def __len__(self):
        return len(self.entries)


class VisualGenomeFeatureDataset(Dataset):
    def __init__(self, name, features, spatials, dictionary, dataroot='data', adaptive=False, pos_boxes=None):
        super(VisualGenomeFeatureDataset, self).__init__()
        # do not use test split images!
        assert name in ['train', 'val']

        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        ans_embedding_path = os.path.join(dataroot, 'ans_embedding.pkl')

        self.ans_embedding = torch.from_numpy(cPickle.load(open(ans_embedding_path, 'rb')))
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive

        self.img_id2idx = cPickle.load(
                open(os.path.join(dataroot, '%s%s_imgid2idx.pkl' % (name, '' if self.adaptive else '36')), 'rb'))

        self.features = features
        self.spatials = spatials
        if self.adaptive:
            self.pos_boxes = pos_boxes

        self.entries = _load_visualgenome(dataroot, name, self.img_id2idx, self.label2ans)
        self.tokenize()
        self.tensorize()
        self.v_dim = self.features.size(1 if self.adaptive else 2)
        self.s_dim = self.spatials.size(1 if self.adaptive else 2)

    def tokenize(self, max_length=14):
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

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
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

        question = entry['q_token']
        question_id = entry['question_id']
        answer = entry['answer']
        labels = answer['labels']
        scores = answer['scores']
        ans_emb = self.ans_embedding[answer['labels'][np.argmax(answer['scores'])], :]
        target = torch.zeros(self.num_ans_candidates)
        if labels is not None:
            target.scatter_(0, labels, scores)
        return features, spatials, question, target, 0, ans_emb

    def __len__(self):
        return len(self.entries)


def tfidf_from_questions(names, dictionary, dataroot='data_vqa', target=['vqa', 'vg', 'cap']):
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
            assert name in ['train', 'val', 'test-dev2015', 'test2015']
            question_path = os.path.join(
                dataroot, 'v2_OpenEnded_mscoco_%s_questions.json' % \
                (name + '2014' if 'test'!=name[:4] else name))
            questions = json.load(open(question_path))['questions']

            for question in questions:
                populate(inds, df, question['question'])

    if 'vg' in target: # Visual Genome
        question_path = os.path.join(dataroot, 'question_answers.json')
        vgq = json.load(open(question_path, 'r'))
        for vg in vgq:
            for q in vg['qas']:
                populate(inds, df, q['question'])

    if 'cap' in target: # MSCOCO Caption
        for split in ['train2017', 'val2017']:
            captions = json.load(open('data/annotations/captions_%s.json' % split, 'r'))
            for caps in captions['annotations']:
                populate(inds, df, caps['caption'])

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
    weights, word2emb = utils.create_bert_embedding_init(dictionary.idx2word[N:])
    print('tf-idf stochastic matrix (%d x %d) is generated.' % (tfidf.size(0), tfidf.size(1)))

    return tfidf, weights

