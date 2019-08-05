from language_model import QuestionEmbedding, WordEmbedding
import os
import _pickle as cPickle
from dataset import Dictionary
import utils
import torch
import numpy as np

def tokenize(ans_list, dictionary, max_length=3):
    """Tokenizes the answers.

    This will add q_token in each entry of the dataset.
    -1 represent nil, and should be treated as padding_idx in embedding
    """
    ans_tokens = []
    for ans in ans_list:
        tokens = dictionary.tokenize(ans, False)
        tokens = tokens[:max_length]
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [dictionary.padding_idx] * (max_length - len(tokens))
            tokens = tokens + padding
        utils.assert_eq(len(tokens), max_length)
        ans_tokens.append(tokens)
    return ans_tokens

def create_answer_embedding(ans_list, dictionary, w_emb, ans_emb):
    ans_tokens = tokenize(ans_list, dictionary)
    ans_tokens = torch.from_numpy(np.array(ans_tokens))
    answer_embedding = torch.zeros(3129, 1024)
    for idx, ans in enumerate(ans_tokens):
        ans = ans.unsqueeze(0)
        w = w_emb(ans)
        ans = ans_emb(w)
        answer_embedding[idx] = ans.squeeze()

    with open('data/answer_embedding.pkl','wb') as f:
        cPickle.dump(answer_embedding, f)

if __name__ == '__main__':
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    w_emb = WordEmbedding(dictionary.ntoken, 300, .0, 'c')
    w_emb.init_embedding('data/glove6b_init_300d.npy', None, None)
    ans_emb = QuestionEmbedding(600 , 1024, 1, False, .0)

    ans2label_path = ans2label_path = os.path.join('data', 'cache', 'trainval_ans2label.pkl')
    ans2label = cPickle.load(open(ans2label_path, 'rb'))
    ans_list = [ans for ans in ans2label]
    create_answer_embedding(ans_list, dictionary, w_emb, ans_emb)
