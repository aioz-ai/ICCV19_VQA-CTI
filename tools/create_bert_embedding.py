import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import _pickle as pickle
from dataset import Dictionary
import numpy as np

def create_bert_embedding(idx2word, model, tokenizer):
    weights = np.zeros((len(idx2word), 768), dtype=np.float32)
    for idx, word in enumerate(idx2word):
        tokenize_text = tokenizer.tokenize(word)
        index_token = tokenizer.convert_tokens_to_ids(tokenize_text)
        tokens_tensor = torch.tensor([index_token])
        weights[idx] = model(tokens_tensor)[1].detach().numpy()
    return weights

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    d = Dictionary.load_from_file('../data/dictionary.pkl')
    # weights = create_bert_embedding(d.idx2word, model, tokenizer)
    # np.save('../data/bert_embedding.npy',weights)


