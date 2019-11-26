"""
This code is modified from jnhwkim's repository.
https://github.com/jnhwkim/ban-vqa
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset

from src.FFOE.dataset import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset, TDIUCFeatureDataset
import src.FFOE.base_model as base_model
from src.FFOE.train import train
import src.utils as utils

try:
    import _pickle as pickle
except:
    import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='specify index of GPU using for training, to use CPU: -1')
    # Basic training hyperparams
    parser.add_argument('--epochs', type=int, default=13)
    parser.add_argument('--batch_size', type=int, default=256)
    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024)
    # Choices of models
    parser.add_argument('--model', type=str, default='ban', choices=['ban', 'san', 'cti'],
                        help='the model we use')
    parser.add_argument('--op', type=str, default='c')
    # Data
    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true', help='use visual genome dataset to train?')
    parser.add_argument('--tfidf', type=bool, default=True, help='tfidf word embedding?')
    # Model loading/saving
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/ban')
    # General training hyperparameters
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM', help='clip threshold of gradients')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='lr', help='initial learning rate')
    # Delayed updates
    parser.add_argument('--update_freq', default='1', metavar='N',help='update parameters every N_i batches, when in epoch i')
    # BAN
    parser.add_argument('--gamma', type=int, default=2, help='glimpse')
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',help='number of maximum bounding boxes for K-adaptive')
    parser.add_argument('--use_counter', action='store_true', default=False, help='use counter module')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'swish'], help='The activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout', help='Dropout of rate of final classifier')
    parser.add_argument('--question_len', default=12, type=int, metavar='N', help='maximum length of input question')
    # Utilities
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    parser.add_argument('--print_interval', default=200, type=int, metavar='N',help='print per certain number of steps')

    # Train with TDIUC
    parser.add_argument('--use_TDIUC', action='store_true', default=False, help='Using TDIUC dataset to train')
    parser.add_argument('--TDIUC_dir', type=str, help='TDIUC dir')

    # CTI
    parser.add_argument('--rank', default=32, type=int, help='number of rank decomposition')
    parser.add_argument('--h_out', default=1, type=int)
    parser.add_argument('--h_mm', default=512, type=int)
    parser.add_argument('--k', default=1, type=int)

    # Distributed
    parser.add_argument('--local_rank', type=int)

    # Distillation
    parser.add_argument('--distillation', default=False, action='store_true', help='use KD loss')
    parser.add_argument('--T', default=1.5, type=float)
    parser.add_argument('--alpha', default=0.2, type=float)

    # SAN
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())

    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(args.gpu)

    if args.use_TDIUC:
        dictionary = Dictionary.load_from_file('data_TDIUC/dictionary.pkl')
        train_dset = TDIUCFeatureDataset('train', args, dictionary, dataroot='data_TDIUC', adaptive=True,
                                         max_boxes=args.max_boxes, question_len=args.question_len)
        val_dset = TDIUCFeatureDataset('val', args, dictionary, dataroot='data_TDIUC', adaptive=True,
                                       max_boxes=args.max_boxes, question_len=args.question_len)
    else:
        dictionary = Dictionary.load_from_file('data_vqa/dictionary.pkl')
        train_dset = VQAFeatureDataset('train', args, dictionary, adaptive=True, max_boxes=args.max_boxes,
                                       question_len=args.question_len)
        val_dset = VQAFeatureDataset('val', args, dictionary, adaptive=True, max_boxes=args.max_boxes,
                                    question_len=args.question_len)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(args, train_dset)
    model = model.to(device)
    # Comment because do not use multi gpu

    optim = None
    epoch = 0
    sampler = None
    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        model.to(device)
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

    if args.use_both:  # use train & val splits to optimize
        if args.use_vg:  # use a portion of Visual Genome dataset
            vg_dsets = [
                VisualGenomeFeatureDataset('train',
                    train_dset.features, train_dset.spatials, dictionary, adaptive=True, pos_boxes=train_dset.pos_boxes),
                VisualGenomeFeatureDataset('val',
                    val_dset.features, val_dset.spatials, dictionary, adaptive=True, pos_boxes=val_dset.pos_boxes)]
            trainval_dset = ConcatDataset([train_dset, val_dset]+vg_dsets)
        else:
            trainval_dset = ConcatDataset([train_dset, val_dset])
        train_loader = DataLoader(trainval_dset, batch_size, shuffle=sampler is None, num_workers=0,
                                  collate_fn=utils.trim_collate, pin_memory=True)
        eval_loader = None
    else:
        train_loader = DataLoader(train_dset, batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0,
                                  collate_fn=utils.trim_collate, pin_memory=True)
        eval_loader = DataLoader(val_dset, batch_size*2, shuffle=False,sampler=sampler, num_workers=0,
                                 collate_fn=utils.trim_collate, pin_memory=False)

    train(args, model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)
