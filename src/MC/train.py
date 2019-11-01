"""
This code is modified from jnhwkim's repository.
https://github.com/jnhwkim/ban-vqa
"""
import os
import time
import torch
import src.utils
import torch.nn as nn
from src.MC.trainer import Trainer
warmup_updates = 4000


def compute_score_mc(logits, labels):
    prob_preds = torch.softmax(logits, 1)
    result = [torch.max(prob_preds[idx*4:idx*4+4, 0], 0)[1]+(idx*4) for idx in range(int(logits.size(0)/4))]
    idx = torch.stack(result)
    scores = labels[:, 0].gather(0, idx)
    return scores


def train(args, model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0):
    device = args.device
    lr_default = args.lr
    lr_decay_step = 2
    lr_decay_rate = .25
    lr_decay_epochs = range(10, 20, lr_decay_step) if eval_loader is not None else range(10, 20, lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 0
    grad_clip = args.clip_norm
    utils.create_dir(output)
    optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        if opt is None else opt

    # Initial loss function
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    logger = utils.Logger(os.path.join(output, 'log.txt'))
    logger.write(args.__repr__())
    best_eval_score = 0

    utils.print_model(model, logger)
    logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        (lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    # Create trainer
    trainer = Trainer(args, model, criterion, optim)
    update_freq = int(args.update_freq)
    wall_time_start = time.time()
    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        num_updates = 0
        t = time.time()
        N = len(train_loader.dataset)
        num_batches = int(N/args.batch_size + 1)
        if epoch < len(gradual_warmup_steps):
            trainer.optimizer.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            logger.write('gradual warmup lr: %.8f' % trainer.optimizer.param_groups[0]['lr'])
        elif epoch in lr_decay_epochs:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay_rate
            logger.write('decreased lr: %.8f' % trainer.optimizer.param_groups[0]['lr'])
        else:
            logger.write('lr: %.8f' % trainer.optimizer.param_groups[0]['lr'])
        for i, (v, b, q, a, ans_mc, ans_gt) in enumerate(train_loader):
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            a = a.to(device)
            ans_mc = ans_mc.to(device)

            # Clone each sample to 4 samples
            v = v.unsqueeze(1).expand(v.size(0), 4, v.size(1), v.size(2)).contiguous().view(v.size(0) * 4, v.size(1),
                                                                                        v.size(2))
            q = q.unsqueeze(1).expand(q.size(0), 4, q.size(1)).contiguous().view(q.size(0) * 4, q.size(1))
            ans_mc = ans_mc.view(ans_mc.size(0) * ans_mc.size(1), ans_mc.size(2))
            a = a.view(ans_mc.size(0), 1)
            labels = torch.cat([a, 1-a], 1)
            labels = labels.to(device)

            sample = [v, b, q, labels, ans_mc]
            if i < num_batches - 1 and (i + 1) % update_freq > 0:
                trainer.train_step(sample, update_params=False)
            else:
                loss, grad_norm, batch_score = trainer.train_step(sample, update_params=True)
                total_norm += grad_norm
                count_norm += 1
                total_loss += loss.item()
                train_score += batch_score
                num_updates += 1
                if num_updates % int(args.print_interval / update_freq) == 0:
                    print("Iter: {}, Loss {:.4f}, Norm: {:.4f}, Total norm: {:.4f}, Num updates: {}, Wall time: {:.2f},"
                          " ETA: {}".format(i + 1, total_loss / ((num_updates + 1)), grad_norm, total_norm,
                                            num_updates, time.time() - wall_time_start,
                                            utils.time_since(t, i / num_batches)))

        total_loss /= num_updates
        train_score = 100 * train_score / (num_updates * args.batch_size)
        if eval_loader is not None:
            print("Evaluating...")
            trainer.model.train(False)
            eval_score, bound = evaluate(model, eval_loader, args)
            trainer.model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        # Save per epoch
        if epoch >= saving_epoch:
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, trainer.optimizer)
            # Save best epoch
            if eval_loader is not None and eval_score > best_eval_score:
                model_path = os.path.join(output, 'model_epoch_best.pth')
                utils.save_model(model_path, model, epoch, trainer.optimizer)
                best_eval_score = eval_score


def evaluate(model, dataloader, args):
    device = args.device
    score = 0
    upper_bound = 0
    num_data = 0
    with torch.no_grad():
        for v, b, q, a, ans_mc, ans_gt in iter(dataloader):
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            ans_mc = ans_mc.to(device)
            final_preds = None
            ans_mc = ans_mc.to(device)
            # Clone each sample to 4 samples
            v = v.unsqueeze(1).expand(v.size(0), 4, v.size(1), v.size(2)).contiguous().view(v.size(0) * 4, v.size(1),
                                                                                            v.size(2))
            q = q.unsqueeze(1).expand(q.size(0), 4, q.size(1)).contiguous().view(q.size(0) * 4, q.size(1))
            ans_mc = ans_mc.view(ans_mc.size(0) * ans_mc.size(1), ans_mc.size(2))
            a = a.view(ans_mc.size(0), 1)
            labels = torch.cat([a, 1 - a], 1)
            labels = labels.to(device)

            if args.model == "cti":
                final_preds, _ = model(v, b, q, ans_mc)

            elif args.model == "san":
                final_preds = model(v, q, ans_mc)

            elif args.model == "ban":
                final_preds, _ = model(v, b, q, ans_mc)

            batch_score = compute_score_mc(final_preds, labels).sum()
            score += batch_score
            upper_bound += (a.max(1)[0]).sum()
            num_data += final_preds.size(0)

    score = score.float() / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
