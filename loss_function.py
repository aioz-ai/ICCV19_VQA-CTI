import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscriminativeLoss(nn.Module):
    def __init__(self, alpha):
        super(DiscriminativeLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.alpha = alpha
        self.relu = nn.ReLU()
    def forward(self, logits, labels, ans_emb, print_info):
        '''
        logits: joint semantic spaces between images and questions dx1
        labels: answers for each question-image pairs 3129x1
        ans_emb: embedding for answer 3129xd
        Create a criterion to measure the distance between image-question pair and correct/uncorrect answers
        '''
        # N = labels.size(0)
        C = labels.size(1) #number of centroids
        scale = self.alpha*(1.0/(C-1))
        #logits = self.softmax(logits)

        #calculate distance between joint semantic space and correct answer
        #ans_emb = ans_emb.unsqueeze(0).expand(logits.size(0), ans_emb.size(0), ans_emb.size(1)) #BxCxD
        max_val, idx = labels.max(dim=1)
        idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, ans_emb.size(2))
        correct_class = ans_emb.gather(1, idx)
        correct = (logits.unsqueeze(1) - correct_class).pow(2).sum(2) #distance between image_question pair and correct answer

        #calculate distance between joint semantic space and uncorrect answers
        uncorrect_class = (labels < max_val.unsqueeze(1).float())
        _, idx = uncorrect_class.topk(3128, dim=1, sorted=False)
        idx = idx.unsqueeze(2).expand(-1, idx.size(1), ans_emb.size(2))
        uncorrect_class = ans_emb.gather(1, idx)
        uncorrect = (logits.unsqueeze(1) - uncorrect_class).pow(2).sum(2) #distance between image_question pair and uncorrect answers
        hardest_negative = self.alpha*torch.min(uncorrect, 1)[0]
        loss = correct.squeeze(1) - hardest_negative
        loss = self.relu(loss)
        if print_info:
            print ('Distance to correct centroid: {}, min_uncorrect: {}'.format(correct.squeeze(1), hardest_negative))
        loss = loss.sum()
        return loss


class Distillation_Loss(nn.Module):
    def __init__(self, T, alpha):
        super(Distillation_Loss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.T = T
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, input, knowledge, target):
        # loss = nn.KLDivLoss(reduction='none')(nn.functional.log_softmax(input/self.T, dim=1), \
        #                       nn.functional.softmax(knowledge/self.T, dim=1)).sum(1).mean()*(self.alpha * self.T * self.T) -\
        #         torch.mul(input.log_softmax(1), target).sum(1).mean()*(1. - self.alpha)

        loss = nn.KLDivLoss(reduction='none')(nn.functional.log_softmax(input / self.T, dim=1), \
                                              nn.functional.softmax(knowledge / self.T, dim=1)).sum(1).mean() * (
                           self.alpha * self.T * self.T) + (self.bce(input, target) / input.size(0)) * (1. - self.alpha)

        return loss

