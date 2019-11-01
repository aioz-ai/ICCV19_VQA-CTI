from __future__ import division
import json
import csv
import numpy as np
from collections import defaultdict
from scipy import stats
import sys,argparse

def load_json(json_preds,gt_ann_a,answerkey):
    preds_json = json.load(open(json_preds))
    lut_preds = dict()
    for pred in preds_json:
        lut_preds[pred['question_id']]=pred['answer']
    predictions = []
    for idx,ans in enumerate(gt_ann_a):
        if idx%1000==0:
            sys.stdout.write('\rAligning.. '+str(idx))
            sys.stdout.flush()
        try:
            predictions.append(lut_preds[ans['question_id']])
        except:
            print ("THIS SHOULD NOT HAPPEN")
    predictions = [int(answerkey[p]) for p in predictions]
    return predictions

def mean_per_class(predictions,gt_ann,answerkey):
    res = defaultdict(list)
    gt_answers_idx = []
    notfound=0
    score = 0
    num_sample = 0
    for idx,pred in enumerate(predictions):
        gt_answer = gt_ann[idx]['answers'][0]['answer']
        gt_type = gt_ann[idx]['question_type']
        res[gt_type+'_pred'].append(pred)
        if gt_answer in answerkey:
            gt_idx = int(answerkey[gt_answer])
            res[gt_type+'_gt'].append(gt_idx)
            gt_answers_idx.append(gt_idx)
            if gt_idx == pred:
                res[gt_type+'_t'].append(pred)
            else:
                res[gt_type+'_f'].append(pred)
        else:
            gt_answers_idx.append(-1)
            res[gt_type+'_f'].append(pred)
            res[gt_type+'_gt'].append(-1)
            notfound+=1
    print ("\n %d of validation answers were not in the answerkey" % notfound)
    types = list(set([a['question_type'] for a in gt_ann]))
    for t in types:
        #if t != 'absurd':
        score += len(res[t+'_t'])
        num_sample += len(res[t+'_gt'])
    sum_acc = []
    eps = 1e-10
    print ('\nNOT USING PER-ANSWER NORMALIZATION\n')
    for tp in types:
        acc = 100*(len(res[tp+'_t'])/len(res[tp+'_t']+res[tp+'_f']))
        sum_acc.append(acc+eps)
        print ('Accuracy for %s is %.2f' % (tp,acc)) #'%.2f'%acc#
    #sum_acc = sum_acc[:7] + sum_acc[8:]
    print ('Arithmetic MPT Accuracy is %.2f'%(np.mean(np.array(sum_acc)))) #'%.2f'%np.mean(np.array(sum_acc))#'Arithmetic MPT Accuracy is %.2f'%(np.mean(np.array(sum_acc))) #
    print ('Harmonic MPT Accuracy is %.2f'%(stats.hmean(sum_acc))) #'%.2f'%stats.hmean(sum_acc)#'Harmonic MPT Accuracy is %.2f'%(stats.hmean(sum_acc)) #
    n_acc = 100*np.mean(predictions==np.array(gt_answers_idx))
    print ('%.2f'%n_acc)#'Overall Traditional Accuracy is %.2f'%n_acc
    #print 'Overall Traditional Accuracy is %.2f' % (score * 100.0 / num_sample)
    print ('\n---------------------------------------')
    print ('USING PER-ANSWER NORMALIZATION\n')
    types = list(set([a['question_type'] for a in gt_ann]))
    sum_acc = []
    eps = 1e-10
    for tp in types:
        per_ans_stat = defaultdict(int)
        for g,p in zip(res[tp+'_gt'],res[tp+'_pred']):
            per_ans_stat[str(g)+'_gt']+=1
            if g==p:
                per_ans_stat[str(g)]+=1
        unq_acc = 0
        for unq_ans in set(res[tp+'_gt']):
            acc_curr_ans = per_ans_stat[str(unq_ans)]/per_ans_stat[str(unq_ans)+'_gt']
            unq_acc +=acc_curr_ans
        acc = 100*unq_acc/len(set(res[tp+'_gt']))
        sum_acc.append(acc+eps)
        print ('Accuracy for %s is %.2f' % (tp,acc)) #'%.2f' % acc#'Accuracy for %s is %.2f'%(tp,acc) #

    #sum_acc = sum_acc[:7] + sum_acc[8:]
    print ('Arithmetic MPT Accuracy is %.2f'%(np.mean(np.array(sum_acc)))) #'%.2f' % np.mean(np.array(sum_acc))  # 'Arithmetic MPT Accuracy is %.2f'%(np.mean(np.array(sum_acc))) #
    print ('Harmonic MPT Accuracy is %.2f'%(stats.hmean(sum_acc))) #'%.2f' % stats.hmean(sum_acc)  # 'Harmonic MPT Accuracy is %.2f'%(stats.hmean(sum_acc)) #
    n_acc = 100 * np.mean(predictions == np.array(gt_answers_idx))
    print ('%.2f' % n_acc)  # 'Overall Traditional Accuracy is %.2f'%n_acc
    #print 'Overall Traditional Accuracy is %.2f' % (score * 100.0 / num_sample)

    return
#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_ann', required=True,
                        help='path to ground truth annotation JSON file')
    parser.add_argument('--pred_ann', required=True,
                        help='path to the predictions JSON file')
    parser.add_argument('--answerkey', required=True,
                        help='answerkey CSV file')
    args = parser.parse_args()
    answerkey_csv = csv.reader(open(args.answerkey))
    answerkey = dict((rows[0],rows[1]) for rows in answerkey_csv)
    gt_ann = json.load(open(args.gt_ann))['annotations']
    predictions = load_json(args.pred_ann,gt_ann,answerkey)
    predictions = np.array(predictions)
    mean_per_class(predictions,gt_ann,answerkey)
    print ('------------------------------------------')

if __name__ == "__main__":
    main()
