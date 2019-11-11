# -*- coding: utf-8 -*-
# @Time    : 2019/11/4 15:02
# @Author  : moMing.yang
# @File    : evaluate.py
# @Software: PyCharm
#
import re
import os
import shutil
import torch
import mir_eval
import collections
import config as cfg
import numpy as np

thresh = cfg.threshold
class evaluate(object):
    """docstring for Evaluate"""

    def __init__(self):
        super(evaluate, self).__init__()

    def __call__(self, predict, target):
        prob = predict.squeeze().cpu().numpy().reshape(-1)#单行排列
        ground_truth = target.squeeze().cpu().numpy().reshape(-1)
        prob_idx = set(np.where(prob > thresh)[0])
        truth_idx = set(np.where(ground_truth > thresh)[0])
        true_positive = prob_idx.intersection(truth_idx)#返回真正例集合

        true_positive_num = len(true_positive)
        predict_all_num = len(prob_idx)
        target_all_num = len(truth_idx)

        P = self.precise(true_positive_num, predict_all_num)
        R = self.recall(true_positive_num, target_all_num)
        Fscore = self.F_score(P, R)

        eval_res = {
            'P': P,
            'R': R,
            'F': Fscore
        }
        return eval_res

    def precise(self, true_positive_num, predict_all_num):
        P = (true_positive_num + 0.1) / (predict_all_num + 0.1)
        return P

    def recall(self, true_positive_num, target_all_num):
        R = (true_positive_num + 0.1) / (target_all_num + 0.1)
        return R

    def F_score(self, P, R):
        return 2 * P * R / (P + R)
'''
	测试集使用的onset评估，允许容错为onset_tolerance
'''
'''
def evaluate_onset(self,
                   ref_filepath,
                   est_filepath,
                   onset_combined=True,
                   onset_tolerance=0.1,
                   length='full'):

    ref_intervals, _ = mir_eval.io.load_valued_intervals(ref_filepath)
    if est_filepath[-3:] == 'dat' or est_filepath[-3:] == 'est':
        est_intervals = np.loadtxt(est_filepath)
    else:
        est_intervals, _ = mir_eval.io.load_valued_intervals(est_filepath)

    if length == '30':
        # row = cut_30(ref_intervals)
        ref_intervals = ref_intervals[0:row, :]
        # row = cut_30(est_intervals)
        est_intervals = est_intervals[0:row, :]

    ref_onset = np.unique(ref_intervals[:, 0])
    if est_filepath[-3:] == 'dat' or est_filepath[-3:] == 'est':
        est_onset = est_intervals
    else:
        est_onset = est_intervals[:, 0]
    if onset_combined == True:
        # ref_onset = combine_onset(ref_onset)

    onset_scores = mir_eval.onset.evaluate(ref_onset,
                                           est_onset,
                                           window=onset_tolerance)

    return onset_scores
'''
if __name__ == '__main__':
    predict = torch.randn(78,1)
    target = torch.randn(78,1).reshape(-1)
    evaluator = evaluate()
    eval = evaluator(predict,target)
    print(eval['P'],eval['R'],eval['F'])