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
def modify_refile(dirpath, ext):
    '''
    在MAPS数据集中，txt文件第一行有OnsetTime,OffsetTime,MidiPitch
    在评价是应该删掉
    '''
    files = os.listdir(dirpath)
    for file in files:
        filepath = os.path.join(dirpath, file)[0:-3] + ext
        filetemp = os.path.join(dirpath, file)[0:-3] + "est"
        index = 0
        DeleteFile = True
        with open(filetemp, "w") as ft:
            with open(filepath, "r") as f:
                for content in f.readlines():
                    if index == 0:
                        content = content.strip().split('\t')
                        if content[0] == 'OnsetTime':
                            DeleteFile = False
                        index += 1
                    else:
                        ft.write(content)
        if not DeleteFile:
            shutil.move(filetemp, filepath)
        else:
            os.remove(filetemp)

def combine_onset(ref_onset):
    '''
    合并间隔<10ms的onset
    '''
    ref_onset_new = [[ref_onset[0]]]
    onset_diff = [y - x for (x, y) in zip(ref_onset, ref_onset[1:])]
    for (i, diff) in enumerate(onset_diff):
        if diff < 0.01:
            ref_onset_new[-1].append(ref_onset[i + 1])
        else:
            ref_onset_new.append([ref_onset[i + 1]])
    ref_onset = []
    for item in ref_onset_new:
        sum = 0
        for item2 in item:
            sum += item2
        ref_onset.append(sum / len(item))
    return np.array(ref_onset)

def cut_30(intervals):
    '''
    返回前30s的数据行数（从1计数）
    '''
    for row in range(intervals.shape[0]):
        if intervals[row, 0] > 30:
            return row

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

	# '''
	# 	测试集使用的onset评估，允许容错为onset_tolerance
    #   将间隔小于10ms 的onset合并
	# '''
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
		    est_intervals, _ = mir_eval.io.load_valued_intervals(est_filepath)  # get annotations of [onset,offset]

	    if length == '30':
		    row = cut_30(ref_intervals)
		    ref_intervals = ref_intervals[0:row, :]
		    row = cut_30(est_intervals)
		    est_intervals = est_intervals[0:row, :]

	    ref_onset = np.unique(ref_intervals[:, 0])
	    if est_filepath[-3:] == 'dat' or est_filepath[-3:] == 'est':
		    est_onset = est_intervals
	    else:
		    est_onset = est_intervals[:, 0]
	    if onset_combined == True:
		    ref_onset = combine_onset(ref_onset)

	    onset_scores = mir_eval.onset.evaluate(ref_onset,
	                                           est_onset,
	                                           window=onset_tolerance)

	    return onset_scores

    def evaluate_onset_dir(self,
	                       ref_dir,
	                       est_dir,
	                       onset_combined=True,
	                       onset_tolerance=0.1,
	                       length='full',
	                       beta=1.0):
	    modify_refile(ref_dir, "txt")
	    result = []
	    files = [file for file in os.listdir(ref_dir) if file.endswith('txt')]
	    for idx, file in enumerate(files):
		    ref_filepath = os.path.join(ref_dir, file)  # testset onset label txt
		    est_filepath = os.path.join(est_dir, file)  # testset predict onset txt
		    onset_scores = self.evaluate_onset(ref_filepath,
		                                       est_filepath,
		                                       onset_combined=onset_combined,
		                                       onset_tolerance=onset_tolerance,
		                                       length=length)
		    result.append([onset_scores['Precision'], onset_scores['Recall'],
		                   onset_scores['F-measure']])
		    print('%s\tp:%.4f \t r:%.4f \t f:%.4f' % (file,
		                                              onset_scores[
			                                              'Precision'],
		                                              onset_scores[
			                                              'Recall'],
		                                              onset_scores['F-measure']))
	    result = np.array(result)
	    result_mean = np.mean(result, axis=0)
	    print('p-mean:%.4f \t r-mean:%.4f \t f-mean:%.4f' % (result_mean[0],
	                                                         result_mean[1],
	                                                         result_mean[2]))


if __name__ == '__main__':
    predict = torch.randn(78,1)
    target = torch.randn(78,1).reshape(-1)
    evaluator = evaluate()
    eval = evaluator(predict,target)
    print(eval['P'],eval['R'],eval['F'])