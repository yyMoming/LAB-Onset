# -*- coding: utf-8 -*-
# @Time    : 2019/11/15 11:00
# @Author  : moMing.yang
# @File    : test.py.py
# @Software: PyCharm

import os
import time
import torch
import argparse
import numpy as np
import config as cfg
import torch.nn as nn
from torch.autograd import Variable
from model import onsetnet
from input_data_proc import data_proc
from torch.utils.data import DataLoader
from Dataset import onset_dataset
from model import factory_net
from losses import focalloss
from evaluate import evaluate
import multiprocessing
parser = argparse.ArgumentParser()

parser.add_argument('--onset', help='whether evaluate onset',
                    action="store_true")
parser.add_argument('--dual', help='whether evaluate onset',
                    action="store_true")
args = parser.parse_args()

evaluator = evaluate()
use_cuda = torch.cuda.is_available()

def get_solf_wav_anno_files(root_path):
	test_root = root_path   #os.path.join(root_path,'test')

	test_wav_list = [os.path.join(test_root,file) for file in os.listdir(test_root)\
	                  if file.endswith('.wav') or file.endswith('.mp3')]
	test_anno_list = [(wav_file[:-4]+'.txt') for wav_file in test_wav_list]
	test_wav_list = np.array(test_wav_list)
	test_anno_list = np.array(test_anno_list)

	return  test_wav_list, test_anno_list
class predictor_onset(object):
    """docstring for pre"""

    def __init__(self,
                 model_path,
                 pad_length,
                 spec_style,
                 dual_channel):
        super(predictor_onset, self).__init__()

        self.pad_length = pad_length
        self.spec_style = spec_style
        self.net = onsetnet(pad_length=pad_length,
                            spec_style=spec_style,
                            dual_channel=dual_channel)
        self.load_model(model_path)

        self.proc = data_proc(pad_length=pad_length,
                              spec_style=spec_style,
                              dual_channel=dual_channel,
                              is_filter=False)

        self.hopsize_t = cfg.hopsize_t
        self.thresh = cfg.threshold

    @property
    def onset_frame(self):
        return self._onset_frame

    @property
    def onset_time(self):
        return self._onset_time

    @property
    def onset_prob(self):
        return self._onset_prob

    @property
    def onset_pred(self):
        return self._onset_pred

    def input_cqt_data(self, wav_file):
        if self.spec_style == 'cqt':
            spec = self.proc.cal_cqt_spec(wav_file)
        elif self.spec_style == 'fft':
            spec = self.proc.cal_fft_spectrum(wav_file)
        input_specs = []
        totol_frame = spec.shape[-1]
        for i in range(self.pad_length, totol_frame - self.pad_length):
            input_spec = spec[
                :, :, i - self.pad_length:i + 1 + self.pad_length]
            smax = np.max(input_spec)
            smin = np.min(input_spec)
            input_spec = (input_spec - smin) / (smax - smin + 1e-9)

            input_specs.append(np.expand_dims(input_spec, axis=0))
        input_specs = np.concatenate(input_specs, axis=0)
        input_specs = Variable(torch.from_numpy(input_specs).float())
        return input_specs

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.net.eval()
        if use_cuda:
            self.net=nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def predict(self, wav_file):
        inputs = self.input_cqt_data(wav_file)
        pred = []
        if inputs.size()[0] > 10000:
            for i in range(0, inputs.size()[0], 10000):
                _inputs = inputs[i:i + 10000]
                if use_cuda:
                    _inputs = _inputs.cuda()
                _pred = self.net(_inputs)
                pred += [_pred.data.squeeze().cpu().numpy()]
        else:
            if use_cuda:
                inputs = inputs.cuda()
            _pred = self.net(inputs)
            pred += [_pred.data.squeeze().cpu().numpy()]
        pred = np.concatenate(pred, axis=0)
        onset_time = self.post_process(pred)
        return onset_time

    def detect_onset(self, wav_file, res_file):
        onset_time = self.predict(wav_file)
        with open(res_file, 'w') as fw:
            for val in onset_time:
                fw.write(str(val) + '\t')
                fw.write(str(val + 0.03) + '\t')
                fw.write(str(50) + '\n')

    def post_process(self, pred):
        self._onset_prob = np.where(pred > self.thresh)[0]
        self._onset_pred = pred
        onset_prob = self._onset_prob.copy()
        onset_frame = []
        i = 0
        while i < len(onset_prob):
            candi_frame = []
            j = i
            while j < len(onset_prob):
                if (onset_prob[j] - onset_prob[i]) <= 15:
                    candi_frame.append(onset_prob[j])
                else:
                    break
                j += 1
            maxprob, max_onset = pred[candi_frame[0]], candi_frame[0]
            for frame in candi_frame:
                if pred[frame] > maxprob:
                    max_onset = frame
            onset_frame.append(max_onset)
            i = j
        self._onset_time = np.array(onset_frame) * self.hopsize_t
        self._onset_frame = np.array(onset_frame)
        return self._onset_time

def get_best_weights():
    dual = 'dual' if args.dual else 'mono'
    model_path = os.path.join('./model', '{}_onset_{}_{}_fold1test_{}'.format(
        args.spec_style,args.data, args.pad, dual))

    return os.path.join(model_path, 'best.pth')

def evaluate_solf_test(predictor):
    path = os.path.join(cfg.solf_dataset)
    wav_files, anno_files = get_solf_wav_anno_files(path)
    est_dir_ = os.path.join(cfg.est_dir, 'solf')
    if not os.path.exists(est_dir_):
        os.makedirs(est_dir_)

    for wav_file in wav_files:
        print(wav_file)
        filename = os.path.splitext(os.path.basename(wav_file))[0] + ".txt"
        filename = os.path.join(est_dir_, filename)
        predictor.detect_onset(wav_file, filename)

    ref_dir_ = path
    evaluator.evaluate_onset_dir(ref_dir_, est_dir_, onset_tolerance=0.05)

if __name__ == '__main__':
    model_name = os.path.join('./Models','best.pth')
    predictor = predictor_onset(model_name,4,'cqt',dual_channel=False)
    evaluate_solf_test(predictor)