# -*- coding:utf-8 -*-
import torch
import numpy as np
import librosa
from torch.autograd import Variable
from model import onsetnet
from config import model_path
import config as cfg

use_cuda = torch.cuda.is_available()


cqt_config = cfg.cqt_config


class predictor_onset(object):
    """docstring for pre"""

    def __init__(self):
        super(predictor_onset, self).__init__()
        self.net = onsetnet()
        self.load_model(model_path)

        self.pad_length = 4
        self.hopsize_t = cfg.hopsize_t #时间
        self.thresh = cfg.threshold #阈值0.5

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
        spec = self.cal_cqt_spectrum(wav_file) #CQT频谱图
        input_specs = []
        totol_frame = spec.shape[-1]
        for i in range(self.pad_length, totol_frame - self.pad_length):
            input_spec = spec[
                :, :, i - self.pad_length:i + 1 + self.pad_length] #9个帧
            smax = np.max(input_spec)
            smin = np.min(input_spec)
            input_spec = (input_spec - smin) / (smax - smin + 1e-9)#？

            input_specs.append(np.expand_dims(input_spec, axis=0))
        input_specs = np.concatenate(input_specs, axis=0)
        input_specs = Variable(torch.from_numpy(input_specs).float())#频谱转换为间隔为1帧的N个9帧数组，并且变换为torch变量
        return input_specs  #2d

    def cal_cqt_spectrum(self, wav_file): #计算CQT频谱图

        def padding_cqt_spec(cqt_spec):
            padding_zeros = np.zeros((cqt_config['n_bins'], self.pad_length))#n_bins=267,pad_length=4
            cqt_spec = np.concatenate(#数组拼接
                (padding_zeros, cqt_spec, padding_zeros), axis=1)
            return cqt_spec

        y, sr = librosa.load(wav_file, cqt_config[
                             'sample_rate'], mono=True)
        cqt = librosa.core.cqt(y, sr=sr, hop_length=cqt_config['hop_length'],
                               fmin=librosa.note_to_hz('A0'),
                               n_bins=cqt_config['n_bins'],
                               bins_per_octave=cqt_config['bins_per_octave'])
        cqt = np.abs(cqt)
        cqt = padding_cqt_spec(cqt)
        cqt = np.expand_dims(cqt, axis=0)
        return cqt

    def load_model(self, model_path):
        self.net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        self.net.eval()#设置预测模式
        if use_cuda:
            self.net = self.net.cuda()

    def predict(self, wav_file):#预测onset时间
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
            print(_pred.data)
            pred += [_pred.data.squeeze().cpu().numpy()]
        pred = np.concatenate(pred, axis=0)     #1d，得到此帧onset的概率
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
        self._onset_prob = np.where(pred > self.thresh)[0] #onset帧索引
        self._onset_pred = pred
        onset_prob = self._onset_prob.copy()
        onset_frame = []
        i = 0
        while i < len(onset_prob):
            candi_frame = []
            j = i
            while j < len(onset_prob):
                if (onset_prob[j] - onset_prob[i]) <= 15:   #0.15内的onset处理
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

if __name__ == '__main__':
    wav_file = 'F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\408\\408.wav'
    predictor = predictor_onset()
    onset_time = predictor.predict(wav_file)
