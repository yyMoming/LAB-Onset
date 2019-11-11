import re
import librosa
import collections
import numpy as np
per_sec_frame = 44100 / float(512)
sample_rate = 44100
hop_length = 512
bins_per_octave = 36
n_bins = 267
pad_length = 4
class data_proc(object):
    def __init__(self,pad_length,
                 spec_style="cqt",
                 dual_channel=False,
                 is_filter=True):
        super(data_proc, self).__init__()
        self.pad_length = pad_length
        self.dual_channel = dual_channel
        self.is_filter = is_filter
        self.spec_style = spec_style

    def input_onset_data(self,wav_file,anno_file):
        cqt_spec = self.cal_cqt_spec(wav_file)
        onset_values = self.parse_anno_file_onset(anno_file)
        total_frame_num = cqt_spec.shape[-1]
        input_specs = []
        for i in range(self.pad_length, total_frame_num - self.pad_length):
            input_spec = cqt_spec[:, :, i - self.pad_length:i + 1 + self.pad_length]
            smax = np.max(input_spec) # max in one frame
            smin = np.min(input_spec)
            input_spec = (input_spec - smin) / (smax - smin + 1e-9) # normalize to 0-1
            input_specs.append(np.expand_dims(input_spec, axis=0))
        input_specs = np.concatenate(input_specs,axis=0)
        frames = input_specs.shape[0]
        onset_labels = np.zeros(frames)
        filter_frame = []
        for value in onset_values:
            index = value - 1 if value >= 1 else 0
            onset_loc = np.array(
                [index - 2, index - 1, index, index + 1, index + 2])
            onset_loc = onset_loc.clip(min=0, max=frames - 1)
            onset_labels[onset_loc] = 1
            filter_frame.extend([onset_loc[0], onset_loc[1],
                                 onset_loc[3], onset_loc[4]])
        if self.is_filter:
            input_specs = np.delete(input_specs, filter_frame, axis=0)
            onset_labels = np.delete(onset_labels, filter_frame, axis=0)

        input_labels = onset_labels.copy()

        return input_specs, input_labels

    def parse_anno_file_onset(self,anno_file):
        '''
            get onset labels
        '''
        with open(anno_file,'r+') as f:
            onsets = f.readlines()
        onsets = onsets[1:]
        onset_values = []
        for onset in onsets:
            onset = onset.strip().strip('\n').split('\t')[0]
            onset = float(onset)*per_sec_frame
            onset_values.append(int(round(onset)))
        onset_values = sorted(onset_values,key=onset_values.index)
        return onset_values

    def cal_cqt_spec(self,wav_file,dual_channel=False):
        def padding_cqt_spec(cqt_spec):
            padding_zeros = np.zeros((n_bins, self.pad_length))
            cqt_spec = np.concatenate(
                (padding_zeros, cqt_spec, padding_zeros), axis=1)
            return cqt_spec

        if dual_channel:
            y,sr = librosa.load(wav_file,sr=sample_rate,mono=True)
            cqt1 = librosa.core.cqt(y[0], sr=sr, hop_length=hop_length,
                                        fmin=librosa.note_to_hz('A0'),
                                        n_bins=n_bins,
                                    bins_per_octave=bins_per_octave)
            cqt2 = librosa.core.cqt(y[1], sr=sr, hop_length=hop_length,
                                        fmin=librosa.note_to_hz('A0'),
                                        n_bins=n_bins,
                                    bins_per_octave=bins_per_octave)
            cqt1 = padding_cqt_spec(np.abs(cqt1))
            cqt2 = padding_cqt_spec(np.abs(cqt2))
            cqt1 = np.expand_dims(cqt1, axis=0)
            cqt2 = np.expand_dims(cqt2, axis=0)
            cqt = np.concatenate((cqt1, cqt2), axis=0)
        else:
            y,sr = librosa.load(wav_file,sr=sample_rate,mono=True)
            cqt = librosa.core.cqt(y, sr=sr, hop_length=hop_length,
                                        fmin=librosa.note_to_hz('A0'),
                                        n_bins=n_bins,
                                   bins_per_octave=bins_per_octave)
            cqt = np.abs(cqt)
            cqt = padding_cqt_spec(cqt)
            cqt = np.expand_dims(cqt, axis=0)
        return cqt

if __name__ == '__main__':
    anno_file = "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\364\\364_onset.txt"
    wav_file = "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\308\\308.wav"
    instance = data_proc(pad_length)
    input_specs, input_labels = instance.input_onset_data(wav_file,anno_file)
    print(input_specs.shape,input_labels.shape)