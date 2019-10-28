import re
import librosa
import collections
import numpy as np

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
        if self.spec_style == "cqt":
            spec = self.cal_cqt_spec(wav_file)

        onset_values = self.parse_anno_file_onset(anno_file)

def parse_anno_file_onset(anno_file):
    with open(anno_file,'r+') as f:
        onsets = f.readlines()
    onset_values = []
    for onset in onsets:
        onset = onset.strip().strip('\n').split('\t')[0]
