import os
import torch
import time
import numpy as np
import config as cfg
from torch.utils.data import Dataset
from multiprocessing import Process, cpu_count, Manager

from input_data_proc import data_proc


class onset_dataset(Dataset):
    def __init__(self,wav_files,
                 anno_files,
                 pad_length,
                 spec_style="cqt",
                 dual_channel=False,
                 is_filter=False,
                 is_training=True):
        super(onset_dataset, self).__init__()
        self.pad_length = pad_length
        self.is_filter = is_filter
        self.is_training = is_training

        self.spec_style = spec_style
        self.wav_files = wav_files
        self.anno_files = anno_files
        proc_data = data_proc(pad_length=self.pad_length,
                                    spec_style=self.spec_style,
                                    dual_channel=False,
                                    is_filter=self.is_filter)
        self.input_onset_data = proc_data.input_onset_data
        '''
            multiprocess load datas
        '''
        manager = Manager()
        self.datas = manager.list()
        self.labels = manager.list()
        process_num = cpu_count() if len(self.wav_files) > cpu_count()\
                            else len(self.wav_files)
        process_list = []
        for x in range(process_num):
            start = x * len(self.wav_files) // process_num
            end = (x + 1) * len(self.wav_files) // process_num
            p = Process(target=self.get_data,args=(start,end))
            p.start()
            process_list.append(p)
        for p in process_list:
            p.join()
        self.combine_data()
    def get_data(self,start,end):
        for i in range(start,end):
            wav_file = self.wav_files[i]
            anno_file = wav_file[:-4] + '_onset.txt'
            input_specs, onset_labels = self.input_onset_data(wav_file,anno_file)

            self.datas.append(input_specs)
            self.labels.append(onset_labels)
    def combine_data(self):
        self.datas = np.concatenate(self.datas,axis=0)
        self.labels = np.concatenate(self.labels,axis=0)
    def pull_item(self,index):
        inputs = self.datas[index]
        label = self.labels[index]
        return torch.FloatTensor(inputs),torch.FloatTensor([label])
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, index):
        return self.pull_item(index)

if __name__ == '__main__':
    anno_file = ["F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\308\\308_onset.txt",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\364\\364_onset.txt",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\366\\366_onset.txt",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\406\\406_onset.txt",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\408\\408_onset.txt"]

    wav_files = ["F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\308\\308.wav",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\364\\364.wav",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\366\\366.wav",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\406\\406.wav",\
                 "F:\\YWM_work\\Music Data\\HUST_Solfege_onset+wav\\408\\408.wav"]
    test = onset_dataset(wav_files,anno_file,pad_length=4)
    print("datas:",test.datas.shape)
    print(test[2])