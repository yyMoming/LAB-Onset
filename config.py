import os

note_type_param = {
    0.0625: 0.5,
    0.125: 1,
    0.25: 2,
    0.375: 3,
    0.5: 4,
    0.625: 5,
    0.75: 6,
    0.875: 7,
    1.0: 8
}

pitch_det_param = {
    'hopsize_t': 512/float(44100),
    'fftLength': 8192,
    'windowLength': 2048,
    'sampleRate': 44100,
    'frameSize': 2048,
    'hopSize': 512,
    'H': 5,
    'h': 0.8
}

cqt_config = {
    'sample_rate': 44100,
    'hop_length': 512,
    'n_bins': 267,
    'bins_per_octave': 36,
}

alignment_param = {
    'MATCH_COST': 0,
    'INSERT_COST': 1,
    'DELETE_COST': 2
}

post_process_param = {
	'sample_ratio':0.3,
	'hopsize_t':512/float(44100),
}

model_path = './Models/onset.pth'

sr = 44100
hop_length = 512
bins_per_octave = 36
per_sec_frame = 44100 / float(512)
hopsize_t = 512 / float(44100)
threshold = 0.5


score_json_name = ['1A_35', '1A_22', 'C_01', 'C_02', 'C_03',
                   'C_04', 'C_05', 'Jingle_Bells', 'Poem_Chorus_final',
                   'Yankee_Doodle_final', 'You_and_Me', 'G1-1', 'G1-2', 'G1-4', 'G1-9', 'G1-5', 'LAW']
