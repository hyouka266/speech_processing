#%%
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../speech_reg_uet'))
	print(os.getcwd())
except:
	pass
#%%
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import hmmlearn.hmm as hmm
from math import exp

def record_sound(filename, duration=1, fs=44100, play=False):
    # input('start')
    sd.play( np.sin( 2*np.pi*940*np.arange(fs)/fs )  , samplerate=fs, blocking=True)
    sd.play( np.zeros( int(fs*0.2) ), samplerate=fs, blocking=True)
    data = sd.rec(frames=duration*fs, samplerate=fs, channels=1, blocking=True)
    if play:
        sd.play(data, samplerate=fs, blocking=True)

    sf.write('train_data/' + filename, data=data, samplerate=fs)

def record_data(prefix, i=0, n=10, duration=1):
    for i in range(i, n):
        print('{}_{}.wav'.format(prefix, i))
        record_sound('{}_{}.wav'.format(prefix, i), duration=duration)
        # if i % 10 == 9:
        #     input("Press Enter to continue...")

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T

# def get_prob(log_x1, log_x2, log_x3, log_x4, log_x5, log_x6):
#     arr = [log_x1, log_x2, log_x3, log_x4, log_x5, log_x6]
#     log_min = min(arr)

#         exp_x1_x2 = exp(log_x1-log_x2)
#         exp_x1_x3 = exp(log_x1-log_x3)
#         exp_x1_x4 = exp(log_x1-log_x4)
#         exp_x1_x5 = exp(log_x1-log_x5)
#         exp_x1_x6 = exp(log_x1-log_x6)
#         return  (exp_x1_x2 / (1+exp_x1_x2), 1 / (1+exp_x1_x2)), 
#                 (exp_x1_x3 / (1+exp_x1_x3), 1 / (1+exp_x1_x3)), 
#                 (exp_x1_x4 / (1+exp_x1_x4), 1 / (1+exp_x1_x4)), 
#                 (exp_x1_x5 / (1+exp_x1_x5), 1 / (1+exp_x1_x5)), 
#                 (exp_x1_x6 / (1+exp_x1_x6), 1 / (1+exp_x1_x6)), 
#     # else:
#     #     p = get_prob(log_x2, log_x1)
#     #     return p[1], p[0]

#%%
def get_prob(log_x1, log_x2):
    if log_x1[0] < log_x2[0]:
        exp_x1_x2 = exp(log_x1[0]-log_x2[0])
        return (exp_x1_x2 / (1+exp_x1_x2), log_x1[1]), (1 / (1+exp_x1_x2), log_x2[1])
    else:
        p = get_prob(log_x2, log_x1)
        return p[1], p[0]

#%%
# record_data('C')
# record_data('D')
# record_data('E')
# record_data('F')
# record_data('G')
# record_data('A')

#%%
n_sample = 10
data_C = [get_mfcc('train_data/C_{}.wav'.format(i)) for i in range(n_sample)]
data_D = [get_mfcc('train_data/D_{}.wav'.format(i)) for i in range(n_sample)]
data_E = [get_mfcc('train_data/E_{}.wav'.format(i)) for i in range(n_sample)]
data_F = [get_mfcc('train_data/F_{}.wav'.format(i)) for i in range(n_sample)]
data_G = [get_mfcc('train_data/G_{}.wav'.format(i)) for i in range(n_sample)]
data_A = [get_mfcc('train_data/A_{}.wav'.format(i)) for i in range(n_sample)]

#%%
model_C = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_C.fit(X=np.vstack(data_C), lengths=[x.shape[0] for x in data_C])

model_D = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_D.fit(X=np.vstack(data_D), lengths=[x.shape[0] for x in data_D])

model_E = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_E.fit(X=np.vstack(data_E), lengths=[x.shape[0] for x in data_E])

model_F = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_F.fit(X=np.vstack(data_F), lengths=[x.shape[0] for x in data_F])

model_G = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_G.fit(X=np.vstack(data_G), lengths=[x.shape[0] for x in data_G])

model_A = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_A.fit(X=np.vstack(data_A), lengths=[x.shape[0] for x in data_A])

#%%
# print(model_C.score(mfcc) - model_D.score(mfcc))

#%%
def transcribe_audio(filename):
    mfcc = get_mfcc(filename)
    tuple_c = (model_C.score(mfcc), 'c major')
    tuple_d = (model_D.score(mfcc), 'd minor')
    tuple_e = (model_E.score(mfcc), 'e minor')
    tuple_f = (model_F.score(mfcc), 'f major')
    tuple_g = (model_G.score(mfcc), 'g major')
    tuple_a = (model_A.score(mfcc), 'a minor')

    for i in range(1):
        # record_sound('test.wav')
        log_pC, log_pD, log_pE, log_pF, log_pG, log_pA = tuple_c, tuple_d, tuple_e, tuple_f, tuple_g, tuple_a 
        arr = [log_pC, log_pD, log_pE, log_pF, log_pG, log_pA]
        log_max = max(arr)
        print(log_max[1])

#%%
transcribe_audio('train_data/G_9.wav')
