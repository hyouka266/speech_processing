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

    sf.write(filename, data=data, samplerate=fs)

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

#%%
def get_prob(log_x1, log_x2):
    if log_x1[0] < log_x2[0]:
        exp_x1_x2 = exp(log_x1[0]-log_x2[0])
        return (exp_x1_x2 / (1+exp_x1_x2), log_x1[1]), (1 / (1+exp_x1_x2), log_x2[1])
    else:
        p = get_prob(log_x2, log_x1)
        return p[1], p[0]

#%%
# record_data('train_data/C')
# record_data('train_data/D')
# record_data('train_data/E')
# record_data('train_data/F')
# record_data('train_data/G')
# record_data('train_data/A')

#%%
n_sample = 10
data_C = [get_mfcc('train_data/C_{}.wav'.format(i)) for i in range(n_sample)]
data_Dm = [get_mfcc('train_data/Dm_{}.wav'.format(i)) for i in range(n_sample)]
data_Em = [get_mfcc('train_data/Em_{}.wav'.format(i)) for i in range(n_sample)]
data_F = [get_mfcc('train_data/F_{}.wav'.format(i)) for i in range(n_sample)]
data_G = [get_mfcc('train_data/G_{}.wav'.format(i)) for i in range(n_sample)]
data_Am = [get_mfcc('train_data/Am_{}.wav'.format(i)) for i in range(n_sample)]

#%%
model_C = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_C.fit(X=np.vstack(data_C), lengths=[x.shape[0] for x in data_C])

model_Dm = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Dm.fit(X=np.vstack(data_Dm), lengths=[x.shape[0] for x in data_Dm])

model_Em = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Em.fit(X=np.vstack(data_Em), lengths=[x.shape[0] for x in data_Em])

model_F = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_F.fit(X=np.vstack(data_F), lengths=[x.shape[0] for x in data_F])

model_G = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_G.fit(X=np.vstack(data_G), lengths=[x.shape[0] for x in data_G])

model_Am = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Am.fit(X=np.vstack(data_Am), lengths=[x.shape[0] for x in data_Am])

#%%
# print(model_C.score(mfcc) - model_Dm.score(mfcc))

#%%
def transcribe_audio(filename):
    mfcc = get_mfcc(filename)
    tuple_c = (model_C.score(mfcc), 'c major')
    tuple_dm = (model_Dm.score(mfcc), 'd minor')
    tuple_em = (model_Em.score(mfcc), 'e minor')
    tuple_f = (model_F.score(mfcc), 'f major')
    tuple_g = (model_G.score(mfcc), 'g major')
    tuple_am = (model_Am.score(mfcc), 'a minor')

    # for i in range(1):
    log_pC, log_pD, log_pE, log_pF, log_pG, log_pA = tuple_c, tuple_dm, tuple_em, tuple_f, tuple_g, tuple_am 
    arr = [log_pC, log_pD, log_pE, log_pF, log_pG, log_pA]
    log_max = max(arr)
    print(log_max[0], log_max[1])

#%%
record_sound('audio_test.wav')
transcribe_audio('audio_test.wav')
#ddsds


#%%
