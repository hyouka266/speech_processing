#%%
import os
try:
	os.chdir(os.path.join(os.getcwd(), '../speech_reg_uet'))
	print(os.getcwd())
except:
	pass

#%%
import tkinter as tk
import sounddevice as sd
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import hmmlearn.hmm as hmm
from math import exp
import simpleaudio as sa
import pickle

#%%
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

#%%
def playback(chordname):
    tup = [('piano', 'piano'),
        ('c major', 'c'), ('c minor', 'cm'),
        ('d major', 'd'), ('d minor', 'dm'),
        ('e major', 'e'), ('e minor', 'em'),
        ('f major', 'f'), ('f minor', 'fm'),
        ('g major', 'g'), ('g minor', 'gm'),
        ('a major', 'a'), ('a minor', 'am'),
        ('b major', 'b'), ('b minor', 'bm')]
    filename = ['chords/{}.wav'.format(i[1]) for i in tup if i[0] == chordname][0]
    wav_obj = sa.WaveObject.from_wave_file(filename)
    play_obj = wav_obj.play()
    play_obj.wait_done()

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T #transpose

#%%
def get_prob(log_x1, log_x2):
    if log_x1[0] < log_x2[0]:
        exp_x1_x2 = exp(log_x1[0]-log_x2[0])
        return (exp_x1_x2 / (1+exp_x1_x2), log_x1[1]), (1 / (1+exp_x1_x2), log_x2[1])
    else:
        p = get_prob(log_x2, log_x1)
        return p[1], p[0]

#%%
n_sample = 30
data_C = [get_mfcc('train_data/C_{}.wav'.format(i)) for i in range(n_sample)]
data_Cm = [get_mfcc('train_data/Cm_{}.wav'.format(i)) for i in range(n_sample)]
data_D = [get_mfcc('train_data/D_{}.wav'.format(i)) for i in range(n_sample)]
data_Dm = [get_mfcc('train_data/Dm_{}.wav'.format(i)) for i in range(n_sample)]
data_E = [get_mfcc('train_data/E_{}.wav'.format(i)) for i in range(n_sample)]
data_Em = [get_mfcc('train_data/Em_{}.wav'.format(i)) for i in range(n_sample)]
data_F = [get_mfcc('train_data/F_{}.wav'.format(i)) for i in range(n_sample)]
data_Fm = [get_mfcc('train_data/Fm_{}.wav'.format(i)) for i in range(n_sample)]
data_G = [get_mfcc('train_data/G_{}.wav'.format(i)) for i in range(n_sample)]
data_Gm = [get_mfcc('train_data/Gm_{}.wav'.format(i)) for i in range(n_sample)]
data_A = [get_mfcc('train_data/A_{}.wav'.format(i)) for i in range(n_sample)]
data_Am = [get_mfcc('train_data/Am_{}.wav'.format(i)) for i in range(n_sample)]
data_B = [get_mfcc('train_data/B_{}.wav'.format(i)) for i in range(n_sample)]
data_Bm = [get_mfcc('train_data/Bm_{}.wav'.format(i)) for i in range(n_sample)]

#%%
model_C = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_C.fit(X=np.vstack(data_C), lengths=[x.shape[0] for x in data_C])
with open('pkl/model_c.pkl', 'wb') as file: pickle.dump(model_C, file)

model_Cm = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Cm.fit(X=np.vstack(data_Cm), lengths=[x.shape[0] for x in data_Cm])
with open('pkl/model_cm.pkl', 'wb') as file: pickle.dump(model_Cm, file)

model_D = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_D.fit(X=np.vstack(data_D), lengths=[x.shape[0] for x in data_D])
with open('pkl/model_d.pkl', 'wb') as file: pickle.dump(model_D, file)

model_Dm = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Dm.fit(X=np.vstack(data_Dm), lengths=[x.shape[0] for x in data_Dm])
with open('pkl/model_dm.pkl', 'wb') as file: pickle.dump(model_Dm, file)

model_E = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_E.fit(X=np.vstack(data_E), lengths=[x.shape[0] for x in data_E])
with open('pkl/model_e.pkl', 'wb') as file: pickle.dump(model_E, file)

model_Em = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Em.fit(X=np.vstack(data_Em), lengths=[x.shape[0] for x in data_Em])
with open('pkl/model_em.pkl', 'wb') as file: pickle.dump(model_Em, file)

model_F = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_F.fit(X=np.vstack(data_F), lengths=[x.shape[0] for x in data_F])
with open('pkl/model_f.pkl', 'wb') as file: pickle.dump(model_F, file)

model_Fm = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Fm.fit(X=np.vstack(data_Fm), lengths=[x.shape[0] for x in data_Fm])
with open('pkl/model_fm.pkl', 'wb') as file: pickle.dump(model_Fm, file)

model_G = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_G.fit(X=np.vstack(data_G), lengths=[x.shape[0] for x in data_G])
with open('pkl/model_g.pkl', 'wb') as file: pickle.dump(model_G, file)

model_Gm = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Gm.fit(X=np.vstack(data_Gm), lengths=[x.shape[0] for x in data_Gm])
with open('pkl/model_gm.pkl', 'wb') as file: pickle.dump(model_Gm, file)

model_A = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_A.fit(X=np.vstack(data_A), lengths=[x.shape[0] for x in data_A])
with open('pkl/model_a.pkl', 'wb') as file: pickle.dump(model_A, file)

model_Am = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Am.fit(X=np.vstack(data_Am), lengths=[x.shape[0] for x in data_Am])
with open('pkl/model_am.pkl', 'wb') as file: pickle.dump(model_Am, file)

model_B = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_B.fit(X=np.vstack(data_B), lengths=[x.shape[0] for x in data_B])
with open('pkl/model_b.pkl', 'wb') as file: pickle.dump(model_B, file)

model_Bm = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
model_Bm.fit(X=np.vstack(data_Bm), lengths=[x.shape[0] for x in data_Bm])
with open('pkl/model_bm.pkl', 'wb') as file: pickle.dump(model_Bm, file)

#%%
with open('pkl/model_c.pkl', 'rb') as file: model_C = pickle.load(file)
with open('pkl/model_cm.pkl', 'rb') as file: model_Cm = pickle.load(file)
with open('pkl/model_d.pkl', 'rb') as file: model_D = pickle.load(file)
with open('pkl/model_dm.pkl', 'rb') as file: model_Dm = pickle.load(file)
with open('pkl/model_e.pkl', 'rb') as file: model_E = pickle.load(file)
with open('pkl/model_em.pkl', 'rb') as file: model_Em = pickle.load(file)
with open('pkl/model_f.pkl', 'rb') as file: model_F = pickle.load(file)
with open('pkl/model_fm.pkl', 'rb') as file: model_Fm = pickle.load(file)
with open('pkl/model_g.pkl', 'rb') as file: model_G = pickle.load(file)
with open('pkl/model_gm.pkl', 'rb') as file: model_Gm = pickle.load(file)
with open('pkl/model_a.pkl', 'rb') as file: model_A = pickle.load(file)
with open('pkl/model_am.pkl', 'rb') as file: model_Am = pickle.load(file)
with open('pkl/model_b.pkl', 'rb') as file: model_B = pickle.load(file)
with open('pkl/model_bm.pkl', 'rb') as file: model_Bm = pickle.load(file)

#%%
def transcribe_audio(filename):
    mfcc = get_mfcc(filename)
    tuple_c = (model_C.score(mfcc), 'c major')
    tuple_cm = (model_Cm.score(mfcc), 'c minor')
    tuple_d = (model_D.score(mfcc), 'd major')
    tuple_dm = (model_Dm.score(mfcc), 'd minor')
    tuple_e = (model_E.score(mfcc), 'e major')
    tuple_em = (model_Em.score(mfcc), 'e minor')
    tuple_f = (model_F.score(mfcc), 'f major')
    tuple_fm = (model_Fm.score(mfcc), 'f minor')
    tuple_g = (model_G.score(mfcc), 'g major')
    tuple_gm = (model_Gm.score(mfcc), 'g minor')
    tuple_a = (model_A.score(mfcc), 'a major')
    tuple_am = (model_Am.score(mfcc), 'a minor')
    tuple_b = (model_B.score(mfcc), 'b major')
    tuple_bm = (model_Bm.score(mfcc), 'b minor')

    # for i in range(1):
    log_pC, log_pCm = tuple_c, tuple_cm
    log_pD, log_pDm = tuple_d, tuple_dm
    log_pE, log_pEm = tuple_e, tuple_em
    log_pF, log_pFm = tuple_f, tuple_fm
    log_pG, log_pGm = tuple_g, tuple_gm
    log_pA, log_pAm = tuple_a, tuple_am
    log_pB, log_pBm = tuple_b, tuple_bm
    arr = [log_pC, log_pD, log_pE, log_pF, log_pG, log_pA, log_pB, log_pCm, log_pDm, log_pEm, log_pFm, log_pGm, log_pAm, log_pBm]
    log_max = max(arr)
    return log_max
    # print(log_max[0], log_max[1])

#%%
# record_sound('audio_test.wav')
print(transcribe_audio('train_data/Am_12.wav'))
# playback('e major')

#%%
root = tk.Tk()
root.title('Music Chord')
root.geometry('600x500')
root.resizable(0,0)
frame = tk.Frame(master=root, bg='white')
frame.pack_propagate(0)
frame.pack(fill=tk.BOTH, expand=1)

menu = tk.Menu(root, bg='white', fg='black', activebackground='lightblue', activeforeground='black')
root.config(menu=menu, bg='white')
helpmenu = tk.Menu(menu, bg='white', fg='black', activebackground='lightblue', activeforeground='black')
menu.add_cascade(label='Help', menu=helpmenu)
helpmenu.add_command(label='About')
helpmenu.add_command(label='Exit', command=root.quit)

photo = tk.PhotoImage(file='pics/piano.png')
chord_pic = tk.Label(frame, image=photo)
chord_pic.pack()

def changepic(chordname):
    tup = [('piano', 'piano'),
        ('c major', 'c'), ('c minor', 'cm'),
        ('d major', 'd'), ('d minor', 'dm'),
        ('e major', 'e'), ('e minor', 'em'),
        ('f major', 'f'), ('f minor', 'fm'),
        ('g major', 'g'), ('g minor', 'gm'),
        ('a major', 'a'), ('a minor', 'am'),
        ('b major', 'b'), ('b minor', 'bm')]
    filename = ['pics/{}.png'.format(i[1]) for i in tup if i[0] == chordname][0]
    pic = tk.PhotoImage(file=filename)
    chord_pic.configure(image=pic)
    chord_pic.image = pic


labelText = tk.StringVar()
def rec_func(event):
    record_sound('audio_test.wav')
    log_max, _chord_name = transcribe_audio('audio_test.wav')
    chord_name = _chord_name
    changepic(_chord_name)
    labelText.set(_chord_name)
    playback(_chord_name)


def play_func(event):
    # print(labelText.get())
    playback(labelText.get())

chord_label = tk.Label(frame, textvariable=labelText, fg='black', font=20, bg='white')
chord_label.pack()

bot_frame = tk.Frame(master=root, bg='white', height=100)
bot_frame.pack(side='bottom')

play_button = tk.Button(master=bot_frame, 
    text='Play Chord', 
    width=25, 
    fg='black',
    activeforeground='black', 
    activebackground='lightblue',
    bg='white')
play_button.pack()
play_button.bind('<Button-1>', play_func)

pad_frame2 = tk.Frame(master=bot_frame, bg='white', height=30)
pad_frame2.pack()

rec_button = tk.Button(master=bot_frame,
    text='Record', 
    width=25, 
    fg='black',
    activeforeground='black', 
    activebackground='lightgreen',
    bg='white')
rec_button.pack()
rec_button.bind('<Button-1>', rec_func)

pad_frame = tk.Frame(master=bot_frame, bg='white', height=50)
pad_frame.pack()



root.mainloop()

#%%
