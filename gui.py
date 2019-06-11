import simpleaudio as sa

filename = 'chords/bm.wav'
wav_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wav_obj.play()
play_obj.wait_done()

bot_frame = tk.Frame(master=root, bg='white', height=150)
bot_frame.pack(side='bottom')

play_button = tk.Button(master=bot_frame, 
    text='Play', 
    width=25, 
    fg='black',
    activeforeground='black', 
    activebackground='lightblue',
    bg='white')
play_button.pack()

pad_frame2 = tk.Frame(master=bot_frame, bg='white', height=30)
pad_frame2.pack()

pad_frame = tk.Frame(master=bot_frame, bg='white', height=50)
pad_frame.pack()