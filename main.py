# import speech_recognition as sr
from os import path
import os
import time
import hmm as my_hmm

r = sr.Recognizer()
m = sr.Microphone()

keyword = [('a minor', 0.3), ('c major', 0.3), ('d minor', 0.3), ('e minor', 0.3), ('f major', 0.3), ('g major', 0.3)]

def recorded_audio(filename):
    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), filename)
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)
    return audio

def playback(filename):
    import wave
    import pyaudio

    CHUNK = 1024

    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(CHUNK)

    while data != '':
        stream.write(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()

def playchord(chord):
    if chord == 'a minor':
        playback('chords/Am.wav')
    elif chord == 'c major':
        playback('chords/C.wav')
    elif chord == 'd minor':
        playback('chords/Dm.wav')
    elif chord == 'e minor':
        playback('chords/Em.wav')
    elif chord == 'g major':
        playback('chords/G.wav')
    elif chord == 'f major':
        playback('chords/F.wav')
    else:
        pass

def main():
    # audio = recorded_audio('test_audio/record apple.wav')

    with m as source:
        print("Please wait")
        r.adjust_for_ambient_noise(source, duration=1)  # listen for 1 second to calibrate the energy threshold for ambient noise levels
        print("Say something!")
        # audio = r.listen(source)

    content = []
    def callback(recognizer, audio):
        try:
            text = recognizer.recognize_sphinx(audio, language="en-US", keyword_entries=keyword)
            # content.append(text)
            # playback(text)
            print(text)
        except sr.UnknownValueError:
            print("could not understand audio")

    # def transcribe():


    stop_listening = r.listen_in_background(m, callback)

    for _ in range(50):
        time.sleep(0.1)

if __name__ == "__main__":
    main()
