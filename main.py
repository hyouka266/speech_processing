import speech_recognition as sr
from os import path
import os

r = sr.Recognizer()

def sphinx_recognizer(audio):
    try:
        text = r.recognize_sphinx(audio)
        return text
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))

def record(filename):
    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), filename)
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)
    return audio

def capture_audio():
    with sr.Microphone() as source:
        print("Please wait")
        r.adjust_for_ambient_noise(source, duration=1)  # listen for 1 second to calibrate the energy threshold for ambient noise levels
        print("Say something!")
        audio = r.listen(source)
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

def main():
    audio = record('output.wav')
    print(sphinx_recognizer(audio))
    # playback('chords/Am.wav')

if __name__ == "__main__":
    main()
