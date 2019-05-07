import speech_recognition as sr
from os import path

r = sr.Recognizer()

def google_recognizer(audio):
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""{
        "type": "service_account",
        "project_id": "discord-bot-224114",
        "private_key_id": "5e814c8e4b03fcd8a97e87c71d07ba8dd872d726",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCxQmo/6hIInpF5\nIkxAAOFdNA36TCMRFT2Rl9DM3wuIC9VsmrKhMhj6oDJnYsfslli68knphQ0vDNCT\n9a3n9uiQTBPyRQa5HMzzJvmbywAONDaFZmbbS8gQCShwPdM5Y9kOiGsMFdCbvBbX\nMUARPvoZuj1HlGJdd+RZFW9mVS8RyVDRmwNPBimSSAlOfSWIqZ5DihoG2Lm9W62d\nJKo6x8hwN8i38IVdar9HbP4Q9MG3zoYUL3U4iH8UFR5feHCDpXtx6sfT6fvb7+WC\na2xTiDqFf2Vfz5VXlh6sIp6aiht+bP5wKL09cwd/ztCYcRpI0NoHsn0S3rzxOjnW\nCLo3Avg9AgMBAAECggEAFsNQoaSNbE3woH2nDREP38gvg1UDq83ueiPkzGpYMMBU\nR/4Vsl2cl5K65kvpIJiuDFPQ6gwMBV6/6kBCdOdgNieO6C21D1CDgTYgF8GZ4DkU\nR00/KCozJtwGNWO7HEOWPCeIl6S1hUpCJKBOeeh8dZFVDTwg5Z/f0xDNnXaS9NWO\nGliuJDusQKjf0q3JSls2U2c9Ugsa25UHG7gJefML6QPNTnVbwRQwqw3WPAOZkpam\n6xbKSuFiwT/X/KeTMFYSE36eKuEPpqjXVTHZZlGpvfDpISLC1CRx53IyBD7kvp/N\n8CTFaxURKrOQ71oZF6ZjUjTBZTT4vIQ9aQeZU+xkgQKBgQD3IvayHIKIeeDi1kXj\nm63xhWuEQ6XTd0LDN1zXqES1ZsTZj5C5A4lnanA/T1Ljc5ys8zXz4Gu2ww+6vo4I\nOL/b7T02w56r8ts1JniKp+lV43KTpeeV0f33Epwc+hLITWTLyDa+9a0ity8M38mJ\nagkMBdQAqjgkJda+d2PYjYBq0wKBgQC3neNUNhvy+Jyr9V5Qw/bJf+Weui6dEQ0n\nNhxZU1kKZXFIJ1HcN1OppoXz/ituoCw0ku4ly5cDnN2vqLdPOwqJ0BgaBZxE9+jy\nm+mtWIvpBnimWS32EJEoNh2UaXE2EMWgKc9E2uv4QicymSuwc7lbX44Sflu1dsvR\nmOUNmcMGrwKBgQC9PrfcUjqe2X2dFmn0Tj1Xykw3vzmXgibMqHNe7QqYQncRBn/T\nPWMVnwsPX+XgKKLcSW2SL9Mr45kC7nKC1zoL5lJOwmOZ1mGIBMqfms7yJzaQ26VI\nM8KfVU/YXKLPYXyDE/DgL+8Bu3a7DA8fO+RroXbjf3V4MMWNmo0JwemJYwKBgCvT\nGWEOERmq0OoSBFLOkuaBCjMaSOngGf2T4qxHQmdC0wjfqaAf7G3/etVDguZCgIqD\nzydiMkcAd8DnSek1NEy0SCxdznB/oy/Umq9vBOW3T7CUdG3YgmzQjbrQd97pneGe\nWQQcJFn6oBRpjo3s8P6oDebIFed31SnPjkvyuSOtAoGBALe6FRIK5X22sIlCw3v+\nAqCHwg1Koau23i5zoa0ONjnMQN7h8wjthxs6V8XNl3TC14T+Qwi/2efH8heYTGc1\nf+8E+qzjEVld98JYqL5owoLSKhWl8KMqIekRdjMXvHPlFQ0pffoA+sDkd1m9yxZq\nncP53dvpLUOt6xvn2lr92AtZ\n-----END PRIVATE KEY-----\n",
        "client_email": "discord-bot-224114@appspot.gserviceaccount.com",
        "client_id": "111153145478547139353",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/discord-bot-224114%40appspot.gserviceaccount.com"
    }"""
    try:
        text = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS)
        print(text)
    except sr.UnknownValueError:
        print("Google Cloud Speech could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Cloud Speech service; {0}".format(e))

def sphinx_recognizer(audio):
    try:
        text = r.recognize_sphinx(audio)
        print(text)
    except sr.UnknownValueError:
        print("Sphinx could not understand audio")
    except sr.RequestError as e:
        print("Sphinx error; {0}".format(e))

def record(filename):
    AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), filename)
    with sr.AudioFile(AUDIO_FILE) as source:
        audio = r.record(source)
    return audio

def capture():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)  # listen for 1 second to calibrate the energy threshold for ambient noise levels
        print("Say something!")
        audio = r.listen(source)
    return audio

def main():
    audio = record('output.wav')
    google_recognizer(audio)

if __name__ == "__main__":
    main()
