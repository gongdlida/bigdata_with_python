import os
import time
from time import ctime # function to convert a time in seconds to a string

# requires pyAudio when using microphone inputs
import speech_recognition as sr # google speech recognition

from gtts import gTTS # google text-to-speech

from pygame import mixer # loading and playing sounds
mixer.init()

# 오디오 처리
def speak(audioString):
  print(audioString)
  tts= gTTS(text= audioString, lang="en")
  tts.save("audio.mp3")

  mixer.music.load("audio.mp3")
  mixer.music.play()
  while mixer.music.get_busy(): # check if the file is playing
    pass
  mixer.music.load("empty.mp3") # for releasing audio.mp3

# 음성 녹음과 인식
def recordAudio():
  r = sr.Recognizer()
  with sr.Microphone() as source:
      print("Say something!")
      audio = r.listen(source)
      #Speech recognition using google speech recognition
      data = ""
      try:
        data= r.recognize_google(audio)
        print("You said" + data)
      except sr.UnknownValueError:
          print("google speech recognition could not understand audio")
      except sr.RequestError as e :
          print ("could not request result from google speech recognition service;, {0}".format(e))
      return(data)

# 대답 처리
  def jarvis(data):
    if "how are you" in data:
      spaek("I'm fine")
    if "what time is it" in data:
      speak(ctime())
    if "where is" in data:
      data = data.split(" ")
      location = data[2]
      speak("Hold on, please. I will show you where " + location + " is.") 
      os.system("chromium-browser https://www.google.nl/maps/place/" + location + "/&amp;")
    if "bye" in data:
      speak("Have a nice day!") 
      return(True)
  return(False)

  while True:
    data = recordAudio() 
    if jarvis(data)==True:
      break
# pip install chatterbot 설치 시 python3 version이 3.9라서 현재 가장 최근 chatterbot 라이브러리가 설치에 이슈가 있음
# 하단의 stack overflow 글에서 설치하는 방뻡을 제시해주었음 -> 설치는 되었으나, 자료의 예시처럼 모든 결곽값이 출력되고 있지는 않고 있음
# https://stackoverflow.com/questions/64827019/pip-fails-to-install-library-and-i-dont-know-why-chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer

myBot = ChatBot(name="JARVIS", read_only=True,
logic_adapters=["chatterbot.logic.MathematicalEvaluation","chatterbot.logic.TimeLogicAdapter","chatterbot.logic.BestMatch"]
)

smallTalk = [
"Hi there!",
"Hi!",
"How do you do?",
"How are you?",
"I'm cool.",
"Fine, you?",
"Always cool.",
"I'm ok.",
"Glad to hear that.",
"I'm fine.",
"Glad to hear that.",
"I feel awesome.",
"Excellent, glad to hear that.",
"Not so good!",
"Sorry to hear that.",
"What's your name?",
"I'm jarvis. ask me a math question, please."]

# 훈련시키기
list_trainer = ListTrainer(myBot)
list_trainer.train(smallTalk)

corpus_trainer = ChatterBotCorpusTrainer(myBot)
corpus_trainer.train("chatterbot.corpus.english.conversations")

# 현재 
print(myBot.get_response("Always cool."))
print(myBot.get_response("What's your name?"))

print(myBot.get_response("What time is it?"))
print(myBot.get_response("Did I ever live?"))
