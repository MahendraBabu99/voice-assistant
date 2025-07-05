import speech_recognition as sr
import pyttsx3 
import datetime
import pywhatkit as pw 
import wikipedia as wiki
import os
import time
import threading
import torch

from tokenizers import Tokenizer
from mahe import Transformer

  # adjust import path if needed

def generate_reply(model, tokenizer, input_text, max_len=50):
                input_ids = tokenizer.encode(input_text).ids[:max_len-2]
                input_tensor = torch.tensor([[1] + input_ids + [2]])  # Add [SOS] and [EOS]
                src_mask = (input_tensor != 0).unsqueeze(1).unsqueeze(2)

                enc_output = model.encode(input_tensor, src_mask)

                decoded = [1]  # Start with [SOS]
                for _ in range(max_len):
                    tgt_input = torch.tensor([decoded]).long()
                    tgt_mask = torch.tril(torch.ones(tgt_input.size(1), tgt_input.size(1))).bool().unsqueeze(0)
                    dec_output = model.decode(enc_output, src_mask, tgt_input, tgt_mask)
                    logits = model.project(dec_output)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
                    if next_token == 2:  # [EOS]
                        break
                    decoded.append(next_token)

                return tokenizer.decode(decoded[1:])  # Skip [SOS]

# Load tokenizer
tokenizer = Tokenizer.from_file("chat_tokenizer.json")

# Load model
model = Transformer.build_transformer(
    src_vocab_size=len(tokenizer.get_vocab()),
    tgt_vocab_size=len(tokenizer.get_vocab()),
    src_seq_len=50,
    tgt_seq_len=50
)
model.load_state_dict(torch.load("chatbot_transformer.pth", map_location=torch.device('cpu')))
model.eval()

# Get assistant name from user
va_name = "vrinda"
    
# Initialize the engine
engine = pyttsx3.init()
engine.setProperty('rate', 125)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

def speak(text):
    engine.say(text)
    engine.runAndWait()
def start_timer(seconds):
    time.sleep(seconds)
    speak("Time's up, boss!")

def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Tell me, boss.")
        print("Listening...")
        r.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = r.listen(source, phrase_time_limit=5)
            command = r.recognize_google(audio).lower()
            print("Raw:", command)

            if va_name in command:
                print("You:", command)
                return command
            else:
                print(f"(Ignored) Assistant name '{va_name}' not mentioned.")
                speak(f"not specified"'{va_name}')
                return ""
        
        except sr.UnknownValueError:
            print("Could not understand the audio.")
            speak("Sorry, I didn't get that, boss.")
            return ""
        except sr.RequestError as e:
            print(f"RequestError: {e}")
            speak("Speech server is down, boss.")
            return ""

# Main command loop
while True:
    cmd = take_command()
    if cmd:
        if "exit" in cmd or "stop" in cmd:
            speak("Shutting down, boss.")
            break
        elif "what is time " in cmd:
            now =  datetime.datetime.now()
            cur_time = now.strftime("%I:%M:%P")
            speak("The current time is"+cur_time+"boss")
        elif "who is" in cmd or "what is" in cmd or "tell me about" in cmd or "is it real" in cmd:
            topic = cmd.replace(" vrinda who is", "").replace(" vrinda what is", "").replace(" vrinda tell me about", "").replace("vrinda is it real", "").strip()
            try:
                info = wiki.summary(topic, 2)
                print(info)
                speak(info)
            except:
                speak("Sorry, I couldn't find anything about that, boss.")

        elif "turn on" in cmd:
            device = cmd.replace("vrinda turn on", "").strip()
            speak("Turning on " + device + ", boss.")
            if "wifi" in device:
                os.system('netsh interface set interface name="Wi-Fi" admin=enabled')    
        elif "time in" in cmd:
            cmd1 = cmd.replace("vrinda time in ","").strip()
            cur_time2 = wiki.summary(cmd1)
            speak(cur_time2+"boss")
        elif "turn off" in cmd:
            device = cmd.replace("vrinda turn off", "").strip()
            speak("Turning off " + device + ", boss.")
            if "wifi" in device:
                os.system('netsh interface set interface name="Wi-Fi" admin=disabled')

        elif "shutdown" in cmd:
            speak("Shutting down the system.")
            os.system('shutdown /s /t 1')

        elif "restart" in cmd:
            speak("Restarting the system.")
            os.system('shutdown /r /t 1')
            
        elif "open youtube" in cmd:
            speak("Opening YouTube, boss.")
            pw.playonyt("YouTube")

        elif "open" in cmd:
            item = cmd.replace("vrinda open", "").strip()
            speak("Opening " + item + ", boss.")
            pw.search(item)

        elif "play" in cmd:
            song = cmd.replace("vrinda play", "").strip()
            speak("Playing " + song + ", boss.")
            pw.playonyt(song)

        elif "search" in cmd or "google" in cmd:
            query = cmd.replace("vrinda search", "").replace("vrinda google", "").strip()
            speak("Searching " + query + ", boss.")
            pw.search(query)

        elif "give me code" in cmd:
            topic = cmd.replace("vrinda give me code", "").strip()
            speak("Here is the code for " + topic + ", boss.")
            pw.search(topic)
        elif "set timer" in cmd:
            try:
                timer_text = cmd.replace("vrinda set timer for", "").replace("vrinda set timer", "").strip()
                seconds = 0
                if "minute" in timer_text:
                    num = int(''.join(filter(str.isdigit, timer_text)))
                    seconds = num * 60
                elif "second" in timer_text:
                    num = int(''.join(filter(str.isdigit, timer_text)))
                    seconds = num
                else:
                    seconds = int(timer_text)

                speak(f"Setting a timer for {seconds} seconds, boss.")
                threading.Thread(target=start_timer, args=(seconds,)).start()

            except:
                speak("I couldn't set the timer, boss. Please try again with a valid time.")
        elif "your name" in cmd:
            speak("my name is"+va_name+"boss")
        elif "what is time " in cmd:
            now =  datetime.datetime.now()
            cur_time = now.strftime("%I:%M:%P")
            speak("The current time is"+cur_time+"boss")
       
        else:
            input_text = cmd.replace(va_name, "").strip()
            response = generate_reply(model, tokenizer, input_text)
            print("🤖:", response)
            speak(response)

