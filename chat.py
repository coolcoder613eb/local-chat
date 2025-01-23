import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
import ollama
from kokoro_onnx import Kokoro
import nltk
import queue
from dataclasses import dataclass
from typing import Optional

@dataclass
class AudioConfig:
    rate: int = 16000
    channels: int = 1
    dtype: str = 'float32'
    filename: str = "input.wav"
    
@dataclass
class TTSConfig:
    voice: str = "bm_george"
    speed: float = 1.0
    lang: str = "en-us"

class AudioRecorder:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.recording = False
        self.audio_queue = queue.Queue()
        
    def callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_queue.put(indata.copy())
            
    def record(self) -> Optional[str]:
        self.recording = True
        audio_data = []
        print("Press Enter to record.")
        input()
        with sd.InputStream(callback=self.callback, 
                          channels=self.config.channels,
                          samplerate=self.config.rate,
                          dtype=self.config.dtype):
            print("Recording... Press Enter to stop.")
            input()
            
        self.recording = False
        
        while not self.audio_queue.empty():
            audio_data.append(self.audio_queue.get())
            
        if audio_data:
            audio_array = np.concatenate(audio_data)
            sf.write(self.config.filename, audio_array, self.config.rate)
            return self.config.filename
        return None

class VoiceChat:
    def __init__(self, model_path: str, voice_path: str, 
                 audio_config: AudioConfig, tts_config: TTSConfig):
        self.audio_config = audio_config
        self.tts_config = tts_config
        self.recorder = AudioRecorder(audio_config)
        self.model = WhisperModel('medium', device='cuda')
        self.kokoro = Kokoro(model_path, voice_path)
        self.conversation_history = []
        
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        
    def transcribe_audio(self, filename: str) -> str:
        segments, _ = self.model.transcribe(filename, language='en')
        return ''.join(segment.text for segment in segments).strip()
        
    def process_speech(self, text: str):
        samples, sample_rate = self.kokoro.create(
            text, 
            voice=self.tts_config.voice, 
            speed=self.tts_config.speed, 
            lang=self.tts_config.lang
        )
        sd.wait()
        sd.play(samples, sample_rate)
        
    def chat(self, user_input: str):
        self.conversation_history.append({'role': 'user', 'content': user_input})
        
        stream = ollama.chat(
            model='llama3.2',
            messages=self.conversation_history,
            stream=True,
        )
        
        assistant_reply = ""
        sentence_buffer = ""
        
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
            assistant_reply += content
            sentence_buffer += content.replace('*', '').replace('\n', '.\n')
            
            sentences = nltk.sent_tokenize(sentence_buffer)
            
            while len(sentences) > 1:
                self.process_speech(sentences[0])
                sentence_buffer = " ".join(sentences[1:])
                sentences = nltk.sent_tokenize(sentence_buffer)
        
        if sentence_buffer.strip():
            self.process_speech(sentence_buffer)
            
        self.conversation_history.append({'role': 'assistant', 'content': assistant_reply})
        
    def run(self):
        print("Starting voice chat. Press Enter to start/stop recording.")
        while True:
            filename = self.recorder.record()
            if filename and os.path.exists(filename):
                user_input = self.transcribe_audio(filename)
                if user_input:
                    print(f"\nUser: {user_input}")
                    self.chat(user_input)
                    print("\n")
                else:
                    print("No speech detected.")
            else:
                print("Recording failed. Please try again.")

if __name__ == "__main__":
    audio_config = AudioConfig()
    tts_config = TTSConfig()
    
    chat = VoiceChat(
        model_path="kokoro-v0_19.onnx",
        voice_path="voices.bin",
        audio_config=audio_config,
        tts_config=tts_config
    )
    chat.run()
