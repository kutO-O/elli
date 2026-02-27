"""
Голосовой модуль (Silero TTS).
Локальный, качественный, русский.
"""

import os
import torch
import threading
import queue

try:
    import winsound
except ImportError:
    winsound = None


class Voice:
    
    def __init__(self):
        self.enabled = False
        self.queue = queue.Queue()
        self.thread = None
        self.model = None
        
        model_path = "voice_models/v4_ru.pt"
        
        if not os.path.exists(model_path):
            print(f"⚠️ Модель голоса не найдена: {model_path}")
            print("   Скачай: https://models.silero.ai/models/tts/ru/v4_ru.pt")
            return
        
        try:
            device = torch.device('cpu')
            
            # Загружаем локально
            self.model = torch.package.PackageImporter(model_path).load_pickle("tts_models", "model")
            self.model.to(device)
            
            self.speaker = 'baya'
            self.sample_rate = 48000
            
            self.enabled = True
            print(f"🎤 Голос загружен: Silero ({self.speaker})")
            
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            
        except Exception as e:
            print(f"❌ Ошибка загрузки голоса: {e}")
    
    def say(self, text, emotion_valence=0.0, emotion_arousal=0.0):
        if not self.enabled or not text.strip():
            return
        self.queue.put((text, emotion_valence, emotion_arousal))
    
    def _worker(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            text, valence, arousal = item
            self._speak_now(text, valence, arousal)
            self.queue.task_done()
    
    def _speak_now(self, text, valence, arousal):
        try:
            temp_file = "temp_speech.wav"
            
            # Генерируем и сразу сохраняем
            self.model.save_wav(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
                audio_path=temp_file
            )
            
            if winsound:
                winsound.PlaySound(temp_file, winsound.SND_FILENAME)
            else:
                os.system(f"aplay {temp_file}")
            
            try:
                os.remove(temp_file)
            except:
                pass
                
        except Exception as e:
            print(f"❌ Ошибка синтеза: {e}")