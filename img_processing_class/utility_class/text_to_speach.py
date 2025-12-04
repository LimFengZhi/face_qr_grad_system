import threading
import time
import queue


class TextToSpeech:
    """Text-to-Speech class for announcing student names"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._speaking = False
        self._last_spoken = None
        self._last_spoken_time = 0
        self._cooldown = 5  # seconds between announcements for same person
        self._speech_queue = queue.Queue()
        self._running = True
        
        # Start background speech worker thread
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker_thread.start()
        
        print("✓ Text-to-Speech initialized")
    
    def _speech_worker(self):
        """Background worker that processes speech requests"""
        import pyttsx3
        
        while self._running:
            try:
                # Wait for speech request (timeout to check _running flag)
                try:
                    name = self._speech_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Create fresh engine for each speech (fixes threading issues)
                engine = None
                try:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.setProperty('volume', 1.0)
                    
                    self._speaking = True
                    engine.say(f"Congratulation, {name}")
                    engine.runAndWait()
                except Exception as e:
                    print(f"Speech error: {e}")
                finally:
                    self._speaking = False
                    if engine:
                        try:
                            engine.stop()
                        except:
                            pass
                    
                self._speech_queue.task_done()
                
            except Exception as e:
                print(f"Speech worker error: {e}")
    
    def speak(self, name):
        """Speak the name asynchronously"""
        if self._speaking:
            return
        
        # Check cooldown for same person
        current_time = time.time()
        if name == self._last_spoken and (current_time - self._last_spoken_time) < self._cooldown:
            return
        
        self._last_spoken = name
        self._last_spoken_time = current_time
        
        # Add to speech queue
        self._speech_queue.put(name)
    
    def set_cooldown(self, seconds):
        """Set cooldown between announcements for same person"""
        self._cooldown = seconds
    
    def stop(self):
        """Stop the speech worker"""
        self._running = False