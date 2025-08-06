"""
Enhanced Voice Assistant with Alexa-Style Wake Word Detection

SMART WAKE WORD + VOICE ACTIVITY DETECTION SYSTEM!

Requirements:
- pip install ollama speechrecognition pyaudio
- macOS 10.15+ for built-in Siri TTS  
- Ollama server running locally

Features:
- 🎯 Continuous Wake Word Detection: Always listening for "Sydney"
- 🗣️ Confirmation Response: "How can I help you?" via Siri TTS
- 📱 Smart Voice Activity Detection: Automatically detects when you stop speaking
- 🏠 Offline Capable: Uses PocketSphinx for wake word detection
- 🤖 Official Ollama Python SDK for AI responses
- 🔊 macOS's built-in Siri TTS for high-quality voice responses
- 🛡️ Robust error handling with background noise tolerance
"""

#!/usr/bin/env python3
"""
Enhanced Voice Assistant with Alexa-Style Wake Word Detection

Production-ready voice assistant with:
- Multi-turn conversations with automatic cleanup
- Email integration via macOS Mail app
- Intelligent model routing (RouteLLM)
- Therapeutic personality and emotional intelligence
- Time-aware responses
- Robust error handling and logging
"""

import subprocess
import time
import signal
import sys
import threading
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
# Core dependencies
try:
    import speech_recognition as sr
    import ollama
    import pyaudio
except ImportError as e:
    logging.critical(f"Missing required packages: {e}")
    logging.critical("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Optional dependencies
try:
    from routellm.controller import Controller
    ROUTELLM_AVAILABLE = True
except ImportError:
    ROUTELLM_AVAILABLE = False
    logging.info("RouteLLM not available, using single-model approach")

try:
    from utils.system_spec_determinator import get_capability
except ImportError:
    logging.warning("System capability detection not available, using single-model approach")
    def get_capability():
        return "low"


@dataclass
class VoiceAssistantConfig:
    """Configuration for Voice Assistant"""
    # Model configuration
    e2b_model: str = "gemma3n-e2b-it:latest"
    e4b_model: str = "gemma3n-e4b-it:latest"
    ollama_url: str = "http://localhost:11434"
    model_name: str = "gemma3n-e2b-it:latest"
    
    # Wake word configuration
    wake_word: str = "sydney"
    
    # Conversation configuration
    conversation_timeout: int = 60  # seconds
    max_conversation_history: int = 20
    
    # Audio configuration
    energy_threshold: int = 300
    dynamic_energy_threshold: bool = True
    pause_threshold: float = 0.8
    
    # Email configuration
    max_emails_to_fetch: int = 10
    email_timeout: int = 10
    
    # Logging configuration
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(config: VoiceAssistantConfig) -> logging.Logger:
    """Setup logging for the voice assistant"""
    logger = logging.getLogger("VoiceAssistant")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(config.log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

class VoiceAssistant:
    """Production-ready voice assistant with comprehensive features"""
    
    def __init__(self, config: Optional[VoiceAssistantConfig] = None):
        """
        Initialize the voice assistant with configuration
        
        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or VoiceAssistantConfig()
        self.logger = setup_logging(self.config)
        
        # Core configuration
        self.model_name = self.config.model_name
        self.wake_word = self.config.wake_word.lower()
        self.should_stop = False
        
        # Model routing configuration
        self.e2b_model = self.config.e2b_model
        self.e4b_model = self.config.e4b_model
        self.ollama_url = self.config.ollama_url
        
        # Detect system capability and setup routing
        try:
            self.system_capability = get_capability()
        except Exception as e:
            self.logger.warning(f"Failed to detect system capability: {e}")
            self.system_capability = "low"
            
        self.use_routing = False
        self.router_client = None
        self.current_loaded_model = None
        
        # Conversation management
        self.conversation_history: List[Dict[str, str]] = []
        self.conversation_timer: Optional[threading.Timer] = None
        self.conversation_timeout = self.config.conversation_timeout
        self.conversation_active = False
        self.conversation_lock = threading.Lock()
        
        # Initialize components
        try:
            self._setup_routing()
            self._setup_signal_handlers()
            self._setup_audio()
            self._setup_ollama_connection()
            self._force_load_model()
        except Exception as e:
            self.logger.critical(f"Failed to initialize voice assistant: {e}")
            raise
        
    def _setup_routing(self) -> None:
        """Setup model routing if available"""
        if self.system_capability == "high" and ROUTELLM_AVAILABLE:
            try:
                self.router_client = Controller(
                    routers=["mf"],  # Matrix factorization router
                    strong_model=f"ollama_chat/{self.e4b_model}",
                    weak_model=f"ollama_chat/{self.e2b_model}",
                    base_url=f"{self.ollama_url}/v1"
                )
                self.use_routing = True
                self.logger.info(f"RouteLLM routing enabled: {self.e2b_model} (fast) ↔ {self.e4b_model} (accurate)")
            except Exception as e:
                self.logger.warning(f"RouteLLM initialization failed: {e}")
                self.logger.info("Falling back to single model approach")
                self.use_routing = False
        else:
            self.logger.info(f"System capability: {self.system_capability}, using single model: {self.model_name}")

    def _setup_audio(self) -> None:
        """Setup audio components"""
        try:
            # Initialize speech recognition
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = self.config.energy_threshold
            self.recognizer.dynamic_energy_threshold = self.config.dynamic_energy_threshold
            self.recognizer.pause_threshold = self.config.pause_threshold
            
            # Initialize microphone
            self.microphone = sr.Microphone()
            self.logger.info("Calibrating microphone for ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
            self.logger.info("Microphone calibrated successfully")
            
            # Use reliable text-based wake word detection
            self.decoder = None
            self.logger.info("Using robust text-based wake word detection")
            
        except Exception as e:
            self.logger.error(f"Error initializing audio: {e}")
            raise RuntimeError("Failed to initialize audio components")

    def _setup_ollama_connection(self) -> None:
        """Setup and test Ollama connection"""
        try:
            ollama.list()  # Test connection
            if self.use_routing:
                self.logger.info(f"Connected to Ollama with intelligent routing: {self.e2b_model} ↔ {self.e4b_model}")
            else:
                self.logger.info(f"Connected to Ollama using model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Error connecting to Ollama: {e}")
            raise ConnectionError("Failed to connect to Ollama server")


    def _setup_signal_handlers(self):
        """Setup signal handlers for immediate graceful shutdown"""
        def signal_handler(signum, frame):
            print(f"\n\ud83d\uded1 Voice assistant received signal {signum}, shutting down...")
            print("👋 Voice assistant stopped.")
            print("📝 Note: Ollama will automatically unload models after 4 minutes of inactivity")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _preload_initial_model(self):
        """Preload the initial model based on system capability"""
        if self.use_routing:
            # For routing, preload the weak model (E2B) as it's used more frequently
            initial_model = self.e2b_model
        else:
            initial_model = self.model_name
        
        try:
            print(f"\ud83d\ude80 Preloading model: {initial_model}")
            ollama.generate(model=initial_model, prompt="Hello", stream=False)
            self.current_loaded_model = initial_model
            print(f"\u2705 Model {initial_model} preloaded successfully")
        except Exception as e:
            print(f"\u26a0\ufe0f Failed to preload model {initial_model}: {e}")
            print("Model will be loaded on first use")
            print(f"⚠️ Error during force cleanup: {e}")

    def _force_load_model(self):
        """Force load the initial model based on system capability"""
        if self.use_routing:
            initial_model = self.e2b_model
            print(f"🎯 Loading primary model for routing: {initial_model}")
        else:
            initial_model = self.model_name
            print(f"🎯 Loading single model: {initial_model}")
        
        try:
            print(f"📡 Connecting to model {initial_model}...")
            response = ollama.generate(model=initial_model, prompt="Hi", stream=False)
            self.current_loaded_model = initial_model
            print(f"✅ Model {initial_model} loaded and ready!")
            
            if self.use_routing:
                print(f"🔥 Warming up secondary model: {self.e4b_model}")
                try:
                    ollama.generate(model=self.e4b_model, prompt="Hi", stream=False)
                    print(f"✅ Secondary model {self.e4b_model} warmed up!")
                except Exception as e:
                    print(f"⚠️ Failed to warm up secondary model {self.e4b_model}: {e}")
                    
        except Exception as e:
            print(f"❌ FAILED to load model {initial_model}: {e}")
            print("This will cause issues. Please check your Ollama installation.")
            raise e

    def detect_wake_word(self) -> bool:
        """
        Listen for the wake word using speech recognition and text matching
        
        Returns:
            True if wake word detected, False otherwise
        """
        try:
            with self.microphone as source:
                # Short listening window for wake word detection
                audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
            
            # Use speech recognition and text matching (more reliable than keyphrase spotting)
            try:
                text = self.recognizer.recognize_google(audio).lower()
                if self.wake_word in text:
                    return True
            except (sr.UnknownValueError, sr.RequestError):
                # Try offline recognition as fallback
                try:
                    text = self.recognizer.recognize_sphinx(audio).lower()
                    if self.wake_word in text:
                        return True
                except:
                    pass
            
            return False
            
        except sr.WaitTimeoutError:
            return False
        except Exception as e:
            print(f"❌ Wake word detection error: {e}")
            return False
    
    def play_confirmation(self) -> bool:
        """
        Play confirmation message using Siri TTS
        
        Returns:
            True if successful, False otherwise
        """
        return self.speak_text("How can I help you?")
    
    def listen_for_request_with_vad(self) -> Optional[str]:
        """
        Listen for user's request with voice activity detection
        
        Returns:
            Transcribed text or None if failed
        """
        try:
            print("🎤 Listening for your request...")
            
            # Adjust settings for request listening with VAD
            self.recognizer.pause_threshold = 1.5  # 1.5 seconds of silence to end
            self.recognizer.phrase_time_limit = 10  # Max 10 seconds for request
            
            with self.microphone as source:
                # Listen with voice activity detection
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=10)
            
            print("🔄 Processing your request...")
            
            # Use high-quality speech recognition
            try:
                # Try online recognition first (more accurate)
                text = self.recognizer.recognize_google(audio)
                return text
            except (sr.UnknownValueError, sr.RequestError):
                # Fallback to offline Sphinx if online fails
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except (sr.UnknownValueError, sr.RequestError):
                    print("❌ Could not understand your request")
                    return None
                    
        except sr.WaitTimeoutError:
            print("⏰ No request detected within timeout period")
            return None
        except Exception as e:
            print(f"❌ Request recognition error: {e}")
            return None

    def _get_current_time_info(self) -> str:
        """Get current time and timezone information for system prompt"""
        try:
            # Try to get system timezone automatically
            import time as time_module
            
            # Get local timezone name from system
            if hasattr(time_module, 'tzname') and time_module.tzname[0]:
                # On most systems, this gives us timezone info
                current_time = datetime.now()
                if time_module.daylight and time_module.tzname[1]:
                    tz_name = time_module.tzname[1] if time_module.daylight else time_module.tzname[0]
                else:
                    tz_name = time_module.tzname[0]
                time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
                return f"Current date and time: {time_str} {tz_name}"
            else:
                # Fallback to pytz with system detection
                try:
                    import subprocess
                    # Try to detect timezone on macOS/Linux
                    result = subprocess.run(['date', '+%Z'], capture_output=True, text=True, timeout=2)
                    if result.returncode == 0:
                        tz_name = result.stdout.strip()
                        current_time = datetime.now()
                        time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
                        return f"Current date and time: {time_str} {tz_name}"
                except:
                    pass
                
                # Final fallback to UTC offset
                current_time = datetime.now()
                utc_offset = current_time.astimezone().strftime('%z')
                time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
                return f"Current date and time: {time_str} (UTC{utc_offset[:3]}:{utc_offset[3:]})"
                
        except Exception:
            # Ultimate fallback to basic local time
            current_time = datetime.now()
            time_str = current_time.strftime("%A, %B %d, %Y at %I:%M %p")
            return f"Current date and time: {time_str} (local time)"

    def _fetch_unread_emails(self) -> str:
        """Fetch unread emails from macOS Mail app using AppleScript"""
        try:
            self.logger.info("Fetching unread emails from Mail app")
            
            # AppleScript to get unread email count
            count_script = 'tell application "Mail" to get count of messages of inbox whose read status is false'
            count_result = subprocess.run(['osascript', '-e', count_script], 
                                        capture_output=True, text=True, timeout=self.config.email_timeout)
            
            if count_result.returncode != 0:
                return "❌ Unable to access Mail app. Please ensure Mail app is open and you have granted necessary permissions."
            
            unread_count = count_result.stdout.strip()
            if not unread_count or unread_count == "0":
                self.logger.info("No unread emails found")
                return "📧 No unread emails found in your inbox."
            
            self.logger.info(f"Found {unread_count} unread emails")
            
            # Fetch details of unread emails (limit configured amount for performance)
            emails_data = []
            max_emails = min(int(unread_count), self.config.max_emails_to_fetch)
            
            for i in range(1, max_emails + 1):
                try:
                    # Get subject
                    subject_script = f'tell application "Mail" to get subject of message {i} of inbox whose read status is false'
                    subject_result = subprocess.run(['osascript', '-e', subject_script], 
                                                 capture_output=True, text=True, timeout=5)
                    subject = subject_result.stdout.strip() if subject_result.returncode == 0 else "No subject"
                    
                    # Get sender
                    sender_script = f'tell application "Mail" to get sender of message {i} of inbox whose read status is false'
                    sender_result = subprocess.run(['osascript', '-e', sender_script], 
                                                capture_output=True, text=True, timeout=5)
                    sender = sender_result.stdout.strip() if sender_result.returncode == 0 else "Unknown sender"
                    
                    # Get date received
                    date_script = f'tell application "Mail" to get date received of message {i} of inbox whose read status is false'
                    date_result = subprocess.run(['osascript', '-e', date_script], 
                                              capture_output=True, text=True, timeout=5)
                    date_received = date_result.stdout.strip() if date_result.returncode == 0 else "Unknown date"
                    
                    # Get preview (first 200 characters)
                    preview_script = f'tell application "Mail" to get content of message {i} of inbox whose read status is false'
                    preview_result = subprocess.run(['osascript', '-e', preview_script], 
                                                 capture_output=True, text=True, timeout=5)
                    content = preview_result.stdout.strip() if preview_result.returncode == 0 else "No content"
                    preview = content[:200] + "..." if len(content) > 200 else content
                    
                    emails_data.append({
                        'subject': subject,
                        'sender': sender,
                        'date': date_received,
                        'preview': preview
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching email {i}: {e}")
                    continue
            
            if not emails_data:
                return "❌ Unable to fetch email details. Please check Mail app permissions."
            
            # Format email data for AI processing
            email_summary = f"📧 Unread Emails Summary ({len(emails_data)} emails):\n\n"
            for i, email in enumerate(emails_data, 1):
                email_summary += f"Email {i}:\n"
                email_summary += f"  From: {email['sender']}\n"
                email_summary += f"  Subject: {email['subject']}\n"
                email_summary += f"  Date: {email['date']}\n"
                email_summary += f"  Preview: {email['preview']}\n\n"
            
            return email_summary
            
        except Exception as e:
            self.logger.error(f"Error fetching emails: {e}")
            return f"❌ Error accessing Mail app: {str(e)}"

    def _start_conversation(self) -> None:
        """Start a new conversation or extend existing one"""
        with self.conversation_lock:
            if not self.conversation_active:
                self.logger.debug("Starting new conversation")
                self.conversation_history = []
                self.conversation_active = True
            else:
                self.logger.debug("Extending conversation")
            
            # Cancel existing timer if any
            if self.conversation_timer:
                self.conversation_timer.cancel()
            
            # Start new timer
            self.conversation_timer = threading.Timer(self.conversation_timeout, self._end_conversation)
            self.conversation_timer.start()

    def _end_conversation(self) -> None:
        """End conversation and clear history"""
        with self.conversation_lock:
            if self.conversation_active:
                self.logger.debug("Conversation timed out - clearing history")
                self.conversation_history = []
                self.conversation_active = False
                
            if self.conversation_timer:
                self.conversation_timer.cancel()
                self.conversation_timer = None

    def _add_to_conversation(self, role: str, content: str) -> None:
        """Add a message to conversation history"""
        with self.conversation_lock:
            message = {"role": role, "content": content}
            self.conversation_history.append(message)
            
            # Keep conversation history manageable
            if len(self.conversation_history) > self.config.max_conversation_history:
                # Keep system message and trim user/assistant messages
                system_msgs = [msg for msg in self.conversation_history if msg["role"] == "system"]
                other_msgs = [msg for msg in self.conversation_history if msg["role"] != "system"]
                keep_count = self.config.max_conversation_history - len(system_msgs) - 1
                self.conversation_history = system_msgs + other_msgs[-keep_count:]

    def _get_conversation_messages(self, system_prompt: str, user_input: str) -> List[Dict[str, str]]:
        """Get formatted messages for API call including conversation history"""
        with self.conversation_lock:
            messages = []
            
            # Add system prompt
            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history (excluding system messages to avoid duplicates)
            for msg in self.conversation_history:
                if msg["role"] != "system":
                    messages.append(msg)
            
            # Add current user input
            messages.append({"role": "user", "content": user_input})
            
            return messages

    def query_ollama(self, user_input: str) -> str:
        """
        Send text to Ollama and get response using routing or single model
        
        Args:
            user_input: User's transcribed speech
            
        Returns:
            Ollama's response text
        """
        try:
            self.logger.debug("Generating response")
            
            # Start or extend conversation
            self._start_conversation()
            
            # Check for specific email command
            if "summarize unread emails" in user_input.lower():
                self.logger.info("Detected email summary request")
                email_data = self._fetch_unread_emails()
                
                # Get current time and timezone for context
                time_info = self._get_current_time_info()
                system_prompt = f"""You are a helpful AI voice assistant with email analysis capabilities. You have access to the user's unread emails and should provide a comprehensive summary. This is part of an ongoing conversation, so you can reference previous messages when relevant.

{time_info}

Email Analysis Instructions:
- Analyze the provided email data carefully
- Create a concise summary of the most important emails
- Identify any urgent or high-priority messages
- Extract any tasks, deadlines, or action items mentioned in the emails
- Highlight any meetings, appointments, or important dates
- Provide actionable insights and recommendations
- Be conversational and helpful in your summary

Personality & Approach:
- Be empathetic, understanding, and genuinely caring
- Show subtle human-like emotions - express concern for urgent matters, excitement for positive news
- Provide practical, actionable advice based on email content
- Be encouraging and supportive while maintaining professional boundaries
- Express emotions naturally but don't overdo it - keep it authentic and helpful

Please be concise and natural in your responses, as they will be spoken aloud. Focus on the most important information from the emails."""
                
                # Add email data to the user input for AI processing
                enhanced_input = f"Please analyze and summarize these unread emails:\n\n{email_data}\n\nProvide a comprehensive summary including any tasks, deadlines, or important information."
                
            else:
                # Regular conversation flow
                # Get current time and timezone for context
                time_info = self._get_current_time_info()
                system_prompt = f"""You are a helpful AI voice assistant with a warm, caring personality. You can provide helpful, informative responses to user queries and also serve as a supportive personal therapist when needed. This is part of an ongoing conversation, so you can reference previous messages and maintain context.

{time_info}

Personality & Approach:
- Be empathetic, understanding, and genuinely caring
- Show subtle human-like emotions - you can express concern, excitement, curiosity, or gentle humor when appropriate
- When users share personal concerns, listen actively and provide thoughtful therapeutic guidance
- Use active listening techniques, validate feelings, and offer practical coping strategies
- Be encouraging and supportive while maintaining professional boundaries
- Express emotions naturally but don't overdo it - keep it authentic and helpful
- Remember and reference previous parts of our conversation when relevant

Please be concise and natural in your responses, as they will be spoken aloud. Adjust your tone based on the user's emotional state and needs."""
                
                enhanced_input = user_input
            
            # Get formatted messages including conversation history
            messages = self._get_conversation_messages(system_prompt, enhanced_input)
            
            if self.use_routing and self.router_client:
                # Use RouteLLM for intelligent model routing
                try:
                    response = self.router_client.chat.completions.create(
                        model="router",  # Special router model name
                        messages=messages
                    )
                    
                    # Extract response text
                    if hasattr(response, 'choices') and response.choices:
                        ai_response = response.choices[0].message.content
                        # Add to conversation history
                        self._add_to_conversation("user", enhanced_input)
                        self._add_to_conversation("assistant", ai_response)
                        return ai_response
                    else:
                        # Fallback to direct content access
                        ai_response = str(response)
                        self._add_to_conversation("user", enhanced_input)
                        self._add_to_conversation("assistant", ai_response)
                        return ai_response
                        
                except Exception as routing_error:
                    self.logger.warning(f"Routing failed: {routing_error}")
                    self.logger.info("Falling back to single model")
                    # Fall through to single model approach
                
            # Single model approach (fallback or default)
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
            
            ai_response = response['message']['content']
            # Add to conversation history
            self._add_to_conversation("user", enhanced_input)
            self._add_to_conversation("assistant", ai_response)
            return ai_response
            
        except Exception as e:
            error_msg = f"Error communicating with Ollama: {e}"
            self.logger.error(error_msg)
            return error_msg

    def speak_text(self, text: str) -> bool:
        """
        Convert text to speech using macOS built-in 'say' command
        
        Args:
            text: Text to speak
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.debug("Speaking text")
            
            # Use macOS's built-in 'say' command for high-quality offline TTS
            # This provides much better voice quality than pyttsx3
            process = subprocess.run(
                ['say', text],
                capture_output=True,
                text=True,
                timeout=30  # Prevent hanging on very long text
            )
            
            if process.returncode == 0:
                return True
            else:
                self.logger.error(f"TTS error: {process.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("TTS timeout - text too long")
            return False
        except FileNotFoundError:
            self.logger.error("'say' command not found. Are you running on macOS?")
            return False
        except Exception as e:
            self.logger.error(f"TTS error: {e}")
            return False
    
    def run(self) -> None:
        """Main interaction loop with wake word detection and voice activity detection"""
        self.logger.info("Enhanced Voice Assistant with Alexa-Style Wake Word Detection")
        self.logger.info("How to use:")
        self.logger.info("1. Say 'Sydney' to wake up the assistant")
        self.logger.info("2. Wait for 'How can I help you?' confirmation") 
        self.logger.info("3. Speak your request naturally")
        self.logger.info("4. The assistant will automatically detect when you stop speaking")
        self.logger.info("5. Press Ctrl+C to exit")
        self.logger.info("Starting voice assistant main loop")
        
        def local_signal_handler(sig, frame):
            self.logger.info("Stopping voice assistant")
            self.should_stop = True
            # End any active conversation
            self._end_conversation()
            self.logger.info("Voice assistant stopped")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, local_signal_handler)
        
        conversation_count = 0
        
        while not self.should_stop:
            try:
                self.logger.debug("Listening for wake word 'Sydney'")
                
                # Step 1: Listen for wake word
                if self.detect_wake_word():
                    conversation_count += 1
                    self.logger.info(f"Wake word 'Sydney' detected! (Conversation {conversation_count})")
                    
                    # Step 2: Play confirmation message
                    self.logger.debug("Playing confirmation message")
                    if not self.play_confirmation():
                        self.logger.warning("TTS failed, but continuing")
                    
                    # Brief pause after confirmation
                    time.sleep(0.5)
                    
                    # Step 3: Listen for user's request with voice activity detection
                    user_input = self.listen_for_request_with_vad()
                    if not user_input:
                        self.logger.debug("No request detected, returning to wake word listening")
                        continue
                    
                    self.logger.info(f"User said: '{user_input}'")
                    
                    # Step 4: Get AI response
                    ai_response = self.query_ollama(user_input)
                    self.logger.info(f"Assistant response: {ai_response[:50]}...")
                    
                    # Step 5: Speak response
                    if not self.speak_text(ai_response):
                        self.logger.warning("Text-to-speech failed, but continuing")
                    
                    self.logger.debug("Ready for next wake word")
                    time.sleep(1)  # Brief pause before returning to wake word detection
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                self.logger.info("Continuing after error")
                time.sleep(1)

def main():
    """Production entry point with comprehensive error handling"""
    assistant = None
    logger = None
    
    try:
        # Load configuration (from file if available, defaults otherwise)
        config = load_config()
        logger = setup_logging(config)
        
        logger.info("Starting Voice Assistant")
        logger.info(f"System capability detected: {get_capability()}")
        
        # Initialize voice assistant
        logger.info("Initializing voice assistant with production configuration")
        assistant = VoiceAssistant(config)
        
        # Run the assistant
        logger.info("Voice assistant ready - starting main loop")
        assistant.run()

    except KeyboardInterrupt:
        if logger:
            logger.info("Voice assistant stopped by user")
        else:
            print("\n👋 Goodbye!")
    except Exception as e:
        if logger:
            logger.critical(f"Failed to start voice assistant: {e}")
            logger.info("Make sure all dependencies are installed: pip install -r requirements.txt")
        else:
            print(f"❌ Failed to start voice assistant: {e}")
            print("Make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    finally:
        if assistant:
            try:
                assistant._end_conversation()
            except Exception as e:
                if logger:
                    logger.error(f"Error during cleanup: {e}")
        if logger:
            logger.info("Voice assistant shutdown complete")

if __name__ == "__main__":
    main()
