# 🎙️ Enhanced Voice Assistant

A production-ready voice assistant with **Alexa-style wake word detection**, multi-turn conversations, email integration, and intelligent model routing using local Ollama AI models.

## 🚀 Key Features

### ✅ **Alexa-Style Wake Word Detection**
- **🎯 Continuous Listening**: Always listening for "Sydney" wake word
- **🗣️ Smart Confirmation**: Plays "How can I help you?" when activated
- **📱 Voice Activity Detection**: Automatically detects when you stop speaking
- **🏠 Reliable Detection**: Speech recognition + text matching for accuracy
- **🔊 High-Quality TTS**: macOS Siri voices for natural responses
- **🛡️ Background Noise Tolerance**: Works with ambient noise

### ✅ **Multi-Turn Conversations**
- **💬 Conversation Memory**: Maintains context across interactions
- **⏰ Auto-Cleanup**: Conversations expire after 1 minute of inactivity
- **🧠 Context Awareness**: AI remembers previous messages
- **🔄 Conversation Renewal**: Each interaction extends the conversation

### ✅ **Email Integration**
- **📧 macOS Mail App**: Fetch and summarize unread emails
- **🤖 AI Analysis**: Intelligent email summarization with tasks/deadlines
- **🎯 Voice Command**: Say "summarize unread emails" for analysis
- **📋 Action Items**: Extracts tasks, deadlines, and important information

### ✅ **Intelligent Model Routing**
- **🧠 RouteLLM Integration**: Dynamic model selection based on query complexity
- **⚡ Performance Optimization**: Fast models for simple queries, accurate for complex
- **🔄 Automatic Fallback**: Falls back to single model if routing fails
- **📊 System Detection**: Automatically detects system capabilities

### ✅ **Therapeutic Personality**
- **❤️ Emotional Intelligence**: Subtle human-like emotions and empathy
- **🧘‍♀️ Therapeutic Support**: Acts as a supportive personal therapist
- **⏰ Time Awareness**: Includes current time/date in all responses
- **🎭 Natural Personality**: Warm, caring, and genuinely helpful

## 🎯 How It Works

### **Alexa-Style Interaction Flow:**
1. **Wake Word Listening**: Continuously listening for "Sydney"
2. **Wake Word Detection**: Speech recognition detects "Sydney"
3. **Confirmation**: Plays "How can I help you?" via TTS
4. **Request Listening**: Listens for your request with voice activity detection
5. **Automatic End Detection**: Detects when you stop speaking
6. **AI Processing**: Sends request to local Ollama model with conversation context
7. **Voice Response**: Speaks the AI's response using macOS Siri TTS
8. **Return to Listening**: Automatically returns to wake word detection

### **Example Usage:**
```
You: "Sydney"
Assistant: "How can I help you?"
You: "What's the weather like today?"
Assistant: [Processes with AI and speaks response]
You: "Can you summarize my unread emails?"
Assistant: [Fetches emails and provides AI analysis]
[Returns to listening for "Sydney"]
```

## 📋 Requirements

### **System Requirements**
- **macOS 10.15+ (Catalina or later)** - Required for TTS and Mail integration
- **Python 3.8+**
- **Ollama** installed and running locally

### **Python Dependencies**

#### **Production Installation:**
```bash
pip install -r requirements.txt
```

#### **Development Installation:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

#### **Full Enterprise Stack:**
```bash
pip install -r requirements*.txt
```

## 📦 Installation

### **1. Install Ollama**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### **2. Pull AI Models**
```bash
# Pull the recommended models
ollama pull gemma3n-e2b-it:latest
ollama pull gemma3n-e4b-it:latest
```

### **3. Install Python Dependencies**
```bash
# Navigate to project directory
cd stt_tts_ollama/

# Install production dependencies
pip install -r requirements.txt
```

### **4. Run the Voice Assistant**
```bash
python3 voice_ollama.py
```

## 🎤 Using the Voice Assistant

### **Basic Usage:**
1. **Start**: Run `python3 voice_ollama.py`
2. **Wait**: Let it initialize and calibrate the microphone (wait 2-3 minutes for it to fully activate)
3. **Say Wake Word**: Say "Sydney" clearly
4. **Wait for Confirmation**: Listen for "How can I help you?"
5. **Speak Request**: Speak your request naturally
6. **Listen**: The AI will respond and speak back to you
7. **Repeat**: The assistant returns to listening for "Sydney"
8. **Exit**: Press `Ctrl+C` when done

### **Email Integration:**
- Say "Sydney" to wake up
- Say "summarize unread emails" when prompted
- The assistant will fetch your unread emails and provide an AI analysis

### **Multi-Turn Conversations:**
- Have natural conversations with context memory
- Ask follow-up questions - the AI remembers previous context
- Conversations automatically clear after 1 minute of inactivity

## 🔧 Configuration

### **Environment Variables:**
```bash
# Custom configuration file
export VOICE_ASSISTANT_CONFIG=/path/to/config.json

# Custom log level
export LOG_LEVEL=INFO
```

### **Configuration File (`config.json`):**
```json
{
  "wake_word": "sydney",
  "conversation_timeout": 60,
  "max_conversation_history": 20,
  "log_level": "INFO",
  "model_name": "gemma3n-e2b-it:latest",
  "ollama_url": "http://localhost:11434"
}
```

## 🚀 Production Deployment

### **Systemd Service (Linux):**
```ini
[Unit]
Description=Voice Assistant
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/stt_tts_ollama
ExecStart=/usr/bin/python3 voice_ollama.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### **Launchd Service (macOS):**
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.voiceassistant</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/stt_tts_ollama/voice_ollama.py</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
</dict>
</plist>
```

## 🔧 Troubleshooting

### **Common Issues:**

| Problem | Solution |
|---------|----------|
| "No module named 'speechrecognition'" | Run: `pip install -r requirements.txt` |
| "Error connecting to Ollama" | Make sure Ollama is running: `ollama serve` |
| "No models found" | Install models: `ollama pull gemma3n-e2b-it` |
| "Wake word not detected" | Speak louder, ensure quiet environment |
| Microphone not working | Check System Preferences > Security & Privacy > Microphone |
| Email integration fails | Ensure Mail app is open and permissions granted |

### **Debug Mode:**
```bash
LOG_LEVEL=DEBUG python3 voice_ollama.py
```

### **Component Testing:**
```bash
python3 test_components.py
```

## 🏗️ Architecture

### **Core Components:**
- **`voice_ollama.py`**: Main application with production-ready features
- **`config.py`**: Configuration management system
- **`requirements.txt`**: Production dependencies
- **`requirements-dev.txt`**: Development tools
- **`requirements-optional.txt`**: Enterprise features

### **Key Features:**
- **Multi-turn conversations** with automatic cleanup
- **Email integration** via macOS Mail app
- **Intelligent model routing** with RouteLLM
- **Therapeutic personality** with emotional intelligence
- **Time-aware responses** with current context
- **Robust error handling** and recovery
- **Production logging** with configurable levels
- **Thread-safe conversation** management

## 📊 Performance

### **System Requirements:**
- **Low-end systems**: Uses efficient E2B model
- **High-end systems**: Automatic routing to E4B for complex queries
- **Memory efficient**: Conversation history limits and auto-cleanup
- **CPU optimized**: Intelligent model selection based on query complexity

### **Resource Usage:**
- **Memory**: ~500MB base + model size
- **CPU**: Minimal during listening, spikes during AI processing
- **Network**: Local only (Ollama runs locally)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ollama** for local AI model serving
- **SpeechRecognition** for robust speech processing
- **RouteLLM** for intelligent model routing
- **macOS** for high-quality TTS and system integration

---

**🎉 Ready to experience the future of voice assistants!**
