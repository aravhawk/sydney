#!/usr/bin/env python3
"""
Configuration module for Voice Assistant

Provides configuration management and validation for production deployment.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class VoiceAssistantConfig:
    """Production configuration for Voice Assistant"""
    
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
    
    @classmethod
    def load_from_file(cls, config_path: Optional[str] = None) -> 'VoiceAssistantConfig':
        """Load configuration from JSON file"""
        if config_path is None:
            config_path = os.getenv('VOICE_ASSISTANT_CONFIG', 'config.json')
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            return cls(**config_data)
        
        # Return default configuration
        return cls()
    
    def save_to_file(self, config_path: Optional[str] = None) -> None:
        """Save configuration to JSON file"""
        if config_path is None:
            config_path = 'config.json'
        
        config_file = Path(config_path)
        with open(config_file, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    def validate(self) -> None:
        """Validate configuration parameters"""
        if self.conversation_timeout <= 0:
            raise ValueError("conversation_timeout must be positive")
        
        if self.max_conversation_history <= 0:
            raise ValueError("max_conversation_history must be positive")
        
        if self.energy_threshold <= 0:
            raise ValueError("energy_threshold must be positive")
        
        if not self.wake_word:
            raise ValueError("wake_word cannot be empty")
        
        if self.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError(f"Invalid log_level: {self.log_level}")


def load_config() -> VoiceAssistantConfig:
    """Load and validate configuration"""
    config = VoiceAssistantConfig.load_from_file()
    config.validate()
    return config