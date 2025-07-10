"""
Settings Manager - Centralized configuration for the robot brain system.

Loads settings from settings.json in the root directory and provides
easy access to configuration values throughout the system.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class SettingsManager:
    """
    Manages configuration settings for the robot brain system.
    """
    
    def __init__(self, settings_file: str = "settings.json"):
        """
        Initialize settings manager.
        
        Args:
            settings_file: Path to the settings JSON file
        """
        self.settings_file = Path(settings_file)
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from JSON file."""
        if not self.settings_file.exists():
            # Return default settings if file doesn't exist
            return self._get_default_settings()
        
        try:
            with open(self.settings_file, 'r') as f:
                settings = json.load(f)
            return settings
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not load settings from {self.settings_file}: {e}")
            print("   Using default settings instead.")
            return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings if no file exists."""
        return {
            "memory": {
                "persistent_memory_path": "./robot_memory",
                "enable_persistence": True
            },
            "logging": {
                "log_directory": "./logs"
            },
            "system": {
                "use_gpu": True,
                "adaptive_gpu_switching": True,
                "base_time_budget": 0.1
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a setting value using dot notation.
        
        Args:
            key_path: Dot-separated path to the setting (e.g., "memory.persistent_memory_path")
            default: Default value if setting not found
            
        Returns:
            The setting value or default
        """
        keys = key_path.split('.')
        value = self.settings
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_memory_path(self) -> str:
        """Get the persistent memory path."""
        return self.get("memory.persistent_memory_path", "./robot_memory")
    
    def get_log_directory(self) -> str:
        """Get the log directory path."""
        return self.get("logging.log_directory", "./logs")
    
    def is_persistence_enabled(self) -> bool:
        """Check if persistence is enabled."""
        return self.get("memory.enable_persistence", True)
    
    def should_use_gpu(self) -> bool:
        """Check if GPU should be used."""
        return self.get("system.use_gpu", True)
    
    def should_use_adaptive_gpu_switching(self) -> bool:
        """Check if adaptive GPU switching should be used."""
        return self.get("system.adaptive_gpu_switching", True)
    
    def get_base_time_budget(self) -> float:
        """Get the base time budget for predictions."""
        return self.get("system.base_time_budget", 0.1)


# Global settings instance
_settings = None

def get_settings() -> SettingsManager:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = SettingsManager()
    return _settings