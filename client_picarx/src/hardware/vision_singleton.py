#!/usr/bin/env python3
"""
Vision Module Singleton

Ensures only one camera instance exists to prevent resource conflicts.
"""

import threading
from typing import Optional

class VisionSingleton:
    """Singleton manager for vision module."""
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        """Get or create the single vision instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check pattern
                    try:
                        from hardware.configurable_vision import ConfigurableVision
                        cls._instance = ConfigurableVision()
                        print("📷 Vision singleton created")
                    except Exception as e:
                        print(f"📷 Vision initialization failed: {e}")
                        cls._instance = None
        return cls._instance
    
    @classmethod
    def cleanup(cls):
        """Clean up the vision instance."""
        with cls._lock:
            if cls._instance is not None:
                try:
                    if hasattr(cls._instance, 'cleanup'):
                        cls._instance.cleanup()
                except:
                    pass
                cls._instance = None
                print("📷 Vision singleton cleaned up")


class AudioSingleton:
    """Singleton manager for audio module."""
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls, sample_rate: int = 44100):
        """Get or create the single audio instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check pattern
                    try:
                        from hardware.audio_module import AudioModule
                        # Try different sample rates
                        for rate in [44100, 48000, 16000, 8000]:
                            try:
                                cls._instance = AudioModule(sample_rate=rate)
                                print(f"🎵 Audio singleton created at {rate}Hz")
                                break
                            except:
                                continue
                    except Exception as e:
                        print(f"🎵 Audio initialization failed: {e}")
                        cls._instance = None
        return cls._instance
    
    @classmethod  
    def cleanup(cls):
        """Clean up the audio instance."""
        with cls._lock:
            if cls._instance is not None:
                try:
                    if hasattr(cls._instance, 'cleanup'):
                        cls._instance.cleanup()
                except:
                    pass
                cls._instance = None
                print("🎵 Audio singleton cleaned up")