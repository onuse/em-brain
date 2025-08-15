#!/usr/bin/env python3
"""
Audio Module for PiCar-X

Provides audio input (microphone) and output (speaker) capabilities.
Converts audio to brain-friendly features and generates sounds from brain commands.

Philosophy: Raw audio experience, not speech recognition.
The brain learns what sounds mean through experience.
"""

import numpy as np
import time
import threading
import queue
from typing import List, Optional, Tuple
from dataclasses import dataclass

try:
    import os
    import pyaudio
    # Suppress ALSA errors
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ PyAudio not available - audio disabled")

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


@dataclass
class AudioFeatures:
    """Audio features for brain input."""
    volume: float           # Overall volume (0-1)
    frequency_bands: List[float]  # 4 frequency bands (low to high)
    pitch_estimate: float   # Dominant frequency normalized
    onset_detected: float   # Sound onset/change detection
    left_right_balance: float  # Stereo balance if available


class AudioModule:
    """
    Bare-metal audio I/O for robot.
    
    Provides:
    - Microphone input → frequency analysis → brain features
    - Brain commands → sound generation → speaker output
    """
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Initialize audio system.
        
        Args:
            sample_rate: Audio sample rate (16kHz default for efficiency)
            channels: 1 for mono, 2 for stereo
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = 1024  # Samples per chunk
        
        # Audio I/O
        self.audio = None
        self.input_stream = None
        self.output_stream = None
        self.enabled = False
        
        # Audio buffers
        self.input_buffer = queue.Queue(maxsize=10)
        self.last_features = AudioFeatures(
            volume=0.0,
            frequency_bands=[0.0] * 4,
            pitch_estimate=0.5,
            onset_detected=0.0,
            left_right_balance=0.5
        )
        
        # Sound generation state
        self.current_frequency = 440.0  # A4
        self.current_volume = 0.0
        self.sound_thread = None
        self.sound_active = False
        
        if AUDIO_AVAILABLE:
            self._init_audio()
    
    def _init_audio(self):
        """Initialize PyAudio for input and output."""
        try:
            # Suppress error output during initialization
            import sys
            import io
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            
            self.audio = pyaudio.PyAudio()
            
            # Restore stderr
            sys.stderr = old_stderr
            
            # Find default devices
            input_device = None
            output_device = None
            
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0 and input_device is None:
                    input_device = i
                if info['maxOutputChannels'] > 0 and output_device is None:
                    output_device = i
            
            # Initialize input stream (microphone)
            if input_device is not None:
                self.input_stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=input_device,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_input_callback
                )
                print(f"✅ Microphone initialized at {self.sample_rate}Hz")
            else:
                print("⚠️ No microphone found")
            
            # Initialize output stream (speaker)
            if output_device is not None:
                self.output_stream = self.audio.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=self.sample_rate,
                    output=True,
                    output_device_index=output_device,
                    frames_per_buffer=self.chunk_size
                )
                print(f"✅ Speaker initialized at {self.sample_rate}Hz")
            else:
                print("⚠️ No speaker found")
            
            self.enabled = (input_device is not None or output_device is not None)
            
        except Exception as e:
            print(f"❌ Audio initialization failed: {e}")
            self.enabled = False
    
    def _audio_input_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio input stream."""
        if status and status != 2:  # 2 is normal input overflow, ignore it
            print(f"Audio input status: {status}")
        
        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Add to buffer if not full
        if not self.input_buffer.full():
            self.input_buffer.put(audio_data)
        
        return (in_data, pyaudio.paContinue)
    
    def get_audio_features(self) -> List[float]:
        """
        Get audio features for brain input channels.
        
        Returns 7 features:
        - Overall volume (RMS)
        - 4 frequency bands (bass, low-mid, high-mid, treble)
        - Pitch estimate (dominant frequency)
        - Onset detection (sudden changes)
        """
        if not self.enabled or self.input_buffer.empty():
            # Return last known features if no new audio
            return self._features_to_list(self.last_features)
        
        try:
            # Get latest audio chunk
            audio_chunk = self.input_buffer.get_nowait()
            
            # Normalize to -1 to 1
            audio_normalized = audio_chunk / 32768.0
            
            # Feature 1: Volume (RMS)
            volume = np.sqrt(np.mean(audio_normalized**2))
            
            # Feature 2-5: Frequency bands via FFT
            fft = np.fft.rfft(audio_normalized)
            fft_mag = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_normalized), 1/self.sample_rate)
            
            # Divide spectrum into 4 bands
            bands = []
            band_edges = [20, 250, 1000, 4000, 8000]  # Hz
            for i in range(4):
                mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
                if np.any(mask):
                    band_energy = np.mean(fft_mag[mask])
                    bands.append(min(1.0, band_energy / 100))  # Normalize
                else:
                    bands.append(0.0)
            
            # Feature 6: Pitch estimate (dominant frequency)
            if len(fft_mag) > 0:
                dominant_idx = np.argmax(fft_mag[1:]) + 1  # Skip DC
                dominant_freq = freqs[dominant_idx]
                # Normalize to 0-1 (20Hz to 8kHz on log scale)
                pitch = np.log10(max(20, min(8000, dominant_freq)) / 20) / np.log10(400)
            else:
                pitch = 0.5
            
            # Feature 7: Onset detection (compare to previous volume)
            onset = max(0, volume - self.last_features.volume) * 5  # Amplify changes
            
            # Update last features
            self.last_features = AudioFeatures(
                volume=volume,
                frequency_bands=bands,
                pitch_estimate=pitch,
                onset_detected=onset,
                left_right_balance=0.5  # Mono for now
            )
            
            return self._features_to_list(self.last_features)
            
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
            return [0.0] * 7
    
    def _features_to_list(self, features: AudioFeatures) -> List[float]:
        """Convert AudioFeatures to list for brain."""
        return [
            features.volume,
            *features.frequency_bands,  # 4 values
            features.pitch_estimate,
            features.onset_detected
        ]
    
    def generate_sound(self, frequency: float, volume: float, duration: float = 0.1):
        """
        Generate a tone at specified frequency and volume.
        
        Args:
            frequency: Tone frequency in Hz (20-8000)
            volume: Volume 0-1
            duration: Duration in seconds
        """
        if not self.enabled or self.output_stream is None:
            return
        
        try:
            # Clamp values
            frequency = max(20, min(8000, frequency))
            volume = max(0, min(1, volume))
            
            # Generate sine wave
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            wave = volume * np.sin(2 * np.pi * frequency * t)
            
            # Play through speaker
            self.output_stream.write(wave.astype(np.float32).tobytes())
            
        except Exception as e:
            print(f"Sound generation error: {e}")
    
    def generate_sound_from_brain(self, brain_output: List[float]):
        """
        Generate sound based on brain output.
        
        Expected brain outputs (2 channels):
        - Channel 0: Frequency control (0-1 → 100-2000 Hz)
        - Channel 1: Volume control (0-1)
        """
        if len(brain_output) < 2:
            return
        
        # Map brain output to sound parameters
        freq_normalized = max(0, min(1, brain_output[0]))
        volume = max(0, min(1, brain_output[1]))
        
        # Exponential frequency mapping for better perception
        frequency = 100 * (20 ** freq_normalized)  # 100Hz to 2000Hz
        
        # Generate short tone (non-blocking)
        if volume > 0.01:  # Threshold to avoid noise
            self.generate_sound(frequency, volume, 0.05)  # 50ms tone
    
    def play_beep(self, frequency: int = 1000, duration: float = 0.2):
        """Play a simple beep for feedback."""
        self.generate_sound(frequency, 0.5, duration)
    
    def cleanup(self):
        """Clean up audio resources."""
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        self.enabled = False


# Example integration with brainstem:
"""
# In brainstem.py:

def sensors_to_brain_format(self, raw: RawSensorData,
                           vision: Optional[VisionModule] = None,
                           audio: Optional[AudioModule] = None) -> List[float]:
    brain_input = [0.5] * 24
    
    # Basic sensors (0-4)
    # ... existing code ...
    
    # Vision features (5-18)
    if vision:
        brain_input[5:19] = vision.get_vision_features()
    
    # Audio features (19-23 + overflow to spare channels)
    if audio:
        audio_features = audio.get_audio_features()
        brain_input[19:24] = audio_features[:5]  # First 5 audio features
    
    return brain_input
"""


if __name__ == "__main__":
    # Test audio module
    print("Testing audio module...")
    
    audio = AudioModule()
    
    if audio.enabled:
        print("\nPlaying test tones...")
        for freq in [440, 880, 1760]:
            print(f"  {freq}Hz")
            audio.play_beep(freq, 0.3)
            time.sleep(0.5)
        
        print("\nListening for 5 seconds...")
        for i in range(10):
            features = audio.get_audio_features()
            print(f"  Volume: {features[0]:.3f}, Bands: {features[1:5]}")
            time.sleep(0.5)
        
        audio.cleanup()
    else:
        print("No audio devices available")