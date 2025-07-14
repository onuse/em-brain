#!/usr/bin/env python3
"""
Mac Audio Vocal Driver - "Advanced Mock" Implementation

A "mock" implementation that actually uses your Mac speakers for audio output!
This creates a funny development situation where your Mac becomes the robot's
vocal cords during testing. Perfect for experiencing the emotional expressions
while developing the real robot system.

This is technically a mock because it's not the target hardware (PiCar-X speaker),
but it provides real audio feedback for development and demonstration.
"""

import time
import threading
import math
from typing import Dict, Optional
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

from ..interfaces.vocal_interface import VocalInterface, VocalParameters, VocalSafetyConstraints


class MacAudioVocalDriver(VocalInterface):
    """Mac speaker vocal driver - your Mac becomes the robot's voice!"""
    
    def __init__(self, safety_constraints: Optional[VocalSafetyConstraints] = None):
        """Initialize Mac audio vocal driver."""
        super().__init__(safety_constraints)
        
        # Audio system state
        self._initialized = False
        self._current_volume = 0.5
        self._is_currently_vocalizing = False
        self._vocalization_start_time = 0.0
        self._current_vocalization_duration = 0.0
        self._vocalization_thread = None
        self._stop_requested = False
        
        # Audio synthesis parameters
        self.sample_rate = 22050  # Hz - good enough for vocal synthesis
        self.buffer_size = 1024   # samples
        
        # Statistics for testing
        self.vocalization_count = 0
        self.total_vocalization_time = 0.0
        self.last_parameters = None
        self.vocalization_history = []
        
        if not PYGAME_AVAILABLE:
            print("❌ pygame not available - install with: pip install pygame")
            print("🎵 MacAudioVocalDriver will simulate instead of playing audio")
        else:
            print("🎵 MacAudioVocalDriver initialized - your Mac will be the robot's voice!")
    
    def initialize_vocal_system(self) -> bool:
        """Initialize Mac audio system."""
        if self._initialized:
            return True
        
        if not PYGAME_AVAILABLE:
            print("⚠️  pygame not available - running in simulation mode")
            self._initialized = True
            return True
            
        print("🔧 Initializing Mac audio system...")
        
        try:
            # Initialize pygame mixer for audio output
            pygame.mixer.pre_init(
                frequency=self.sample_rate,
                size=-16,  # 16-bit signed
                channels=1,  # Mono
                buffer=self.buffer_size
            )
            pygame.mixer.init()
            
            # Test audio output
            self._test_audio_system()
            
            self._initialized = True
            print("✅ Mac audio system initialized successfully")
            print("🔊 Your Mac speakers are now the robot's vocal cords!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize Mac audio: {e}")
            print("🎵 Falling back to simulation mode")
            self._initialized = True  # Still works, just simulated
            return True
    
    def _test_audio_system(self):
        """Test Mac audio output with a brief tone."""
        if not PYGAME_AVAILABLE:
            return
            
        # Generate a brief test tone (200ms, 440Hz)
        duration = 0.2
        frequency = 440.0  # A4
        samples = int(self.sample_rate * duration)
        
        # Generate sine wave
        wave_array = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
        wave_array = (wave_array * 0.1 * 32767).astype(np.int16)  # Low volume, 16-bit
        
        # Play test tone
        sound = pygame.sndarray.make_sound(wave_array)
        sound.play()
        time.sleep(duration + 0.1)  # Wait for completion
        
        print("🎵 Mac audio test tone completed")
    
    def synthesize_vocalization(self, params: VocalParameters) -> bool:
        """Synthesize and play vocalization through Mac speakers."""
        if not self._initialized:
            print("❌ Vocal system not initialized")
            return False
        
        # Validate parameters
        if not self.validate_parameters(params):
            print(f"❌ Invalid vocal parameters: {params}")
            return False
        
        # Apply safety constraints
        safe_params = self.apply_safety_constraints(params)
        
        # Check if already vocalizing
        if self._is_currently_vocalizing:
            print("⚠️ Already vocalizing - stopping current vocalization")
            self.stop_vocalization()
        
        # Start new vocalization
        self._start_vocalization(safe_params)
        return True
    
    def _start_vocalization(self, params: VocalParameters):
        """Start vocalization synthesis and playback."""
        self._is_currently_vocalizing = True
        self._vocalization_start_time = time.time()
        self._current_vocalization_duration = params.duration
        self._stop_requested = False
        self.last_parameters = params
        
        # Store in history
        self.vocalization_history.append({
            'timestamp': self._vocalization_start_time,
            'parameters': params,
            'duration': params.duration
        })
        
        # Print what we're about to synthesize
        self._print_vocalization_info(params)
        
        # Start synthesis thread
        self._vocalization_thread = threading.Thread(
            target=self._synthesize_and_play, 
            args=(params,)
        )
        self._vocalization_thread.start()
        
        # Update statistics
        self.vocalization_count += 1
    
    def _synthesize_and_play(self, params: VocalParameters):
        """Synthesize audio and play through Mac speakers."""
        if not PYGAME_AVAILABLE:
            # Fall back to simulation with timing
            print("🎵 [Simulated audio - pygame not available]")
            time.sleep(params.duration)
            self._end_vocalization()
            return
        
        try:
            # Generate audio samples
            audio_samples = self._generate_audio_samples(params)
            
            # Convert to pygame sound and play
            sound = pygame.sndarray.make_sound(audio_samples)
            sound.set_volume(self._current_volume * params.amplitude)
            
            # Play the sound
            sound.play()
            
            # Wait for completion (or stop request)
            start_time = time.time()
            while (time.time() - start_time) < params.duration and not self._stop_requested:
                time.sleep(0.01)  # Check stop request every 10ms
            
            # Stop sound if still playing
            sound.stop()
            
        except Exception as e:
            print(f"❌ Audio synthesis error: {e}")
            print("🎵 Continuing with timing simulation...")
            time.sleep(params.duration)
        
        self._end_vocalization()
    
    def _generate_audio_samples(self, params: VocalParameters) -> np.ndarray:
        """Generate audio samples based on vocal parameters."""
        samples = int(self.sample_rate * params.duration)
        time_array = np.linspace(0, params.duration, samples)
        
        # Start with fundamental frequency
        fundamental = params.fundamental_frequency
        
        # Generate base waveform (sine wave)
        waveform = np.sin(2 * np.pi * fundamental * time_array)
        
        # Add harmonics
        for i, harmonic_amplitude in enumerate(params.harmonics[1:], start=2):
            if harmonic_amplitude > 0:
                harmonic_freq = fundamental * i
                harmonic_wave = np.sin(2 * np.pi * harmonic_freq * time_array)
                waveform += harmonic_wave * harmonic_amplitude
        
        # Apply frequency modulation (vibrato)
        freq_mod_rate, freq_mod_depth = params.frequency_modulation
        if freq_mod_rate > 0 and freq_mod_depth > 0:
            freq_mod = np.sin(2 * np.pi * freq_mod_rate * time_array)
            freq_deviation = freq_mod * (freq_mod_depth / 100.0) * fundamental
            
            # Apply frequency modulation
            phase_mod = np.cumsum(freq_deviation) * 2 * np.pi / self.sample_rate
            waveform = np.sin(2 * np.pi * fundamental * time_array + phase_mod)
        
        # Apply amplitude modulation (tremolo)
        amp_mod_rate, amp_mod_depth = params.amplitude_modulation
        if amp_mod_rate > 0 and amp_mod_depth > 0:
            amp_mod = np.sin(2 * np.pi * amp_mod_rate * time_array)
            amplitude_envelope = 1.0 + amp_mod * amp_mod_depth
            waveform *= amplitude_envelope
        
        # Add noise component
        if params.noise_component > 0:
            noise = np.random.normal(0, 1, samples) * params.noise_component
            waveform += noise
        
        # Apply ADSR envelope
        envelope = self._generate_adsr_envelope(samples, params.attack_time, params.decay_time, params.duration)
        waveform *= envelope
        
        # Apply overall amplitude and convert to 16-bit
        waveform *= params.amplitude
        
        # Normalize and convert to 16-bit signed integers
        waveform = np.clip(waveform, -1.0, 1.0)
        audio_samples = (waveform * 32767 * 0.8).astype(np.int16)  # 80% max volume for safety
        
        return audio_samples
    
    def _generate_adsr_envelope(self, samples: int, attack_time: float, decay_time: float, total_duration: float) -> np.ndarray:
        """Generate ADSR (Attack, Decay, Sustain, Release) amplitude envelope."""
        envelope = np.ones(samples)
        
        attack_samples = int(attack_time * self.sample_rate)
        decay_samples = int(decay_time * self.sample_rate)
        release_samples = int(0.1 * self.sample_rate)  # 100ms release
        
        # Attack phase (0 to 1)
        if attack_samples > 0:
            attack_end = min(attack_samples, samples)
            envelope[:attack_end] = np.linspace(0, 1, attack_end)
        
        # Decay phase (1 to sustain level)
        sustain_level = 0.7  # 70% of peak amplitude
        if decay_samples > 0 and attack_samples < samples:
            decay_start = attack_samples
            decay_end = min(decay_start + decay_samples, samples)
            if decay_end > decay_start:
                envelope[decay_start:decay_end] = np.linspace(1, sustain_level, decay_end - decay_start)
        
        # Sustain phase (constant level)
        sustain_start = attack_samples + decay_samples
        release_start = max(sustain_start, samples - release_samples)
        if release_start > sustain_start:
            envelope[sustain_start:release_start] = sustain_level
        
        # Release phase (sustain to 0)
        if release_samples > 0 and release_start < samples:
            envelope[release_start:] = np.linspace(sustain_level, 0, samples - release_start)
        
        return envelope
    
    def stop_vocalization(self) -> bool:
        """Stop current vocalization immediately."""
        if not self._is_currently_vocalizing:
            return True
        
        print("⏹️ Stopping Mac audio vocalization...")
        self._stop_requested = True
        
        # Wait for thread to finish
        if self._vocalization_thread and self._vocalization_thread.is_alive():
            self._vocalization_thread.join(timeout=0.1)
        
        self._end_vocalization()
        return True
    
    def _end_vocalization(self):
        """End current vocalization."""
        if self._is_currently_vocalizing:
            actual_duration = time.time() - self._vocalization_start_time
            self.total_vocalization_time += actual_duration
            
            print(f"🔇 Mac audio vocalization ended (duration: {actual_duration:.2f}s)")
            
        self._is_currently_vocalizing = False
        self._stop_requested = False
    
    def is_vocalizing(self) -> bool:
        """Check if currently vocalizing."""
        return self._is_currently_vocalizing
    
    def get_vocal_capabilities(self) -> Dict[str, any]:
        """Get Mac audio vocal system capabilities."""
        return {
            'type': 'mac_audio_vocal_driver',
            'version': '1.0.0',
            'audio_backend': 'pygame' if PYGAME_AVAILABLE else 'simulation',
            'features': [
                'real_audio_synthesis',
                'harmonic_generation',
                'frequency_modulation',
                'amplitude_modulation',
                'adsr_envelope',
                'noise_component',
                'mac_speaker_output'
            ],
            'frequency_range': [50.0, 2000.0],  # Hz
            'amplitude_range': [0.0, 1.0],
            'max_duration': 5.0,  # seconds
            'sample_rate': self.sample_rate,
            'latency': 0.05,  # seconds (estimated)
            'harmonic_count': 10,
            'special_feature': 'Your Mac becomes the robot voice!'
        }
    
    def set_volume(self, volume: float) -> bool:
        """Set Mac audio vocal volume."""
        if volume < 0.0 or volume > 1.0:
            print(f"❌ Invalid volume: {volume} (must be 0.0-1.0)")
            return False
        
        old_volume = self._current_volume
        self._current_volume = volume
        print(f"🔊 Mac speaker volume changed: {old_volume:.2f} → {volume:.2f}")
        return True
    
    def _print_vocalization_info(self, params: VocalParameters):
        """Print information about the vocalization being synthesized."""
        print(f"\n🎵 MAC AUDIO SYNTHESIS:")
        print(f"   🔊 Playing through Mac speakers")
        print(f"   🎼 Frequency: {params.fundamental_frequency:.1f} Hz")
        print(f"   📢 Amplitude: {params.amplitude:.2f}")
        print(f"   ⏱️  Duration: {params.duration:.2f}s")
        print(f"   🎶 Harmonics: {len(params.harmonics)} components")
        print(f"   🌊 Noise: {params.noise_component:.2f}")
        
        # Show modulation if present
        freq_mod_rate, freq_mod_depth = params.frequency_modulation
        if freq_mod_rate > 0:
            print(f"   🎵 Vibrato: {freq_mod_rate:.1f}Hz, {freq_mod_depth:.1f} cents")
        
        amp_mod_rate, amp_mod_depth = params.amplitude_modulation
        if amp_mod_rate > 0:
            print(f"   📊 Tremolo: {amp_mod_rate:.1f}Hz, {amp_mod_depth:.2f} depth")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get Mac audio driver statistics for testing."""
        return {
            'total_vocalizations': self.vocalization_count,
            'total_time_vocalizing': self.total_vocalization_time,
            'currently_vocalizing': self._is_currently_vocalizing,
            'current_volume': self._current_volume,
            'vocalization_history_count': len(self.vocalization_history),
            'last_vocalization': self.last_parameters.__dict__ if self.last_parameters else None,
            'audio_backend': 'pygame' if PYGAME_AVAILABLE else 'simulation',
            'special_status': 'Your Mac is the robot voice!'
        }
    
    def clear_statistics(self):
        """Clear statistics for testing."""
        self.vocalization_count = 0
        self.total_vocalization_time = 0.0
        self.vocalization_history.clear()
        self.last_parameters = None
        print("📊 Mac audio vocal driver statistics cleared")