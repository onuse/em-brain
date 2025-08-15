#!/usr/bin/env python3
"""
Audio Field Injector - Direct audio to field injection.

Processes audio chunks and injects frequency/amplitude features into field.
Enables the brain to "hear" without blocking main processing.
"""

import socket
import struct
import threading
import time
import torch
import numpy as np
from typing import Dict, Optional, Tuple


class AudioFieldInjector:
    """
    Audio sensor thread with direct field injection.
    Receives audio chunks and injects spectral features into auditory region.
    """
    
    def __init__(self, brain_field: torch.Tensor, port: int = 10006,
                 sample_rate: int = 16000, chunk_size: int = 512):
        """
        Args:
            brain_field: Reference to brain's field tensor
            port: UDP port to listen on for audio chunks
            sample_rate: Expected audio sample rate (Hz)
            chunk_size: Audio chunk size in samples
        """
        self.field = brain_field
        self.port = port
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.thread = None
        
        # Audio gets its own region - the "auditory cortex"
        self.field_region = self._allocate_region()
        
        # Audio processing state
        self.audio_buffer = np.zeros(chunk_size * 2)  # Double buffer for FFT
        
        # Stats
        self.chunks_received = 0
        self.chunks_injected = 0
        self.last_chunk_time = 0
        self.loudness_warnings = 0
        
    def _allocate_region(self) -> Dict:
        """
        Allocate field region for auditory processing.
        
        Audio needs:
        - Frequency bands (low to high)
        - Temporal patterns
        - Spatial localization (if stereo)
        """
        field_shape = self.field.shape
        
        # Use a different region from vision
        # Start from middle of field
        spatial_start = field_shape[0] // 2
        spatial_end = spatial_start + 4  # 4x4x4 region for audio
        
        return {
            'spatial': (slice(spatial_start, spatial_end),
                       slice(0, 4),
                       slice(0, 4)),
            'channels': slice(16, 24),  # 8 channels for audio features
            'decay_rate': 0.85,  # Faster decay than vision (sound is transient)
            'injection_strength': 0.4,
            'loudness_threshold': 0.7  # For detecting loud sounds
        }
    
    def start(self):
        """Start the audio injection thread."""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._injection_loop, daemon=True)
        self.thread.start()
        print(f"🔊 Audio field injector started on port {self.port}")
        print(f"   Sample rate: {self.sample_rate}Hz")
        print(f"   Chunk size: {self.chunk_size} samples")
        print(f"   Field region: {self.field_region['spatial']}")
    
    def stop(self):
        """Stop the injection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        print(f"🔊 Audio injector stopped. Chunks: {self.chunks_received}, "
              f"Injected: {self.chunks_injected}, Loud events: {self.loudness_warnings}")
    
    def _injection_loop(self):
        """
        Main loop - receives audio chunks and injects into field.
        Processes audio in parallel without blocking brain.
        """
        # Setup UDP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', self.port))
        sock.settimeout(0.05)  # 50ms timeout
        
        print(f"🔊 Audio injector listening on UDP:{self.port}")
        
        while self.running:
            try:
                # Receive audio chunk
                data, addr = sock.recvfrom(4096)  # Up to 1024 float32 samples
                
                if len(data) < 4:
                    continue
                
                # Parse header
                num_samples = struct.unpack('!I', data[:4])[0]
                
                if num_samples > 0 and len(data) >= 4 + num_samples * 4:
                    # Extract audio samples (float32)
                    samples = np.frombuffer(data[4:4+num_samples*4], dtype=np.float32)
                    
                    # Process and inject
                    self._process_audio(samples)
                    
            except socket.timeout:
                # No audio - apply decay
                self._inject_decay()
                
            except Exception as e:
                if self.chunks_received % 100 == 0:
                    print(f"⚠️  Audio injector error: {e}")
        
        sock.close()
    
    def _process_audio(self, samples: np.ndarray):
        """
        Process audio chunk and inject features into field.
        Extracts frequency bands and temporal features.
        """
        self.chunks_received += 1
        
        try:
            # Shift buffer and add new samples
            self.audio_buffer[:-len(samples)] = self.audio_buffer[len(samples):]
            self.audio_buffer[-len(samples):] = samples
            
            # Extract features
            features = self._extract_audio_features(self.audio_buffer)
            
            # Inject into field
            self._inject_into_field(features)
            
            self.chunks_injected += 1
            self.last_chunk_time = time.time()
            
            # Log periodically
            if self.chunks_injected % 100 == 0:  # Every ~3 seconds at 30Hz
                print(f"🔊 Chunk #{self.chunks_injected}: "
                      f"amplitude={features['amplitude']:.3f}, "
                      f"spectral_centroid={features['spectral_centroid']:.1f}Hz")
                
        except Exception as e:
            print(f"⚠️  Audio processing error: {e}")
    
    def _extract_audio_features(self, audio: np.ndarray) -> Dict:
        """
        Extract audio features for field injection.
        
        Returns:
            Dictionary of audio features
        """
        # Basic amplitude
        amplitude = np.abs(audio).mean()
        peak = np.abs(audio).max()
        
        # FFT for frequency analysis
        fft = np.fft.rfft(audio)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        magnitude = np.abs(fft)
        
        # Frequency bands (like cochlea)
        bands = []
        band_edges = [20, 100, 250, 500, 1000, 2000, 4000, 8000, 16000]
        for i in range(len(band_edges)-1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            if mask.any():
                bands.append(magnitude[mask].mean())
            else:
                bands.append(0.0)
        
        # Spectral centroid (brightness)
        if magnitude.sum() > 0:
            spectral_centroid = np.sum(freqs * magnitude) / magnitude.sum()
        else:
            spectral_centroid = 0.0
        
        # Zero crossing rate (percussiveness)
        zero_crossings = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
        
        # Detect loud events
        if amplitude > self.field_region['loudness_threshold']:
            self.loudness_warnings += 1
            if self.loudness_warnings % 10 == 1:
                print(f"🔊 LOUD SOUND! Amplitude: {amplitude:.2f}")
        
        return {
            'amplitude': amplitude,
            'peak': peak,
            'bands': np.array(bands),
            'spectral_centroid': spectral_centroid,
            'zero_crossings': zero_crossings,
            'is_loud': amplitude > self.field_region['loudness_threshold']
        }
    
    def _inject_into_field(self, features: Dict):
        """
        Inject audio features into field.
        
        Maps frequency bands and temporal patterns to field regions.
        """
        spatial = self.field_region['spatial']
        channels = self.field_region['channels']
        decay = self.field_region['decay_rate']
        strength = self.field_region['injection_strength']
        
        with torch.no_grad():
            # Decay old audio activation
            self.field[spatial][..., channels] *= decay
            
            # Get field dimensions
            field_x = spatial[0].stop - spatial[0].start
            field_y = spatial[1].stop - spatial[1].start
            field_z = spatial[2].stop - spatial[2].start
            
            # Create audio feature tensor
            audio_field = torch.zeros(field_x, field_y, field_z, 8)
            
            # Channel 0: Overall amplitude
            audio_field[0, 0, 0, 0] = features['amplitude']
            
            # Channel 1: Peak (for transients)
            audio_field[0, 0, 0, 1] = features['peak']
            
            # Channels 2-5: Frequency bands (spatial mapping)
            # Map frequency bands to spatial positions
            bands = features['bands'][:4]  # Use first 4 bands
            for i, band_amp in enumerate(bands):
                if i < field_y:
                    audio_field[0, i, 0, 2] = band_amp
            
            # Channel 6: Spectral centroid (brightness)
            audio_field[0, 0, 0, 6] = features['spectral_centroid'] / 8000.0  # Normalize
            
            # Channel 7: Loudness alert
            if features['is_loud']:
                audio_field[:, :, :, 7] = 1.0  # Flood the region for attention
            
            # Inject into field
            self.field[spatial][..., channels] += audio_field * strength
            
            # Clamp to prevent explosion
            self.field[spatial][..., channels] = torch.clamp(
                self.field[spatial][..., channels], -10, 10
            )
    
    def _inject_decay(self):
        """
        Apply decay when no audio received.
        Sound fades quickly in absence of input.
        """
        spatial = self.field_region['spatial']
        channels = self.field_region['channels']
        
        with torch.no_grad():
            # Fast decay for audio (sound doesn't persist like vision)
            self.field[spatial][..., channels] *= 0.7


# Test standalone
if __name__ == "__main__":
    print("AUDIO FIELD INJECTOR TEST")
    print("=" * 50)
    
    # Create test field
    field = torch.zeros(16, 16, 16, 64)
    print(f"Created field: {field.shape}")
    
    # Create audio injector
    injector = AudioFieldInjector(field, port=10006)
    injector.start()
    
    # Simulate audio stream
    def send_test_audio():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Generate test tones
        sample_rate = 16000
        duration = 0.03  # 30ms chunks
        
        for freq in [440, 880, 440, 220, 440]:  # A4, A5, A4, A3, A4
            t = np.linspace(0, duration, int(sample_rate * duration))
            samples = np.sin(2 * np.pi * freq * t).astype(np.float32)
            
            # Add some noise
            samples += np.random.normal(0, 0.1, len(samples)).astype(np.float32)
            
            # Pack and send
            packet = struct.pack('!I', len(samples)) + samples.tobytes()
            sock.sendto(packet, ('localhost', 10006))
            
            print(f"  Sent {freq}Hz tone")
            time.sleep(0.5)
        
        sock.close()
    
    print("\nSending test audio...")
    audio_thread = threading.Thread(target=send_test_audio, daemon=True)
    audio_thread.start()
    
    # Monitor field changes
    print("\nMonitoring audio field region...")
    for i in range(10):
        spatial = injector.field_region['spatial']
        channels = injector.field_region['channels']
        
        region = field[spatial][..., channels]
        mean_val = region.mean().item()
        max_val = region.max().item()
        
        print(f"t={i}s: chunks={injector.chunks_received}, "
              f"field: mean={mean_val:.3f}, max={max_val:.3f}")
        
        time.sleep(1)
    
    injector.stop()
    
    if injector.chunks_injected > 0:
        print("\n✅ Audio injector works!")
    else:
        print("\n⚠️  No audio chunks processed")