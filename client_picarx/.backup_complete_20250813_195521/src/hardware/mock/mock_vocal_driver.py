#!/usr/bin/env python3
"""
Mock Vocal Driver - Testing Implementation

Provides a mock implementation of the VocalInterface for hardware-free
development and testing. Simulates vocal output behavior without requiring
actual audio hardware.
"""

import time
import threading
from typing import Dict, Optional
import numpy as np

from ..interfaces.vocal_interface import VocalInterface, VocalParameters, VocalSafetyConstraints


class MockVocalDriver(VocalInterface):
    """Mock implementation of vocal interface for testing."""
    
    def __init__(self, safety_constraints: Optional[VocalSafetyConstraints] = None):
        """Initialize mock vocal driver."""
        super().__init__(safety_constraints)
        
        # Mock state
        self._initialized = False
        self._current_volume = 0.5
        self._is_currently_vocalizing = False
        self._vocalization_start_time = 0.0
        self._current_vocalization_duration = 0.0
        self._vocalization_thread = None
        self._stop_requested = False
        
        # Statistics for testing
        self.vocalization_count = 0
        self.total_vocalization_time = 0.0
        self.last_parameters = None
        self.vocalization_history = []
        
        print("🎵 MockVocalDriver initialized - vocal output will be simulated")
    
    def initialize_vocal_system(self) -> bool:
        """Initialize mock vocal system."""
        if self._initialized:
            return True
            
        print("🔧 Initializing mock vocal system...")
        
        # Simulate initialization delay
        time.sleep(0.1)
        
        self._initialized = True
        print("✅ Mock vocal system initialized successfully")
        return True
    
    def synthesize_vocalization(self, params: VocalParameters) -> bool:
        """Simulate vocalization synthesis."""
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
        """Start simulated vocalization in separate thread."""
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
        
        # Print simulation of vocal output
        self._print_vocalization_simulation(params)
        
        # Start timer thread to automatically stop
        self._vocalization_thread = threading.Thread(
            target=self._vocalization_timer, 
            args=(params.duration,)
        )
        self._vocalization_thread.start()
        
        # Update statistics
        self.vocalization_count += 1
    
    def _vocalization_timer(self, duration: float):
        """Timer thread to stop vocalization after duration."""
        time.sleep(duration)
        if not self._stop_requested:
            self._end_vocalization()
    
    def _end_vocalization(self):
        """End current vocalization."""
        if self._is_currently_vocalizing:
            actual_duration = time.time() - self._vocalization_start_time
            self.total_vocalization_time += actual_duration
            
            print(f"🔇 Vocalization ended (duration: {actual_duration:.2f}s)")
            
        self._is_currently_vocalizing = False
        self._stop_requested = False
    
    def stop_vocalization(self) -> bool:
        """Stop current vocalization immediately."""
        if not self._is_currently_vocalizing:
            return True
        
        print("⏹️ Stopping vocalization...")
        self._stop_requested = True
        
        # Wait for thread to finish
        if self._vocalization_thread and self._vocalization_thread.is_alive():
            self._vocalization_thread.join(timeout=0.1)
        
        self._end_vocalization()
        return True
    
    def is_vocalizing(self) -> bool:
        """Check if currently vocalizing."""
        return self._is_currently_vocalizing
    
    def get_vocal_capabilities(self) -> Dict[str, any]:
        """Get mock vocal system capabilities."""
        return {
            'type': 'mock_vocal_driver',
            'version': '1.0.0',
            'features': [
                'digital_vocal_synthesis',
                'harmonic_generation',
                'formant_filtering',
                'real_time_parameter_control',
                'safety_constraint_enforcement'
            ],
            'frequency_range': [50.0, 2000.0],  # Hz
            'amplitude_range': [0.0, 1.0],
            'max_duration': 5.0,  # seconds
            'sample_rate': 22050,  # Hz
            'latency': 0.001,  # seconds (simulated low latency)
            'harmonic_count': 10,
            'formant_count': 3
        }
    
    def set_volume(self, volume: float) -> bool:
        """Set mock vocal volume."""
        if volume < 0.0 or volume > 1.0:
            print(f"❌ Invalid volume: {volume} (must be 0.0-1.0)")
            return False
        
        old_volume = self._current_volume
        self._current_volume = volume
        print(f"🔊 Volume changed: {old_volume:.2f} → {volume:.2f}")
        return True
    
    def _print_vocalization_simulation(self, params: VocalParameters):
        """Print visual simulation of vocal output."""
        print(f"\n🎵 VOCAL OUTPUT SIMULATION:")
        print(f"   Frequency: {params.fundamental_frequency:.1f} Hz")
        print(f"   Amplitude: {params.amplitude:.2f}")
        print(f"   Duration: {params.duration:.2f}s")
        print(f"   Harmonics: {[f'{h:.2f}' for h in params.harmonics[:3]]}...")
        print(f"   Noise: {params.noise_component:.2f}")
        
        # Create ASCII visualization of waveform
        self._print_waveform_ascii(params)
    
    def _print_waveform_ascii(self, params: VocalParameters):
        """Print ASCII art representation of waveform."""
        # Generate simplified waveform representation
        width = 40
        height = 7
        
        # Create basic sine wave pattern
        waveform = []
        for i in range(width):
            # Base sine wave
            phase = (i / width) * 2 * np.pi * 3  # 3 cycles across display
            amplitude = np.sin(phase)
            
            # Add frequency modulation if present
            freq_mod_rate, freq_mod_depth = params.frequency_modulation
            if freq_mod_rate > 0:
                mod_phase = (i / width) * 2 * np.pi * freq_mod_rate * 0.1
                freq_mod = np.sin(mod_phase) * (freq_mod_depth / 100.0)
                amplitude *= (1.0 + freq_mod)
            
            # Add amplitude modulation if present
            amp_mod_rate, amp_mod_depth = params.amplitude_modulation
            if amp_mod_rate > 0:
                mod_phase = (i / width) * 2 * np.pi * amp_mod_rate * 0.1
                amp_mod = np.sin(mod_phase) * amp_mod_depth
                amplitude *= (1.0 + amp_mod)
            
            # Scale by overall amplitude
            amplitude *= params.amplitude
            
            # Add noise component
            if params.noise_component > 0:
                noise = (np.random.random() - 0.5) * params.noise_component
                amplitude += noise
            
            # Convert to display row (center = height//2)
            row = int((amplitude + 1.0) * height / 2)
            row = max(0, min(height - 1, row))
            waveform.append(row)
        
        # Print ASCII waveform
        print("   Waveform:")
        for row in range(height - 1, -1, -1):
            line = "   "
            for col in range(width):
                if waveform[col] == row:
                    line += "█"
                elif abs(waveform[col] - row) <= 0.5:
                    line += "▓"
                else:
                    line += " "
            print(line)
        print("   " + "─" * width)
        print(f"   0s{' ' * (width-6)}{params.duration:.1f}s")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get mock driver statistics for testing."""
        return {
            'total_vocalizations': self.vocalization_count,
            'total_time_vocalizing': self.total_vocalization_time,
            'currently_vocalizing': self._is_currently_vocalizing,
            'current_volume': self._current_volume,
            'vocalization_history_count': len(self.vocalization_history),
            'last_vocalization': self.last_parameters.__dict__ if self.last_parameters else None
        }
    
    def clear_statistics(self):
        """Clear statistics for testing."""
        self.vocalization_count = 0
        self.total_vocalization_time = 0.0
        self.vocalization_history.clear()
        self.last_parameters = None
        print("📊 Mock vocal driver statistics cleared")


class MockVocalTester:
    """Test utility for mock vocal driver."""
    
    def __init__(self):
        """Initialize vocal tester."""
        self.driver = MockVocalDriver()
        
    def run_basic_test(self):
        """Run basic functionality test."""
        print("\n🧪 MOCK VOCAL DRIVER TEST")
        print("=" * 50)
        
        # Initialize
        success = self.driver.initialize_vocal_system()
        assert success, "Initialization failed"
        
        # Test basic vocalization
        test_params = VocalParameters(
            fundamental_frequency=440.0,  # A4 note
            amplitude=0.5,
            duration=1.0,
            harmonics=[1.0, 0.5, 0.25],
            frequency_modulation=(2.0, 10.0),
            amplitude_modulation=(1.0, 0.1),
            noise_component=0.1,
            attack_time=0.1,
            decay_time=0.5,
            formant_frequencies=[500.0, 1000.0, 2000.0],
            formant_amplitudes=[0.8, 0.6, 0.4]
        )
        
        print("\n🎼 Testing basic vocalization...")
        success = self.driver.synthesize_vocalization(test_params)
        assert success, "Vocalization failed"
        
        # Wait for completion
        time.sleep(1.2)
        
        # Check statistics
        stats = self.driver.get_statistics()
        print(f"\n📊 Statistics: {stats['total_vocalizations']} vocalizations, {stats['total_time_vocalizing']:.2f}s total")
        
        # Test volume control
        print("\n🔊 Testing volume control...")
        self.driver.set_volume(0.8)
        
        # Test stop functionality
        print("\n⏹️ Testing stop functionality...")
        self.driver.synthesize_vocalization(test_params)
        time.sleep(0.2)
        self.driver.stop_vocalization()
        
        print("\n✅ All basic tests passed!")
        return True


if __name__ == "__main__":
    # Run test if executed directly
    tester = MockVocalTester()
    tester.run_basic_test()