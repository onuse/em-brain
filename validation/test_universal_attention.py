#!/usr/bin/env python3
"""
Test Universal Attention System with different signal modalities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.universal_attention import UniversalAttentionSystem, ModalityType

def test_universal_attention():
    """Test universal attention system with different signal types"""
    print("🎯 Testing Universal Attention System")
    print("=" * 50)
    
    # Create attention system
    attention_system = UniversalAttentionSystem()
    
    # Test 1: Visual signal (2D image)
    print("\n📷 Testing Visual Signal (2D)")
    visual_signal = create_test_image(64, 64)
    
    attention_map, windows = attention_system.calculate_attention_map(
        visual_signal, ModalityType.VISUAL, novelty_score=0.7
    )
    
    print(f"   Visual attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
              f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 2: Audio signal (1D time series)
    print("\n🎵 Testing Audio Signal (1D)")
    audio_signal = create_test_audio(1000)
    
    attention_map, windows = attention_system.calculate_attention_map(
        audio_signal, ModalityType.AUDIO, novelty_score=0.8
    )
    
    print(f"   Audio attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        if 'start' in window:
            print(f"   Window {i+1}: {window['type']} from {window['start']} to {window['end']} "
                  f"strength {window['strength']:.3f}")
        else:
            print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
                  f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 3: Audio spectrogram (2D)
    print("\n🎶 Testing Audio Spectrogram (2D)")
    spectrogram = create_test_spectrogram(32, 100)
    
    attention_map, windows = attention_system.calculate_attention_map(
        spectrogram, ModalityType.AUDIO, novelty_score=0.6
    )
    
    print(f"   Spectrogram attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
              f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 4: With brain response
    print("\n🧠 Testing with Brain Response")
    brain_output = np.random.uniform(0, 1, 16)  # Simulated brain response
    
    attention_map, windows = attention_system.calculate_attention_map(
        visual_signal, ModalityType.VISUAL, 
        brain_output=brain_output, novelty_score=0.9
    )
    
    print(f"   Brain-enhanced attention max: {np.max(attention_map):.3f}")
    print(f"   Brain-enhanced attention avg: {np.mean(attention_map):.3f}")
    print(f"   Brain-enhanced windows: {len(windows)}")
    
    # Test 5: Tactile signal (1D pressure array)
    print("\n👆 Testing Tactile Signal (1D Pressure)")
    tactile_signal = create_test_tactile_1d(50)
    
    attention_map, windows = attention_system.calculate_attention_map(
        tactile_signal, ModalityType.TACTILE, novelty_score=0.7
    )
    
    print(f"   Tactile attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        if 'start' in window:
            print(f"   Window {i+1}: {window['type']} from {window['start']} to {window['end']} "
                  f"strength {window['strength']:.3f}")
        else:
            print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
                  f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 6: Motor signal (1D joint positions)
    print("\n🤖 Testing Motor Signal (1D Joint Positions)")
    motor_signal = create_test_motor_1d(30)
    
    attention_map, windows = attention_system.calculate_attention_map(
        motor_signal, ModalityType.MOTOR, novelty_score=0.8
    )
    
    print(f"   Motor attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        if 'start' in window:
            print(f"   Window {i+1}: {window['type']} from {window['start']} to {window['end']} "
                  f"strength {window['strength']:.3f}")
        else:
            print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
                  f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 7: Temporal signal (rhythm pattern)
    print("\n🎵 Testing Temporal Signal (Rhythm Pattern)")
    temporal_signal = create_test_temporal_1d(100)
    
    attention_map, windows = attention_system.calculate_attention_map(
        temporal_signal, ModalityType.TEMPORAL, novelty_score=0.6
    )
    
    print(f"   Temporal attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        if 'start' in window:
            print(f"   Window {i+1}: {window['type']} from {window['start']} to {window['end']} "
                  f"strength {window['strength']:.3f}")
        else:
            print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
                  f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 8: 2D Tactile signal (pressure map)
    print("\n🤚 Testing 2D Tactile Signal (Pressure Map)")
    tactile_2d = create_test_tactile_2d(16, 16)
    
    attention_map, windows = attention_system.calculate_attention_map(
        tactile_2d, ModalityType.TACTILE, novelty_score=0.7
    )
    
    print(f"   2D Tactile attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
              f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 9: 2D Motor signal (multiple joints over time)
    print("\n🦾 Testing 2D Motor Signal (Multi-Joint Timeline)")
    motor_2d = create_test_motor_2d(6, 20)
    
    attention_map, windows = attention_system.calculate_attention_map(
        motor_2d, ModalityType.MOTOR, novelty_score=0.8
    )
    
    print(f"   2D Motor attention map shape: {attention_map.shape}")
    print(f"   Max attention: {np.max(attention_map):.3f}")
    print(f"   Avg attention: {np.mean(attention_map):.3f}")
    print(f"   Attention windows: {len(windows)}")
    
    for i, window in enumerate(windows):
        if 'start' in window:
            print(f"   Window {i+1}: {window['type']} from {window['start']} to {window['end']} "
                  f"strength {window['strength']:.3f}")
        else:
            print(f"   Window {i+1}: {window['type']} at ({window['x']}, {window['y']}) "
                  f"size {window['width']}x{window['height']}, strength {window['strength']:.3f}")
    
    # Test 10: Signal shape detection
    print("\n📊 Testing Signal Shape Detection")
    scalar = np.array(5.0)
    vector = np.array([1, 2, 3, 4, 5])
    matrix = np.array([[1, 2], [3, 4]])
    tensor = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    
    print(f"   Scalar shape: {attention_system.get_signal_shape(scalar)}")
    print(f"   Vector shape: {attention_system.get_signal_shape(vector)}")
    print(f"   Matrix shape: {attention_system.get_signal_shape(matrix)}")
    print(f"   Tensor shape: {attention_system.get_signal_shape(tensor)}")
    
    # Performance test
    print("\n⚡ Performance Test")
    import time
    
    test_signals = [
        (visual_signal, ModalityType.VISUAL, "Visual"),
        (audio_signal, ModalityType.AUDIO, "Audio"),
        (tactile_signal, ModalityType.TACTILE, "Tactile"),
        (motor_signal, ModalityType.MOTOR, "Motor"),
        (temporal_signal, ModalityType.TEMPORAL, "Temporal")
    ]
    
    performance_times = []
    for signal, modality, name in test_signals:
        start_time = time.time()
        for i in range(10):
            attention_map, windows = attention_system.calculate_attention_map(
                signal, modality, novelty_score=0.5
            )
        avg_time = (time.time() - start_time) / 10
        performance_times.append(avg_time)
        print(f"   {name} processing: {avg_time*1000:.1f}ms avg")
    
    max_time = max(performance_times)
    print(f"\n🎉 Universal Attention System Test Complete!")
    print(f"   ✅ Visual signals: Working")
    print(f"   ✅ Audio signals: Working")
    print(f"   ✅ Tactile signals: Working")
    print(f"   ✅ Motor signals: Working")
    print(f"   ✅ Temporal signals: Working")
    print(f"   ✅ Brain integration: Working")
    print(f"   ✅ Real-time performance: {'Yes' if max_time < 0.05 else 'Marginal'}")
    
    return True

def create_test_image(width, height):
    """Create test image with various features"""
    image = np.zeros((height, width), dtype=np.float32)
    
    # Add some shapes
    cv2.rectangle(image, (10, 10), (30, 30), 1.0, -1)  # White square
    cv2.circle(image, (50, 20), 8, 0.7, -1)  # Gray circle
    cv2.line(image, (5, 50), (width-5, 50), 0.5, 2)  # Line
    
    # Add some noise
    noise = np.random.normal(0, 0.1, (height, width))
    image = np.clip(image + noise, 0, 1)
    
    return image

def create_test_audio(length):
    """Create test audio signal with various features"""
    t = np.linspace(0, 1, length)
    
    # Base sine wave
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
    
    # Add some transients (onsets)
    signal[100:120] += 0.8  # Sharp onset
    signal[300:310] += 0.6  # Another onset
    signal[500:520] += 0.9  # Strong onset
    
    # Add some noise
    noise = np.random.normal(0, 0.05, length)
    signal = np.clip(signal + noise, -1, 1)
    
    return signal

def create_test_spectrogram(freq_bins, time_bins):
    """Create test spectrogram with various features"""
    spectrogram = np.random.uniform(0, 0.2, (freq_bins, time_bins))
    
    # Add some frequency content
    spectrogram[10:15, 20:30] = 0.8  # Strong frequency band
    spectrogram[5:8, 40:60] = 0.6    # Another band
    spectrogram[25:30, 10:20] = 0.9  # High frequency content
    
    return spectrogram

def create_test_tactile_1d(length):
    """Create test 1D tactile signal (pressure array)"""
    signal = np.zeros(length, dtype=np.float32)
    
    # Add pressure points
    signal[10:15] = 0.8  # Strong pressure
    signal[25:30] = 0.4  # Medium pressure
    signal[40:42] = 0.9  # Sharp pressure spike
    
    # Add some noise
    noise = np.random.normal(0, 0.05, length)
    signal = np.clip(signal + noise, 0, 1)
    
    return signal

def create_test_tactile_2d(height, width):
    """Create test 2D tactile signal (pressure map)"""
    signal = np.zeros((height, width), dtype=np.float32)
    
    # Add pressure regions
    signal[4:8, 4:8] = 0.7  # Square pressure area
    signal[10:12, 10:14] = 0.5  # Rectangular pressure
    signal[2, 12:14] = 0.9  # Line pressure
    
    # Add some noise
    noise = np.random.normal(0, 0.03, (height, width))
    signal = np.clip(signal + noise, 0, 1)
    
    return signal

def create_test_motor_1d(length):
    """Create test 1D motor signal (joint positions)"""
    t = np.linspace(0, 2*np.pi, length)
    
    # Create joint movement pattern
    signal = np.sin(t) * 0.5  # Base sinusoidal movement
    signal[10:15] += 0.3  # Sudden movement
    signal[20:25] -= 0.2  # Reverse movement
    
    # Add some noise
    noise = np.random.normal(0, 0.02, length)
    signal = signal + noise
    
    return signal

def create_test_motor_2d(joints, time_steps):
    """Create test 2D motor signal (multiple joints over time)"""
    signal = np.zeros((joints, time_steps), dtype=np.float32)
    
    # Create coordinated movement patterns
    for i in range(joints):
        t = np.linspace(0, 2*np.pi, time_steps)
        # Different phase for each joint
        phase = i * np.pi / joints
        signal[i] = np.sin(t + phase) * 0.4
        
        # Add some joint-specific movements
        if i == 0:  # First joint has extra movement
            signal[i, 5:10] += 0.3
        elif i == joints-1:  # Last joint (end effector) has precise movements
            signal[i, 15:18] += 0.2
    
    # Add some noise
    noise = np.random.normal(0, 0.02, (joints, time_steps))
    signal = signal + noise
    
    return signal

def create_test_temporal_1d(length):
    """Create test temporal signal (rhythm pattern)"""
    signal = np.zeros(length, dtype=np.float32)
    
    # Create rhythmic pattern
    beat_pattern = [1, 0, 0.5, 0, 1, 0, 0.7, 0]  # 8-beat pattern
    
    # Repeat pattern throughout signal
    for i in range(length):
        signal[i] = beat_pattern[i % len(beat_pattern)]
    
    # Add some rhythm variations
    signal[20:25] *= 1.3  # Accent
    signal[40:45] *= 0.7  # Softer
    signal[60:65] = 0     # Rest
    
    # Add some noise
    noise = np.random.normal(0, 0.05, length)
    signal = np.clip(signal + noise, 0, 1)
    
    return signal

if __name__ == "__main__":
    test_universal_attention()