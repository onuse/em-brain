#!/usr/bin/env python3
"""
Vocal Demo Diagnostic - Test audio system step by step

Helps diagnose audio issues in the visual demo by testing each component separately.
"""

import sys
import os
import time

# Add client source to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("❌ pygame not available")
    sys.exit(1)

from hardware.interfaces.vocal_interface import EmotionalVocalMapper
from hardware.mock.mac_audio_vocal_driver import MacAudioVocalDriver


def test_audio_before_pygame():
    """Test audio system before pygame initialization."""
    
    print("🔧 STEP 1: Testing MacAudioVocalDriver before pygame.init()")
    print("=" * 60)
    
    # Initialize vocal driver BEFORE pygame
    vocal_driver = MacAudioVocalDriver()
    success = vocal_driver.initialize_vocal_system()
    
    if not success:
        print("❌ Failed to initialize vocal driver")
        return False
    
    # Test a simple sound
    emotional_mapper = EmotionalVocalMapper()
    brain_state = {
        'prediction_confidence': 0.9,
        'prediction_method': 'consensus', 
        'total_experiences': 100,
        'collision_detected': False
    }
    
    vocal_params = emotional_mapper.map_brain_state_to_vocal_params(brain_state)
    
    print(f"🎵 Playing test sound (confidence emotion)...")
    print(f"   Frequency: {vocal_params.fundamental_frequency:.1f} Hz")
    print(f"   Duration: {vocal_params.duration:.1f}s")
    
    vocal_driver.synthesize_vocalization(vocal_params)
    
    # Wait for completion
    time.sleep(vocal_params.duration + 0.5)
    
    print("✅ Audio test before pygame completed")
    return True


def test_audio_after_pygame():
    """Test audio system after pygame initialization."""
    
    print("\n🔧 STEP 2: Testing after pygame.init()")
    print("=" * 60)
    
    # Now initialize pygame
    print("🎮 Initializing pygame...")
    pygame.init()
    
    # Check mixer status
    print(f"   pygame.mixer initialized: {pygame.mixer.get_init()}")
    
    # Try to create a new vocal driver
    print("🎵 Creating new MacAudioVocalDriver after pygame.init()...")
    vocal_driver2 = MacAudioVocalDriver()
    success = vocal_driver2.initialize_vocal_system()
    
    if not success:
        print("❌ Failed to initialize vocal driver after pygame")
        return False
    
    # Test sound again
    emotional_mapper = EmotionalVocalMapper()
    brain_state = {
        'prediction_confidence': 0.2,
        'prediction_method': 'bootstrap_random',
        'total_experiences': 5,
        'collision_detected': False  
    }
    
    vocal_params = emotional_mapper.map_brain_state_to_vocal_params(brain_state)
    
    print(f"🎵 Playing test sound (curiosity emotion)...")
    print(f"   Frequency: {vocal_params.fundamental_frequency:.1f} Hz")
    print(f"   Duration: {vocal_params.duration:.1f}s")
    
    vocal_driver2.synthesize_vocalization(vocal_params)
    
    # Wait for completion
    time.sleep(vocal_params.duration + 0.5)
    
    print("✅ Audio test after pygame completed")
    return True


def test_mixer_conflict():
    """Test for pygame mixer conflicts."""
    
    print("\n🔧 STEP 3: Testing mixer conflict resolution")
    print("=" * 60)
    
    # Quit existing mixer
    print("🔄 Quitting pygame.mixer...")
    pygame.mixer.quit()
    
    # Reinitialize with vocal driver settings
    print("🎵 Reinitializing mixer with vocal settings...")
    pygame.mixer.pre_init(
        frequency=22050,
        size=-16,  # 16-bit signed
        channels=1,  # Mono
        buffer=1024
    )
    pygame.mixer.init()
    
    print(f"   New mixer settings: {pygame.mixer.get_init()}")
    
    # Test with direct pygame sound
    print("🎵 Testing direct pygame sound generation...")
    import numpy as np
    
    # Generate test tone
    duration = 1.0
    frequency = 440.0
    sample_rate = 22050
    samples = int(sample_rate * duration)
    
    wave_array = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
    wave_array = (wave_array * 0.5 * 32767).astype(np.int16)
    
    sound = pygame.sndarray.make_sound(wave_array)
    
    print(f"🔊 Playing 440Hz test tone for {duration}s...")
    sound.play()
    time.sleep(duration + 0.2)
    
    print("✅ Direct pygame audio test completed")
    return True


def main():
    """Run complete diagnostic."""
    
    print("🔍 VOCAL DEMO AUDIO DIAGNOSTIC")
    print("=" * 60)
    print("Testing audio system components to identify issues")
    print()
    
    try:
        # Step 1: Test before pygame
        if not test_audio_before_pygame():
            print("❌ Audio failed before pygame - basic audio issue")
            return
        
        # Step 2: Test after pygame  
        if not test_audio_after_pygame():
            print("❌ Audio failed after pygame - mixer conflict detected")
            
            # Step 3: Try to fix conflict
            if test_mixer_conflict():
                print("✅ Mixer conflict resolved!")
            else:
                print("❌ Could not resolve mixer conflict")
        else:
            print("✅ No mixer conflict detected")
        
        print("\n🎯 DIAGNOSTIC COMPLETE")
        print("=" * 60)
        print("If you heard sounds in Step 1 but not Step 2,")
        print("there's a pygame mixer conflict that we can fix.")
        print()
        print("If you heard no sounds at all, check:")
        print("   • Mac volume is up")
        print("   • No other audio apps using the sound system")
        print("   • pygame installation is working")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()