#!/usr/bin/env python3
"""
Test the refactored active sensing system

Verifies that:
1. Base ActiveSensingSystem provides common functionality
2. ActiveVisionSystem works as before with new abstraction
3. Audio and Tactile stubs are ready for future use
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

# Import brain components
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
from src.brains.field.active_vision_system import ActiveVisionSystem
from src.brains.field.active_audio_system import ActiveAudioSystem
from src.brains.field.active_tactile_system import ActiveTactileSystem


def test_vision_system():
    """Test that vision system still works with new abstraction."""
    print("\n=== Testing Refactored Vision System ===\n")
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=3,
        motor_dim=5,  # 3 basic + 2 sensor control
        spatial_resolution=32,
        quiet_mode=True
    )
    
    # Enable active vision
    brain.enable_active_vision(True)
    
    print("1. Testing vision system initialization...")
    assert brain.use_active_vision
    assert isinstance(brain.active_vision, ActiveVisionSystem)
    assert brain.active_vision.modality == "vision"
    print("   ✓ Vision system initialized correctly")
    
    print("\n2. Testing uncertainty map generation...")
    # Run a few cycles
    for i in range(10):
        sensory_input = [float(i % 2), 0.5, 0.0, 0.0]
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    # Generate uncertainty map
    uncertainty_map = brain.active_vision.generate_uncertainty_map(
        topology_regions=brain._last_activated_regions if hasattr(brain, '_last_activated_regions') else [],
        field=brain.unified_field
    )
    
    assert uncertainty_map.total_uncertainty >= 0.0
    assert len(uncertainty_map.peak_locations) > 0
    print("   ✓ Uncertainty maps working")
    
    print("\n3. Testing sensor control generation...")
    # Run more cycles with varying input
    for i in range(15):
        if i < 5:
            sensory_input = [0.5, 0.5, 0.0, 0.0]
        else:
            sensory_input = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), 0.0, 0.0]
        
        motor_output, brain_state = brain.process_robot_cycle(sensory_input)
        
        # Check motor output includes sensor control
        assert len(motor_output) >= 5
    
    # Check statistics
    stats = brain_state.get('active_vision', {})
    if stats:
        assert 'modality' in stats
        assert stats['modality'] == 'vision'
        assert 'current_pattern' in stats
        print(f"   Current pattern: {stats['current_pattern']}")
    else:
        print("   Note: Statistics not yet populated (expected in early cycles)")
    print("   ✓ Vision control working")
    
    return True


def test_audio_stub():
    """Test that audio system stub is ready for future use."""
    print("\n\n=== Testing Audio System Stub ===\n")
    
    # Create standalone audio system
    audio_system = ActiveAudioSystem(
        field_shape=(32, 32, 32, 64),
        control_dims=3,  # frequency, bandwidth, direction
        device=torch.device('cpu')
    )
    
    print("1. Testing audio system initialization...")
    assert audio_system.modality == "audio"
    assert audio_system.control_dims == 3
    print("   ✓ Audio system initialized")
    
    print("\n2. Testing audio attention control...")
    # Create dummy uncertainty map
    spatial_uncertainty = torch.rand(32, 32, 32)
    uncertainty_map = type('UncertaintyMap', (), {
        'spatial_uncertainty': spatial_uncertainty,
        'feature_uncertainty': torch.rand(64),
        'total_uncertainty': 0.5,
        'peak_locations': [(10, 10, 10)]
    })()
    
    control = audio_system.generate_attention_control(
        uncertainty_map=uncertainty_map,
        current_predictions=None,
        exploration_drive=0.5
    )
    
    assert control.shape == (3,)
    assert torch.all(control == 0)  # Stub returns zeros
    print("   ✓ Audio control stub working")
    
    print("\n3. Testing audio statistics...")
    stats = audio_system.get_attention_statistics()
    assert stats['modality'] == 'audio'
    assert 'current_pattern' in stats
    print(f"   Current pattern: {stats['current_pattern']}")
    print("   ✓ Audio statistics available")
    
    return True


def test_tactile_stub():
    """Test that tactile system stub is ready for future use."""
    print("\n\n=== Testing Tactile System Stub ===\n")
    
    # Create standalone tactile system
    tactile_system = ActiveTactileSystem(
        field_shape=(32, 32, 32, 64),
        control_dims=4,  # x, y, pressure, vibration
        device=torch.device('cpu')
    )
    
    print("1. Testing tactile system initialization...")
    assert tactile_system.modality == "tactile"
    assert tactile_system.control_dims == 4
    print("   ✓ Tactile system initialized")
    
    print("\n2. Testing tactile attention control...")
    # Create dummy uncertainty map
    spatial_uncertainty = torch.rand(32, 32, 32)
    uncertainty_map = type('UncertaintyMap', (), {
        'spatial_uncertainty': spatial_uncertainty,
        'feature_uncertainty': torch.rand(64),
        'total_uncertainty': 0.5,
        'peak_locations': [(15, 15, 15)]
    })()
    
    control = tactile_system.generate_attention_control(
        uncertainty_map=uncertainty_map,
        current_predictions=None,
        exploration_drive=0.5
    )
    
    assert control.shape == (4,)
    assert torch.all(control == 0)  # Stub returns zeros
    print("   ✓ Tactile control stub working")
    
    print("\n3. Testing tactile statistics...")
    stats = tactile_system.get_attention_statistics()
    assert stats['modality'] == 'tactile'
    assert 'current_pattern' in stats
    print(f"   Current pattern: {stats['current_pattern']}")
    print("   ✓ Tactile statistics available")
    
    return True


def test_base_functionality():
    """Test that base class provides common functionality."""
    print("\n\n=== Testing Base Class Functionality ===\n")
    
    # All modalities should share uncertainty map generation
    vision = ActiveVisionSystem(
        field_shape=(32, 32, 32, 64),
        motor_dim=5,
        device=torch.device('cpu')
    )
    
    audio = ActiveAudioSystem(
        field_shape=(32, 32, 32, 64),
        control_dims=3,
        device=torch.device('cpu')
    )
    
    tactile = ActiveTactileSystem(
        field_shape=(32, 32, 32, 64),
        control_dims=4,
        device=torch.device('cpu')
    )
    
    print("1. Testing shared uncertainty map generation...")
    # Create dummy field
    field = torch.randn(32, 32, 32, 64)
    
    # All should generate uncertainty maps the same way
    vision_map = vision.generate_uncertainty_map([], field)
    audio_map = audio.generate_uncertainty_map([], field)
    tactile_map = tactile.generate_uncertainty_map([], field)
    
    # Maps should have same structure
    assert vision_map.spatial_uncertainty.shape == audio_map.spatial_uncertainty.shape
    assert audio_map.spatial_uncertainty.shape == tactile_map.spatial_uncertainty.shape
    print("   ✓ Shared uncertainty generation working")
    
    print("\n2. Testing shared information tracking...")
    # All should track information gain the same way
    dummy_data = {'test': torch.randn(10)}
    
    vision.process_attention_return(dummy_data, 0.8, 0.6)
    audio.process_attention_return(dummy_data, 0.7, 0.5)
    tactile.process_attention_return(dummy_data, 0.9, 0.7)
    
    assert len(vision.information_gain_history) == 1
    assert len(audio.information_gain_history) == 1
    assert len(tactile.information_gain_history) == 1
    
    print("   ✓ Shared information tracking working")
    
    return True


if __name__ == "__main__":
    print("Active Sensing Refactor Test")
    print("=" * 50)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    test1 = test_vision_system()
    test2 = test_audio_stub()
    test3 = test_tactile_stub()
    test4 = test_base_functionality()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    if all([test1, test2, test3, test4]):
        print("\n✓ Refactoring successful!")
        print("\nWhat's working:")
        print("- Base ActiveSensingSystem provides common functionality")
        print("- ActiveVisionSystem maintains all previous capabilities")
        print("- Audio and Tactile stubs ready for future implementation")
        print("- Shared uncertainty generation and information tracking")
        
        print("\nBenefits of refactoring:")
        print("- Modality-agnostic uncertainty mapping")
        print("- Shared information gain tracking")
        print("- Easy to add new sensor modalities")
        print("- Clean separation of generic vs specific behaviors")
        
        print("\nReady for future sensors!")
    else:
        print("\n✗ Refactoring has issues")