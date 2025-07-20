#!/usr/bin/env python3
"""
Test Field Brain Persistence

Validate the intrinsic field persistence methods in UnifiedFieldBrain.
Tests field state save/load, delta compression, and consolidation.
"""

import sys
import os
import time
import tempfile
import torch
import numpy as np

# Add path for imports
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server/src'))

try:
    from field_native_brain import create_unified_field_brain
    from field_native_robot_interface import create_field_native_robot_interface
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_field_state_persistence():
    """Test basic field state save and load."""
    print("🧠 TESTING FIELD STATE PERSISTENCE")
    
    # Create field brain and add some activity
    brain1 = create_unified_field_brain(spatial_resolution=8, quiet_mode=True)
    
    # Simulate some robot experiences to create field patterns
    for i in range(10):
        # Create varied sensory input
        sensory_input = [
            0.5 + 0.3 * np.sin(i),  # X position
            0.2 + 0.2 * np.cos(i),  # Y position
            0.0,                    # Z position
            0.8 + 0.1 * np.sin(i*2), # Distance forward
            0.5, 0.5,              # Distance left, right
            0.7 + 0.2 * np.sin(i/2), # Red
            0.3 + 0.1 * np.cos(i/2), # Green
            0.5,                    # Blue
            0.4 + 0.2 * np.sin(i*3), # Audio
            0.6,                    # Temperature
            0.8,                    # Battery
        ] + [0.5] * 13  # Fill remaining sensors
        
        # Process robot cycle
        motor_commands, brain_state = brain1.process_robot_cycle(sensory_input)
    
    # Get original field stats
    original_stats = brain1.get_field_memory_stats()
    original_field = brain1.unified_field.clone()
    
    print(f"   Original field stats:")
    print(f"      Memory size: {original_stats['memory_size_mb']:.2f} MB")
    print(f"      Nonzero elements: {original_stats['nonzero_elements']:,}")
    print(f"      Field energy: {original_stats['field_energy']:.3f}")
    print(f"      Brain cycles: {original_stats['brain_cycles']}")
    
    # Test field state save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.field.gz') as tmp_file:
        field_file = tmp_file.name
    
    print(f"\n   Testing field save...")
    save_success = brain1.save_field_state(field_file, compress=True)
    print(f"   Save success: {'✅ YES' if save_success else '❌ NO'}")
    
    if save_success:
        file_size_mb = os.path.getsize(field_file) / (1024 * 1024)
        compression_ratio = original_stats['memory_size_mb'] / file_size_mb if file_size_mb > 0 else 1.0
        print(f"   File size: {file_size_mb:.2f} MB")
        print(f"   Compression ratio: {compression_ratio:.1f}x")
    
    # Test field state load
    print(f"\n   Testing field load...")
    brain2 = create_unified_field_brain(spatial_resolution=8, quiet_mode=True)
    load_success = brain2.load_field_state(field_file)
    print(f"   Load success: {'✅ YES' if load_success else '❌ NO'}")
    
    if load_success:
        loaded_stats = brain2.get_field_memory_stats()
        
        # Verify field integrity
        field_match = torch.allclose(original_field, brain2.unified_field, atol=1e-6)
        stats_match = (
            original_stats['nonzero_elements'] == loaded_stats['nonzero_elements'] and
            abs(original_stats['field_energy'] - loaded_stats['field_energy']) < 1e-3 and
            original_stats['brain_cycles'] == loaded_stats['brain_cycles']
        )
        
        print(f"   Field tensor match: {'✅ YES' if field_match else '❌ NO'}")
        print(f"   Statistics match: {'✅ YES' if stats_match else '❌ NO'}")
        print(f"   Loaded field energy: {loaded_stats['field_energy']:.3f}")
        print(f"   Loaded brain cycles: {loaded_stats['brain_cycles']}")
        
        persistence_success = save_success and load_success and field_match and stats_match
    else:
        persistence_success = False
    
    # Cleanup
    try:
        os.unlink(field_file)
    except:
        pass
    
    return {
        'save_success': save_success,
        'load_success': load_success,
        'persistence_success': persistence_success,
        'original_stats': original_stats
    }


def test_field_delta_compression():
    """Test incremental field updates with delta compression."""
    print("\n🔄 TESTING FIELD DELTA COMPRESSION")
    
    # Create field brain
    brain = create_unified_field_brain(spatial_resolution=8, quiet_mode=True)
    
    # Create initial field state
    for i in range(5):
        sensory_input = [0.5] * 24
        brain.process_robot_cycle(sensory_input)
    
    initial_field = brain.unified_field.clone()
    
    # Make small changes
    for i in range(3):
        sensory_input = [
            0.5 + 0.1 * np.sin(i),
            0.5 + 0.1 * np.cos(i),
            0.0
        ] + [0.5] * 21
        brain.process_robot_cycle(sensory_input)
    
    # Test delta save
    with tempfile.NamedTemporaryFile(delete=False, suffix='.delta.gz') as tmp_file:
        delta_file = tmp_file.name
    
    print(f"   Testing delta save...")
    delta_save_success = brain.save_field_delta(delta_file, initial_field)
    print(f"   Delta save success: {'✅ YES' if delta_save_success else '❌ NO'}")
    
    if delta_save_success:
        delta_size_kb = os.path.getsize(delta_file) / 1024
        print(f"   Delta file size: {delta_size_kb:.2f} KB")
        
        # Calculate compression ratio
        full_field_size = brain.unified_field.element_size() * brain.unified_field.nelement()
        compression_ratio = full_field_size / os.path.getsize(delta_file) if delta_size_kb > 0 else 1.0
        print(f"   Delta compression: {compression_ratio:.1f}x")
    
    # Test delta application
    print(f"\n   Testing delta application...")
    
    # Reset brain to initial state
    brain.unified_field = initial_field.clone()
    modified_field = brain.unified_field.clone()
    
    # Apply delta
    delta_apply_success = brain.apply_field_delta(delta_file)
    print(f"   Delta apply success: {'✅ YES' if delta_apply_success else '❌ NO'}")
    
    # Verify delta application
    if delta_apply_success:
        change_magnitude = torch.norm(brain.unified_field - modified_field).item()
        print(f"   Applied change magnitude: {change_magnitude:.6f}")
        
        delta_success = delta_save_success and delta_apply_success and change_magnitude > 1e-6
    else:
        delta_success = False
    
    # Cleanup
    try:
        os.unlink(delta_file)
    except:
        pass
    
    return {
        'delta_save_success': delta_save_success,
        'delta_apply_success': delta_apply_success,
        'delta_success': delta_success
    }


def test_field_consolidation():
    """Test field consolidation and memory strengthening."""
    print("\n🌙 TESTING FIELD CONSOLIDATION")
    
    # Create field brain with some activity
    brain = create_unified_field_brain(spatial_resolution=8, quiet_mode=True)
    
    # Create varied patterns to establish field regions
    for i in range(15):
        # Create patterns with different strengths
        strength = 0.3 + 0.5 * (i % 3)  # Three strength levels
        sensory_input = [
            strength * np.sin(i * 0.5),
            strength * np.cos(i * 0.5),
            0.1 * i / 15
        ] + [strength] * 21
        
        brain.process_robot_cycle(sensory_input)
    
    # Get field stats before consolidation
    pre_stats = brain.get_field_memory_stats()
    pre_energy = pre_stats['field_energy']
    pre_max = pre_stats['max_activation']
    pre_nonzero = pre_stats['nonzero_elements']
    
    print(f"   Pre-consolidation:")
    print(f"      Field energy: {pre_energy:.3f}")
    print(f"      Max activation: {pre_max:.3f}")
    print(f"      Nonzero elements: {pre_nonzero:,}")
    print(f"      Topology regions: {pre_stats['topology_regions']}")
    
    # Perform consolidation
    print(f"\n   Performing field consolidation...")
    consolidated_regions = brain.consolidate_field(consolidation_strength=0.15)
    
    # Get field stats after consolidation
    post_stats = brain.get_field_memory_stats()
    post_energy = post_stats['field_energy']
    post_max = post_stats['max_activation']
    post_nonzero = post_stats['nonzero_elements']
    
    print(f"\n   Post-consolidation:")
    print(f"      Field energy: {post_energy:.3f}")
    print(f"      Max activation: {post_max:.3f}")
    print(f"      Nonzero elements: {post_nonzero:,}")
    print(f"      Topology regions: {post_stats['topology_regions']}")
    print(f"      Consolidated regions: {consolidated_regions:,}")
    
    # Analyze consolidation effects
    energy_change = (post_energy - pre_energy) / pre_energy if pre_energy > 0 else 0
    max_change = (post_max - pre_max) / pre_max if pre_max > 0 else 0
    
    print(f"\n   Consolidation analysis:")
    print(f"      Energy change: {energy_change:.3f} ({energy_change*100:.1f}%)")
    print(f"      Max activation change: {max_change:.3f} ({max_change*100:.1f}%)")
    
    consolidation_success = (
        consolidated_regions > 0 and
        abs(energy_change) > 0.001  # Some change occurred
    )
    
    return {
        'consolidated_regions': consolidated_regions,
        'energy_change': energy_change,
        'max_change': max_change,
        'consolidation_success': consolidation_success
    }


def test_session_persistence():
    """Test field persistence across simulated robot sessions."""
    print("\n🔄 TESTING SESSION PERSISTENCE")
    
    # Session 1: Learn some patterns
    print(f"   Session 1: Learning patterns...")
    brain1 = create_unified_field_brain(spatial_resolution=8, quiet_mode=True)
    
    # Learn pattern A (forward movement with visual input)
    for i in range(8):
        sensory_input = [
            0.8, 0.0, 0.0,  # Forward position
            0.9, 0.2, 0.2,  # Distance sensors
            0.8, 0.3, 0.1,  # Visual (reddish)
            0.6,            # Audio
        ] + [0.5] * 14
        brain1.process_robot_cycle(sensory_input)
    
    # Learn pattern B (turning with different visual)
    for i in range(8):
        sensory_input = [
            0.2, 0.8, 0.0,  # Side position
            0.3, 0.9, 0.3,  # Distance sensors
            0.1, 0.8, 0.2,  # Visual (greenish)
            0.4,            # Audio
        ] + [0.5] * 14
        brain1.process_robot_cycle(sensory_input)
    
    session1_stats = brain1.get_field_memory_stats()
    
    # Save session 1 state
    with tempfile.NamedTemporaryFile(delete=False, suffix='.session1.gz') as tmp_file:
        session1_file = tmp_file.name
    
    save_success = brain1.save_field_state(session1_file)
    
    print(f"      Session 1 field energy: {session1_stats['field_energy']:.3f}")
    print(f"      Session 1 brain cycles: {session1_stats['brain_cycles']}")
    print(f"      Save success: {'✅ YES' if save_success else '❌ NO'}")
    
    # Session 2: Load and continue learning
    print(f"\n   Session 2: Loading and continuing...")
    brain2 = create_unified_field_brain(spatial_resolution=8, quiet_mode=True)
    load_success = brain2.load_field_state(session1_file)
    
    if load_success:
        # Continue with new pattern C
        for i in range(6):
            sensory_input = [
                0.5, 0.5, 0.5,  # Center position
                0.5, 0.5, 0.5,  # Balanced sensors
                0.5, 0.5, 0.8,  # Visual (bluish)
                0.8,            # Audio
            ] + [0.5] * 14
            brain2.process_robot_cycle(sensory_input)
        
        session2_stats = brain2.get_field_memory_stats()
        
        print(f"      Load success: {'✅ YES' if load_success else '❌ NO'}")
        print(f"      Session 2 field energy: {session2_stats['field_energy']:.3f}")
        print(f"      Session 2 brain cycles: {session2_stats['brain_cycles']}")
        print(f"      Cycle continuity: {'✅ YES' if session2_stats['brain_cycles'] > session1_stats['brain_cycles'] else '❌ NO'}")
        
        # Test pattern recall
        print(f"\n   Testing pattern recall...")
        
        # Present pattern A stimulus
        pattern_a_input = [0.8, 0.0, 0.0, 0.9, 0.2, 0.2, 0.8, 0.3, 0.1, 0.6] + [0.5] * 14
        motor_a, state_a = brain2.process_robot_cycle(pattern_a_input)
        
        # Present pattern B stimulus
        pattern_b_input = [0.2, 0.8, 0.0, 0.3, 0.9, 0.3, 0.1, 0.8, 0.2, 0.4] + [0.5] * 14
        motor_b, state_b = brain2.process_robot_cycle(pattern_b_input)
        
        # Check if patterns produce different responses
        motor_diff = np.linalg.norm(np.array(motor_a) - np.array(motor_b))
        pattern_discrimination = motor_diff > 0.1
        
        print(f"      Pattern A motor: {motor_a}")
        print(f"      Pattern B motor: {motor_b}")
        print(f"      Motor difference: {motor_diff:.3f}")
        print(f"      Pattern discrimination: {'✅ YES' if pattern_discrimination else '⚠️ WEAK'}")
        
        session_success = (
            save_success and load_success and
            session2_stats['brain_cycles'] > session1_stats['brain_cycles'] and
            session2_stats['field_energy'] > session1_stats['field_energy'] * 0.5
        )
    else:
        session_success = False
        pattern_discrimination = False
    
    # Cleanup
    try:
        os.unlink(session1_file)
    except:
        pass
    
    return {
        'save_success': save_success,
        'load_success': load_success,
        'pattern_discrimination': pattern_discrimination,
        'session_success': session_success
    }


def run_field_persistence_tests():
    """Run comprehensive field persistence tests."""
    print("🧠 COMPREHENSIVE FIELD PERSISTENCE TESTS")
    print("=" * 60)
    
    # Test 1: Basic field state persistence
    persistence_results = test_field_state_persistence()
    
    # Test 2: Delta compression
    delta_results = test_field_delta_compression()
    
    # Test 3: Field consolidation
    consolidation_results = test_field_consolidation()
    
    # Test 4: Session persistence
    session_results = test_session_persistence()
    
    # Overall assessment
    print(f"\n📊 PERSISTENCE TEST RESULTS")
    print(f"=" * 40)
    
    print(f"\n   💾 Field State Persistence:")
    print(f"      Save success: {'✅ YES' if persistence_results['save_success'] else '❌ NO'}")
    print(f"      Load success: {'✅ YES' if persistence_results['load_success'] else '❌ NO'}")
    print(f"      Overall success: {'✅ YES' if persistence_results['persistence_success'] else '❌ NO'}")
    
    print(f"\n   🔄 Delta Compression:")
    print(f"      Delta save: {'✅ YES' if delta_results['delta_save_success'] else '❌ NO'}")
    print(f"      Delta apply: {'✅ YES' if delta_results['delta_apply_success'] else '❌ NO'}")
    print(f"      Overall success: {'✅ YES' if delta_results['delta_success'] else '❌ NO'}")
    
    print(f"\n   🌙 Field Consolidation:")
    print(f"      Consolidated regions: {consolidation_results['consolidated_regions']:,}")
    print(f"      Energy change: {consolidation_results['energy_change']:.3f}")
    print(f"      Overall success: {'✅ YES' if consolidation_results['consolidation_success'] else '❌ NO'}")
    
    print(f"\n   🔄 Session Persistence:")
    print(f"      Session continuity: {'✅ YES' if session_results['session_success'] else '❌ NO'}")
    print(f"      Pattern discrimination: {'✅ YES' if session_results['pattern_discrimination'] else '⚠️ WEAK'}")
    
    # Calculate overall success rate
    success_metrics = [
        persistence_results['persistence_success'],
        delta_results['delta_success'],
        consolidation_results['consolidation_success'],
        session_results['session_success']
    ]
    
    success_count = sum(success_metrics)
    success_rate = success_count / len(success_metrics)
    
    print(f"\n   🌟 OVERALL ASSESSMENT:")
    print(f"      Success metrics: {success_count}/{len(success_metrics)}")
    print(f"      Success rate: {success_rate:.3f}")
    print(f"      Field persistence: {'✅ FULLY FUNCTIONAL' if success_rate >= 0.75 else '⚠️ DEVELOPING'}")
    
    if success_rate >= 0.75:
        print(f"\n🚀 INTRINSIC FIELD PERSISTENCE SUCCESSFULLY IMPLEMENTED!")
        print(f"🎯 Key achievements:")
        print(f"   ✓ Field state save/load with compression")
        print(f"   ✓ Incremental delta compression for efficiency")
        print(f"   ✓ Field consolidation and memory strengthening")
        print(f"   ✓ Session continuity and pattern persistence")
        print(f"   ✓ The field IS the memory - no separate storage needed!")
    else:
        print(f"\n⚠️ Field persistence system needs refinement")
        print(f"🔧 Areas for improvement:")
        if not persistence_results['persistence_success']:
            print(f"   • Basic field state save/load reliability")
        if not delta_results['delta_success']:
            print(f"   • Delta compression and application")
        if not consolidation_results['consolidation_success']:
            print(f"   • Field consolidation effectiveness")
        if not session_results['session_success']:
            print(f"   • Session continuity and pattern recall")
    
    return {
        'persistence_results': persistence_results,
        'delta_results': delta_results,
        'consolidation_results': consolidation_results,
        'session_results': session_results,
        'success_rate': success_rate,
        'fully_functional': success_rate >= 0.75
    }


if __name__ == "__main__":
    results = run_field_persistence_tests()
    
    print(f"\n🔬 FIELD PERSISTENCE TEST COMPLETE")
    print(f"   Success rate: {results['success_rate']:.3f}")
    print(f"   Field persistence: {'✅ ACHIEVED' if results['fully_functional'] else '⚠️ DEVELOPING'}")
    
    if results['fully_functional']:
        print(f"\n🌊 Field-native intrinsic memory persistence is working!")
        print(f"💡 The unified field contains all memory - no separate storage needed!")