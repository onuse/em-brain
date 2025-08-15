#!/usr/bin/env python3
"""
Test Universal Memory System with different signal modalities
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import numpy as np
import cv2
import time
from src.universal_memory import UniversalMemorySystem, ModalityType
from src.universal_attention import UniversalAttentionSystem

def test_universal_memory():
    """Test universal memory system with different signal types"""
    print("🧠 Testing Universal Memory System")
    print("=" * 50)
    
    # Create memory and attention systems
    memory_system = UniversalMemorySystem(memory_capacity=100)
    attention_system = UniversalAttentionSystem()
    
    # Test 1: Visual memory formation
    print("\n📷 Testing Visual Memory Formation")
    visual_signal = create_test_image(64, 64)
    
    attention_map, windows = attention_system.calculate_attention_map(
        visual_signal, ModalityType.VISUAL, novelty_score=0.8
    )
    
    brain_output = np.random.uniform(0, 1, 16)  # Simulated brain response
    
    visual_memory = memory_system.form_memory(
        visual_signal, ModalityType.VISUAL, attention_map, brain_output,
        metadata={'source': 'test_image', 'complexity': 'medium'}
    )
    
    if visual_memory:
        print(f"   ✅ Visual memory formed: {visual_memory.pattern_id}")
        print(f"   Attention weight: {visual_memory.attention_weight:.3f}")
        print(f"   Novelty score: {visual_memory.novelty_score:.3f}")
        print(f"   Sparse pattern size: {np.sum(visual_memory.sparse_representation != 0)}")
        print(f"   Metadata keys: {list(visual_memory.metadata.keys())}")
    else:
        print("   ❌ Visual memory not formed")
    
    # Test 2: Audio memory formation
    print("\n🎵 Testing Audio Memory Formation")
    audio_signal = create_test_audio(1000)
    
    attention_map, windows = attention_system.calculate_attention_map(
        audio_signal, ModalityType.AUDIO, novelty_score=0.9
    )
    
    audio_memory = memory_system.form_memory(
        audio_signal, ModalityType.AUDIO, attention_map, brain_output,
        metadata={'source': 'test_audio', 'frequency': '440Hz'}
    )
    
    if audio_memory:
        print(f"   ✅ Audio memory formed: {audio_memory.pattern_id}")
        print(f"   Attention weight: {audio_memory.attention_weight:.3f}")
        print(f"   Novelty score: {audio_memory.novelty_score:.3f}")
        print(f"   Sparse pattern size: {np.sum(audio_memory.sparse_representation != 0)}")
        print(f"   Metadata keys: {list(audio_memory.metadata.keys())}")
    else:
        print("   ❌ Audio memory not formed")
    
    # Test 3: Spectrogram memory formation
    print("\n🎶 Testing Spectrogram Memory Formation")
    spectrogram = create_test_spectrogram(32, 100)
    
    attention_map, windows = attention_system.calculate_attention_map(
        spectrogram, ModalityType.AUDIO, novelty_score=0.7
    )
    
    spectrogram_memory = memory_system.form_memory(
        spectrogram, ModalityType.AUDIO, attention_map, brain_output,
        metadata={'source': 'test_spectrogram', 'type': '2D'}
    )
    
    if spectrogram_memory:
        print(f"   ✅ Spectrogram memory formed: {spectrogram_memory.pattern_id}")
        print(f"   Attention weight: {spectrogram_memory.attention_weight:.3f}")
        print(f"   Novelty score: {spectrogram_memory.novelty_score:.3f}")
        print(f"   Sparse pattern size: {np.sum(spectrogram_memory.sparse_representation != 0)}")
    else:
        print("   ❌ Spectrogram memory not formed")
    
    # Test 3b: Tactile memory formation
    print("\n👆 Testing Tactile Memory Formation")
    tactile_signal = create_test_tactile_1d(50)
    
    attention_map, windows = attention_system.calculate_attention_map(
        tactile_signal, ModalityType.TACTILE, novelty_score=0.8
    )
    
    tactile_memory = memory_system.form_memory(
        tactile_signal, ModalityType.TACTILE, attention_map, brain_output,
        metadata={'source': 'test_tactile', 'sensor_type': 'pressure'}
    )
    
    if tactile_memory:
        print(f"   ✅ Tactile memory formed: {tactile_memory.pattern_id}")
        print(f"   Attention weight: {tactile_memory.attention_weight:.3f}")
        print(f"   Novelty score: {tactile_memory.novelty_score:.3f}")
        print(f"   Sparse pattern size: {np.sum(tactile_memory.sparse_representation != 0)}")
        print(f"   Metadata keys: {list(tactile_memory.metadata.keys())}")
    else:
        print("   ❌ Tactile memory not formed")
    
    # Test 3c: Motor memory formation
    print("\n🤖 Testing Motor Memory Formation")
    motor_signal = create_test_motor_1d(30)
    
    attention_map, windows = attention_system.calculate_attention_map(
        motor_signal, ModalityType.MOTOR, novelty_score=0.9
    )
    
    motor_memory = memory_system.form_memory(
        motor_signal, ModalityType.MOTOR, attention_map, brain_output,
        metadata={'source': 'test_motor', 'joint_count': 6}
    )
    
    if motor_memory:
        print(f"   ✅ Motor memory formed: {motor_memory.pattern_id}")
        print(f"   Attention weight: {motor_memory.attention_weight:.3f}")
        print(f"   Novelty score: {motor_memory.novelty_score:.3f}")
        print(f"   Sparse pattern size: {np.sum(motor_memory.sparse_representation != 0)}")
        print(f"   Metadata keys: {list(motor_memory.metadata.keys())}")
    else:
        print("   ❌ Motor memory not formed")
    
    # Test 3d: Temporal memory formation
    print("\n🎵 Testing Temporal Memory Formation")
    temporal_signal = create_test_temporal_1d(100)
    
    attention_map, windows = attention_system.calculate_attention_map(
        temporal_signal, ModalityType.TEMPORAL, novelty_score=0.6
    )
    
    temporal_memory = memory_system.form_memory(
        temporal_signal, ModalityType.TEMPORAL, attention_map, brain_output,
        metadata={'source': 'test_temporal', 'rhythm_type': 'beat'}
    )
    
    if temporal_memory:
        print(f"   ✅ Temporal memory formed: {temporal_memory.pattern_id}")
        print(f"   Attention weight: {temporal_memory.attention_weight:.3f}")
        print(f"   Novelty score: {temporal_memory.novelty_score:.3f}")
        print(f"   Sparse pattern size: {np.sum(temporal_memory.sparse_representation != 0)}")
        print(f"   Metadata keys: {list(temporal_memory.metadata.keys())}")
    else:
        print("   ❌ Temporal memory not formed")
    
    # Test 4: Low attention - should not form memory
    print("\n⚠️  Testing Low Attention (Should Not Form Memory)")
    low_attention_map = np.ones_like(attention_map) * 0.1  # Very low attention
    
    no_memory = memory_system.form_memory(
        visual_signal, ModalityType.VISUAL, low_attention_map, brain_output
    )
    
    if no_memory:
        print("   ❌ Memory formed despite low attention")
    else:
        print("   ✅ Memory correctly rejected due to low attention")
    
    # Test 5: Memory retrieval
    print("\n🔍 Testing Memory Retrieval")
    
    # Retrieve all patterns
    all_patterns = memory_system.retrieve_patterns()
    print(f"   Total patterns: {len(all_patterns)}")
    
    # Retrieve visual patterns only
    visual_patterns = memory_system.retrieve_patterns(modality=ModalityType.VISUAL)
    print(f"   Visual patterns: {len(visual_patterns)}")
    
    # Retrieve audio patterns only
    audio_patterns = memory_system.retrieve_patterns(modality=ModalityType.AUDIO)
    print(f"   Audio patterns: {len(audio_patterns)}")
    
    # Retrieve tactile patterns only
    tactile_patterns = memory_system.retrieve_patterns(modality=ModalityType.TACTILE)
    print(f"   Tactile patterns: {len(tactile_patterns)}")
    
    # Retrieve motor patterns only
    motor_patterns = memory_system.retrieve_patterns(modality=ModalityType.MOTOR)
    print(f"   Motor patterns: {len(motor_patterns)}")
    
    # Retrieve temporal patterns only
    temporal_patterns = memory_system.retrieve_patterns(modality=ModalityType.TEMPORAL)
    print(f"   Temporal patterns: {len(temporal_patterns)}")
    
    # Retrieve high attention patterns
    high_attention = memory_system.retrieve_patterns(min_attention=0.7)
    print(f"   High attention patterns: {len(high_attention)}")
    
    # Test 6: Cross-modal associations
    print("\n🔗 Testing Cross-Modal Associations")
    
    # Create multiple patterns with similar brain responses
    for i in range(5):
        # Similar brain response for association
        similar_brain = brain_output + np.random.normal(0, 0.1, 16)
        
        # Create visual pattern
        visual_signal_2 = create_test_image(64, 64)
        attention_map_2, _ = attention_system.calculate_attention_map(
            visual_signal_2, ModalityType.VISUAL, novelty_score=0.6
        )
        
        memory_system.form_memory(
            visual_signal_2, ModalityType.VISUAL, attention_map_2, similar_brain,
            metadata={'batch': i}
        )
        
        # Create audio pattern
        audio_signal_2 = create_test_audio(1000)
        attention_map_3, _ = attention_system.calculate_attention_map(
            audio_signal_2, ModalityType.AUDIO, novelty_score=0.6
        )
        
        memory_system.form_memory(
            audio_signal_2, ModalityType.AUDIO, attention_map_3, similar_brain,
            metadata={'batch': i}
        )
    
    # Check for cross-modal associations
    all_patterns = memory_system.retrieve_patterns()
    total_associations = 0
    for pattern in all_patterns:
        associations = memory_system.get_cross_modal_associations(pattern.pattern_id)
        total_associations += len(associations)
        
        if associations:
            print(f"   Pattern {pattern.pattern_id[:20]}... has {len(associations)} associations")
    
    print(f"   Total cross-modal associations: {total_associations}")
    
    # Test 7: Memory statistics
    print("\n📊 Testing Memory Statistics")
    stats = memory_system.get_statistics()
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    # Test 8: Memory persistence
    print("\n💾 Testing Memory Persistence")
    
    # Save memories
    save_path = "/tmp/test_memories.json"
    memory_system.save_memories(save_path)
    
    # Create new system and load
    new_memory_system = UniversalMemorySystem()
    new_memory_system.load_memories(save_path)
    
    # Compare
    original_count = len(memory_system.memory_patterns)
    loaded_count = len(new_memory_system.memory_patterns)
    
    print(f"   Original patterns: {original_count}")
    print(f"   Loaded patterns: {loaded_count}")
    print(f"   Persistence test: {'✅ Passed' if original_count == loaded_count else '❌ Failed'}")
    
    # Test 9: Memory consolidation
    print("\n🗂️  Testing Memory Consolidation")
    
    # Fill memory beyond capacity
    consolidation_system = UniversalMemorySystem(memory_capacity=10)
    
    patterns_created = 0
    for i in range(15):  # Create more than capacity
        test_signal = create_test_image(32, 32)
        attention_map, _ = attention_system.calculate_attention_map(
            test_signal, ModalityType.VISUAL, novelty_score=0.5
        )
        
        pattern = consolidation_system.form_memory(
            test_signal, ModalityType.VISUAL, attention_map
        )
        
        if pattern:
            patterns_created += 1
    
    final_count = len(consolidation_system.memory_patterns)
    print(f"   Patterns created: {patterns_created}")
    print(f"   Final stored patterns: {final_count}")
    print(f"   Consolidation triggered: {'✅ Yes' if final_count <= 10 else '❌ No'}")
    
    # Test 10: Performance test
    print("\n⚡ Performance Test")
    
    performance_system = UniversalMemorySystem()
    formation_times = []
    
    for i in range(20):
        test_signal = create_test_image(64, 64)
        attention_map, _ = attention_system.calculate_attention_map(
            test_signal, ModalityType.VISUAL, novelty_score=0.6
        )
        
        start_time = time.time()
        pattern = performance_system.form_memory(
            test_signal, ModalityType.VISUAL, attention_map
        )
        formation_time = (time.time() - start_time) * 1000  # ms
        
        if pattern:
            formation_times.append(formation_time)
    
    if formation_times:
        avg_time = np.mean(formation_times)
        max_time = np.max(formation_times)
        print(f"   Average formation time: {avg_time:.2f}ms")
        print(f"   Max formation time: {max_time:.2f}ms")
        print(f"   Real-time capable: {'✅ Yes' if avg_time < 10 else '⚠️ Marginal' if avg_time < 20 else '❌ No'}")
    
    # Cleanup
    memory_system.cleanup()
    new_memory_system.cleanup()
    consolidation_system.cleanup()
    performance_system.cleanup()
    
    print(f"\n🎉 Universal Memory System Test Complete!")
    
    # Final assessment
    success_criteria = [
        visual_memory is not None,
        audio_memory is not None,
        spectrogram_memory is not None,
        tactile_memory is not None,
        motor_memory is not None,
        temporal_memory is not None,
        no_memory is None,
        len(all_patterns) > 0,
        original_count == loaded_count,
        final_count <= 10,
        avg_time < 20 if formation_times else True
    ]
    
    passed = sum(success_criteria)
    total = len(success_criteria)
    
    print(f"   ✅ Tests passed: {passed}/{total}")
    print(f"   🎯 Overall success: {'✅ Yes' if passed >= total * 0.8 else '❌ No'}")
    
    return passed >= total * 0.8

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
    success = test_universal_memory()
    if success:
        print("\n✅ Universal Memory System ready for deployment!")
    else:
        print("\n⚠️  System needs further optimization")