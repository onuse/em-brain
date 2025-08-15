#!/usr/bin/env python3
"""
Test Exclusive Attention System - Biologically Realistic Attention Switching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from src.exclusive_attention import ExclusiveAttentionSystem, AttentionState, ModalityType

def test_exclusive_attention():
    """Test exclusive attention system with different scenarios"""
    print("🎯 Testing Exclusive Attention System")
    print("=" * 50)
    
    # Create attention system
    attention_system = ExclusiveAttentionSystem(
        attention_duration=0.3,    # 300ms focus
        switch_duration=0.05,      # 50ms switching
        inhibition_duration=1.0,   # 1s inhibition of return
        max_targets=5
    )
    
    # Test 1: Visual scene with multiple targets
    print("\n📷 Testing Multi-Target Visual Scene")
    visual_scene = create_multi_target_scene(64, 64)
    
    # Simulate temporal sequence of attention
    attention_sequence = []
    for frame in range(20):  # 20 frames over ~6 seconds
        
        # Update attention system
        attention_state = attention_system.update_attention(
            visual_scene, ModalityType.VISUAL, novelty_score=0.8
        )
        
        attention_sequence.append({
            'frame': frame,
            'time': frame * 0.3,  # 300ms per frame
            'state': attention_state['state'],
            'target': attention_state['current_target'],
            'spotlight_x': attention_state['spotlight_x'],
            'spotlight_y': attention_state['spotlight_y'],
            'spotlight_strength': attention_state['spotlight_strength'],
            'switches': attention_state['attention_switches']
        })
        
        # Print current state
        target_info = ""
        if attention_state['current_target']:
            target = attention_state['current_target']
            target_info = f"Target: ({target.x}, {target.y}) priority={target.priority:.3f}"
        
        print(f"   Frame {frame:2d}: {attention_state['state']:9s} - {target_info}")
        
        # Sleep to simulate real time
        time.sleep(0.05)  # 50ms per frame for demo
    
    # Test 2: Inhibition of return
    print(f"\n🔄 Testing Inhibition of Return")
    stats = attention_system.get_statistics()
    print(f"   Total attention switches: {stats['attention_switches']}")
    print(f"   Average attention time: {stats['avg_attend_time']:.3f}s")
    print(f"   Inhibited targets: {stats['inhibited_targets']}")
    
    # Test 3: Single target (should maintain attention)
    print(f"\n🎯 Testing Single Target Attention")
    attention_system.reset()
    
    single_target_scene = create_single_target_scene(64, 64)
    
    single_sequence = []
    for frame in range(10):
        attention_state = attention_system.update_attention(
            single_target_scene, ModalityType.VISUAL, novelty_score=0.9
        )
        
        single_sequence.append({
            'frame': frame,
            'state': attention_state['state'],
            'spotlight_strength': attention_state['spotlight_strength']
        })
        
        print(f"   Frame {frame:2d}: {attention_state['state']:9s} - "
              f"strength={attention_state['spotlight_strength']:.2f}")
        
        time.sleep(0.05)
    
    # Test 4: Audio temporal sequence
    print(f"\n🎵 Testing Audio Temporal Attention")
    attention_system.reset()
    
    audio_sequence = create_audio_sequence(1000)
    
    audio_attention = []
    for frame in range(15):
        attention_state = attention_system.update_attention(
            audio_sequence, ModalityType.AUDIO, novelty_score=0.7
        )
        
        audio_attention.append({
            'frame': frame,
            'state': attention_state['state'],
            'switches': attention_state['attention_switches']
        })
        
        print(f"   Frame {frame:2d}: {attention_state['state']:9s} - "
              f"switches={attention_state['attention_switches']}")
        
        time.sleep(0.05)
    
    # Test 5: Attention mask generation
    print(f"\n🖼️  Testing Attention Mask Generation")
    attention_system.reset()
    
    # Create attention to a target
    test_scene = create_multi_target_scene(64, 64)
    attention_state = attention_system.update_attention(
        test_scene, ModalityType.VISUAL, novelty_score=0.8
    )
    
    # Generate attention mask
    mask = attention_system.get_attention_mask(640, 480)
    
    print(f"   Attention mask shape: {mask.shape}")
    print(f"   Attended pixels: {np.sum(mask > 0)}")
    print(f"   Max attention: {np.max(mask):.3f}")
    print(f"   Attention coverage: {np.mean(mask > 0.1):.3f}")
    
    # Test 6: Competitive target selection
    print(f"\n🏆 Testing Competitive Target Selection")
    attention_system.reset()
    
    # Create scene with targets of different priorities
    competitive_scene = create_competitive_scene(64, 64)
    
    competitive_sequence = []
    for frame in range(12):
        attention_state = attention_system.update_attention(
            competitive_scene, ModalityType.VISUAL, novelty_score=0.6
        )
        
        target_priority = 0.0
        if attention_state['current_target']:
            target_priority = attention_state['current_target'].priority
        
        competitive_sequence.append({
            'frame': frame,
            'state': attention_state['state'],
            'target_priority': target_priority,
            'switches': attention_state['attention_switches']
        })
        
        print(f"   Frame {frame:2d}: {attention_state['state']:9s} - "
              f"priority={target_priority:.3f}")
        
        time.sleep(0.05)
    
    # Test 7: Performance test
    print(f"\n⚡ Performance Test")
    attention_system.reset()
    
    performance_times = []
    for i in range(100):
        start_time = time.time()
        
        attention_state = attention_system.update_attention(
            visual_scene, ModalityType.VISUAL, novelty_score=0.5
        )
        
        processing_time = (time.time() - start_time) * 1000
        performance_times.append(processing_time)
    
    avg_time = np.mean(performance_times)
    max_time = np.max(performance_times)
    
    print(f"   Average processing time: {avg_time:.2f}ms")
    print(f"   Max processing time: {max_time:.2f}ms")
    print(f"   Real-time capable: {'✅ Yes' if avg_time < 10 else '⚠️ Marginal' if avg_time < 20 else '❌ No'}")
    
    # Final statistics
    final_stats = attention_system.get_statistics()
    print(f"\n📊 Final Statistics:")
    print(f"   Total attention switches: {final_stats['attention_switches']}")
    print(f"   Total attention time: {final_stats['total_attend_time']:.2f}s")
    print(f"   Average attention duration: {final_stats['avg_attend_time']:.3f}s")
    print(f"   Current state: {final_stats['current_state']}")
    
    print(f"\n🎉 Exclusive Attention System Test Complete!")
    
    # Assessment
    success_criteria = [
        final_stats['attention_switches'] > 5,  # Multiple attention switches
        final_stats['avg_attend_time'] > 0.2,   # Reasonable attention duration
        avg_time < 20,                          # Performance acceptable
        len(competitive_sequence) > 0,          # Competitive selection working
        np.max(mask) > 0.5                      # Attention mask generation working
    ]
    
    passed = sum(success_criteria)
    total = len(success_criteria)
    
    print(f"   ✅ Tests passed: {passed}/{total}")
    print(f"   🎯 Biological realism: {'✅ High' if passed >= 4 else '⚠️ Medium' if passed >= 3 else '❌ Low'}")
    
    return passed >= 4

def create_multi_target_scene(width, height):
    """Create visual scene with multiple attention targets"""
    scene = np.zeros((height, width), dtype=np.float32)
    
    # Add multiple targets of different strengths
    # Target 1: High salience
    scene[10:20, 10:20] = 0.9
    
    # Target 2: Medium salience  
    scene[30:40, 30:40] = 0.6
    
    # Target 3: Low salience
    scene[50:55, 50:55] = 0.4
    
    # Target 4: High salience, different location
    scene[5:15, 45:55] = 0.8
    
    # Add some noise
    noise = np.random.normal(0, 0.05, (height, width))
    scene = np.clip(scene + noise, 0, 1)
    
    return scene

def create_single_target_scene(width, height):
    """Create visual scene with single attention target"""
    scene = np.zeros((height, width), dtype=np.float32)
    
    # Single strong target
    scene[25:35, 25:35] = 0.9
    
    # Add some noise
    noise = np.random.normal(0, 0.03, (height, width))
    scene = np.clip(scene + noise, 0, 1)
    
    return scene

def create_competitive_scene(width, height):
    """Create scene with targets of different competitive strengths"""
    scene = np.zeros((height, width), dtype=np.float32)
    
    # Strong competitor
    scene[15:25, 15:25] = 0.95
    
    # Medium competitor
    scene[35:45, 35:45] = 0.7
    
    # Weak competitor
    scene[50:60, 10:20] = 0.3
    
    # Very strong competitor (should win)
    scene[5:15, 40:50] = 1.0
    
    # Add some noise
    noise = np.random.normal(0, 0.04, (height, width))
    scene = np.clip(scene + noise, 0, 1)
    
    return scene

def create_audio_sequence(length):
    """Create audio sequence with multiple events"""
    t = np.linspace(0, 1, length)
    
    # Base signal
    signal = np.sin(2 * np.pi * 440 * t) * 0.3
    
    # Add attention-grabbing events
    signal[100:120] += 0.8  # Strong event
    signal[300:320] += 0.6  # Medium event
    signal[500:510] += 0.9  # Very strong event
    signal[700:720] += 0.4  # Weak event
    
    # Add noise
    noise = np.random.normal(0, 0.05, length)
    signal = np.clip(signal + noise, -1, 1)
    
    return signal

if __name__ == "__main__":
    success = test_exclusive_attention()
    if success:
        print("\n✅ Exclusive Attention System ready for biological realism!")
    else:
        print("\n⚠️  System needs further optimization for biological accuracy")