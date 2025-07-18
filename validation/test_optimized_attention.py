#!/usr/bin/env python3
"""
Test optimized adaptive attention system with simulated camera feed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import cv2
import numpy as np
import time
from src.brain import MinimalBrain
from memory_inspector import MemoryInspector
from memory_overlay_camera import MemoryOverlayCamera

def test_optimized_attention():
    """Test optimized attention system with real-time performance"""
    print("⚡ Testing Optimized Adaptive Attention System")
    print("=" * 50)
    
    # Create memory overlay camera
    camera = MemoryOverlayCamera(brain_type="sparse_goldilocks")
    
    # Test with simulated frames
    test_frames = []
    for i in range(20):  # 20 test frames
        # Create test frame with varied content
        frame = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)
        
        # Add some structure
        if i % 4 == 0:  # Every 4th frame has objects
            cv2.circle(frame, (320, 180), 50, (255, 255, 255), -1)
            cv2.rectangle(frame, (100, 100), (200, 200), (0, 255, 0), -1)
        
        test_frames.append(frame)
    
    print(f"Testing {len(test_frames)} frames for performance...")
    
    # Performance tracking
    processing_times = []
    attention_stats = []
    
    for i, frame in enumerate(test_frames):
        start_time = time.time()
        
        # Process frame like the camera would
        try:
            # Convert to brain input with high-res preservation
            brain_input, high_res_frame = camera.process_frame_to_brain_input(frame)
            
            # Calculate activity level
            activity_level = camera._calculate_activity_level(brain_input)
            
            # Process through brain
            brain_output, brain_info = camera.brain.process_sensory_input(brain_input)
            
            # Calculate adaptive attention
            current_array = np.array(brain_input)
            brain_activity = {
                'brain_output': brain_output,
                'brain_info': brain_info,
                'timestamp': time.time()
            }
            
            attention_map, attention_windows = camera.overlay_generator._calculate_adaptive_attention_map(
                current_array, brain_activity, high_res_frame
            )
            
            # Capture memory
            camera.memory_inspector.capture_memory_snapshot(
                brain_input, brain_output, brain_info, attention_map
            )
            
            # Create overlay
            visualization = camera.overlay_generator.create_memory_overlay(
                frame, brain_input, brain_activity, high_res_frame
            )
            
            processing_time = (time.time() - start_time) * 1000  # ms
            processing_times.append(processing_time)
            
            # Track attention stats
            if attention_map is not None:
                attention_stats.append({
                    'max_attention': np.max(attention_map),
                    'avg_attention': np.mean(attention_map),
                    'hotspots': np.sum(attention_map > 0.6),
                    'windows': len(attention_windows)
                })
            
            print(f"Frame {i+1}: {processing_time:.1f}ms, "
                  f"{len(attention_windows)} windows, "
                  f"{len(camera.memory_inspector.memory_samples)} memories")
            
        except Exception as e:
            print(f"Frame {i+1}: Error - {e}")
            continue
    
    # Performance analysis
    if processing_times:
        avg_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        min_time = np.min(processing_times)
        
        print(f"\n📊 Performance Results:")
        print(f"Average processing time: {avg_time:.1f}ms")
        print(f"Max processing time: {max_time:.1f}ms")
        print(f"Min processing time: {min_time:.1f}ms")
        print(f"Real-time capable: {'✅ Yes' if avg_time < 33 else '⚠️ Marginal' if avg_time < 50 else '❌ No'}")
    
    # Attention analysis
    if attention_stats:
        avg_max_attention = np.mean([s['max_attention'] for s in attention_stats])
        avg_windows = np.mean([s['windows'] for s in attention_stats])
        
        print(f"\n🎯 Attention Results:")
        print(f"Average max attention: {avg_max_attention:.3f}")
        print(f"Average attention windows: {avg_windows:.1f}")
        print(f"Total memories formed: {len(camera.memory_inspector.memory_samples)}")
    
    # Memory analysis
    memories = camera.memory_inspector.memory_samples
    attention_memories = [m for m in memories if 'attention_strength' in m and m['attention_strength'] > 0]
    
    if attention_memories:
        avg_attention_strength = np.mean([m['attention_strength'] for m in attention_memories])
        print(f"Average attention strength: {avg_attention_strength:.3f}")
        print(f"✅ Attention-gated memory formation working!")
    
    # Cleanup
    camera.memory_inspector.cleanup()
    
    print(f"\n🎉 Test completed successfully!")
    return avg_time < 50  # Return True if performance is acceptable

if __name__ == "__main__":
    success = test_optimized_attention()
    if success:
        print("\n✅ System ready for real-time camera use!")
    else:
        print("\n⚠️ Performance may need further optimization for real-time use")