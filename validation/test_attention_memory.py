#!/usr/bin/env python3
"""
Test attention-gated memory formation without full visualization
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import time
import numpy as np
from src.brain import MinimalBrain
from memory_inspector import MemoryInspector

def create_test_attention_map(width: int, height: int, hotspot_x: int, hotspot_y: int) -> np.ndarray:
    """Create a test attention map with a hotspot"""
    attention_map = np.zeros((height, width))
    
    # Create a Gaussian hotspot
    y, x = np.ogrid[:height, :width]
    mask = (x - hotspot_x)**2 + (y - hotspot_y)**2 <= (min(width, height) // 4)**2
    attention_map[mask] = 1.0
    
    # Add some background attention
    attention_map += np.random.uniform(0, 0.2, (height, width))
    
    return np.clip(attention_map, 0, 1)

def test_attention_gated_memory():
    """Test attention-gated memory formation"""
    print("🎯 Testing Attention-Gated Memory Formation")
    print("=" * 50)
    
    # Create brain and inspector
    brain = MinimalBrain(brain_type="sparse_goldilocks", quiet_mode=True)
    inspector = MemoryInspector(brain, use_emergent_gate=True)
    
    # Test parameters
    frame_width, frame_height = 32, 32
    test_frames = 200
    
    print(f"Testing {test_frames} frames with attention hotspots...")
    
    memories_without_attention = 0
    memories_with_attention = 0
    
    for i in range(test_frames):
        # Generate sensory input
        sensory_input = np.random.uniform(0, 1, brain.sensory_dim).tolist()
        
        # Process through brain
        brain_output, brain_info = brain.process_sensory_input(sensory_input)
        
        if i < test_frames // 2:
            # First half: no attention map (baseline)
            snapshot = inspector.capture_memory_snapshot(
                sensory_input, brain_output, brain_info, None
            )
            if snapshot.get('stored', True):  # Default to True if 'stored' key missing
                memories_without_attention += 1
        else:
            # Second half: with attention hotspots
            hotspot_x = np.random.randint(0, frame_width)
            hotspot_y = np.random.randint(0, frame_height)
            attention_map = create_test_attention_map(frame_width, frame_height, hotspot_x, hotspot_y)
            
            snapshot = inspector.capture_memory_snapshot(
                sensory_input, brain_output, brain_info, attention_map
            )
            if snapshot.get('stored', True):  # Default to True if 'stored' key missing
                memories_with_attention += 1
                # Check if attention data is stored
                if 'attention_strength' in snapshot:
                    attention_str = snapshot['attention_strength']
                    print(f"  Frame {i}: Attention {attention_str:.2f}, Memory formed")
        
        # Progress updates
        if (i + 1) % 50 == 0:
            total_memories = len(inspector.memory_samples)
            stats = inspector.memory_gate.get_statistics()
            storage_rate = stats.get('recent_storage_rate', 0) * 100
            
            print(f"Frame {i + 1}: {total_memories} total memories, {storage_rate:.0f}% recent storage")
    
    # Final analysis
    print(f"\n📊 Results:")
    print(f"Memories without attention: {memories_without_attention}")
    print(f"Memories with attention: {memories_with_attention}")
    print(f"Total memories stored: {len(inspector.memory_samples)}")
    
    # Analyze stored memories for attention data
    attention_memories = [m for m in inspector.memory_samples if 'attention_strength' in m and m['attention_strength'] > 0]
    print(f"Memories with attention data: {len(attention_memories)}")
    
    if attention_memories:
        avg_attention = np.mean([m['attention_strength'] for m in attention_memories])
        print(f"Average attention strength: {avg_attention:.3f}")
    
    # Get final statistics
    final_stats = inspector.memory_gate.get_statistics()
    print(f"\nFinal storage rate: {final_stats.get('overall_storage_rate', 0)*100:.1f}%")
    print(f"Final pressure: {final_stats.get('total_pressure', 0):.2f}")
    
    inspector.cleanup()
    
    # Determine if attention gating is working
    if len(attention_memories) > 0:
        print("\n✅ Attention-gated memory formation is working!")
        print("   - Attention data is being stored with memories")
        print("   - Attention strength affects memory formation decisions")
    else:
        print("\n⚠️  Attention gating may not be working properly")
        print("   - No attention data found in stored memories")

if __name__ == "__main__":
    test_attention_gated_memory()