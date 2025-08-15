#!/usr/bin/env python3
"""
Simple test of adaptive attention system with variable window sizes
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
from memory_overlay_camera import MemoryOverlayGenerator

def test_adaptive_attention():
    """Test adaptive attention with simulated high-res frames"""
    print("🎯 Testing Adaptive Attention System")
    print("=" * 50)
    
    # Create brain and components
    brain = MinimalBrain(brain_type="sparse_goldilocks", quiet_mode=True)
    inspector = MemoryInspector(brain, use_emergent_gate=True)
    overlay_generator = MemoryOverlayGenerator(inspector, frame_width=640, frame_height=360)
    
    # Create test images with different complexity levels
    test_cases = [
        ("Simple uniform", np.ones((360, 640, 3), dtype=np.uint8) * 128),
        ("High contrast edges", create_edge_image(640, 360)),
        ("Complex pattern", create_complex_pattern(640, 360)),
        ("Multiple objects", create_multi_object_image(640, 360))
    ]
    
    print("Testing different image complexities...")
    
    for test_name, test_frame in test_cases:
        print(f"\n🔍 Testing: {test_name}")
        
        # Convert to brain input with high-res preservation
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        high_res = cv2.resize(gray, (64, 64)).astype(np.float32) / 255.0
        
        # Create brain input (16x16 for meaningful patterns)
        brain_input = cv2.resize(gray, (16, 16)).astype(np.float32) / 255.0
        brain_input = brain_input.flatten()[:brain.sensory_dim].tolist()
        
        # Process through brain
        brain_output, brain_info = brain.process_sensory_input(brain_input)
        
        # Create brain activity
        brain_activity = {
            'brain_output': brain_output,
            'brain_info': brain_info,
            'timestamp': time.time()
        }
        
        # Calculate adaptive attention
        current_array = np.array(brain_input)
        attention_map, attention_windows = overlay_generator._calculate_adaptive_attention_map(
            current_array, brain_activity, high_res
        )
        
        # Analyze results
        if attention_map is not None:
            max_attention = np.max(attention_map)
            avg_attention = np.mean(attention_map)
            hotspot_count = np.sum(attention_map > 0.6)
            
            print(f"   Max attention: {max_attention:.3f}")
            print(f"   Avg attention: {avg_attention:.3f}")
            print(f"   Hotspots: {hotspot_count}")
            print(f"   Attention windows: {len(attention_windows)}")
            
            # Show window types
            for i, window in enumerate(attention_windows):
                print(f"   Window {i+1}: {window['type']} ({window['width']}x{window['height']}) strength={window['strength']:.3f}")
        else:
            print("   ❌ Attention calculation failed")
        
        # Test memory formation with attention
        inspector.capture_memory_snapshot(
            brain_input, brain_output, brain_info, attention_map
        )
        
        print(f"   Memories formed: {len(inspector.memory_samples)}")
        
        # Brief pause between tests
        time.sleep(0.1)
    
    # Final summary
    print(f"\n📊 Final Results:")
    print(f"Total memories: {len(inspector.memory_samples)}")
    
    # Check for attention data in memories
    attention_memories = [m for m in inspector.memory_samples if 'attention_strength' in m]
    if attention_memories:
        avg_attention_strength = np.mean([m['attention_strength'] for m in attention_memories])
        print(f"Average attention strength: {avg_attention_strength:.3f}")
        print("✅ Adaptive attention system working!")
    else:
        print("⚠️ No attention data found in memories")
    
    inspector.cleanup()

def create_edge_image(width, height):
    """Create image with strong edges"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Vertical edge
    img[:, width//3:width//3+20] = 255
    
    # Horizontal edge
    img[height//3:height//3+20, :] = 128
    
    # Diagonal line
    for i in range(min(width, height)):
        if i < height and i < width:
            img[i, i] = 200
    
    return img

def create_complex_pattern(width, height):
    """Create complex pattern with multiple features"""
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    
    # Add some structure
    center_x, center_y = width//2, height//2
    cv2.circle(img, (center_x, center_y), 50, (255, 255, 255), -1)
    cv2.rectangle(img, (center_x-30, center_y-30), (center_x+30, center_y+30), (0, 0, 0), -1)
    
    return img

def create_multi_object_image(width, height):
    """Create image with multiple distinct objects"""
    img = np.zeros((height, width, 3), dtype=np.uint8) + 50
    
    # Object 1: Circle
    cv2.circle(img, (width//4, height//4), 30, (255, 0, 0), -1)
    
    # Object 2: Rectangle
    cv2.rectangle(img, (3*width//4-25, height//4-25), (3*width//4+25, height//4+25), (0, 255, 0), -1)
    
    # Object 3: Triangle (as polygon)
    points = np.array([[width//2, height//2-30], [width//2-30, height//2+30], [width//2+30, height//2+30]])
    cv2.fillPoly(img, [points], (0, 0, 255))
    
    # Object 4: Line
    cv2.line(img, (width//4, 3*height//4), (3*width//4, 3*height//4), (255, 255, 0), 5)
    
    return img

if __name__ == "__main__":
    test_adaptive_attention()