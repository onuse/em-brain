#!/usr/bin/env python3
"""
Test enhanced resolution brain input with edge detection and contour tracing
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import cv2
import numpy as np
import time
from src.brain import MinimalBrain
from memory_overlay_camera import MemoryOverlayCamera

def test_enhanced_resolution():
    """Test enhanced resolution brain input capabilities"""
    print("🔬 Testing Enhanced Resolution Brain Input")
    print("=" * 50)
    
    # Create test image with clear contours
    test_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    
    # Add geometric shapes with clear contours
    cv2.rectangle(test_frame, (100, 100), (200, 200), (255, 255, 255), -1)  # White square
    cv2.circle(test_frame, (400, 150), 60, (128, 128, 128), -1)  # Gray circle
    cv2.ellipse(test_frame, (500, 250), (80, 40), 0, 0, 360, (200, 200, 200), -1)  # Light gray ellipse
    
    # Add some edges and lines
    cv2.line(test_frame, (50, 300), (590, 300), (255, 255, 255), 3)  # Horizontal line
    cv2.line(test_frame, (320, 50), (320, 310), (255, 255, 255), 2)  # Vertical line
    
    print("Created test frame with geometric shapes for contour detection")
    
    # Create camera system
    camera = MemoryOverlayCamera(brain_type="sparse_goldilocks")
    
    # Test frame processing
    print("\nTesting frame processing with different resolutions...")
    
    # Process with enhanced system
    start_time = time.time()
    brain_input, high_res_frame, edge_frame = camera.process_frame_to_brain_input(test_frame)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"✅ Frame processing completed in {processing_time:.1f}ms")
    print(f"📊 Resolution Analysis:")
    print(f"   Original frame: {test_frame.shape[1]}x{test_frame.shape[0]} = {test_frame.shape[1]*test_frame.shape[0]} pixels")
    print(f"   High-res attention: {high_res_frame.shape[1]}x{high_res_frame.shape[0]} = {high_res_frame.shape[1]*high_res_frame.shape[0]} pixels")
    print(f"   Edge detection: {edge_frame.shape[1]}x{edge_frame.shape[0]} = {edge_frame.shape[1]*edge_frame.shape[0]} pixels")
    print(f"   Brain input: {len(brain_input)} values")
    
    # Estimate brain input resolution
    brain_res = int(np.sqrt(len(brain_input)))
    print(f"   Brain input grid: ~{brain_res}x{brain_res}")
    
    # Compression ratios
    original_pixels = test_frame.shape[1] * test_frame.shape[0]
    high_res_pixels = high_res_frame.shape[1] * high_res_frame.shape[0]
    brain_pixels = len(brain_input)
    
    print(f"📈 Compression Ratios:")
    print(f"   Original to high-res: {original_pixels/high_res_pixels:.0f}:1")
    print(f"   Original to brain: {original_pixels/brain_pixels:.0f}:1")
    print(f"   High-res to brain: {high_res_pixels/brain_pixels:.0f}:1")
    
    # Test edge detection
    edge_pixels = np.sum(edge_frame > 0.1)
    edge_density = edge_pixels / (edge_frame.shape[0] * edge_frame.shape[1])
    print(f"🔍 Edge Detection:")
    print(f"   Edge pixels: {edge_pixels}")
    print(f"   Edge density: {edge_density:.3f}")
    
    # Test contour detection
    contours, _ = cv2.findContours(
        (edge_frame * 255).astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    significant_contours = [c for c in contours if cv2.contourArea(c) > 50]
    print(f"📐 Contour Detection:")
    print(f"   Total contours: {len(contours)}")
    print(f"   Significant contours: {len(significant_contours)}")
    
    if significant_contours:
        areas = [cv2.contourArea(c) for c in significant_contours]
        print(f"   Contour areas: {[f'{a:.0f}' for a in areas]}")
        print(f"   Largest contour: {max(areas):.0f} pixels")
    
    # Test memory formation
    print(f"\n🧠 Testing Memory Formation:")
    brain_output, brain_info = camera.brain.process_sensory_input(brain_input)
    
    # Create attention map
    current_array = np.array(brain_input)
    brain_activity = {
        'brain_output': brain_output,
        'brain_info': brain_info,
        'timestamp': time.time()
    }
    
    attention_map, attention_windows = camera.overlay_generator._calculate_adaptive_attention_map(
        current_array, brain_activity, high_res_frame
    )
    
    if attention_map is not None:
        max_attention = np.max(attention_map)
        avg_attention = np.mean(attention_map)
        attention_pixels = np.sum(attention_map > 0.6)
        
        print(f"   Max attention: {max_attention:.3f}")
        print(f"   Avg attention: {avg_attention:.3f}")
        print(f"   High attention pixels: {attention_pixels}")
        print(f"   Attention windows: {len(attention_windows)}")
    
    # Test memory capture
    memory_snapshot = camera.memory_inspector.capture_memory_snapshot(
        brain_input, brain_output, brain_info, attention_map
    )
    
    if memory_snapshot and memory_snapshot.get('stored', True):
        print(f"   ✅ Memory formed successfully")
        if 'attention_strength' in memory_snapshot:
            print(f"   Attention strength: {memory_snapshot['attention_strength']:.3f}")
    else:
        print(f"   ⚠️  Memory not formed")
    
    # Assessment
    print(f"\n🎯 Assessment:")
    can_detect_edges = edge_pixels > 100
    can_trace_contours = len(significant_contours) > 0
    reasonable_compression = original_pixels/brain_pixels < 20000  # Less than 20,000:1
    
    print(f"   Edge detection capable: {'✅' if can_detect_edges else '❌'}")
    print(f"   Contour tracing capable: {'✅' if can_trace_contours else '❌'}")
    print(f"   Reasonable compression: {'✅' if reasonable_compression else '❌'}")
    
    # Overall assessment
    if can_detect_edges and can_trace_contours and reasonable_compression:
        print(f"\n🎉 SUCCESS: System can detect edges and trace contours!")
        print(f"   Brain input resolution is sufficient for meaningful pattern recognition")
    else:
        print(f"\n⚠️  PARTIAL: System needs further resolution improvements")
    
    # Cleanup
    camera.memory_inspector.cleanup()
    
    return can_detect_edges and can_trace_contours

if __name__ == "__main__":
    success = test_enhanced_resolution()
    if success:
        print("\n✅ Enhanced resolution system ready for deployment!")
    else:
        print("\n⚠️  System needs further optimization")