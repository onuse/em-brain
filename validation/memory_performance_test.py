#!/usr/bin/env python3
"""
Test memory overlay camera performance with gating
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'server'))

import time
import numpy as np
from memory_overlay_camera import MemoryOverlayCamera

def test_performance():
    """Run performance test"""
    print("🧠 Memory Overlay Performance Test")
    print("=" * 50)
    print("Testing memory gating impact on performance...")
    print("Press 'q' to stop, '1-6' to change modes")
    print()
    
    # Create camera
    camera = MemoryOverlayCamera(brain_type="sparse_goldilocks")
    
    # Run visualization
    camera.run_memory_visualization()

if __name__ == "__main__":
    test_performance()