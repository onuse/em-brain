#!/usr/bin/env python3
"""
Hardware Adaptation Test - Check if hardware adaptation is the 2.2s bottleneck
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

from src.brain import MinimalBrain
from src.utils.hardware_adaptation import record_brain_cycle_performance

def test_hardware_adaptation_bottleneck():
    """Test if hardware adaptation is causing the 2.2s delay."""
    print("🔍 HARDWARE ADAPTATION BOTTLENECK TEST")
    print("=" * 45)
    
    # Test hardware adaptation function directly
    print("Testing hardware adaptation function...")
    
    # Time the hardware adaptation call
    start_time = time.time()
    record_brain_cycle_performance(100.0, 50.0)  # 100ms cycle, 50MB memory
    adaptation_time = (time.time() - start_time) * 1000
    
    print(f"Hardware adaptation call: {adaptation_time:.1f}ms")
    
    if adaptation_time > 1000:
        print(f"🚨 BOTTLENECK FOUND: Hardware adaptation!")
        print(f"   This explains the 2.2s delay!")
        return True
    else:
        print(f"✅ Hardware adaptation is fast")
    
    # Test multiple calls
    print("\nTesting multiple hardware adaptation calls...")
    
    total_time = 0
    for i in range(10):
        start_time = time.time()
        record_brain_cycle_performance(100.0 + i, 50.0)
        elapsed = (time.time() - start_time) * 1000
        total_time += elapsed
        
        if elapsed > 100:
            print(f"  Call {i+1}: {elapsed:.1f}ms (SLOW)")
        else:
            print(f"  Call {i+1}: {elapsed:.2f}ms")
    
    avg_time = total_time / 10
    print(f"\nAverage hardware adaptation time: {avg_time:.1f}ms")
    
    if avg_time > 200:
        print(f"🚨 CUMULATIVE BOTTLENECK: Hardware adaptation")
        print(f"   Multiple calls are adding up to significant delay")
        return True
    
    return False

if __name__ == "__main__":
    print("Testing if hardware adaptation is the 2.2s bottleneck...")
    is_bottleneck = test_hardware_adaptation_bottleneck()
    
    if is_bottleneck:
        print(f"\n🎯 BOTTLENECK IDENTIFIED: Hardware adaptation system")
        print(f"   The fast path bypasses this by using minimal brain state")
        print(f"   This explains the 908x speedup!")
    else:
        print(f"\n❓ Hardware adaptation is not the bottleneck")
        print(f"   Need to investigate other components")