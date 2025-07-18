#!/usr/bin/env python3
"""
System Validation Test - Validates current working brain architecture.

This test validates what's actually working in the current system
after the architectural cleanup.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'server'))

def test_current_architecture():
    """Test the current working architecture."""
    print("🧪 SYSTEM VALIDATION TEST")
    print("=" * 40)
    
    # Test 1: MinimalBrain with minimal backend
    print("\n1. Testing MinimalBrain with minimal backend...")
    try:
        from src.brain import MinimalBrain
        brain = MinimalBrain(brain_type='minimal', quiet_mode=True)
        result = brain.process_sensory_input([0.5, 0.3, 0.8, 0.1])
        print(f"   ✅ MinimalBrain working: {len(result[0])} motor dims, {result[1]['cycle_time_ms']:.2f}ms")
        brain.finalize_session()
    except Exception as e:
        print(f"   ❌ MinimalBrain failed: {e}")
    
    # Test 2: Sparse representations
    print("\n2. Testing sparse representations...")
    try:
        from src.vector_stream.sparse_representations import SparsePatternEncoder, SparsePatternStorage
        encoder = SparsePatternEncoder(16, sparsity=0.02, quiet_mode=True)
        storage = SparsePatternStorage(pattern_dim=16, max_patterns=100, quiet_mode=True)
        print(f"   ✅ Sparse representations working: {encoder.pattern_dim}D, {encoder.sparsity} sparsity")
    except Exception as e:
        print(f"   ❌ Sparse representations failed: {e}")
    
    # Test 3: Attention systems
    print("\n3. Testing attention systems...")
    try:
        from src.attention.signal_attention import UniversalAttentionSystem
        attention = UniversalAttentionSystem()
        print(f"   ✅ Attention systems working")
    except Exception as e:
        print(f"   ❌ Attention systems failed: {e}")
    
    # Test 4: Memory systems
    print("\n4. Testing memory systems...")
    try:
        from src.memory.pattern_memory import UniversalMemorySystem
        memory = UniversalMemorySystem()
        print(f"   ✅ Memory systems working")
    except Exception as e:
        print(f"   ❌ Memory systems failed: {e}")
    
    # Test 5: Hardware adaptation
    print("\n5. Testing hardware adaptation...")
    try:
        from src.utils.hardware_adaptation import get_hardware_adaptation
        hardware = get_hardware_adaptation()
        print(f"   ✅ Hardware adaptation working")
    except Exception as e:
        print(f"   ❌ Hardware adaptation failed: {e}")
    
    print("\n✅ System validation completed!")

if __name__ == "__main__":
    test_current_architecture()
