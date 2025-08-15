#!/usr/bin/env python3
"""
Test Minimal Field Brain - Biological Optimization

Test the field brain with minimal infrastructure overhead to see
if the threading bottleneck comes from logging/persistence systems.

This is the first "biological hack" - removing systems that real brains don't have.
"""

import sys
import os
import time
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

def test_field_brain_minimal_infrastructure():
    """Test field brain with minimal logging and no persistence."""
    print("🧠 Testing Field Brain with Minimal Infrastructure")
    print("=" * 55)
    print("Biological Insight: Real brains don't have async loggers!")
    print()
    
    try:
        from src.brain import MinimalBrain
        
        # Minimal configuration - disable heavy infrastructure
        config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 10,  # Reduced resolution
                "field_temporal_window": 5.0,   # Reduced temporal window
                "field_evolution_rate": 0.05,   # Reduced evolution rate
                "constraint_discovery_rate": 0.05  # Reduced constraint discovery
            },
            "memory": {
                "enable_persistence": False  # DISABLE PERSISTENCE
            },
            "logging": {
                "log_brain_cycles": False,      # DISABLE CYCLE LOGGING
                "log_pattern_storage": False,   # DISABLE PATTERN LOGGING
                "log_performance": False        # DISABLE PERFORMANCE LOGGING
            }
        }
        
        print("🔧 Configuration: Minimal field brain")
        print("   - Persistence: DISABLED")
        print("   - Logging: MINIMAL")
        print("   - Spatial resolution: 10 (reduced)")
        print("   - Temporal window: 5.0s (reduced)")
        
        # Test brain creation
        print("\n⏱️ Creating brain...")
        start_time = time.time()
        brain = MinimalBrain(config=config, quiet_mode=True, enable_logging=False)
        creation_time = time.time() - start_time
        print(f"   ✅ Brain created in {creation_time:.3f}s")
        
        # Test processing with multiple cycles
        print("\n🔄 Testing processing cycles...")
        sensory_input = [0.1] * 16
        
        processing_times = []
        for i in range(5):
            print(f"   Cycle {i+1}/5...", end=" ")
            start_time = time.time()
            
            action, brain_state = brain.process_sensory_input(sensory_input)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            print(f"{processing_time:.3f}s")
        
        # Analysis
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        print(f"\n📊 Performance Results:")
        print(f"   Average processing: {avg_time:.3f}s")
        print(f"   Min processing: {min_time:.3f}s")
        print(f"   Max processing: {max_time:.3f}s")
        
        # Compare to baseline (62s)
        baseline = 62.0
        improvement = baseline / avg_time
        
        print(f"\n🎯 Performance Improvement:")
        print(f"   Baseline (full infrastructure): {baseline:.1f}s")
        print(f"   Minimal infrastructure: {avg_time:.3f}s")
        print(f"   Improvement factor: {improvement:.1f}x faster!")
        
        if avg_time < 1.0:
            print("   ✅ USABLE FOR REAL-TIME ROBOT CONTROL!")
        elif avg_time < 5.0:
            print("   🔧 GETTING CLOSE - needs more optimization")
        else:
            print("   ❌ Still too slow - need more biological hacks")
        
        brain.finalize_session()
        return avg_time
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_even_more_minimal():
    """Test with even more aggressive reduction."""
    print("\n🔬 Ultra-Minimal Field Brain Test")
    print("=" * 40)
    
    try:
        # Try to access field brain directly without MinimalBrain wrapper
        from src.brains.field.tcp_adapter import FieldBrainTCPAdapter
        from src.brains.field.field_brain_config import FieldBrainConfig
        
        # Ultra-minimal config
        field_config = FieldBrainConfig(
            sensory_dimensions=16,
            motor_dimensions=4,
            spatial_resolution=5,      # VERY small field
            temporal_window=2.0,       # VERY short window
            field_evolution_rate=0.01, # VERY slow evolution
            constraint_discovery_rate=0.01,  # VERY slow discovery
            quiet_mode=True
        )
        
        print("⚡ Ultra-minimal configuration:")
        print("   - Spatial resolution: 5x5 (tiny!)")
        print("   - Temporal window: 2.0s")
        print("   - Minimal dynamics")
        
        start_time = time.time()
        adapter = FieldBrainTCPAdapter(field_config)
        creation_time = time.time() - start_time
        print(f"   ✅ Direct adapter created in {creation_time:.3f}s")
        
        # Test direct processing
        sensory_input = [0.1] * 16
        
        start_time = time.time()
        action, brain_state = adapter.process_sensory_input(sensory_input)
        processing_time = time.time() - start_time
        
        print(f"   ✅ Direct processing: {processing_time:.3f}s")
        
        if processing_time < 0.1:
            print("   🚀 EXCELLENT - under 100ms!")
        elif processing_time < 1.0:
            print("   ✅ GOOD - under 1 second")
        else:
            print("   🔧 Still needs work")
        
        return processing_time
        
    except Exception as e:
        print(f"❌ Ultra-minimal test failed: {e}")
        return None

def main():
    """Run biological optimization tests."""
    print("🧬 BIOLOGICAL FIELD BRAIN OPTIMIZATION")
    print("=" * 60)
    print("Testing the hypothesis: Infrastructure overhead is the bottleneck")
    print("Biological insight: Real brains don't have async loggers!")
    print()
    
    # Test 1: Minimal infrastructure
    minimal_time = test_field_brain_minimal_infrastructure()
    
    # Test 2: Ultra-minimal
    ultra_time = test_even_more_minimal()
    
    # Summary
    print(f"\n{'=' * 60}")
    print("🎯 BIOLOGICAL OPTIMIZATION RESULTS")
    print("=" * 60)
    
    baseline = 62.0
    
    if minimal_time:
        improvement = baseline / minimal_time
        print(f"📊 Minimal Infrastructure:")
        print(f"   Baseline: {baseline:.1f}s")
        print(f"   Minimal: {minimal_time:.3f}s")
        print(f"   Improvement: {improvement:.1f}x faster")
    
    if ultra_time:
        improvement = baseline / ultra_time
        print(f"\n📊 Ultra-Minimal:")
        print(f"   Baseline: {baseline:.1f}s") 
        print(f"   Ultra-minimal: {ultra_time:.3f}s")
        print(f"   Improvement: {improvement:.1f}x faster")
    
    print(f"\n🧠 Biological Insights:")
    if minimal_time and minimal_time < 5.0:
        print("✅ Infrastructure overhead was indeed the major bottleneck!")
        print("✅ Field brain can be fast with biological optimizations!")
    else:
        print("🔧 Infrastructure helps but more biological hacks needed")
    
    print("\n🎯 Next biological optimizations to implement:")
    print("   1. Sparse field updates (only 'interesting' regions)")
    print("   2. Hierarchical processing (coarse-to-fine)")
    print("   3. Predictive caching (pre-computed patterns)")
    print("   4. Event-driven processing (no continuous updates)")

if __name__ == "__main__":
    main()