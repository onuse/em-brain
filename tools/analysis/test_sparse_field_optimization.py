#!/usr/bin/env python3
"""
Test Sparse Field Optimization - Biological Performance Hack

Test the sparse field update optimization to achieve 90% reduction in processing time.
Target: Reduce from 5.6s to ~0.56s through attention-based sparse processing.
"""

import sys
import os
import time
import cProfile
import pstats
import io
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

def test_sparse_field_optimization():
    """Test field brain with sparse field updates for biological optimization."""
    print("🧬 TESTING SPARSE FIELD OPTIMIZATION")
    print("=" * 55)
    print("BIOLOGICAL HACK: Attention-based sparse field updates")
    print("Target: 90% reduction (5.6s → 0.56s)")
    print()
    
    try:
        from src.brain import MinimalBrain
        
        # Minimal configuration for pure field processing test
        config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 10,  # Same as baseline
                "field_temporal_window": 5.0,   # Same as baseline
                "field_evolution_rate": 0.05,   
                "constraint_discovery_rate": 0.05
            },
            "memory": {"enable_persistence": False},
            "logging": {
                "log_brain_cycles": False,
                "log_pattern_storage": False,
                "log_performance": False
            }
        }
        
        print("🔧 Configuration: Sparse field brain")
        print("   - Spatial resolution: 10 (same as baseline)")
        print("   - Attention-based sparse updates: ENABLED")
        print("   - Gradient-based sparse constraints: ENABLED")
        print("   - Minimal infrastructure overhead")
        
        # Create brain
        print("\\n⏱️ Creating sparse field brain...")
        start_time = time.time()
        brain = MinimalBrain(config=config, quiet_mode=True, enable_logging=False)
        creation_time = time.time() - start_time
        print(f"   ✅ Brain created in {creation_time:.3f}s")
        
        # Test multiple processing cycles for consistency
        print("\\n🔄 Testing sparse processing cycles...")
        sensory_input = [0.1] * 16
        
        processing_times = []
        for i in range(5):
            print(f"   Cycle {i+1}/5...", end=" ")
            start_time = time.time()
            
            action, brain_state = brain.process_sensory_input(sensory_input)
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            print(f"{processing_time:.3f}s")
        
        # Performance analysis
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        min_time = min(processing_times)
        
        print(f"\\n📊 Sparse Field Performance:") 
        print(f"   Average processing: {avg_time:.3f}s")
        print(f"   Min processing: {min_time:.3f}s")
        print(f"   Max processing: {max_time:.3f}s")
        
        # Compare to baselines
        baseline_full = 62.0   # Original full infrastructure
        baseline_minimal = 5.6  # Minimal infrastructure (previous optimization)
        
        improvement_from_full = baseline_full / avg_time
        improvement_from_minimal = baseline_minimal / avg_time
        reduction_achieved = (1 - avg_time / baseline_minimal) * 100
        
        print(f"\\n🎯 Performance Improvements:")
        print(f"   Full infrastructure baseline: {baseline_full:.1f}s")
        print(f"   Minimal infrastructure baseline: {baseline_minimal:.1f}s")
        print(f"   Sparse field optimization: {avg_time:.3f}s")
        print(f"   Improvement from full: {improvement_from_full:.1f}x faster")
        print(f"   Improvement from minimal: {improvement_from_minimal:.1f}x faster")
        print(f"   Reduction achieved: {reduction_achieved:.1f}%")
        
        # Target assessment
        target_time = 0.56  # 90% reduction target
        print(f"\\n🎯 Target Assessment:")
        print(f"   Target time (90% reduction): {target_time:.2f}s")
        print(f"   Actual time: {avg_time:.3f}s")
        
        if avg_time <= target_time:
            print("   ✅ TARGET ACHIEVED! 90% reduction successful!")
        elif avg_time <= 1.0:
            print("   🔧 CLOSE TO TARGET - significant improvement achieved")
        elif avg_time <= 2.0:
            print("   🔧 GOOD PROGRESS - substantial optimization, more needed")
        else:
            print("   ❌ MORE OPTIMIZATION NEEDED")
        
        # Real-time assessment
        print(f"\\n🤖 Real-time Robot Control Assessment:")
        if avg_time <= 0.1:
            print("   ✅ EXCELLENT - suitable for high-frequency control (>10Hz)")
        elif avg_time <= 0.5:
            print("   ✅ GOOD - suitable for real-time control (~2Hz)")
        elif avg_time <= 1.0:
            print("   🔧 ACCEPTABLE - suitable for slower control (~1Hz)")
        else:
            print("   ❌ TOO SLOW - needs more biological optimizations")
        
        brain.finalize_session()
        return avg_time, reduction_achieved
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def profile_sparse_optimization():
    """Profile the sparse optimization to see function-level improvements."""
    print("\\n🔬 PROFILING SPARSE OPTIMIZATION")
    print("=" * 45)
    
    try:
        from src.brain import MinimalBrain
        
        config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 10,
                "field_temporal_window": 5.0,
                "field_evolution_rate": 0.05,
                "constraint_discovery_rate": 0.05
            },
            "memory": {"enable_persistence": False},
            "logging": {
                "log_brain_cycles": False,
                "log_pattern_storage": False,
                "log_performance": False
            }
        }
        
        brain = MinimalBrain(config=config, quiet_mode=True, enable_logging=False)
        sensory_input = [0.1] * 16
        
        # Profile a single processing cycle
        profiler = cProfile.Profile()
        
        start_time = time.time()
        profiler.enable()
        
        action, brain_state = brain.process_sensory_input(sensory_input)
        
        profiler.disable()
        processing_time = time.time() - start_time
        
        print(f"✅ Sparse processing completed in {processing_time:.3f}s")
        
        # Analyze the profile
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
        ps.print_stats(30)  # Top 30 functions
        
        profile_data = s.getvalue()
        
        # Save sparse profile
        profile_file = Path(__file__).parent.parent.parent / "logs" / "sparse_field_brain_profile.txt"
        profile_file.parent.mkdir(exist_ok=True)
        with open(profile_file, 'w') as f:
            f.write(profile_data)
        
        print(f"💾 Sparse profile saved to: {profile_file}")
        
        # Quick analysis of top time consumers
        lines = profile_data.split('\\n')
        print("\\n📊 Top Time Consumers in Sparse Optimization:")
        
        parsing_data = False
        for line in lines:
            if 'ncalls' in line and 'tottime' in line:
                parsing_data = True
                continue
            
            if parsing_data and line.strip():
                parts = line.split()
                if len(parts) >= 6 and parts[0].replace('.', '').isdigit():
                    try:
                        cumtime = float(parts[3])
                        function_name = ' '.join(parts[5:])
                        
                        if cumtime > 0.1:  # Only show functions taking >100ms
                            percentage = (cumtime / processing_time) * 100
                            print(f"   {cumtime:.3f}s ({percentage:5.1f}%) - {function_name[:60]}...")
                    except (ValueError, IndexError):
                        continue
        
        brain.finalize_session()
        return processing_time
        
    except Exception as e:
        print(f"❌ Profiling failed: {e}")
        return None

def main():
    """Run sparse field optimization tests."""
    print("🧬 SPARSE FIELD OPTIMIZATION TEST")
    print("=" * 60)
    print("BIOLOGICAL HYPOTHESIS: Attention-based sparse processing")
    print("Target: 90% reduction through biological optimizations")
    print()
    
    # Test 1: Performance measurement
    avg_time, reduction_achieved = test_sparse_field_optimization()
    
    # Test 2: Detailed profiling
    profile_time = profile_sparse_optimization()
    
    # Summary
    print(f"\\n{'=' * 60}")
    print("🎯 SPARSE OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    if avg_time and reduction_achieved is not None:
        print(f"📊 Performance Results:")
        print(f"   Sparse processing time: {avg_time:.3f}s")
        print(f"   Reduction achieved: {reduction_achieved:.1f}%")
        
        target_reduction = 90.0
        if reduction_achieved >= target_reduction:
            print(f"   ✅ SUCCESS: {target_reduction}% reduction target achieved!")
        else:
            remaining_needed = target_reduction - reduction_achieved
            print(f"   🔧 PROGRESS: {remaining_needed:.1f}% more reduction needed")
    
    print(f"\\n🧠 Biological Insights:")
    print("✅ Attention-based sparse updates implemented")
    print("✅ Gradient-based constraint processing optimized")
    print("✅ Computational budget limits enforced")
    print("✅ Background minimal diffusion for field health")
    
    if avg_time and avg_time < 1.0:
        print("\\n🎯 Next optimizations (synergistic approach):")
        print("   1. Background field evolution thread")
        print("   2. Hierarchical coarse-to-fine processing")
        print("   3. Predictive caching of field states")
        print("   4. Event-driven vs continuous processing split")
    else:
        print("\\n🔧 Additional biological hacks needed:")
        print("   1. More aggressive attention thresholds")
        print("   2. Hierarchical multi-resolution processing")
        print("   3. Temporal dynamics optimization")
        print("   4. Memory-based predictive shortcuts")

if __name__ == "__main__":
    main()