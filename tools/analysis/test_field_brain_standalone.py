#!/usr/bin/env python3
"""
Standalone Field Brain Test - 1 Minute Operation Test

Test the field brain in isolation for 1 minute to verify it operates correctly
before attempting brain + brainstem integration.

No socket connections, no external dependencies - just pure field brain operation.
"""

import sys
import os
import time
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

def test_field_brain_standalone():
    """Test field brain standalone operation for 1 minute."""
    print("🧠 STANDALONE FIELD BRAIN TEST - 1 MINUTE OPERATION")
    print("=" * 60)
    print("Testing: Pure field brain functionality without socket dependencies")
    print("Duration: 60 seconds of continuous operation")
    print()
    
    try:
        from src.brain import MinimalBrain
        
        # Configuration for standalone test
        config = {
            "brain": {
                "type": "field",
                "sensory_dim": 16,
                "motor_dim": 4,
                "field_spatial_resolution": 8,   # Reasonable size for testing
                "field_temporal_window": 4.0,   
                "field_evolution_rate": 0.1,    
                "constraint_discovery_rate": 0.1
            },
            "memory": {"enable_persistence": False},
            "logging": {
                "log_brain_cycles": False,
                "log_pattern_storage": False,
                "log_performance": False
            }
        }
        
        print("🔧 Configuration: Standalone field brain")
        print("   - Spatial resolution: 8³ field")
        print("   - Sparse attention-based processing")
        print("   - No persistence, minimal logging")
        print("   - Pure field brain operation")
        
        # Create brain
        print("\\n⏱️ Creating field brain...")
        start_time = time.time()
        brain = MinimalBrain(config=config, quiet_mode=True, enable_logging=False)
        creation_time = time.time() - start_time
        print(f"   ✅ Brain created in {creation_time:.3f}s")
        
        # Test pattern sequences
        test_patterns = [
            # Simple patterns
            [0.8, 0.2, 0.5, 0.7] + [0.1] * 12,  # Pattern A
            [0.2, 0.8, 0.7, 0.5] + [0.1] * 12,  # Pattern B
            [0.5, 0.5, 0.8, 0.2] + [0.1] * 12,  # Pattern C
            
            # Complex patterns  
            [0.9, 0.1, 0.6, 0.4, 0.8, 0.2] + [0.1] * 10,  # Pattern D
            [0.1, 0.9, 0.4, 0.6, 0.2, 0.8] + [0.1] * 10,  # Pattern E
            
            # Random-ish patterns
            [0.7, 0.3, 0.9, 0.1, 0.5, 0.6, 0.4, 0.8] + [0.2] * 8,  # Pattern F
            [0.3, 0.7, 0.1, 0.9, 0.6, 0.5, 0.8, 0.4] + [0.2] * 8,  # Pattern G
        ]
        
        print(f"\\n🔄 Starting 1-minute continuous operation test...")
        print("   Testing with varied sensory patterns")
        
        # Track metrics
        cycle_count = 0
        total_processing_time = 0.0
        processing_times = []
        errors = 0
        actions_generated = []
        
        # Run for 1 minute
        test_start_time = time.time()
        target_duration = 60.0  # 1 minute
        
        pattern_index = 0
        
        while (time.time() - test_start_time) < target_duration:
            cycle_count += 1
            
            # Get current pattern (cycle through test patterns)
            current_pattern = test_patterns[pattern_index % len(test_patterns)]
            pattern_index += 1
            
            try:
                # Process sensory input
                cycle_start = time.time()
                action, brain_state = brain.process_sensory_input(current_pattern)
                cycle_time = time.time() - cycle_start
                
                # Track metrics
                processing_times.append(cycle_time)
                total_processing_time += cycle_time
                actions_generated.append(action)
                
                # Progress indicator every 10 cycles
                if cycle_count % 10 == 0:
                    elapsed = time.time() - test_start_time
                    avg_cycle_time = total_processing_time / cycle_count
                    print(f"     Cycle {cycle_count:3d}: {elapsed:5.1f}s elapsed, avg cycle: {avg_cycle_time:.3f}s")
                
                # Brief pause to prevent overwhelming
                time.sleep(0.05)  # 50ms pause between cycles
                
            except Exception as e:
                errors += 1
                print(f"     ❌ Error in cycle {cycle_count}: {e}")
                if errors > 5:
                    print("     ⚠️ Too many errors, stopping test")
                    break
        
        # Test completed
        elapsed_time = time.time() - test_start_time
        
        print(f"\\n📊 STANDALONE TEST RESULTS:")
        print(f"   Duration: {elapsed_time:.1f}s")
        print(f"   Cycles completed: {cycle_count}")
        print(f"   Errors encountered: {errors}")
        
        if cycle_count > 0:
            avg_cycle_time = total_processing_time / cycle_count
            max_cycle_time = max(processing_times) if processing_times else 0
            min_cycle_time = min(processing_times) if processing_times else 0
            
            print(f"\\n⏱️ Performance Metrics:")
            print(f"   Average cycle time: {avg_cycle_time:.3f}s")
            print(f"   Min cycle time: {min_cycle_time:.3f}s")
            print(f"   Max cycle time: {max_cycle_time:.3f}s")
            print(f"   Theoretical max frequency: {1/avg_cycle_time:.1f}Hz")
            
            # Action consistency check
            if len(actions_generated) >= 2:
                action_changes = 0
                for i in range(1, len(actions_generated)):
                    if actions_generated[i] != actions_generated[i-1]:
                        action_changes += 1
                
                action_stability = (len(actions_generated) - action_changes) / len(actions_generated) * 100
                print(f"\\n🎯 Behavior Metrics:")
                print(f"   Action changes: {action_changes}/{len(actions_generated)}")
                print(f"   Action stability: {action_stability:.1f}%")
                
                # Sample actions
                print(f"   Sample actions:")
                for i in range(min(3, len(actions_generated))):
                    action_str = [f"{x:.2f}" for x in actions_generated[i]]
                    print(f"     Cycle {i+1}: [{', '.join(action_str)}]")
        
        # Overall assessment
        print(f"\\n🏆 OVERALL ASSESSMENT:")
        
        success_rate = (cycle_count - errors) / cycle_count * 100 if cycle_count > 0 else 0
        
        if success_rate >= 95 and avg_cycle_time < 0.5:
            print("   ✅ EXCELLENT - Field brain operating reliably")
            print("   ✅ Ready for brain + brainstem integration")
        elif success_rate >= 90 and avg_cycle_time < 1.0:
            print("   🔧 GOOD - Field brain mostly stable")
            print("   🔧 Should work for brain + brainstem integration")
        elif success_rate >= 80:
            print("   ⚠️ CONCERNING - Some stability issues detected")
            print("   ⚠️ Consider debugging before integration")
        else:
            print("   ❌ PROBLEMATIC - Significant issues detected")
            print("   ❌ Fix standalone issues before integration")
        
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Ready for robot testing: {'Yes' if success_rate >= 90 and avg_cycle_time < 0.5 else 'Maybe' if success_rate >= 80 else 'No'}")
        
        brain.finalize_session()
        return {
            'success': success_rate >= 90,
            'cycle_count': cycle_count,
            'avg_cycle_time': avg_cycle_time,
            'error_count': errors,
            'ready_for_integration': success_rate >= 90 and avg_cycle_time < 0.5
        }
        
    except Exception as e:
        print(f"❌ Standalone test failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'ready_for_integration': False
        }

def main():
    """Run standalone field brain test."""
    print("🚀 FIELD BRAIN STANDALONE TEST")
    print("=" * 60)
    print("Purpose: Verify field brain operates correctly in isolation")
    print("Goal: 1 minute of stable operation before socket integration")
    print()
    
    # Run standalone test
    results = test_field_brain_standalone()
    
    # Summary
    print(f"\\n{'=' * 60}")
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    
    if results['success']:
        print("✅ Standalone test PASSED")
        if results['ready_for_integration']:
            print("✅ Field brain ready for brain + brainstem integration")
            print("\\n🚀 Next steps:")
            print("   1. Start brain server with field brain type")
            print("   2. Connect robot client via binary TCP")
            print("   3. Monitor real-world performance")
        else:
            print("🔧 Field brain stable but needs performance tuning")
    else:
        print("❌ Standalone test FAILED")
        print("🔧 Fix standalone issues before attempting integration")
    
    print(f"\\n📋 Test completed - check results above for next steps")

if __name__ == "__main__":
    main()