#!/usr/bin/env python3
"""
Debug Field Brain Processing

Test the field brain processing directly to see if there are performance issues
or errors that could cause timeouts in the TCP communication.
"""

import sys
import os
import time
from pathlib import Path

# Add server source to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../server'))

def test_field_brain_processing():
    """Test field brain processing performance and error handling."""
    print("🔍 Field Brain Processing Debug Test")
    print("=" * 50)
    
    try:
        from src.brain import MinimalBrain
        import json
        
        # Load current settings (field brain configuration)
        server_dir = Path(__file__).parent.parent.parent / "server"
        settings_file = server_dir / "settings.json"
        
        with open(settings_file, 'r') as f:
            config = json.load(f)
        
        print(f"✅ Configuration loaded: {config['brain']['type']}")
        
        # Create field brain with minimal logging
        print("\n🧠 Creating field brain...")
        start_time = time.time()
        brain = MinimalBrain(config=config, quiet_mode=True)
        init_time = time.time() - start_time
        print(f"✅ Brain created in {init_time:.3f}s")
        print(f"   Type: {type(brain.vector_brain).__name__}")
        
        # Test multiple processing cycles to check for performance issues
        print("\n🔄 Testing processing cycles...")
        
        test_sensory = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
                       0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        
        processing_times = []
        
        for i in range(10):
            print(f"   Cycle {i+1}/10...", end=" ")
            start_time = time.time()
            
            try:
                action, brain_state = brain.process_sensory_input(test_sensory)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                print(f"{processing_time*1000:.1f}ms ✅")
                
                # Check for any obvious issues in brain state
                if 'prediction_confidence' not in brain_state:
                    print(f"      ⚠️ Missing prediction_confidence in brain state")
                
                if len(action) != 4:
                    print(f"      ⚠️ Unexpected action vector length: {len(action)}")
                    
            except Exception as e:
                processing_time = time.time() - start_time
                print(f"❌ ERROR after {processing_time*1000:.1f}ms")
                print(f"      Error: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Analyze results
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
            
            print(f"\n📊 Processing Performance:")
            print(f"   Average: {avg_time*1000:.1f}ms")
            print(f"   Min: {min_time*1000:.1f}ms")
            print(f"   Max: {max_time*1000:.1f}ms")
            print(f"   Cycles completed: {len(processing_times)}/10")
            
            # Check if any cycles are suspiciously slow
            if max_time > 5.0:  # 5 second timeout concern
                print(f"   ⚠️ WARNING: Max processing time {max_time:.1f}s exceeds timeout threshold")
            elif max_time > 1.0:  # 1 second is already quite slow
                print(f"   ⚠️ Some cycles took over 1 second ({max_time:.1f}s)")
            else:
                print(f"   ✅ All cycles completed in reasonable time")
        
        # Test brain statistics
        print(f"\n📈 Testing brain statistics...")
        start_time = time.time()
        try:
            stats = brain.get_brain_stats()
            stats_time = time.time() - start_time
            print(f"   ✅ Stats retrieved in {stats_time*1000:.1f}ms")
            print(f"   Architecture: {stats['brain_summary']['architecture']}")
            print(f"   Total cycles: {stats['brain_summary']['total_cycles']}")
        except Exception as e:
            stats_time = time.time() - start_time
            print(f"   ❌ Stats failed after {stats_time*1000:.1f}ms: {e}")
        
        # Cleanup
        brain.finalize_session()
        print(f"\n✅ Test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tcp_compatibility():
    """Test compatibility with TCP protocol expectations."""
    print("\n🌐 TCP Compatibility Test")
    print("=" * 30)
    
    try:
        from src.brain import MinimalBrain
        import json
        
        # Load settings
        server_dir = Path(__file__).parent.parent.parent / "server"
        settings_file = server_dir / "settings.json"
        
        with open(settings_file, 'r') as f:
            config = json.load(f)
        
        brain = MinimalBrain(config=config, quiet_mode=True)
        
        # Test the exact interface that TCP server expects
        print("Testing TCP server interface methods...")
        
        # Test process_sensory_input with exact parameters TCP server uses
        test_sensory = [0.1] * 16  # 16D input as configured
        action_dimensions = 4      # 4D output as configured
        
        print(f"   Input: {len(test_sensory)}D sensory vector")
        print(f"   Expected output: {action_dimensions}D action vector")
        
        action, brain_state = brain.process_sensory_input(test_sensory, action_dimensions)
        
        print(f"   ✅ Output: {len(action)}D action vector")
        print(f"   ✅ Brain state keys: {list(brain_state.keys())}")
        
        # Test get_brain_stats
        stats = brain.get_brain_stats()
        print(f"   ✅ Stats keys: {list(stats.keys())}")
        
        brain.finalize_session()
        print("   ✅ TCP interface compatibility confirmed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ TCP compatibility test failed: {e}")
        return False

def main():
    """Run all debug tests."""
    print("🚀 FIELD BRAIN PROCESSING DEBUG SUITE")
    print("=" * 60)
    
    tests = [
        ("Processing Performance", test_field_brain_processing),
        ("TCP Compatibility", test_tcp_compatibility),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 60}")
    print("📊 DEBUG SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎯 Field brain processing looks healthy!")
        print("   The timeout issue may be in network/protocol layer")
    else:
        print(f"\n⚠️ Found {total - passed} processing issues")
        print("   These could explain the timeout problems")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)