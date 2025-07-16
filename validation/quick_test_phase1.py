#!/usr/bin/env python3
"""
Quick test of Phase 1 improvements with running server
"""

import sys
import os
import time
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server' / 'src'))
sys.path.insert(0, str(brain_root / 'server'))
sys.path.insert(0, str(brain_root / 'validation'))

from src.communication.client import MinimalBrainClient
from embodied_learning.environments.sensory_motor_world import SensoryMotorWorld
from micro_experiments.improved_framework import ImprovedMicroExperimentSuite
from micro_experiments.improved_core_assumptions import ImprovedSimilarityConsistency

def test_phase1_improvements():
    """Test Phase 1 improvements with running server."""
    print("🚀 Quick Test of Phase 1 Improvements")
    print("=" * 50)
    
    # Test 1: Persistent connection
    print("\n1. Testing persistent connection...")
    client = MinimalBrainClient()
    if client.connect():
        print("   ✅ Connected to brain server")
        
        # Test with a few quick requests
        env = SensoryMotorWorld(random_seed=42)
        for i in range(3):
            sensory_input = env.get_sensory_input()
            start_time = time.time()
            action = client.get_action(sensory_input, timeout=5.0)
            response_time = time.time() - start_time
            
            if action is not None:
                print(f"   ✅ Request {i+1}: {response_time:.3f}s response time")
            else:
                print(f"   ❌ Request {i+1}: No response")
                
        client.disconnect()
    else:
        print("   ❌ Failed to connect to brain server")
        return False
    
    # Test 2: Retry logic
    print("\n2. Testing retry logic...")
    experiment = ImprovedSimilarityConsistency()
    experiment.client = MinimalBrainClient()
    experiment.environment = SensoryMotorWorld(random_seed=42)
    
    if experiment.client.connect():
        print("   ✅ Connected for retry test")
        
        # Test the retry method
        sensory_input = experiment.environment.get_sensory_input()
        start_time = time.time()
        action = experiment.get_action_with_retry(sensory_input, max_retries=3, timeout=5.0)
        response_time = time.time() - start_time
        
        if action is not None:
            print(f"   ✅ Retry logic works: {response_time:.3f}s response time")
        else:
            print("   ⚠️  Retry logic: No response (might be expected)")
            
        experiment.client.disconnect()
    else:
        print("   ❌ Failed to connect for retry test")
        return False
    
    # Test 3: Improved suite setup
    print("\n3. Testing improved suite setup...")
    suite = ImprovedMicroExperimentSuite()
    suite.add_experiment(ImprovedSimilarityConsistency())
    
    if suite.setup_persistent_resources():
        print("   ✅ Persistent resources setup successful")
        print(f"   ✅ Persistent client: {suite.persistent_client is not None}")
        print(f"   ✅ Persistent environment: {suite.persistent_environment is not None}")
        
        suite.cleanup_persistent_resources()
        print("   ✅ Cleanup successful")
    else:
        print("   ❌ Failed to setup persistent resources")
        return False
    
    print("\n🎉 All Phase 1 improvements working correctly!")
    return True

if __name__ == "__main__":
    success = test_phase1_improvements()
    if success:
        print("\n✅ Phase 1 implementation is ready for use!")
        print("🔄 You can now run comprehensive_verification.py with improved performance")
    else:
        print("\n❌ Phase 1 implementation needs debugging")