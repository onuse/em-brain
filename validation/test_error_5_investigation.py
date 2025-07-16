#!/usr/bin/env python3
"""
Test script to investigate Brain Processing Error (5.0)
Attempts to trigger error 5.0 to understand what causes generic brain processing failures.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.communication.client import MinimalBrainClient

def test_error_5_scenarios():
    """Test various scenarios that might trigger error 5.0."""
    
    print("🔍 Investigating Brain Processing Error (5.0)")
    print("=" * 60)
    
    client = MinimalBrainClient()
    
    if not client.connect():
        print("❌ Failed to connect to brain server")
        return
    
    try:
        # Test 1: Very short timeout (might cause timeout in brain processing)
        print("\n📋 Test 1: Ultra-short timeout (0.1s)")
        try:
            sensory_input = [0.5] * 16  # Standard input
            action = client.get_action(sensory_input, timeout=0.1)
            print("   ✅ Succeeded (no error)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if "5.0" in str(e):
                print("   🎯 Triggered error 5.0!")
        
        time.sleep(1)
        
        # Test 2: Rapid consecutive requests (might overwhelm brain)
        print("\n📋 Test 2: Rapid consecutive requests (100 in 1 second)")
        errors = []
        for i in range(100):
            try:
                sensory_input = [np.random.random() for _ in range(16)]
                action = client.get_action(sensory_input, timeout=0.5)
            except Exception as e:
                errors.append(str(e))
                if "5.0" in str(e):
                    print(f"   🎯 Triggered error 5.0 at request {i}!")
                    break
        
        if errors and "5.0" not in str(errors[-1]):
            print(f"   ❌ Got errors but not 5.0: {errors[-1]}")
        elif not errors:
            print("   ✅ All requests succeeded")
        
        time.sleep(1)
        
        # Test 3: Edge case sensory values
        print("\n📋 Test 3: Edge case sensory values")
        edge_cases = [
            ([0.0] * 16, "all zeros"),
            ([1.0] * 16, "all ones"),
            ([0.999999] * 16, "near ones"),
            ([0.5, 0.5] * 8, "repeated pattern"),
        ]
        
        for values, desc in edge_cases:
            try:
                print(f"   Testing {desc}...")
                action = client.get_action(values, timeout=2.0)
                print(f"   ✅ Succeeded")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                if "5.0" in str(e):
                    print(f"   🎯 Triggered error 5.0 with {desc}!")
            time.sleep(0.5)
        
        # Test 4: Large batch to trigger memory pressure
        print("\n📋 Test 4: Memory pressure test (1000 unique experiences)")
        for i in range(1000):
            try:
                # Generate diverse sensory inputs to fill memory
                sensory_input = [
                    np.sin(i * 0.1 + j * 0.5) * 0.5 + 0.5 
                    for j in range(16)
                ]
                action = client.get_action(sensory_input, timeout=1.0)
                
                if i % 100 == 0:
                    print(f"   Progress: {i}/1000")
                    
            except Exception as e:
                print(f"   ❌ Error at experience {i}: {e}")
                if "5.0" in str(e):
                    print("   🎯 Triggered error 5.0 under memory pressure!")
                    break
        
        # Check if error log was created
        print("\n📋 Checking for error logs...")
        error_log = Path("logs/brain_errors.jsonl")
        if error_log.exists():
            print("   ✅ Error log exists! Running viewer...")
            os.system("python3 server/tools/view_errors.py --limit 5")
        else:
            print("   ℹ️  No error log found - no errors were logged to file")
            
    finally:
        client.disconnect()
        print("\n✅ Investigation complete")

if __name__ == "__main__":
    test_error_5_scenarios()