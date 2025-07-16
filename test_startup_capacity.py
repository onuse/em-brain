#!/usr/bin/env python3
"""
Test the startup capacity test system
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from server.src.utils.startup_capacity_test import run_startup_capacity_test

def main():
    print("🧪 Testing Startup Capacity Test")
    print("=" * 50)
    
    # Run the capacity test
    limits = run_startup_capacity_test()
    
    print("\n📊 Final Results:")
    for key, value in limits.items():
        if isinstance(value, int):
            print(f"   {key}: {value:,}")
        else:
            print(f"   {key}: {value}")
    
    # Check if results are reasonable
    print("\n✅ Validation:")
    max_exp = limits['max_experiences']
    
    if max_exp >= 10000:
        print(f"   ✅ Experience limit ({max_exp:,}) is reasonable for hundreds of thousands target")
    else:
        print(f"   ⚠️  Experience limit ({max_exp:,}) seems low")
    
    if max_exp >= 50000:
        print(f"   ✅ Should support substantial learning")
    else:
        print(f"   ⚠️  May limit learning capacity")
    
    if limits['cleanup_threshold'] > max_exp:
        print(f"   ✅ Cleanup threshold properly set")
    else:
        print(f"   ⚠️  Cleanup threshold issue")

if __name__ == "__main__":
    main()