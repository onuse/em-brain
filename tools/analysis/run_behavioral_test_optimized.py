#!/usr/bin/env python3
"""
Run Full Behavioral Test Framework with CPU Optimizations
Uses the complete behavioral_test_framework.py but with CPU-friendly settings.
"""

import sys
import os
sys.path.append('server/tools/testing')

from behavioral_test_framework import BehavioralTestFramework, BASIC_INTELLIGENCE_PROFILE

def main():
    """Run full behavioral test framework with CPU optimizations."""
    print("🧠 Full Behavioral Test Framework (CPU Optimized)")
    print("Complete intelligence assessment with reduced spatial resolution")
    print("=" * 70)
    
    # Create framework with optimized settings
    framework = BehavioralTestFramework(quiet_mode=True)
    
    # Override the create_brain method to use CPU-optimized config
    original_create_brain = framework.create_brain
    
    def create_optimized_brain(config=None):
        """Create brain with CPU optimizations."""
        # Start with any provided config
        optimized_config = config or {}
        
        # Ensure brain config exists
        if 'brain' not in optimized_config:
            optimized_config['brain'] = {}
        
        # Apply CPU optimizations
        optimized_config['brain'].update({
            'type': 'field',
            'sensory_dim': 16,
            'motor_dim': 4,
            'spatial_resolution': 8,  # Reduced from default 20
            'temporal_window': 10.0
        })
        
        # Disable persistence for speed
        if 'memory' not in optimized_config:
            optimized_config['memory'] = {}
        optimized_config['memory']['enable_persistence'] = False
        
        return original_create_brain(optimized_config)
    
    # Replace the method
    framework.create_brain = create_optimized_brain
    
    print(f"🧪 Running complete behavioral assessment with 8³ field resolution")
    
    # Create optimized brain
    brain = framework.create_brain()
    
    # Run full intelligence assessment
    results = framework.run_intelligence_assessment(brain, BASIC_INTELLIGENCE_PROFILE)
    
    # Enhanced results display
    achievement = results['overall_achievement']
    print(f"\n" + "="*60)
    print(f"🎯 COMPLETE BEHAVIORAL ASSESSMENT RESULTS")
    print(f"="*60)
    print(f"Overall Intelligence Achievement: {achievement:.1%}")
    
    if achievement > 0.8:
        print("🏆 OUTSTANDING: Brain exceeds expectations!")
    elif achievement > 0.6:
        print("🎉 EXCELLENT: Brain performing very well!")
    elif achievement > 0.4:
        print("✅ GOOD: Brain functional and learning well")
    else:
        print("⚠️ NEEDS WORK: Brain requires optimization")
    
    print(f"\n📊 Detailed Performance Breakdown:")
    for metric_name, result in results['detailed_results'].items():
        status = "✅ PASS" if result['achieved'] else "❌ FAIL"
        percentage = (result['score'] / result['target']) * 100 if result['target'] > 0 else 0
        print(f"  {status} {metric_name.replace('_', ' ').title()}")
        print(f"       Score: {result['score']:.3f} / Target: {result['target']:.3f} ({percentage:.0f}%)")
        print(f"       {result['description']}")
    
    # Performance summary
    print(f"\n⚡ Performance Summary:")
    print(f"  🧠 Brain Architecture: Unified Field (37D)")
    print(f"  🔢 Field Resolution: 8×8×8 = 512 elements")
    print(f"  🚀 Optimization Level: CPU-optimized for M1 Mac")
    print(f"  🎯 Target Achievement: {achievement:.1%} of behavioral goals")
    
    return achievement > 0.5

if __name__ == "__main__":
    success = main()
    print(f"\n{'🎉 SUCCESS!' if success else '⚠️  NEEDS IMPROVEMENT'}")
    exit(0 if success else 1)