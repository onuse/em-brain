#!/usr/bin/env python3
"""
Performance Diagnostic Tool

Identifies performance bottlenecks causing the severe degradation during sustained operation.
"""

import sys
import os
import time
import threading
from typing import Dict, List, Any

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain

class PerformanceDiagnostic:
    """Diagnose performance bottlenecks in brain operation."""
    
    def __init__(self):
        self.results = {}
        
    def run_diagnostic(self):
        """Run comprehensive performance diagnostic."""
        print("🔍 PERFORMANCE DIAGNOSTIC")
        print("=" * 50)
        
        # Test 1: Baseline brain performance
        print("\n1. Testing baseline brain performance...")
        self.test_baseline_performance()
        
        # Test 2: Test with checkpointing disabled
        print("\n2. Testing with checkpointing disabled...")
        self.test_without_checkpointing()
        
        # Test 3: Test with Phase 2 adaptations disabled
        print("\n3. Testing with Phase 2 adaptations disabled...")
        self.test_without_phase2()
        
        # Test 4: Test with minimal configuration
        print("\n4. Testing minimal configuration...")
        self.test_minimal_configuration()
        
        # Generate diagnostic report
        self.generate_diagnostic_report()
    
    def test_baseline_performance(self):
        """Test current brain configuration."""
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=True,  # This may be causing checkpointing issues
            enable_storage_optimization=True,
            use_utility_based_activation=True,
            enable_phase2_adaptations=True,
            quiet_mode=True
        )
        
        cycle_times = []
        for i in range(20):
            sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
            
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            end_time = time.time()
            
            cycle_time = (end_time - start_time) * 1000
            cycle_times.append(cycle_time)
            
            if i % 5 == 0:
                print(f"   Cycle {i}: {cycle_time:.1f}ms")
        
        avg_time = sum(cycle_times) / len(cycle_times)
        self.results['baseline'] = {
            'avg_cycle_time': avg_time,
            'cycle_times': cycle_times,
            'experiences': len(brain.experience_storage._experiences)
        }
        
        print(f"   Baseline average: {avg_time:.1f}ms")
        brain.finalize_session()
    
    def test_without_checkpointing(self):
        """Test with persistence disabled."""
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,  # Disable checkpointing
            enable_storage_optimization=True,
            use_utility_based_activation=True,
            enable_phase2_adaptations=True,
            quiet_mode=True
        )
        
        cycle_times = []
        for i in range(20):
            sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
            
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            end_time = time.time()
            
            cycle_time = (end_time - start_time) * 1000
            cycle_times.append(cycle_time)
            
            if i % 5 == 0:
                print(f"   Cycle {i}: {cycle_time:.1f}ms")
        
        avg_time = sum(cycle_times) / len(cycle_times)
        self.results['no_checkpointing'] = {
            'avg_cycle_time': avg_time,
            'cycle_times': cycle_times,
            'experiences': len(brain.experience_storage._experiences)
        }
        
        print(f"   No checkpointing average: {avg_time:.1f}ms")
        brain.finalize_session()
    
    def test_without_phase2(self):
        """Test with Phase 2 adaptations disabled."""
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True,
            use_utility_based_activation=True,
            enable_phase2_adaptations=False,  # Disable Phase 2
            quiet_mode=True
        )
        
        cycle_times = []
        for i in range(20):
            sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
            
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            end_time = time.time()
            
            cycle_time = (end_time - start_time) * 1000
            cycle_times.append(cycle_time)
            
            if i % 5 == 0:
                print(f"   Cycle {i}: {cycle_time:.1f}ms")
        
        avg_time = sum(cycle_times) / len(cycle_times)
        self.results['no_phase2'] = {
            'avg_cycle_time': avg_time,
            'cycle_times': cycle_times,
            'experiences': len(brain.experience_storage._experiences)
        }
        
        print(f"   No Phase 2 average: {avg_time:.1f}ms")
        brain.finalize_session()
    
    def test_minimal_configuration(self):
        """Test with minimal configuration."""
        brain = MinimalBrain(
            enable_logging=False,
            enable_persistence=False,
            enable_storage_optimization=True,
            use_utility_based_activation=True,
            enable_phase2_adaptations=False,
            quiet_mode=True
        )
        
        cycle_times = []
        for i in range(20):
            sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
            
            start_time = time.time()
            predicted_action, brain_state = brain.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain.store_experience(sensory, predicted_action, outcome, predicted_action)
            end_time = time.time()
            
            cycle_time = (end_time - start_time) * 1000
            cycle_times.append(cycle_time)
            
            if i % 5 == 0:
                print(f"   Cycle {i}: {cycle_time:.1f}ms")
        
        avg_time = sum(cycle_times) / len(cycle_times)
        self.results['minimal'] = {
            'avg_cycle_time': avg_time,
            'cycle_times': cycle_times,
            'experiences': len(brain.experience_storage._experiences)
        }
        
        print(f"   Minimal config average: {avg_time:.1f}ms")
        brain.finalize_session()
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report."""
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE DIAGNOSTIC REPORT")
        print("=" * 50)
        
        baseline_time = self.results['baseline']['avg_cycle_time']
        no_checkpoint_time = self.results['no_checkpointing']['avg_cycle_time']
        no_phase2_time = self.results['no_phase2']['avg_cycle_time']
        minimal_time = self.results['minimal']['avg_cycle_time']
        
        print(f"\n🔍 CONFIGURATION COMPARISON:")
        print(f"   Baseline (all features):     {baseline_time:.1f}ms")
        print(f"   No checkpointing:           {no_checkpoint_time:.1f}ms")
        print(f"   No Phase 2 adaptations:     {no_phase2_time:.1f}ms")
        print(f"   Minimal configuration:      {minimal_time:.1f}ms")
        
        # Calculate impact of each feature
        checkpoint_impact = ((baseline_time - no_checkpoint_time) / baseline_time) * 100
        phase2_impact = ((no_checkpoint_time - no_phase2_time) / no_checkpoint_time) * 100
        
        print(f"\n💥 PERFORMANCE IMPACT:")
        print(f"   Checkpointing overhead:     {checkpoint_impact:+.1f}%")
        print(f"   Phase 2 adaptations:        {phase2_impact:+.1f}%")
        
        # Identify biggest culprit
        print(f"\n🎯 OPTIMIZATION RECOMMENDATIONS:")
        
        if checkpoint_impact > 50:
            print("   ❌ DISABLE CHECKPOINTING: Major performance killer")
            print("   📝 Recommendation: Set enable_persistence=False for real-time operation")
        
        if phase2_impact > 30:
            print("   ⚠️  DISABLE PHASE 2: Significant overhead")
            print("   📝 Recommendation: Set enable_phase2_adaptations=False")
        
        if minimal_time < 100:
            print(f"   ✅ MINIMAL CONFIG ACHIEVES REAL-TIME: {minimal_time:.1f}ms")
        else:
            print(f"   ❌ STILL TOO SLOW: {minimal_time:.1f}ms > 100ms target")
        
        print(f"\n🏆 BEST PERFORMANCE: {min(baseline_time, no_checkpoint_time, no_phase2_time, minimal_time):.1f}ms")

def main():
    """Run performance diagnostic."""
    diagnostic = PerformanceDiagnostic()
    diagnostic.run_diagnostic()

if __name__ == "__main__":
    main()