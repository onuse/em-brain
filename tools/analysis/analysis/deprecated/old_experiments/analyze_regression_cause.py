#!/usr/bin/env python3
"""
Analyze Regression Cause

Investigates whether the vector mismatch was the actual cause of the massive
performance regression (906%) or if there are other underlying issues.
"""

import sys
import os
import time

# Set up path to access brain modules
brain_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(brain_root, 'server', 'src'))
sys.path.append(os.path.join(brain_root, 'server'))

from src.brain import MinimalBrain


def analyze_regression_cause():
    """Analyze what actually caused the 906% performance regression."""
    print("🔍 ANALYZING REGRESSION ROOT CAUSE")
    print("=" * 60)
    print("Question: Did vector mismatch cause 906% performance regression?")
    print()
    
    # Test 1: Simulate the original performance issue
    print("1. SIMULATING ORIGINAL ISSUE (Vector Mismatch)")
    print("-" * 55)
    
    # Create brain that would have the vector mismatch
    brain_with_mismatch = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    # Temporarily disable hierarchical clustering to see baseline performance
    brain_with_mismatch.similarity_engine.use_hierarchical_indexing = False
    brain_with_mismatch.similarity_engine.hierarchical_index = None
    
    print("   Testing performance with disabled hierarchical clustering...")
    
    # Add experiences and measure performance over time
    performance_timeline = []
    
    for batch in range(5):  # 5 batches of 50 experiences each
        batch_start = batch * 50
        batch_end = (batch + 1) * 50
        
        # Add experiences
        for i in range(batch_start, batch_end):
            sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
            predicted_action, _ = brain_with_mismatch.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain_with_mismatch.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        # Test performance at this experience count
        search_times = []
        for j in range(10):
            query_sensory = [0.5 + 0.01 * j, 0.4 + 0.01 * j, 0.6 + 0.01 * j, 0.3 + 0.01 * j]
            search_start = time.time()
            predicted_action, brain_state = brain_with_mismatch.process_sensory_input(query_sensory)
            search_time = (time.time() - search_start) * 1000
            search_times.append(search_time)
        
        avg_time = sum(search_times) / len(search_times)
        experience_count = len(brain_with_mismatch.experience_storage._experiences)
        performance_timeline.append({
            'experiences': experience_count,
            'avg_time': avg_time,
            'method': 'no_hierarchical'
        })
        
        print(f"   {experience_count} experiences: {avg_time:.1f}ms")
    
    brain_with_mismatch.finalize_session()
    
    # Test 2: Current fixed performance
    print(f"\n2. TESTING CURRENT FIXED PERFORMANCE")
    print("-" * 40)
    
    brain_fixed = MinimalBrain(
        enable_logging=False,
        enable_persistence=False,
        enable_storage_optimization=True,
        use_utility_based_activation=True,
        enable_phase2_adaptations=False,
        quiet_mode=True
    )
    
    print("   Testing performance with fixed hierarchical clustering...")
    
    fixed_performance_timeline = []
    
    for batch in range(5):  # Same test
        batch_start = batch * 50
        batch_end = (batch + 1) * 50
        
        # Add experiences
        for i in range(batch_start, batch_end):
            sensory = [0.1 + 0.01 * i, 0.2 + 0.01 * i, 0.3 + 0.01 * i, 0.4 + 0.01 * i]
            predicted_action, _ = brain_fixed.process_sensory_input(sensory)
            outcome = [a * 0.9 + 0.05 for a in predicted_action]
            brain_fixed.store_experience(sensory, predicted_action, outcome, predicted_action)
        
        # Test performance
        search_times = []
        for j in range(10):
            query_sensory = [0.5 + 0.01 * j, 0.4 + 0.01 * j, 0.6 + 0.01 * j, 0.3 + 0.01 * j]
            search_start = time.time()
            predicted_action, brain_state = brain_fixed.process_sensory_input(query_sensory)
            search_time = (time.time() - search_start) * 1000
            search_times.append(search_time)
        
        avg_time = sum(search_times) / len(search_times)
        experience_count = len(brain_fixed.experience_storage._experiences)
        fixed_performance_timeline.append({
            'experiences': experience_count,
            'avg_time': avg_time,
            'method': 'fixed_hierarchical'
        })
        
        print(f"   {experience_count} experiences: {avg_time:.1f}ms")
    
    # Get hierarchical stats
    final_stats = brain_fixed.similarity_engine.get_performance_stats()
    hierarchical_stats = final_stats.get('hierarchical_indexing', {})
    
    if hierarchical_stats:
        print(f"   Final regions: {hierarchical_stats.get('total_regions', 0)}")
        print(f"   Search efficiency: {hierarchical_stats.get('search_efficiency', 1.0):.1f}x")
    
    brain_fixed.finalize_session()
    
    # Analysis
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print("-" * 35)
    
    print("   Experience Count | No Hierarchical | Fixed Hierarchical | Improvement")
    print("   " + "-" * 70)
    
    total_improvement = 0
    count = 0
    
    for i in range(len(performance_timeline)):
        no_hier = performance_timeline[i]
        fixed = fixed_performance_timeline[i]
        
        if no_hier['experiences'] == fixed['experiences']:
            improvement = ((no_hier['avg_time'] - fixed['avg_time']) / no_hier['avg_time']) * 100
            total_improvement += improvement
            count += 1
            
            print(f"   {no_hier['experiences']:>15} | {no_hier['avg_time']:>14.1f}ms | {fixed['avg_time']:>17.1f}ms | {improvement:>+9.1f}%")
    
    avg_improvement = total_improvement / count if count > 0 else 0
    
    print(f"\n🎯 ANALYSIS RESULTS:")
    print("-" * 25)
    print(f"   Average improvement with fix: {avg_improvement:+.1f}%")
    
    # Check scaling behavior
    if len(performance_timeline) >= 2:
        # No hierarchical scaling
        first_no_hier = performance_timeline[0]['avg_time']
        last_no_hier = performance_timeline[-1]['avg_time']
        no_hier_degradation = ((last_no_hier - first_no_hier) / first_no_hier) * 100
        
        # Fixed hierarchical scaling
        first_fixed = fixed_performance_timeline[0]['avg_time']
        last_fixed = fixed_performance_timeline[-1]['avg_time']
        fixed_degradation = ((last_fixed - first_fixed) / first_fixed) * 100
        
        print(f"   Performance degradation without hierarchical: {no_hier_degradation:+.1f}%")
        print(f"   Performance degradation with hierarchical: {fixed_degradation:+.1f}%")
        
        print(f"\n💡 ROOT CAUSE ANALYSIS:")
        print("-" * 30)
        
        if abs(no_hier_degradation) > 100:
            print("   ❌ MAJOR ISSUE: Even without hierarchical clustering,")
            print("      performance degrades significantly over time")
            print("   🔍 This suggests the 906% regression was NOT caused")
            print("      by the vector mismatch, but by other factors:")
            print("      - O(n²) similarity search without hierarchical indexing")
            print("      - Memory pressure from growing experience storage")
            print("      - Other systems not scaling efficiently")
            
        elif avg_improvement > 50:
            print("   ✅ HIERARCHICAL CLUSTERING HIGHLY EFFECTIVE:")
            print(f"      {avg_improvement:.1f}% average improvement shows hierarchical")
            print("      clustering successfully prevents performance regression")
            
        else:
            print("   📊 MIXED RESULTS: Hierarchical clustering helps but")
            print("      may not fully explain the original regression")
    
    print(f"\n🔧 RECOMMENDATION:")
    print("-" * 20)
    
    if avg_improvement > 50:
        print("   ✅ Vector mismatch fix + hierarchical clustering")
        print("      successfully resolves the performance regression")
        print("   🚀 Ready for production server testing")
    else:
        print("   ⚠️  May need additional investigation into:")
        print("      - Storage optimization effectiveness")
        print("      - Activation system scaling")
        print("      - Memory management under load")


if __name__ == "__main__":
    analyze_regression_cause()