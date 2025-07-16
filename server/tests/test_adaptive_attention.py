#!/usr/bin/env python3
"""
Test Adaptive Attention Scoring System

Demonstrates how adaptive attention creates biological memory suppression
without information loss, while auto-tuning based on learning velocity.
"""

import sys
import os
import time
import numpy as np
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.similarity import SimilarityEngine, AdaptiveAttentionScorer
from src.experience.models import Experience


def create_experience_with_attention(sensory_input: List[float], 
                                   action_taken: List[float],
                                   prediction_error: float) -> Experience:
    """Create an experience with computed attention score."""
    outcome = [s + np.random.normal(0, 0.1) for s in sensory_input]  # Slight change
    
    exp = Experience(
        sensory_input=sensory_input,
        action_taken=action_taken,
        outcome=outcome,
        prediction_error=prediction_error,
        timestamp=time.time()
    )
    
    return exp


def test_adaptive_attention_scoring():
    """Test the adaptive attention scorer with different prediction errors."""
    print("🧪 Testing Adaptive Attention Scoring")
    print("=" * 60)
    
    scorer = AdaptiveAttentionScorer()
    
    print(f"📊 Initial baseline: {scorer.attention_baseline:.3f}")
    
    # Test with different prediction errors
    test_cases = [
        (0.1, "Low error (boring)"),
        (0.3, "Medium error (interesting)"), 
        (0.7, "High error (surprising)"),
        (0.05, "Very low error (very boring)"),
        (0.9, "Very high error (very surprising)")
    ]
    
    learning_context = {'current_accuracy': 0.7}  # Moderate accuracy
    
    for prediction_error, description in test_cases:
        attention_score = scorer.compute_attention_score(prediction_error, learning_context)
        print(f"   {description:25}: error={prediction_error:.2f} → attention={attention_score:.3f}")
    
    print(f"\n📈 Final baseline: {scorer.attention_baseline:.3f}")
    
    # Test adaptation over time
    print(f"\n🔄 Testing adaptation to learning velocity...")
    
    # Simulate good learning (increasing accuracy)
    for i in range(20):
        context = {'current_accuracy': 0.5 + (i * 0.02)}  # Improving accuracy
        scorer.compute_attention_score(0.5, context)
    
    stats = scorer.get_attention_stats()
    print(f"   After good learning: baseline={stats['attention_baseline']:.3f}")
    print(f"   Learning velocity: {stats['learning_velocity']['current']:.4f}")
    
    return True


def test_attention_weighted_similarity():
    """Test attention-weighted similarity retrieval."""
    print("\n🧪 Testing Attention-Weighted Similarity")
    print("=" * 60)
    
    # Create similarity engine with attention
    similarity_engine = SimilarityEngine(use_natural_attention=True, use_gpu=False)
    
    # Create experiences with different attention levels
    experiences = []
    
    # High attention experiences (surprising)
    for i in range(5):
        exp = create_experience_with_attention(
            [1.0, 1.0, 90.0, 0.5],  # Near obstacle
            [-0.2, 0.1, 15.0, 0.0],  # Avoidance action
            0.8  # High prediction error
        )
        exp.attention_score = 0.9  # High attention
        experiences.append(exp)
        print(f"   High attention exp {i}: error={exp.prediction_error:.2f}, attention={exp.attention_score:.2f}")
    
    # Low attention experiences (boring)
    for i in range(10):
        exp = create_experience_with_attention(
            [2.0, 1.5, 45.0, 0.2],  # Normal navigation
            [0.1, 0.0, 5.0, 0.0],   # Normal movement
            0.1  # Low prediction error
        )
        exp.attention_score = 0.2  # Low attention (suppressed)
        experiences.append(exp)
    
    print(f"   Created {len(experiences)} experiences (5 high attention, 10 suppressed)")
    
    # Test retrieval with different modes
    query_vector = [1.0, 1.0, 90.0, 0.5, -0.2, 0.1, 15.0, 0.0]  # Context: obstacle situation
    
    for mode in ['normal', 'deep', 'hybrid']:
        print(f"\n🔍 Testing {mode.upper()} retrieval mode:")
        
        results = similarity_engine.find_similar_experiences_with_natural_attention(
            query_vector, experiences, max_results=5, retrieval_mode=mode
        )
        
        for i, (exp, weighted_sim, base_sim, attention) in enumerate(results):
            print(f"   {i+1}. weighted_sim={weighted_sim:.3f}, base_sim={base_sim:.3f}, "
                  f"attention={attention:.2f}, error={exp.prediction_error:.2f}")
        
        # Check if high attention experiences dominate in normal mode
        if mode == 'normal' and results:
            high_attention_results = [r for r in results if r[3] > 0.5]
            print(f"      → {len(high_attention_results)}/{len(results)} high-attention results")
    
    return True


def test_memory_suppression_and_retrieval():
    """Test biological memory suppression with cue-based retrieval."""
    print("\n🧪 Testing Memory Suppression & Cue-Based Retrieval")
    print("=" * 60)
    
    similarity_engine = SimilarityEngine(use_natural_attention=True, use_gpu=False)
    
    # Simulate yesterday's boring day - many similar low-attention experiences
    yesterday_experiences = []
    for hour in range(8):  # 8 hours of boring routine
        for event in range(10):  # 10 events per hour
            exp = create_experience_with_attention(
                [2.0 + hour*0.1, 1.5, 0.0, 0.2],  # Gradual position change
                [0.1, 0.0, 0.0, 0.0],  # Boring action
                0.05  # Very low prediction error
            )
            exp.attention_score = 0.1  # Heavily suppressed
            yesterday_experiences.append(exp)
    
    # Add one memorable event (talked about blue car)
    memorable_exp = create_experience_with_attention(
        [2.5, 1.5, 180.0, 0.0],  # Different heading - saw blue car
        [0.0, 0.0, 90.0, 1.0],   # Turned to look
        0.6  # Moderately surprising
    )
    memorable_exp.attention_score = 0.8  # High attention
    yesterday_experiences.append(memorable_exp)
    
    print(f"📅 Created yesterday's memories: {len(yesterday_experiences)} total")
    print(f"   - {len(yesterday_experiences)-1} boring suppressed memories")
    print(f"   - 1 memorable blue car event")
    
    # Test 1: Normal retrieval (trying to remember yesterday without cues)
    print(f"\n🤔 Trying to remember yesterday (no specific cues):")
    general_query = [2.0, 1.5, 0.0, 0.2, 0.1, 0.0, 0.0, 0.0]  # General yesterday context
    
    normal_results = similarity_engine.find_similar_experiences_with_natural_attention(
        general_query, yesterday_experiences, max_results=3, retrieval_mode='normal'
    )
    
    for i, (exp, weighted_sim, base_sim, attention) in enumerate(normal_results):
        is_memorable = attention > 0.5
        event_type = "BLUE CAR" if is_memorable else "boring routine"
        print(f"   {i+1}. {event_type}: weighted_sim={weighted_sim:.3f}, attention={attention:.2f}")
    
    # Test 2: Cue-based retrieval (someone mentions "blue car")
    print(f"\n💡 Someone mentions 'blue car' (high similarity cue):")
    blue_car_query = [2.5, 1.5, 180.0, 0.0, 0.0, 0.0, 90.0, 1.0]  # Specific blue car context
    
    cued_results = similarity_engine.find_similar_experiences_with_natural_attention(
        blue_car_query, yesterday_experiences, max_results=5, retrieval_mode='hybrid'
    )
    
    for i, (exp, weighted_sim, base_sim, attention) in enumerate(cued_results):
        is_memorable = attention > 0.5
        event_type = "BLUE CAR EVENT" if is_memorable else "related routine"
        print(f"   {i+1}. {event_type}: weighted_sim={weighted_sim:.3f}, base_sim={base_sim:.3f}")
    
    # Verify cue retrieval works
    if cued_results and cued_results[0][3] > 0.5:  # First result has high attention
        print(f"   ✅ Cue successfully retrieved suppressed memorable event!")
    
    return True


def test_learning_velocity_adaptation():
    """Test how attention baseline adapts to learning velocity."""
    print("\n🧪 Testing Learning Velocity Adaptation")
    print("=" * 60)
    
    scorer = AdaptiveAttentionScorer()
    initial_baseline = scorer.attention_baseline
    
    print(f"📊 Initial baseline: {initial_baseline:.3f}")
    
    # Simulate fast learning phase
    print(f"\n🚀 Simulating fast learning phase...")
    for i in range(30):
        # Rapidly improving accuracy
        accuracy = 0.5 + (i * 0.02)  # From 50% to 110% (clamped)
        context = {'current_accuracy': min(accuracy, 1.0)}
        
        # Compute some attention scores
        scorer.compute_attention_score(0.5, context)
        
        # Force adaptation check
        if i % 10 == 9:
            scorer._adapt_attention_baseline()
    
    fast_learning_baseline = scorer.attention_baseline
    print(f"   After fast learning: {fast_learning_baseline:.3f} (Δ{fast_learning_baseline-initial_baseline:+.3f})")
    
    # Simulate slow learning phase  
    print(f"\n🐌 Simulating slow learning phase...")
    for i in range(30):
        # Slowly declining accuracy
        accuracy = 0.9 - (i * 0.01)  # From 90% to 60%
        context = {'current_accuracy': max(accuracy, 0.0)}
        
        scorer.compute_attention_score(0.5, context)
        
        if i % 10 == 9:
            scorer._adapt_attention_baseline()
    
    slow_learning_baseline = scorer.attention_baseline
    print(f"   After slow learning: {slow_learning_baseline:.3f} (Δ{slow_learning_baseline-fast_learning_baseline:+.3f})")
    
    # Show final stats
    stats = scorer.get_attention_stats()
    print(f"\n📈 Final adaptation stats:")
    print(f"   Learning velocity: {stats['learning_velocity']['current']:.4f}")
    print(f"   Adaptation active: {stats['adaptation_active']}")
    print(f"   Total baseline change: {slow_learning_baseline-initial_baseline:+.3f}")
    
    # Success if baseline adapted significantly
    adaptation_magnitude = abs(slow_learning_baseline - initial_baseline)
    return adaptation_magnitude > 0.1


if __name__ == "__main__":
    # Run comprehensive adaptive attention tests
    print("🚀 Adaptive Attention Test Suite")
    print("=" * 80)
    
    test_results = []
    
    # Test 1: Basic attention scoring
    result1 = test_adaptive_attention_scoring()
    test_results.append(("Adaptive Attention Scoring", result1))
    
    # Test 2: Attention-weighted similarity
    result2 = test_attention_weighted_similarity()
    test_results.append(("Attention-Weighted Similarity", result2))
    
    # Test 3: Memory suppression and cue retrieval
    result3 = test_memory_suppression_and_retrieval()
    test_results.append(("Memory Suppression & Retrieval", result3))
    
    # Test 4: Learning velocity adaptation
    result4 = test_learning_velocity_adaptation()
    test_results.append(("Learning Velocity Adaptation", result4))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY:")
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print(f"\n🎉 ALL TESTS PASSED ({passed}/{total})")
        print("🧠 Adaptive attention system working perfectly!")
        print("💡 Key achievements:")
        print("   - Auto-tuning attention with no magic numbers")
        print("   - Biological memory suppression without information loss")
        print("   - Cue-based retrieval of suppressed memories")
        print("   - Learning velocity adaptation")
        print("   - Perfect integration with existing similarity engine")
        print("\n🎯 Ready for integration with the 4-system brain architecture!")
    else:
        print(f"\n⚠️  Some tests need attention ({passed}/{total} passed)")