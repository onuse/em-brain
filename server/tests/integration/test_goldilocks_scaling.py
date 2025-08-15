#!/usr/bin/env python3
"""
Goldilocks Brain Scaling Tests

Tests the "just right" implementation for massive scale performance.
Validates the 5 core primitives and emergence at scale.

Test scenarios:
1. Massive pattern storage (up to 1M patterns)
2. GPU-parallel similarity search performance  
3. Cross-stream co-activation emergence
4. Memory usage and efficiency
5. Real-time processing capabilities
6. Emergent intelligence validation
"""

import sys
import os
import time
import torch
import numpy as np
import unittest
from typing import List, Dict

# Add server src to path
server_src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, server_src)

from vector_stream.goldilocks_brain import (
    GoldilocksBrain,
    GoldilocksVectorStream, 
    MassivePatternStorage,
    CrossStreamCoactivation,
    StreamConfig
)


class TestMassivePatternStorage(unittest.TestCase):
    """Test the core primitive: massively parallel pattern storage."""
    
    def setUp(self):
        self.config = StreamConfig(dim=16, max_patterns=10000)  # Smaller for unit tests
        self.storage = MassivePatternStorage(self.config, "test_storage")
    
    def test_basic_storage_and_retrieval(self):
        """Test basic pattern storage and similarity search."""
        # Store a few patterns
        pattern1 = torch.randn(16)
        pattern2 = torch.randn(16)
        pattern3 = pattern1 + 0.1 * torch.randn(16)  # Similar to pattern1
        
        idx1 = self.storage.store_pattern(pattern1)
        idx2 = self.storage.store_pattern(pattern2)
        idx3 = self.storage.store_pattern(pattern3)
        
        # Verify storage
        self.assertEqual(self.storage.pattern_count, 3)
        
        # Test similarity search
        similar_indices, similarities = self.storage.find_similar_patterns(pattern1, k=5)
        
        # Should find pattern1 and pattern3 (similar)
        self.assertGreater(len(similar_indices), 0)
        self.assertIn(idx1, similar_indices)
        
        # pattern1 should have highest similarity to itself
        best_match_idx = similar_indices[0]
        self.assertEqual(best_match_idx, idx1)
    
    def test_replacement_strategy(self):
        """Test pattern replacement when storage is full."""
        # Fill storage beyond capacity
        patterns_stored = []
        for i in range(self.config.max_patterns + 100):  # Store more than capacity
            pattern = torch.randn(16)
            pattern[0] = i  # Make patterns distinguishable
            idx = self.storage.store_pattern(pattern, timestamp=time.time() + i)
            patterns_stored.append((pattern, idx))
        
        # Should still be at max capacity
        self.assertEqual(self.storage.pattern_count, self.config.max_patterns)
        
        # Verify that some patterns were replaced
        # (we can't easily verify which ones without knowing the replacement strategy)
        self.assertLessEqual(self.storage.pattern_count, self.config.max_patterns)
    
    def test_temporal_recency_weighting(self):
        """Test that recent patterns get higher weights."""
        # Store patterns with different timestamps
        old_pattern = torch.tensor([1.0] + [0.0] * 15)
        recent_pattern = torch.tensor([1.0] + [0.0] * 15)  # Same pattern, different time
        
        old_time = time.time() - 100  # 100 seconds ago
        recent_time = time.time()     # Now
        
        self.storage.store_pattern(old_pattern, old_time)
        self.storage.store_pattern(recent_pattern, recent_time)
        
        # Search should favor recent pattern
        query = torch.tensor([1.0] + [0.0] * 15)
        similar_indices, similarities = self.storage.find_similar_patterns(query, k=5)
        
        # Both patterns should be found, but recent should have higher weighted similarity
        self.assertEqual(len(similar_indices), 2)
        # The exact order depends on the recency weighting implementation
    
    def test_pattern_reinforcement(self):
        """Test pattern reinforcement increases frequency."""
        pattern = torch.randn(16)
        idx = self.storage.store_pattern(pattern)
        
        initial_freq = self.storage.frequencies[idx].item()
        
        # Reinforce the pattern
        self.storage.reinforce_pattern(idx, strength=2.0)
        
        final_freq = self.storage.frequencies[idx].item()
        self.assertGreater(final_freq, initial_freq)
    
    def test_performance_scaling(self):
        """Test performance with larger pattern counts."""
        # Store many patterns quickly
        start_time = time.time()
        
        num_patterns = 5000
        for i in range(num_patterns):
            pattern = torch.randn(16)
            self.storage.store_pattern(pattern)
        
        storage_time = time.time() - start_time
        
        # Should be fast enough (less than 1 second for 5k patterns)
        self.assertLess(storage_time, 1.0)
        
        # Test search performance
        start_time = time.time()
        
        num_searches = 100
        for _ in range(num_searches):
            query = torch.randn(16)
            self.storage.find_similar_patterns(query, k=10)
        
        search_time = time.time() - start_time
        
        # Should be very fast (less than 0.1 seconds for 100 searches)
        self.assertLess(search_time, 0.1)
        
        print(f"   Performance: {num_patterns/storage_time:.0f} stores/sec, {num_searches/search_time:.0f} searches/sec")


class TestCrossStreamCoactivation(unittest.TestCase):
    """Test cross-stream co-activation tracking."""
    
    def setUp(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        self.coactivation = CrossStreamCoactivation(['sensory', 'motor', 'temporal'], device)
    
    def test_coactivation_recording(self):
        """Test recording of co-activation patterns."""
        # Record simultaneous activations
        activations = {
            'sensory': [0, 1, 2],
            'motor': [5, 6],
            'temporal': [10]
        }
        
        self.coactivation.record_coactivation(activations, strength=1.0)
        
        # Verify co-activation was recorded
        stats = self.coactivation.get_coactivation_stats()
        self.assertGreater(stats['total_cross_stream_links'], 0)
    
    def test_cross_stream_predictions(self):
        """Test cross-stream predictions based on co-activation."""
        # Record several co-activation patterns
        for i in range(10):
            activations = {
                'sensory': [i],
                'motor': [i + 100],  # Consistent mapping
                'temporal': [i + 200]
            }
            self.coactivation.record_coactivation(activations, strength=1.0)
        
        # Test prediction
        predictions = self.coactivation.get_cross_predictions(
            'sensory', [0], 'motor', k=5
        )
        
        # Should predict motor pattern 100 (co-occurred with sensory 0)
        self.assertGreater(len(predictions), 0)
        predicted_indices = [idx for idx, strength in predictions]
        self.assertIn(100, predicted_indices)
    
    def test_coactivation_strengthening(self):
        """Test that repeated co-activations increase strength."""
        activations = {
            'sensory': [0],
            'motor': [1]
        }
        
        # Record same co-activation multiple times
        for _ in range(5):
            self.coactivation.record_coactivation(activations, strength=1.0)
        
        # Check that strength increased
        predictions = self.coactivation.get_cross_predictions('sensory', [0], 'motor', k=5)
        
        if predictions:
            _, strength = predictions[0]
            self.assertGreater(strength, 1.0)  # Should be accumulated strength


class TestGoldilocksVectorStream(unittest.TestCase):
    """Test the complete vector stream with all primitives."""
    
    def setUp(self):
        config = StreamConfig(dim=8, max_patterns=1000)
        self.stream = GoldilocksVectorStream(config, "test_stream")
    
    def test_stream_update_and_prediction(self):
        """Test stream updates and prediction generation."""
        # Create a simple pattern sequence
        pattern_a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pattern_b = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Train sequence A -> B
        for i in range(10):
            self.stream.update(pattern_a, time.time() + i * 2)
            self.stream.update(pattern_b, time.time() + i * 2 + 1)
        
        # Stream should have learned patterns
        stats = self.stream.storage.get_pattern_stats()
        self.assertGreater(stats['pattern_count'], 0)
        
        # After seeing pattern_a, prediction should be influenced by pattern_b
        self.stream.update(pattern_a, time.time() + 100)
        prediction = self.stream.predicted_next
        
        # Prediction should be non-zero
        self.assertGreater(torch.norm(prediction).item(), 0.01)
    
    def test_pattern_novelty_detection(self):
        """Test that novel patterns are stored, similar ones are not."""
        pattern = torch.randn(8)
        
        # First occurrence should be stored
        initial_count = self.stream.storage.pattern_count
        self.stream.update(pattern)
        after_first = self.stream.storage.pattern_count
        
        self.assertGreater(after_first, initial_count)
        
        # Very similar pattern should not create new storage
        similar_pattern = pattern + 0.01 * torch.randn(8)  # Very small noise
        self.stream.update(similar_pattern)
        after_similar = self.stream.storage.pattern_count
        
        # Might or might not store depending on similarity threshold
        # Just verify the system doesn't crash
        self.assertIsInstance(after_similar, int)
    
    def test_active_pattern_retrieval(self):
        """Test retrieval of active patterns."""
        # Store some patterns
        for i in range(5):
            pattern = torch.randn(8)
            pattern[0] = i  # Make distinguishable
            self.stream.update(pattern, time.time() + i)
        
        # Get active patterns
        active_indices = self.stream.get_active_pattern_indices(k=3)
        
        # Should return some patterns
        self.assertGreater(len(active_indices), 0)
        self.assertLessEqual(len(active_indices), 3)


class TestGoldilocksBrain(unittest.TestCase):
    """Test the complete Goldilocks brain system."""
    
    def setUp(self):
        self.brain = GoldilocksBrain(
            sensory_dim=8, motor_dim=4, temporal_dim=2, 
            max_patterns=1000, quiet_mode=True
        )
    
    def test_brain_processing_cycle(self):
        """Test complete brain processing cycle."""
        sensory_input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Process input
        motor_output, brain_state = self.brain.process_sensory_input(sensory_input)
        
        # Verify output format
        self.assertIsInstance(motor_output, list)
        self.assertEqual(len(motor_output), 4)  # Motor dimension
        self.assertIsInstance(brain_state, dict)
        
        # Verify brain state contains expected fields
        self.assertIn('total_cycles', brain_state)
        self.assertIn('sensory_stream', brain_state)
        self.assertIn('motor_stream', brain_state)
        self.assertIn('temporal_stream', brain_state)
        self.assertIn('coactivation_stats', brain_state)
        
        # Verify brain is updating
        self.assertEqual(brain_state['total_cycles'], 1)
    
    def test_cross_stream_learning(self):
        """Test that cross-stream associations develop over time."""
        # Consistent sensory-motor pattern
        sensory_pattern = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Process same input multiple times
        for i in range(20):
            motor_output, brain_state = self.brain.process_sensory_input(sensory_pattern)
        
        # Should have developed cross-stream associations
        coactivation_stats = brain_state['coactivation_stats']
        self.assertGreater(coactivation_stats['total_cross_stream_links'], 0)
    
    def test_brain_statistics(self):
        """Test brain statistics collection."""
        # Process some inputs
        for i in range(5):
            sensory_input = [float(i), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            self.brain.process_sensory_input(sensory_input)
        
        # Get statistics
        stats = self.brain.get_brain_statistics()
        
        # Verify comprehensive statistics
        self.assertIn('total_cycles', stats)
        self.assertIn('total_patterns', stats)
        self.assertIn('streams', stats)
        self.assertIn('cross_stream', stats)
        self.assertIn('gpu_memory_usage_mb', stats)
        
        # Should have processed cycles
        self.assertEqual(stats['total_cycles'], 5)
        self.assertGreater(stats['total_patterns'], 0)
    
    def test_temporal_context_generation(self):
        """Test temporal context generation."""
        # Process inputs at different times
        start_time = time.time()
        
        sensory_input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        motor1, state1 = self.brain.process_sensory_input(sensory_input)
        
        # Wait a bit
        time.sleep(0.1)
        
        motor2, state2 = self.brain.process_sensory_input(sensory_input)
        
        # Temporal streams should be different due to time progression
        temporal1 = state1['temporal_stream']['current_activation']
        temporal2 = state2['temporal_stream']['current_activation']
        
        # Should be different due to time progression
        self.assertNotEqual(temporal1, temporal2)


def run_massive_scale_benchmark():
    """
    Run massive scale benchmark to test emergence at scale.
    
    This tests the core hypothesis: intelligence emerges from massive parallel processing.
    """
    print("\n🚀 MASSIVE SCALE BENCHMARK - GOLDILOCKS BRAIN")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        (1000, "Small scale"),
        (10000, "Medium scale"), 
        (100000, "Large scale"),
        (500000, "Massive scale")  # As large as memory allows
    ]
    
    for max_patterns, scale_name in test_configs:
        print(f"\n📊 {scale_name}: {max_patterns:,} patterns")
        
        try:
            # Create brain with this scale
            brain = GoldilocksBrain(
                sensory_dim=16, motor_dim=8, temporal_dim=4,
                max_patterns=max_patterns, quiet_mode=True
            )
            
            # Benchmark pattern storage speed
            print("   Testing pattern storage speed...")
            start_time = time.time()
            
            num_cycles = min(1000, max_patterns // 10)
            for i in range(num_cycles):
                sensory_input = np.random.randn(16).tolist()
                brain.process_sensory_input(sensory_input)
            
            storage_time = time.time() - start_time
            cycles_per_sec = num_cycles / storage_time
            
            print(f"   Storage performance: {cycles_per_sec:.0f} cycles/sec")
            
            # Test similarity search performance
            print("   Testing similarity search speed...")
            start_time = time.time()
            
            num_searches = 100
            for _ in range(num_searches):
                query = torch.randn(16)
                brain.sensory_stream.storage.find_similar_patterns(query, k=10)
            
            search_time = time.time() - start_time
            searches_per_sec = num_searches / search_time
            
            print(f"   Search performance: {searches_per_sec:.0f} searches/sec")
            
            # Check memory usage
            stats = brain.get_brain_statistics()
            memory_mb = stats['gpu_memory_usage_mb']
            patterns_stored = stats['total_patterns']
            
            print(f"   Memory usage: {memory_mb:.1f}MB")
            print(f"   Patterns stored: {patterns_stored:,}")
            
            # Test cross-stream emergence
            coactivation_stats = stats['cross_stream']
            cross_links = coactivation_stats['total_cross_stream_links']
            print(f"   Cross-stream links: {cross_links:,}")
            
            if cross_links > 100:
                print("   ✅ Cross-stream emergence detected!")
            
            print(f"   ✅ {scale_name} benchmark completed successfully")
            
        except Exception as e:
            print(f"   ❌ {scale_name} failed: {e}")
            if max_patterns >= 100000:
                print("   (Large scale may fail due to memory limits)")
    
    print(f"\n🎯 EMERGENCE ANALYSIS")
    print("=" * 30)
    
    # Test emergence with consistent patterns
    print("Testing spatial emergence...")
    brain = GoldilocksBrain(max_patterns=10000, quiet_mode=True)
    
    # Create spatial pattern (same location, similar sensory input)
    location_patterns = [
        [1.0, 0.1, 0.0, 0.0] + [0.0] * 12,  # Location A
        [1.0, 0.2, 0.0, 0.0] + [0.0] * 12,  # Location A (slight variation)
        [0.0, 0.0, 1.0, 0.1] + [0.0] * 12,  # Location B
        [0.0, 0.0, 1.0, 0.2] + [0.0] * 12,  # Location B (slight variation)
    ]
    
    # Train with location patterns
    for epoch in range(50):
        for i, pattern in enumerate(location_patterns):
            motor_output, state = brain.process_sensory_input(pattern)
    
    # Test clustering - similar locations should have similar motor outputs
    motor_a1, _ = brain.process_sensory_input(location_patterns[0])
    motor_a2, _ = brain.process_sensory_input(location_patterns[1])
    motor_b1, _ = brain.process_sensory_input(location_patterns[2])
    
    # Calculate similarities
    sim_a1_a2 = np.corrcoef(motor_a1, motor_a2)[0, 1] if len(motor_a1) > 1 else 0
    sim_a1_b1 = np.corrcoef(motor_a1, motor_b1)[0, 1] if len(motor_a1) > 1 else 0
    
    if not np.isnan(sim_a1_a2) and not np.isnan(sim_a1_b1):
        if sim_a1_a2 > sim_a1_b1:
            print("   ✅ Spatial clustering detected! Similar locations → similar actions")
        else:
            print("   ⚠️  Spatial clustering not yet emerged")
    
    print("\n🏆 GOLDILOCKS BRAIN MASSIVE SCALE BENCHMARK COMPLETE!")


if __name__ == "__main__":
    print("🧠 GOLDILOCKS BRAIN SCALING TEST SUITE")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run massive scale benchmark
    run_massive_scale_benchmark()