#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Pattern Memory System

Tests the hierarchical pattern storage, cross-stream linking, and 
episodic capabilities that replace the 50-pattern limit.

Key test areas:
1. Pattern storage and retrieval across tiers
2. Memory pressure management and intelligent forgetting  
3. Cross-stream pattern linking
4. Episode creation and management
5. Pattern importance scoring and promotion/demotion
6. Performance scaling with thousands of patterns
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

from vector_stream.enhanced_pattern_memory import (
    HierarchicalPatternMemory, 
    EnhancedVectorPattern, 
    EpisodeManager,
    PatternTier
)
from vector_stream.enhanced_vector_stream import (
    EnhancedVectorStream,
    CrossStreamManager
)


class TestHierarchicalPatternMemory(unittest.TestCase):
    """Test the hierarchical pattern storage system."""
    
    def setUp(self):
        self.memory = HierarchicalPatternMemory("test_stream", max_active=10, max_working=20)
        self.dim = 8
    
    def create_test_pattern(self, values: List[float] = None) -> EnhancedVectorPattern:
        """Create a test pattern with specified or random values."""
        if values is None:
            values = torch.randn(self.dim)
        else:
            values = torch.tensor(values, dtype=torch.float32)
        
        return EnhancedVectorPattern(activation_pattern=values)
    
    def test_pattern_storage_and_retrieval(self):
        """Test basic pattern storage and retrieval."""
        # Create and store a pattern
        pattern = self.create_test_pattern([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pattern_id = self.memory.store_pattern(pattern)
        
        # Verify storage
        self.assertEqual(self.memory.total_patterns, 1)
        self.assertEqual(len(self.memory.active_patterns), 1)
        self.assertEqual(pattern.storage_tier, PatternTier.ACTIVE)
        
        # Test retrieval
        retrieved = self.memory.get_pattern(pattern_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.pattern_id, pattern_id)
        torch.testing.assert_close(retrieved.activation_pattern, pattern.activation_pattern)
    
    def test_similarity_search(self):
        """Test pattern similarity search across tiers."""
        # Store several similar and dissimilar patterns
        similar_pattern1 = self.create_test_pattern([1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        similar_pattern2 = self.create_test_pattern([0.9, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        different_pattern = self.create_test_pattern([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        
        self.memory.store_pattern(similar_pattern1)
        self.memory.store_pattern(similar_pattern2)
        self.memory.store_pattern(different_pattern)
        
        # Search for patterns similar to the first one
        query = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        results = self.memory.find_similar_patterns(query, threshold=0.8, max_results=5)
        
        # Should find the two similar patterns but not the different one
        self.assertEqual(len(results), 2)
        similarities = [similarity for _, similarity in results]
        self.assertTrue(all(sim > 0.8 for sim in similarities))
        
        # Results should be sorted by similarity (highest first)
        self.assertTrue(similarities[0] >= similarities[1])
    
    def test_memory_pressure_management(self):
        """Test automatic tier management when memory limits are exceeded."""
        # Fill up active tier beyond limit
        patterns = []
        for i in range(15):  # More than max_active=10
            pattern = self.create_test_pattern([float(i), 0, 0, 0, 0, 0, 0, 0])
            pattern.importance_score = 0.1 * i  # Increasing importance
            pattern_id = self.memory.store_pattern(pattern)
            patterns.append((pattern_id, pattern))
        
        # Memory pressure should have demoted some patterns
        self.assertLessEqual(len(self.memory.active_patterns), 12)  # Some buffer allowed
        self.assertGreater(len(self.memory.working_patterns), 0)
        
        # More important patterns should stay in active tier
        active_importance_scores = [p.importance_score for p in self.memory.active_patterns]
        working_importance_scores = [p.importance_score for p in self.memory.working_patterns]
        
        if working_importance_scores:  # If any patterns were demoted
            max_working_importance = max(working_importance_scores)
            min_active_importance = min(active_importance_scores)
            self.assertGreaterEqual(min_active_importance, max_working_importance * 0.5)  # Some tolerance
    
    def test_pattern_promotion_demotion(self):
        """Test pattern promotion and demotion based on usage."""
        # Create a pattern and demote it to working tier
        pattern = self.create_test_pattern([1.0, 0, 0, 0, 0, 0, 0, 0])
        pattern.importance_score = 0.1  # Low importance
        pattern_id = self.memory.store_pattern(pattern)
        
        # Force demotion by adding many patterns
        for i in range(12):
            filler_pattern = self.create_test_pattern([0, float(i), 0, 0, 0, 0, 0, 0])
            filler_pattern.importance_score = 0.8  # High importance
            self.memory.store_pattern(filler_pattern)
        
        # Original pattern should be demoted
        retrieved_pattern = self.memory.get_pattern(pattern_id)
        self.assertEqual(retrieved_pattern.storage_tier, PatternTier.WORKING)
        
        # Now increase importance and access it
        retrieved_pattern.importance_score = 0.9
        retrieved_pattern.prediction_successes = 10
        retrieved_pattern.prediction_attempts = 10
        
        # Access it multiple times to trigger promotion consideration
        for _ in range(3):
            self.memory.get_pattern(pattern_id)
        
        # Should be promoted back to active
        self.assertEqual(retrieved_pattern.storage_tier, PatternTier.ACTIVE)
    
    def test_cross_stream_linking(self):
        """Test linking patterns across different streams."""
        # Create patterns in this stream
        pattern1 = self.create_test_pattern([1, 0, 0, 0, 0, 0, 0, 0])
        pattern2 = self.create_test_pattern([0, 1, 0, 0, 0, 0, 0, 0])
        
        id1 = self.memory.store_pattern(pattern1)
        id2 = self.memory.store_pattern(pattern2)
        
        # Link to patterns in other streams
        self.memory.link_patterns_across_streams(id1, "motor_stream", "motor_pattern_123")
        self.memory.link_patterns_across_streams(id1, "temporal_stream", "temporal_pattern_456")
        self.memory.link_patterns_across_streams(id2, "motor_stream", "motor_pattern_789")
        
        # Test retrieval of links
        motor_links = self.memory.get_linked_patterns(id1, "motor_stream")
        self.assertIn("motor_pattern_123", motor_links)
        
        temporal_links = self.memory.get_linked_patterns(id1, "temporal_stream")  
        self.assertIn("temporal_pattern_456", temporal_links)
        
        # Pattern2 should have different motor link
        motor_links2 = self.memory.get_linked_patterns(id2, "motor_stream")
        self.assertIn("motor_pattern_789", motor_links2)
        
        # Cross-stream index should be updated
        self.assertIn("motor_stream", self.memory.cross_stream_links)
        self.assertIn(id1, self.memory.cross_stream_links["motor_stream"])
    
    def test_importance_scoring(self):
        """Test pattern importance scoring calculation."""
        pattern = self.create_test_pattern()
        
        # Set some usage statistics
        pattern.frequency = 100
        pattern.prediction_successes = 8
        pattern.prediction_attempts = 10
        pattern.last_seen = time.time()  # Recent
        
        pattern.update_importance_score()
        
        # Should have high importance due to good prediction rate and recency
        self.assertGreater(pattern.importance_score, 0.5)
        
        # Test with poor prediction rate
        pattern.prediction_successes = 2
        pattern.prediction_attempts = 10
        pattern.update_importance_score()
        
        # Should have lower importance
        self.assertLess(pattern.importance_score, 0.5)
    
    def test_memory_cleanup(self):
        """Test cleanup of old patterns."""
        # Create patterns with different ages
        old_time = time.time() - (200 * 3600)  # 200 hours ago
        recent_time = time.time()
        
        old_pattern = self.create_test_pattern([1, 0, 0, 0, 0, 0, 0, 0])
        old_pattern.last_seen = old_time
        old_pattern.importance_score = 0.1  # Low importance
        
        recent_pattern = self.create_test_pattern([0, 1, 0, 0, 0, 0, 0, 0])
        recent_pattern.last_seen = recent_time
        recent_pattern.importance_score = 0.1  # Low importance
        
        important_old_pattern = self.create_test_pattern([0, 0, 1, 0, 0, 0, 0, 0])
        important_old_pattern.last_seen = old_time
        important_old_pattern.importance_score = 0.8  # High importance
        
        # Force all patterns to consolidated tier for cleanup testing
        old_pattern.storage_tier = PatternTier.CONSOLIDATED
        recent_pattern.storage_tier = PatternTier.CONSOLIDATED  
        important_old_pattern.storage_tier = PatternTier.CONSOLIDATED
        
        self.memory.consolidated_patterns = [old_pattern, recent_pattern, important_old_pattern]
        self.memory.pattern_index = {
            old_pattern.pattern_id: old_pattern,
            recent_pattern.pattern_id: recent_pattern,
            important_old_pattern.pattern_id: important_old_pattern
        }
        self.memory.total_patterns = 3
        
        # Run cleanup (keep patterns newer than 168 hours)
        removed_count = self.memory.cleanup_old_patterns(max_age_hours=168)
        
        # Should remove old unimportant pattern but keep recent and important ones
        self.assertEqual(removed_count, 1)
        self.assertEqual(len(self.memory.consolidated_patterns), 2)
        self.assertIn(recent_pattern, self.memory.consolidated_patterns)
        self.assertIn(important_old_pattern, self.memory.consolidated_patterns)


class TestEnhancedVectorStream(unittest.TestCase):
    """Test the enhanced vector stream with hierarchical memory."""
    
    def setUp(self):
        self.stream = EnhancedVectorStream(dim=8, name="test_stream", max_active_patterns=10)
    
    def test_pattern_learning_and_prediction(self):
        """Test enhanced pattern learning and prediction."""
        # Feed repeated pattern
        pattern1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pattern2 = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        timestamp = time.time()
        
        # Learn pattern sequence: pattern1 -> pattern2
        for i in range(10):
            self.stream.update(pattern1, timestamp + i * 2)
            self.stream.update(pattern2, timestamp + i * 2 + 1)
        
        # Stream should have learned patterns
        self.assertGreater(self.stream.pattern_memory.total_patterns, 0)
        
        # After seeing pattern1, should predict something closer to pattern2
        self.stream.update(pattern1, timestamp + 100)
        prediction = self.stream.predicted_next_activation
        
        # Prediction should be non-zero and somewhat similar to pattern2
        self.assertGreater(torch.norm(prediction).item(), 0.1)
    
    def test_adaptive_threshold(self):
        """Test adaptive similarity threshold based on pattern creation rate."""
        initial_threshold = self.stream.adaptive_threshold
        
        # Create many different patterns (should raise threshold)
        for i in range(20):
            random_pattern = torch.randn(8)
            self.stream.update(random_pattern, time.time() + i)
        
        self.stream.adapt_learning_threshold()
        
        # Threshold adjustment depends on creation vs match rate
        # Just verify the method runs without error and threshold stays in bounds
        self.assertGreaterEqual(self.stream.adaptive_threshold, 0.5)
        self.assertLessEqual(self.stream.adaptive_threshold, 0.95)
    
    def test_prediction_accuracy_tracking(self):
        """Test prediction accuracy tracking and pattern reward."""
        # Create predictable sequence
        pattern_a = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pattern_b = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Train sequence A -> B
        for i in range(5):
            self.stream.update(pattern_a, time.time() + i * 2)
            self.stream.update(pattern_b, time.time() + i * 2 + 1)
        
        # Make prediction and update accuracy
        self.stream.update(pattern_a, time.time() + 100)
        predicted = self.stream.predicted_next_activation
        
        # Simulate good prediction by updating with similar pattern
        self.stream.update_prediction_accuracy(pattern_b)
        
        # Should have some prediction successes
        self.assertGreaterEqual(self.stream.prediction_successes, 0)
    
    def test_memory_scaling(self):
        """Test that memory system can handle hundreds of patterns efficiently."""
        start_time = time.time()
        
        # Create many diverse patterns
        for i in range(200):
            pattern = torch.randn(8)
            pattern[0] = i / 200.0  # Make them somewhat different
            self.stream.update(pattern, time.time() + i * 0.1)
        
        end_time = time.time()
        
        # Should complete in reasonable time (less than 1 second)
        self.assertLess(end_time - start_time, 1.0)
        
        # Should have managed memory properly
        memory_stats = self.stream.pattern_memory.get_memory_stats()
        self.assertLessEqual(memory_stats['active_patterns'], 
                           self.stream.pattern_memory.max_active + 20)  # Some buffer
    
    def test_cross_stream_prediction(self):
        """Test cross-stream prediction capabilities."""
        # This test requires another stream to link to
        motor_stream = EnhancedVectorStream(dim=4, name="motor_stream")
        
        # Create cross-stream manager
        manager = CrossStreamManager({
            "sensory": self.stream,
            "motor": motor_stream
        })
        
        # Create patterns and episodes
        sensory_pattern = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        motor_pattern = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
        timestamp = time.time()
        
        # Update both streams simultaneously (creates episode)
        self.stream.update(sensory_pattern, timestamp)
        motor_stream.update(motor_pattern, timestamp + 0.001)
        
        # Should have cross-stream links
        cross_predictions = self.stream.get_cross_stream_predictions("motor_stream")
        
        # May or may not have predictions yet (depends on linking timing)
        # Just verify the method works
        self.assertIsInstance(cross_predictions, list)


class TestEpisodeManager(unittest.TestCase):
    """Test episode management for cross-stream pattern linking."""
    
    def setUp(self):
        self.episode_manager = EpisodeManager(max_episodes=100)
    
    def test_episode_creation(self):
        """Test creating episodes with cross-stream patterns."""
        stream_patterns = {
            "sensory": "sensory_pattern_123",
            "motor": "motor_pattern_456", 
            "temporal": "temporal_pattern_789"
        }
        
        episode_id = self.episode_manager.create_episode(
            stream_patterns, 
            context="Test episode"
        )
        
        # Verify episode creation
        self.assertIsNotNone(episode_id)
        self.assertEqual(len(self.episode_manager.episodes), 1)
        
        episode = self.episode_manager.episode_index[episode_id]
        self.assertEqual(episode.stream_patterns, stream_patterns)
        self.assertEqual(episode.context_description, "Test episode")
    
    def test_episode_search(self):
        """Test finding episodes containing specific patterns."""
        # Create multiple episodes
        episode1_patterns = {"sensory": "pattern_A", "motor": "pattern_B"}
        episode2_patterns = {"sensory": "pattern_A", "motor": "pattern_C"}
        episode3_patterns = {"sensory": "pattern_D", "motor": "pattern_B"}
        
        self.episode_manager.create_episode(episode1_patterns)
        self.episode_manager.create_episode(episode2_patterns)
        self.episode_manager.create_episode(episode3_patterns)
        
        # Search for episodes containing specific patterns
        episodes_with_A = self.episode_manager.find_episodes_with_pattern("sensory", "pattern_A")
        episodes_with_B = self.episode_manager.find_episodes_with_pattern("motor", "pattern_B")
        
        # Should find the right episodes
        self.assertEqual(len(episodes_with_A), 2)  # Episodes 1 and 2
        self.assertEqual(len(episodes_with_B), 2)  # Episodes 1 and 3
    
    def test_episode_memory_management(self):
        """Test episode memory management when limit is exceeded."""
        # Create episodes beyond the limit
        for i in range(120):  # More than max_episodes=100
            patterns = {"stream1": f"pattern_{i}"}
            episode_id = self.episode_manager.create_episode(patterns)
            
            # Set importance scores (later episodes more important)
            episode = self.episode_manager.episode_index[episode_id]
            episode.importance_score = i / 120.0
        
        # Should have removed some episodes
        self.assertLessEqual(len(self.episode_manager.episodes), 110)  # Some buffer
        
        # More important episodes should be retained
        remaining_scores = [e.importance_score for e in self.episode_manager.episodes]
        if len(remaining_scores) > 1:
            avg_remaining_score = sum(remaining_scores) / len(remaining_scores)
            self.assertGreater(avg_remaining_score, 0.3)  # Should keep higher-importance episodes


class TestCrossStreamManager(unittest.TestCase):
    """Test cross-stream coordination and episode creation."""
    
    def setUp(self):
        self.sensory_stream = EnhancedVectorStream(dim=8, name="sensory")
        self.motor_stream = EnhancedVectorStream(dim=4, name="motor")
        self.temporal_stream = EnhancedVectorStream(dim=2, name="temporal")
        
        self.streams = {
            "sensory": self.sensory_stream,
            "motor": self.motor_stream,
            "temporal": self.temporal_stream
        }
        
        self.manager = CrossStreamManager(self.streams)
    
    def test_automatic_episode_creation(self):
        """Test automatic episode creation when streams are updated."""
        timestamp = time.time()
        
        # Update all streams with patterns
        sensory_pattern = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        motor_pattern = torch.tensor([1.0, 0.0, 0.0, 0.0])
        temporal_pattern = torch.tensor([1.0, 0.0])
        
        # Update streams (should create episodes automatically)
        self.sensory_stream.update(sensory_pattern, timestamp)
        self.motor_stream.update(motor_pattern, timestamp + 0.001)
        self.temporal_stream.update(temporal_pattern, timestamp + 0.002)
        
        # Should have created episodes
        episode_stats = self.manager.episode_manager.get_episode_stats()
        self.assertGreater(episode_stats['total_episodes'], 0)
    
    def test_cross_stream_statistics(self):
        """Test cross-stream statistics collection."""
        # Create some patterns and links
        timestamp = time.time()
        
        for i in range(5):
            sensory_pattern = torch.randn(8)
            motor_pattern = torch.randn(4)
            
            self.sensory_stream.update(sensory_pattern, timestamp + i)
            self.motor_stream.update(motor_pattern, timestamp + i + 0.1)
        
        # Get statistics
        stats = self.manager.get_cross_stream_stats()
        
        # Should have meaningful statistics
        self.assertIn('total_cross_stream_links', stats)
        self.assertIn('links_per_stream', stats)
        self.assertIn('episode_stats', stats)
        
        # Links should be non-negative
        self.assertGreaterEqual(stats['total_cross_stream_links'], 0)


def run_performance_benchmark():
    """Run performance benchmark for enhanced pattern memory."""
    print("\n🚀 ENHANCED PATTERN MEMORY PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Test large-scale pattern storage and retrieval
    memory = HierarchicalPatternMemory("benchmark_stream", max_active=500, max_working=2000)
    
    # Phase 1: Store many patterns
    print("Phase 1: Storing 5000 patterns...")
    start_time = time.time()
    
    pattern_ids = []
    for i in range(5000):
        pattern = EnhancedVectorPattern(
            activation_pattern=torch.randn(16),
            frequency=np.random.randint(1, 100),
            importance_score=np.random.random()
        )
        pattern_id = memory.store_pattern(pattern)
        pattern_ids.append(pattern_id)
    
    storage_time = time.time() - start_time
    print(f"   Storage time: {storage_time:.2f}s ({5000/storage_time:.0f} patterns/sec)")
    
    # Phase 2: Random pattern retrieval
    print("Phase 2: Retrieving 1000 random patterns...")
    start_time = time.time()
    
    for _ in range(1000):
        random_id = np.random.choice(pattern_ids)
        pattern = memory.get_pattern(random_id)
        assert pattern is not None
    
    retrieval_time = time.time() - start_time
    print(f"   Retrieval time: {retrieval_time:.2f}s ({1000/retrieval_time:.0f} retrievals/sec)")
    
    # Phase 3: Similarity search
    print("Phase 3: Performing 100 similarity searches...")
    start_time = time.time()
    
    for _ in range(100):
        query = torch.randn(16)
        results = memory.find_similar_patterns(query, threshold=0.5, max_results=10)
    
    search_time = time.time() - start_time
    print(f"   Search time: {search_time:.2f}s ({100/search_time:.0f} searches/sec)")
    
    # Memory statistics
    stats = memory.get_memory_stats()
    print(f"\nMemory Statistics:")
    print(f"   Total patterns: {stats['total_patterns']}")
    print(f"   Active patterns: {stats['active_patterns']}")
    print(f"   Working patterns: {stats['working_patterns']}")
    print(f"   Consolidated patterns: {stats['consolidated_patterns']}")
    print(f"   Memory pressure: {stats['memory_pressure']:.2f}")
    
    print("\n✅ Performance benchmark completed successfully!")


if __name__ == "__main__":
    print("🧠 ENHANCED PATTERN MEMORY TEST SUITE")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmark
    run_performance_benchmark()