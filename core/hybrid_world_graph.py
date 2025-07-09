"""
Hybrid World Graph - Combines object-based interface with vectorized backend.

This provides 100% API compatibility with the existing WorldGraph while 
delivering massive GPU acceleration for similarity operations.

Key Features:
1. Exact same interface as original WorldGraph
2. Transparent GPU acceleration for similarity searches
3. Automatic fallback to CPU if GPU unavailable
4. Gradual migration path from object-based to vectorized storage
"""

from typing import List, Dict, Optional, Any, Tuple
import time
import random

from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.vectorized_backend import VectorizedBackend, VectorizedExperience


class HybridWorldGraph(WorldGraph):
    """
    Drop-in replacement for WorldGraph with GPU acceleration.
    
    Maintains all existing functionality while adding vectorized backend
    for massive performance improvements in similarity operations.
    """
    
    def __init__(self, **kwargs):
        """Initialize hybrid graph with both object and vectorized storage."""
        # Initialize parent WorldGraph
        super().__init__(**kwargs)
        
        # Add vectorized backend
        self.vectorized_backend = VectorizedBackend(
            initial_capacity=kwargs.get('initial_vectorized_capacity', 10000),
            device=kwargs.get('device', 'auto')
        )
        
        # Hybrid storage mode
        self.use_vectorized_similarity = True  # Can be toggled for comparison
        self.vectorized_indices = {}  # Map node_id -> vectorized_index
        
        # Performance tracking
        self.similarity_method_stats = {
            'vectorized_calls': 0,
            'object_based_calls': 0,
            'vectorized_time': 0.0,
            'object_based_time': 0.0
        }
        
        # Only print if using GPU (the interesting case)
        if str(self.vectorized_backend.device) != 'cpu':
            print(f"Brain Memory: GPU acceleration enabled ({self.vectorized_backend.device})")
    
    def add_experience(self, mental_context: List[float], action_taken: Dict[str, float],
                      predicted_sensory: List[float], actual_sensory: List[float],
                      prediction_error: float) -> ExperienceNode:
        """
        Add experience to both object-based and vectorized storage.
        
        This is a convenience method that creates an ExperienceNode and adds it.
        """
        # Create experience node
        experience_node = ExperienceNode(
            mental_context=mental_context,
            action_taken=action_taken,
            predicted_sensory=predicted_sensory,
            actual_sensory=actual_sensory,
            prediction_error=prediction_error
        )
        
        # Add to parent graph using add_node
        super().add_node(experience_node)
        
        # Also add to vectorized backend for GPU acceleration
        try:
            vectorized_index = self.vectorized_backend.add_experience(experience_node)
            self.vectorized_indices[experience_node.node_id] = vectorized_index
        except Exception as e:
            print(f"Warning: Could not add experience to vectorized backend: {e}")
            # Continue with object-based storage only
        
        return experience_node
    
    def find_similar_experiences(self, query_context: List[float], 
                               similarity_threshold: float = 0.7,
                               max_results: int = 50) -> List[ExperienceNode]:
        """
        Find similar experiences using GPU acceleration when possible.
        
        Automatically falls back to object-based method if vectorized fails.
        """
        # Try vectorized approach first (GPU accelerated)
        if self.use_vectorized_similarity and self.vectorized_backend.get_size() > 0:
            try:
                return self._find_similar_vectorized(query_context, similarity_threshold, max_results)
            except Exception as e:
                print(f"Warning: Vectorized similarity failed: {e}. Falling back to object-based.")
                self.similarity_method_stats['object_based_calls'] += 1
        
        # Fallback to original object-based method (using find_similar_nodes)
        start_time = time.time()
        result = super().find_similar_nodes(query_context, similarity_threshold, max_results)
        self.similarity_method_stats['object_based_time'] += time.time() - start_time
        self.similarity_method_stats['object_based_calls'] += 1
        
        return result
    
    def find_similar_nodes(self, query_context: List[float], 
                          similarity_threshold: float = 0.7,
                          max_results: int = 50) -> List[ExperienceNode]:
        """Alias for backward compatibility with existing WorldGraph API."""
        return self.find_similar_experiences(query_context, similarity_threshold, max_results)
    
    def _find_similar_vectorized(self, query_context: List[float], 
                               similarity_threshold: float = 0.7,
                               max_results: int = 50) -> List[ExperienceNode]:
        """
        GPU-accelerated similarity search using vectorized backend.
        
        This is where the magic happens - one GPU operation across all experiences.
        """
        start_time = time.time()
        
        # Get similarities for ALL experiences simultaneously on GPU
        similarities, indices = self.vectorized_backend.compute_similarities_vectorized(
            query_context, top_k=max_results * 2  # Get extra to filter by threshold
        )
        
        # Filter by threshold and convert back to ExperienceNodes
        results = []
        for sim, idx in zip(similarities, indices):
            if float(sim) >= similarity_threshold:
                # Get the experience from vectorized storage
                vectorized_exp = VectorizedExperience(int(idx), self.vectorized_backend)
                
                # Convert back to traditional ExperienceNode for compatibility
                experience_node = vectorized_exp.to_experience_node()
                results.append(experience_node)
                
                if len(results) >= max_results:
                    break
        
        # Update performance stats
        elapsed_time = time.time() - start_time
        self.similarity_method_stats['vectorized_time'] += elapsed_time
        self.similarity_method_stats['vectorized_calls'] += 1
        
        return results
    
    def get_vectorized_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about hybrid performance."""
        backend_stats = self.vectorized_backend.get_stats()
        
        # Calculate performance ratios
        total_vectorized_time = self.similarity_method_stats['vectorized_time']
        total_object_time = self.similarity_method_stats['object_based_time']
        
        vectorized_calls = self.similarity_method_stats['vectorized_calls']
        object_calls = self.similarity_method_stats['object_based_calls']
        
        avg_vectorized_time = total_vectorized_time / max(1, vectorized_calls)
        avg_object_time = total_object_time / max(1, object_calls)
        
        speedup_ratio = avg_object_time / max(0.001, avg_vectorized_time)
        
        return {
            'backend_stats': backend_stats,
            'similarity_method_stats': self.similarity_method_stats,
            'performance_analysis': {
                'avg_vectorized_time_ms': avg_vectorized_time * 1000,
                'avg_object_time_ms': avg_object_time * 1000,
                'speedup_ratio': speedup_ratio,
                'vectorized_usage_percentage': vectorized_calls / max(1, vectorized_calls + object_calls) * 100
            },
            'storage_comparison': {
                'object_based_nodes': len(self.nodes),
                'vectorized_experiences': self.vectorized_backend.get_size(),
                'sync_percentage': len(self.vectorized_indices) / max(1, len(self.nodes)) * 100
            }
        }
    
    def benchmark_similarity_methods(self, num_queries: int = 100) -> Dict[str, Any]:
        """
        Benchmark vectorized vs object-based similarity search.
        
        This helps validate the performance improvements.
        """
        if self.vectorized_backend.get_size() == 0:
            return {"error": "No vectorized experiences to benchmark"}
        
        print(f"🚀 Benchmarking similarity methods with {num_queries} queries...")
        
        # Generate random query contexts
        context_dim = len(self.nodes[0].mental_context) if self.nodes else 8
        test_queries = []
        for _ in range(num_queries):
            query = [random.uniform(-1, 1) for _ in range(context_dim)]
            test_queries.append(query)
        
        # Benchmark vectorized approach
        self.use_vectorized_similarity = True
        vectorized_start = time.time()
        
        for query in test_queries:
            self.find_similar_experiences(query, similarity_threshold=0.5, max_results=20)
        
        vectorized_time = time.time() - vectorized_start
        
        # Benchmark object-based approach
        self.use_vectorized_similarity = False
        object_start = time.time()
        
        for query in test_queries:
            self.find_similar_experiences(query, similarity_threshold=0.5, max_results=20)
        
        object_time = time.time() - object_start
        
        # Restore vectorized mode
        self.use_vectorized_similarity = True
        
        # Calculate results
        speedup = object_time / max(0.001, vectorized_time)
        
        results = {
            'num_queries': num_queries,
            'vectorized_total_time': vectorized_time,
            'object_based_total_time': object_time,
            'vectorized_avg_time_ms': (vectorized_time / num_queries) * 1000,
            'object_based_avg_time_ms': (object_time / num_queries) * 1000,
            'speedup_factor': speedup,
            'experiences_processed': self.vectorized_backend.get_size(),
            'device': str(self.vectorized_backend.device)
        }
        
        print(f"✅ Benchmark complete:")
        print(f"   Vectorized: {results['vectorized_avg_time_ms']:.2f}ms per query")
        print(f"   Object-based: {results['object_based_avg_time_ms']:.2f}ms per query") 
        print(f"   Speedup: {speedup:.1f}x faster with GPU acceleration")
        
        return results
    
    def toggle_vectorized_similarity(self, enabled: Optional[bool] = None) -> bool:
        """Toggle vectorized similarity on/off for comparison testing."""
        if enabled is None:
            self.use_vectorized_similarity = not self.use_vectorized_similarity
        else:
            self.use_vectorized_similarity = enabled
        
        mode = "enabled" if self.use_vectorized_similarity else "disabled"
        print(f"🔧 Vectorized similarity {mode}")
        
        return self.use_vectorized_similarity
    
    def migrate_to_vectorized(self, batch_size: int = 1000) -> Dict[str, int]:
        """
        Migrate existing experiences from object storage to vectorized backend.
        
        This allows gradual migration of legacy data.
        """
        print(f"🔄 Migrating experiences to vectorized backend...")
        
        migrated = 0
        errors = 0
        
        for i, experience in enumerate(self.nodes):
            if experience.node_id not in self.vectorized_indices:
                try:
                    vectorized_index = self.vectorized_backend.add_experience(experience)
                    self.vectorized_indices[experience.node_id] = vectorized_index
                    migrated += 1
                except Exception as e:
                    print(f"Warning: Could not migrate experience {experience.node_id}: {e}")
                    errors += 1
                
                # Progress update
                if (i + 1) % batch_size == 0:
                    print(f"   Migrated {migrated} experiences ({i+1}/{len(self.nodes)})")
        
        print(f"✅ Migration complete: {migrated} migrated, {errors} errors")
        
        return {
            'migrated': migrated,
            'errors': errors,
            'total_object_experiences': len(self.nodes),
            'total_vectorized_experiences': self.vectorized_backend.get_size()
        }
    
    def validate_consistency(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Validate that vectorized and object-based storage return consistent results.
        
        This ensures the vectorized backend maintains correctness.
        """
        if not self.nodes or self.vectorized_backend.get_size() == 0:
            return {"error": "Insufficient data for validation"}
        
        print(f"🧪 Validating consistency between storage methods...")
        
        # Sample random experiences for testing
        sample_nodes = random.sample(self.nodes, min(sample_size, len(self.nodes)))
        
        consistency_results = {
            'tested_experiences': 0,
            'context_matches': 0,
            'action_matches': 0,
            'strength_matches': 0,
            'max_context_diff': 0.0,
            'max_strength_diff': 0.0
        }
        
        for node in sample_nodes:
            if node.node_id in self.vectorized_indices:
                vectorized_idx = self.vectorized_indices[node.node_id]
                
                # Compare mental contexts
                original_context = node.mental_context
                vectorized_context = self.vectorized_backend.get_mental_context(vectorized_idx)
                
                context_diff = sum(abs(a - b) for a, b in zip(original_context, vectorized_context))
                consistency_results['max_context_diff'] = max(consistency_results['max_context_diff'], context_diff)
                
                if context_diff < 0.001:  # Floating point tolerance
                    consistency_results['context_matches'] += 1
                
                # Compare actions
                original_action = node.action_taken
                vectorized_action = self.vectorized_backend.get_action_taken(vectorized_idx)
                
                if original_action == vectorized_action:
                    consistency_results['action_matches'] += 1
                
                # Compare strength
                original_strength = node.strength
                vectorized_strength = self.vectorized_backend.get_strength(vectorized_idx)
                
                strength_diff = abs(original_strength - vectorized_strength)
                consistency_results['max_strength_diff'] = max(consistency_results['max_strength_diff'], strength_diff)
                
                if strength_diff < 0.001:
                    consistency_results['strength_matches'] += 1
                
                consistency_results['tested_experiences'] += 1
        
        # Calculate match percentages
        tested = consistency_results['tested_experiences']
        if tested > 0:
            consistency_results['context_match_percentage'] = consistency_results['context_matches'] / tested * 100
            consistency_results['action_match_percentage'] = consistency_results['action_matches'] / tested * 100
            consistency_results['strength_match_percentage'] = consistency_results['strength_matches'] / tested * 100
        
        print(f"✅ Validation complete:")
        print(f"   Context matches: {consistency_results['context_match_percentage']:.1f}%")
        print(f"   Action matches: {consistency_results['action_match_percentage']:.1f}%")
        print(f"   Strength matches: {consistency_results['strength_match_percentage']:.1f}%")
        
        return consistency_results