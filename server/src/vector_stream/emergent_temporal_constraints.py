#!/usr/bin/env python3
"""
Emergent Temporal Hierarchies Through Physical Constraints

Instead of explicit temporal layers, temporal hierarchies emerge naturally from:
1. Computational budget constraints (time pressure)
2. Pattern proximity in sparse space (distance costs)
3. Memory access patterns (recency vs frequency trade-offs)
4. Search depth limitations (how far we can explore)

This mimics how biological temporal hierarchies emerge from physical constraints:
- Spinal reflexes: Direct connections (fast, simple)
- Motor cortex: Local circuits (medium speed, moderate complexity)  
- Prefrontal cortex: Global connections (slow, high complexity)

The key insight: Intelligence emerges from optimization under constraints,
not from explicit architectural features.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque

try:
    from .sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage
    from .cortical_columns import CorticalColumnStorage
except ImportError:
    from sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage
    from cortical_columns import CorticalColumnStorage


@dataclass
class ComputationalBudget:
    """Computational budget constraints that create temporal hierarchies."""
    name: str
    max_time_ms: float          # Maximum computation time allowed
    max_pattern_search: int     # Maximum patterns to examine
    max_search_depth: int       # Maximum similarity search depth
    recency_bias: float         # How much to favor recent patterns
    frequency_bias: float       # How much to favor frequent patterns


class ConstraintBasedPredictor:
    """
    Temporal prediction that emerges from computational constraints.
    
    Different temporal behaviors emerge naturally from different budgets:
    - Tight budgets → Fast, simple responses (reflexes)
    - Medium budgets → Local pattern search (habits)
    - Generous budgets → Global pattern analysis (planning)
    """
    
    def __init__(self, unified_storage, quiet_mode: bool = False):
        self.unified_storage = unified_storage
        self.quiet_mode = quiet_mode
        
        # Access unified storage pattern dimension
        if hasattr(unified_storage, 'pattern_dim'):
            self.pattern_dim = unified_storage.pattern_dim
        else:
            self.pattern_dim = unified_storage.base_storage.pattern_dim
        
        # Fast reflex cache for surface-level sensory-motor mappings
        # This bypasses unified storage for routine predictions
        self.reflex_cache = {}  # pattern_hash -> cached_prediction
        self.reflex_cache_hits = 0
        self.reflex_cache_misses = 0
        self.reflex_cache_max_size = 1000  # Limit cache size
        
        # Define computational budgets that create emergent temporal layers
        self.budgets = {
            'reflex': ComputationalBudget(
                name='reflex',
                max_time_ms=1.0,        # 1ms - biological reflex speed
                max_pattern_search=5,    # Only examine closest patterns
                max_search_depth=1,      # Immediate neighbors only
                recency_bias=0.9,        # Heavily favor recent patterns
                frequency_bias=0.1       # Ignore frequency for reflexes
            ),
            'habit': ComputationalBudget(
                name='habit',
                max_time_ms=50.0,       # 50ms - motor habit speed
                max_pattern_search=50,   # Local neighborhood search
                max_search_depth=3,      # Few hops in pattern space
                recency_bias=0.6,        # Moderate recency bias
                frequency_bias=0.4       # Habits emerge from frequency
            ),
            'deliberate': ComputationalBudget(
                name='deliberate', 
                max_time_ms=500.0,      # 500ms - conscious deliberation
                max_pattern_search=500, # Extensive pattern search
                max_search_depth=10,    # Deep exploration
                recency_bias=0.3,       # Less recency bias
                frequency_bias=0.7      # Consider historical patterns
            )
        }
        
        # Pattern access tracking (creates memory stratification)
        self.access_times: Dict[str, float] = {}
        self.access_frequencies: Dict[str, int] = {}
        
        # Emergent behavior tracking
        self.prediction_history = deque(maxlen=1000)
        self.constraint_violations = {'reflex': 0, 'habit': 0, 'deliberate': 0}
        
        if not quiet_mode:
            print(f"🌊 Constraint-based predictor initialized")
            print(f"   Temporal hierarchies emerge from computational budgets:")
            print(f"   Reflex: {self.budgets['reflex'].max_time_ms}ms budget")
            print(f"   Habit: {self.budgets['habit'].max_time_ms}ms budget") 
            print(f"   Deliberate: {self.budgets['deliberate'].max_time_ms}ms budget")
    
    def predict_under_constraint(self, query_pattern: SparsePattern, 
                               budget_name: str, current_time: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate prediction under specific computational constraints.
        
        Temporal behavior emerges from the budget constraints:
        - Tight budgets force fast, simple responses
        - Generous budgets allow complex, thoughtful responses
        """
        budget = self.budgets[budget_name]
        start_time = time.time() * 1000  # Convert to ms
        
        # Initialize prediction tracking
        prediction_info = {
            'budget_used': budget_name,
            'patterns_examined': 0,
            'search_depth_reached': 0,
            'constraint_violations': [],
            'emergent_behavior': 'unknown'
        }
        
        # TRUE FAST REFLEX MODE: Bypass unified storage for routine predictions
        if budget_name == 'reflex':
            prediction = self._fast_reflex_prediction(query_pattern, prediction_info)
        else:
            # Use unified storage for habit/deliberate modes
            candidates = self._search_patterns_under_budget(
                query_pattern, budget, current_time, prediction_info
            )
            prediction = self._generate_prediction_from_candidates(
                candidates, budget, prediction_info
            )
        
        # Track constraint adherence
        elapsed_ms = (time.time() * 1000) - start_time
        if elapsed_ms > budget.max_time_ms:
            self.constraint_violations[budget_name] += 1
            prediction_info['constraint_violations'].append('time_exceeded')
        
        # Record emergent behavior classification
        prediction_info['emergent_behavior'] = self._classify_emergent_behavior(
            prediction_info, elapsed_ms, budget
        )
        
        self.prediction_history.append(prediction_info)
        
        return prediction, prediction_info
    
    def _fast_reflex_prediction(self, query_pattern: SparsePattern, prediction_info: Dict) -> torch.Tensor:
        """
        TRUE FAST REFLEX MODE: Surface-level sensory-motor mappings.
        
        Bypasses unified storage for routine predictions using cached mappings.
        This is biologically plausible - spinal reflexes don't search memory.
        """
        # Create hash of pattern for cache lookup
        pattern_hash = self._hash_pattern(query_pattern)
        
        # Check cache first (surface-level knowledge)
        if pattern_hash in self.reflex_cache:
            self.reflex_cache_hits += 1
            prediction_info['strategy'] = 'reflex_cache_hit'
            prediction_info['patterns_examined'] = 0  # No search needed
            return self.reflex_cache[pattern_hash]
        
        # Cache miss - generate simple prediction using pattern structure
        self.reflex_cache_misses += 1
        
        # Simple reflex: transform input pattern directly (no search)
        # This mimics spinal reflex circuits that bypass the brain
        dense_pattern = query_pattern.to_dense()
        
        # Apply simple transformations based on pattern structure
        prediction = self._generate_reflex_response(dense_pattern)
        
        # Cache the result for future use (surface-level learning)
        self._cache_reflex_prediction(pattern_hash, prediction)
        
        prediction_info['strategy'] = 'direct_reflex_transformation'
        prediction_info['patterns_examined'] = 0  # No storage search
        
        return prediction
    
    def _hash_pattern(self, pattern: SparsePattern) -> str:
        """Create hash of pattern for cache lookup."""
        # Use pattern structure for hashing (active indices only)
        # Since SparsePattern typically uses binary activation (1.0 for active indices)
        # we only need to hash the active indices
        indices_str = ",".join(map(str, sorted(pattern.active_indices.tolist())))
        return f"{pattern.pattern_dim}|{indices_str}"
    
    def _generate_reflex_response(self, dense_pattern: torch.Tensor) -> torch.Tensor:
        """Generate reflex response using simple pattern transformations."""
        # Simple reflex transformations (no memory search)
        prediction = torch.zeros_like(dense_pattern)
        
        # Apply basic transformations that mimic spinal reflexes
        # 1. Amplitude scaling (stronger input -> stronger output)
        amplitude = torch.norm(dense_pattern)
        if amplitude > 0:
            normalized = dense_pattern / amplitude
            # Scale response based on input strength
            response_amplitude = torch.tanh(amplitude * 0.5)  # Saturating response
            prediction = normalized * response_amplitude
        
        # 2. Add simple pattern-based modulation
        # High-frequency components trigger faster responses
        if len(dense_pattern) > 4:
            # Use pattern structure to generate motor-like response
            sensory_part = dense_pattern[:len(dense_pattern)//2]
            motor_part = dense_pattern[len(dense_pattern)//2:]
            
            # Simple sensory-motor mapping
            prediction[:len(motor_part)] = torch.tanh(sensory_part[:len(motor_part)] * 0.3)
        
        return prediction
    
    def _cache_reflex_prediction(self, pattern_hash: str, prediction: torch.Tensor):
        """Cache reflex prediction for surface-level learning."""
        # Limit cache size (biological constraint)
        if len(self.reflex_cache) >= self.reflex_cache_max_size:
            # Remove oldest entry (simple LRU-like behavior)
            oldest_key = next(iter(self.reflex_cache))
            del self.reflex_cache[oldest_key]
        
        self.reflex_cache[pattern_hash] = prediction.clone()
    
    def _search_patterns_under_budget(self, query_pattern: SparsePattern, 
                                    budget: ComputationalBudget, 
                                    current_time: float,
                                    prediction_info: Dict) -> List[Tuple[SparsePattern, float]]:
        """Search for similar patterns within computational budget."""
        
        # Budget-constrained similarity search using unified storage
        candidates = self.unified_storage.find_similar_patterns(
            query_pattern.to_dense(),
            stream_type='hierarchy',
            k=budget.max_pattern_search,  # Limit by search budget
            min_similarity=0.1,
            cross_stream=True
        )
        
        prediction_info['patterns_examined'] = len(candidates)
        
        # Apply constraint-based filtering and scoring
        constrained_candidates = []
        
        for pattern, similarity in candidates:
            # Time pressure creates recency bias
            recency_score = self._calculate_recency_score(
                pattern.pattern_id, current_time, budget.recency_bias
            )
            
            # Memory pressure creates frequency bias  
            frequency_score = self._calculate_frequency_score(
                pattern.pattern_id, budget.frequency_bias
            )
            
            # Distance in pattern space creates proximity bias
            # (similarity already captures this, but we can add topology costs)
            proximity_score = similarity
            
            # Composite score emerges from constraint interactions
            composite_score = (
                proximity_score * 0.4 +
                recency_score * budget.recency_bias +
                frequency_score * budget.frequency_bias
            )
            
            constrained_candidates.append((pattern, composite_score))
            
            # Budget constraint: stop early if we're running out of time
            if len(constrained_candidates) >= budget.max_pattern_search:
                break
        
        # Sort by composite score (emergent prioritization)
        constrained_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return constrained_candidates
    
    def _calculate_recency_score(self, pattern_id: str, current_time: float, recency_bias: float) -> float:
        """Calculate recency score - more recent patterns score higher."""
        if pattern_id not in self.access_times:
            return 0.0
        
        time_delta = current_time - self.access_times[pattern_id]
        # Exponential decay creates natural recency preference
        recency_score = np.exp(-time_delta / 10.0)  # 10 second half-life
        
        return recency_score * recency_bias
    
    def _calculate_frequency_score(self, pattern_id: str, frequency_bias: float) -> float:
        """Calculate frequency score - more frequent patterns score higher."""
        if pattern_id not in self.access_frequencies:
            return 0.0
        
        frequency = self.access_frequencies[pattern_id]
        # Logarithmic scaling prevents frequency from dominating
        frequency_score = np.log1p(frequency) / 10.0  # Normalized
        
        return frequency_score * frequency_bias
    
    def _generate_prediction_from_candidates(self, candidates: List[Tuple[SparsePattern, float]], 
                                           budget: ComputationalBudget,
                                           prediction_info: Dict) -> torch.Tensor:
        """Generate prediction from budget-constrained candidates."""
        
        if not candidates:
            return torch.zeros(self.pattern_dim)
        
        # Different budgets create different prediction strategies
        if budget.name == 'reflex':
            # Reflex: Use only the best candidate (fast, simple)
            best_pattern = candidates[0][0]
            prediction = best_pattern.to_dense()
            prediction_info['strategy'] = 'single_best_match'
            
        elif budget.name == 'habit':
            # Habit: Weighted average of top candidates (moderate complexity)
            prediction = torch.zeros(self.pattern_dim)
            total_weight = 0.0
            
            for pattern, score in candidates[:5]:  # Top 5 patterns
                weight = score
                prediction += weight * pattern.to_dense()
                total_weight += weight
            
            if total_weight > 0:
                prediction = prediction / total_weight
            
            prediction_info['strategy'] = 'weighted_average_top5'
            
        else:  # deliberate
            # Deliberate: Complex integration using pre-computed cortical columns
            prediction = torch.zeros(self.pattern_dim)
            
            # Use cortical columns for fast clustering (O(1) instead of O(n²))
            query_pattern = candidates[0][0] if candidates else None
            if query_pattern:
                # Get pre-clustered patterns from cortical columns
                cortical_clusters = self.column_storage.get_clustered_patterns(
                    query_pattern, max_columns=5
                )
                
                # Weight clusters by internal consistency
                for cluster_patterns in cortical_clusters:
                    if cluster_patterns:
                        cluster_weight = len(cluster_patterns) / max(1, len(candidates))
                        cluster_prediction = torch.zeros(self.pattern_dim)
                        
                        for pattern, score in cluster_patterns[:10]:  # Top 10 per cluster
                            cluster_prediction += score * pattern.to_dense()
                        
                        prediction += cluster_weight * cluster_prediction
                
                prediction_info['strategy'] = 'cortical_column_integration'
                prediction_info['num_clusters'] = len(cortical_clusters)
            else:
                # Fallback to simple prediction if no query pattern
                prediction = torch.zeros(self.pattern_dim)
                prediction_info['strategy'] = 'fallback_zero'
                prediction_info['num_clusters'] = 0
        
        return prediction
    
    def _cluster_patterns_by_similarity(self, candidates: List[Tuple[SparsePattern, float]]) -> List[List[Tuple[SparsePattern, float]]]:
        """Simple clustering of patterns by similarity (for deliberate prediction)."""
        if len(candidates) <= 1:
            return [candidates]
        
        clusters = []
        used_indices = set()
        
        for i, (pattern_i, score_i) in enumerate(candidates):
            if i in used_indices:
                continue
                
            cluster = [(pattern_i, score_i)]
            used_indices.add(i)
            
            # Find similar patterns for this cluster
            for j, (pattern_j, score_j) in enumerate(candidates[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                similarity = pattern_i.jaccard_similarity(pattern_j)
                if similarity > 0.5:  # Similarity threshold for clustering
                    cluster.append((pattern_j, score_j))
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _classify_emergent_behavior(self, prediction_info: Dict, elapsed_ms: float, 
                                  budget: ComputationalBudget) -> str:
        """Classify the emergent temporal behavior from constraint interactions."""
        
        patterns_examined = prediction_info['patterns_examined']
        strategy = prediction_info.get('strategy', 'unknown')
        
        # Emergent behavior emerges from constraint satisfaction patterns
        if elapsed_ms < budget.max_time_ms * 0.1:
            if patterns_examined <= 5:
                return 'reflex_like'  # Fast, simple response
            else:
                return 'cached_response'  # Fast because pattern was readily available
                
        elif elapsed_ms < budget.max_time_ms * 0.5:
            if 'weighted' in strategy:
                return 'habit_like'  # Moderate speed, pattern integration
            else:
                return 'recognition_based'  # Quick pattern matching
                
        else:
            if 'cluster' in strategy:
                return 'deliberative'  # Slow, complex analysis
            else:
                return 'search_intensive'  # Slow because of extensive search
        
        return 'unknown'
    
    def store_pattern_with_columns(self, pattern: SparsePattern) -> str:
        """Store pattern through unified cortical storage."""
        # Store through unified storage system
        pattern_id = self.unified_storage.store_pattern(
            pattern.to_dense(),
            stream_type='hierarchy',
            pattern_id=pattern.pattern_id
        )
        return pattern_id
    
    def update_access_patterns(self, pattern_id: str, current_time: float):
        """Update pattern access tracking (creates memory stratification)."""
        self.access_times[pattern_id] = current_time
        self.access_frequencies[pattern_id] = self.access_frequencies.get(pattern_id, 0) + 1
    
    def get_emergent_behavior_stats(self) -> Dict[str, Any]:
        """Analyze emergent temporal behaviors from constraint interactions."""
        if not self.prediction_history:
            return {'no_data': True}
        
        # Count emergent behavior types
        behavior_counts = {}
        for pred_info in self.prediction_history:
            behavior = pred_info['emergent_behavior']
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
        
        # Calculate constraint violation rates
        total_predictions = len(self.prediction_history)
        violation_rates = {
            budget_name: violations / total_predictions 
            for budget_name, violations in self.constraint_violations.items()
        }
        
        return {
            'total_predictions': total_predictions,
            'emergent_behaviors': behavior_counts,
            'constraint_violation_rates': violation_rates,
            'dominant_behavior': max(behavior_counts.items(), key=lambda x: x[1])[0] if behavior_counts else 'none'
        }


class EmergentTemporalHierarchy:
    """
    Temporal hierarchy that emerges from computational constraints.
    
    Instead of explicit fast/medium/slow layers, temporal behaviors emerge from:
    - Time pressure (computational budgets)
    - Memory access patterns (recency/frequency trade-offs)
    - Pattern space topology (proximity constraints)
    - Search depth limitations (how far we can explore)
    """
    
    def __init__(self, unified_storage, quiet_mode: bool = False):
        self.unified_storage = unified_storage
        self.predictor = ConstraintBasedPredictor(unified_storage, quiet_mode)
        
        # Adaptive budget allocation based on context
        self.context_pressure = 0.5  # 0=relaxed, 1=urgent
        self.recent_accuracies = deque(maxlen=100)
        
        # Emergent layer tracking
        self.emergent_layers = {
            'reflex_activations': 0,
            'habit_activations': 0, 
            'deliberate_activations': 0
        }
        
        self.hierarchy_start_time = time.time()
        
        if not quiet_mode:
            print(f"\n🌊 EMERGENT TEMPORAL HIERARCHY INITIALIZED")
            print(f"   🎯 Temporal layers emerge from computational constraints")
            print(f"   Reflex: 1ms budget → fast, simple responses")
            print(f"   Habit: 50ms budget → local pattern integration")
            print(f"   Deliberate: 500ms budget → global pattern analysis")
            print(f"   🚀 Intelligence emerges from optimization under constraints")
    
    def process_with_adaptive_budget(self, query_pattern: SparsePattern, 
                                   current_time: float) -> Dict[str, Any]:
        """
        Process pattern with adaptive budget allocation.
        
        Budget selection emerges from:
        - Context pressure (urgency)
        - Recent prediction accuracy
        - Pattern novelty
        """
        
        # Determine appropriate budget based on context
        budget_name = self._select_budget_adaptively(query_pattern, current_time)
        
        # Generate prediction under constraints
        prediction, prediction_info = self.predictor.predict_under_constraint(
            query_pattern, budget_name, current_time
        )
        
        # Update access patterns for memory stratification
        self.predictor.update_access_patterns(query_pattern.pattern_id, current_time)
        
        # Track emergent layer usage
        self.emergent_layers[f'{budget_name}_activations'] += 1
        
        return {
            'prediction': prediction,
            'budget_used': budget_name,
            'prediction_info': prediction_info,
            'emergent_behavior': prediction_info['emergent_behavior']
        }
    
    def _select_budget_adaptively(self, query_pattern: SparsePattern, current_time: float) -> str:
        """Select computational budget based on emergent constraints."""
        
        # Context pressure creates urgency bias
        if self.context_pressure > 0.8:
            return 'reflex'  # High pressure → fast response needed
        
        # Aggressive reflex promotion for routine predictions
        # If we've been successful recently, promote to reflex more aggressively
        if len(self.recent_accuracies) > 5:
            recent_accuracy = np.mean(list(self.recent_accuracies)[-5:])
            if recent_accuracy > 0.7:
                # Successful recent predictions → increase reflex usage
                if np.random.random() < 0.7:  # 70% chance of reflex for successful patterns
                    return 'reflex'
        
        # NEW: If we have no accuracy history, default to reflex for simple patterns
        # This makes sense - unknown patterns should be treated as routine initially
        if len(self.recent_accuracies) == 0:
            # For new patterns, bias towards reflex (fast response)
            if np.random.random() < 0.8:  # 80% chance of reflex for new patterns
                return 'reflex'
        
        # Pattern familiarity affects budget allocation
        # Check reflex cache first for fast familiarity assessment
        pattern_hash = self.predictor._hash_pattern(query_pattern)
        
        if pattern_hash in self.predictor.reflex_cache:
            # Pattern is in reflex cache → very familiar → reflex
            return 'reflex'
        
        # For patterns not in reflex cache, use unified storage for deeper analysis
        # But make this more selective to avoid slowdowns
        similar_patterns = self.unified_storage.find_similar_patterns(
            query_pattern.to_dense(),
            stream_type='hierarchy',
            k=3,  # Reduced from 5 to 3 for faster search
            min_similarity=0.5,  # Higher threshold for faster search
            cross_stream=True
        )
        
        # Calculate familiarity score
        if similar_patterns:
            max_similarity = max(similarity for _, similarity in similar_patterns)
            avg_similarity = np.mean([similarity for _, similarity in similar_patterns])
            
            # Very familiar patterns → reflex
            if max_similarity > 0.8 and avg_similarity > 0.6:
                return 'reflex'
            
            # Moderately familiar patterns → habit
            elif max_similarity > 0.5 and avg_similarity > 0.3:
                return 'habit'
            
            # Somewhat familiar but complex → deliberate
            else:
                return 'deliberate'
        
        # No similar patterns found → novel situation
        # But start with habit, not deliberate (most "novel" patterns are just variations)
        
        # Recent accuracy affects confidence in quick responses
        if len(self.recent_accuracies) > 10:
            recent_accuracy = np.mean(list(self.recent_accuracies)[-10:])
            
            if recent_accuracy > 0.8:
                # High accuracy → can use faster budget even for novel patterns
                return 'habit'
            elif recent_accuracy < 0.3:
                # Low accuracy → need more deliberate processing
                return 'deliberate'
        
        # Default to habit-level processing (NOT deliberate!)
        return 'habit'
    
    def update_context_pressure(self, urgency: float):
        """Update context pressure (creates dynamic budget allocation)."""
        self.context_pressure = np.clip(urgency, 0.0, 1.0)
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get emergent temporal hierarchy statistics."""
        
        # Calculate emergent layer usage patterns
        total_activations = sum(self.emergent_layers.values())
        layer_usage = {}
        
        if total_activations > 0:
            for layer, count in self.emergent_layers.items():
                layer_usage[layer] = count / total_activations
        
        # Get emergent behavior analysis
        behavior_stats = self.predictor.get_emergent_behavior_stats()
        
        # Get reflex cache statistics
        reflex_cache_stats = {
            'cache_size': len(self.predictor.reflex_cache),
            'cache_hits': self.predictor.reflex_cache_hits,
            'cache_misses': self.predictor.reflex_cache_misses,
            'cache_hit_rate': self.predictor.reflex_cache_hits / max(1, self.predictor.reflex_cache_hits + self.predictor.reflex_cache_misses)
        }
        
        return {
            'architecture': 'emergent_constraint_based',
            'layer_usage_patterns': layer_usage,
            'emergent_behaviors': behavior_stats,
            'context_pressure': self.context_pressure,
            'total_predictions': total_activations,
            'uptime_seconds': time.time() - self.hierarchy_start_time,
            'reflex_cache': reflex_cache_stats
        }


def demonstrate_emergent_hierarchies():
    """Demonstrate emergent temporal hierarchies from constraints."""
    print("🌊 EMERGENT TEMPORAL HIERARCHY DEMONSTRATION")
    print("=" * 60)
    
    # Create sparse storage and emergent hierarchy
    storage = SparsePatternStorage(pattern_dim=16, max_patterns=1000, quiet_mode=True)
    encoder = SparsePatternEncoder(pattern_dim=16, sparsity=0.02, quiet_mode=True)
    hierarchy = EmergentTemporalHierarchy(storage, quiet_mode=True)
    
    print("Testing emergent temporal behaviors...")
    
    # Store some patterns to create pattern space
    for i in range(20):
        pattern_vec = torch.randn(16)
        pattern = encoder.encode_top_k(pattern_vec, f"seed_{i}")
        storage.store_pattern(pattern)
    
    # Test different context pressures
    test_scenarios = [
        (0.9, "High pressure (emergency)"),
        (0.5, "Medium pressure (normal)"),  
        (0.1, "Low pressure (relaxed)")
    ]
    
    for pressure, description in test_scenarios:
        print(f"\n{description}:")
        hierarchy.update_context_pressure(pressure)
        
        # Test with familiar and novel patterns
        test_pattern = encoder.encode_top_k(torch.randn(16), "test")
        result = hierarchy.process_with_adaptive_budget(test_pattern, time.time())
        
        print(f"  Budget selected: {result['budget_used']}")
        print(f"  Emergent behavior: {result['emergent_behavior']}")
        print(f"  Strategy: {result['prediction_info'].get('strategy', 'unknown')}")
    
    # Show emergent statistics
    stats = hierarchy.get_hierarchy_stats()
    print(f"\nEmergent layer usage:")
    for layer, usage in stats['layer_usage_patterns'].items():
        print(f"  {layer}: {usage:.1%}")
    
    print(f"\nEmergent behaviors observed:")
    if 'emergent_behaviors' in stats and stats['emergent_behaviors']:
        for behavior, count in stats['emergent_behaviors']['emergent_behaviors'].items():
            print(f"  {behavior}: {count} times")
    
    print(f"\n✅ EMERGENT TEMPORAL HIERARCHY DEMONSTRATION COMPLETE")
    print(f"Intelligence emerged from computational constraints, not explicit design!")


if __name__ == "__main__":
    demonstrate_emergent_hierarchies()