#!/usr/bin/env python3
"""
Massive Parallel Action Generation from Experience History.

This module implements the key algorithmic breakthrough from the vectorization brief:
performing a single, massive parallel search across the entire experience graph 
on the GPU to find thousands of past experiences similar to the current situation.
The actions taken in those past experiences become a rich source of relevant 
action candidates for the MotivationSystem to evaluate.

Key Features:
- GPU-accelerated similarity search across entire experience history
- Parallel extraction of action candidates from similar experiences
- Intelligent filtering and ranking of candidates
- Seamless integration with existing MotivationSystem
- 10-100x more action candidates than manual generation
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
from collections import defaultdict

from core.hybrid_world_graph import HybridWorldGraph
from core.experience_node import ExperienceNode
from core.vectorized_backend import VectorizedBackend


@dataclass
class ActionCandidate:
    """Represents a potential action extracted from experience history."""
    action: Dict[str, float]
    source_experience_id: str
    similarity_score: float
    confidence: float
    context_match: float
    success_history: float
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score for this candidate."""
        return (self.similarity_score * 0.4 + 
                self.confidence * 0.3 + 
                self.success_history * 0.3)


@dataclass
class ActionGenerationResult:
    """Results from massive parallel action generation."""
    candidates: List[ActionCandidate]
    total_experiences_searched: int
    gpu_computation_time: float
    candidate_diversity: float
    context_coverage: float
    
    def get_best_candidates(self, n: int = 50) -> List[ActionCandidate]:
        """Get the top N candidates by quality score."""
        sorted_candidates = sorted(self.candidates, 
                                 key=lambda c: c.get_quality_score(), 
                                 reverse=True)
        return sorted_candidates[:n]
    
    def get_diverse_candidates(self, n: int = 50, diversity_threshold: float = 0.1) -> List[ActionCandidate]:
        """Get diverse candidates avoiding too similar actions."""
        diverse_candidates = []
        
        for candidate in sorted(self.candidates, key=lambda c: c.get_quality_score(), reverse=True):
            # Check if this candidate is too similar to existing ones
            is_diverse = True
            for existing in diverse_candidates:
                if self._actions_too_similar(candidate.action, existing.action, diversity_threshold):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_candidates.append(candidate)
                if len(diverse_candidates) >= n:
                    break
        
        return diverse_candidates
    
    def _actions_too_similar(self, action1: Dict[str, float], action2: Dict[str, float], 
                           threshold: float) -> bool:
        """Check if two actions are too similar."""
        keys = set(action1.keys()) | set(action2.keys())
        total_diff = sum(abs(action1.get(k, 0) - action2.get(k, 0)) for k in keys)
        return total_diff < threshold


class ParallelActionGenerator:
    """
    Massive parallel action generation using GPU-accelerated experience search.
    
    This system performs the algorithmic breakthrough described in the vectorization brief:
    instead of manually generating actions, it searches the robot's entire life experience
    to find contextually relevant actions that have been successful in similar situations.
    """
    
    def __init__(self, world_graph: HybridWorldGraph, device: str = 'auto'):
        """
        Initialize parallel action generator.
        
        Args:
            world_graph: Graph containing all experiences
            device: Device to use ('auto', 'cpu', 'cuda', 'mps')
        """
        self.world_graph = world_graph
        self.vectorized_backend = world_graph.vectorized_backend
        
        # Device selection
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        # Generation parameters
        self.max_candidates = 1000  # Much larger than manual generation
        self.similarity_threshold = 0.3  # Lower threshold for more diversity
        self.context_weight = 0.7  # How much to weight context vs action similarity
        self.success_weight = 0.3  # Weight for historical success
        
        # Performance tracking
        self.generation_count = 0
        self.total_gpu_time = 0.0
        self.total_candidates_generated = 0
        
        # Caching for performance
        self.experience_cache = {}
        self.action_success_cache = {}
        self.cache_dirty = True
        
        print(f"🚀 Parallel Action Generator initialized (device: {self.device})")
    
    def generate_massive_action_candidates(self, current_context: List[float],
                                         current_situation: Dict[str, Any] = None,
                                         max_candidates: int = None) -> ActionGenerationResult:
        """
        Generate massive number of action candidates using parallel GPU search.
        
        This is the core breakthrough: search entire experience history in parallel
        to find thousands of relevant action candidates.
        
        Args:
            current_context: Current mental context
            current_situation: Additional situational information
            max_candidates: Maximum candidates to generate
            
        Returns:
            ActionGenerationResult with candidates and statistics
        """
        if max_candidates is None:
            max_candidates = self.max_candidates
        
        start_time = time.time()
        
        # Ensure we have experiences to search
        if self.vectorized_backend.size == 0:
            return ActionGenerationResult([], 0, 0.0, 0.0, 0.0)
        
        try:
            # Perform massive parallel similarity search
            candidates = self._gpu_parallel_action_search(current_context, max_candidates)
            
            # Post-process candidates
            candidates = self._enhance_candidates(candidates, current_context)
            
            # Calculate statistics
            gpu_time = time.time() - start_time
            diversity = self._calculate_diversity(candidates)
            coverage = self._calculate_context_coverage(candidates, current_context)
            
            # Update performance tracking
            self.generation_count += 1
            self.total_gpu_time += gpu_time
            self.total_candidates_generated += len(candidates)
            
            result = ActionGenerationResult(
                candidates=candidates,
                total_experiences_searched=self.vectorized_backend.size,
                gpu_computation_time=gpu_time,
                candidate_diversity=diversity,
                context_coverage=coverage
            )
            
            print(f"🎯 Generated {len(candidates)} action candidates in {gpu_time*1000:.1f}ms")
            return result
            
        except Exception as e:
            print(f"⚠️  GPU action generation failed: {e}, falling back to CPU")
            return self._cpu_fallback_generation(current_context, max_candidates)
    
    def _gpu_parallel_action_search(self, current_context: List[float], 
                                  max_candidates: int) -> List[ActionCandidate]:
        """Perform GPU-accelerated parallel search for action candidates."""
        
        # Convert context to tensor
        context_tensor = torch.tensor(current_context, dtype=torch.float32, device=self.device)
        
        # Get all mental contexts from vectorized backend
        all_contexts = self.vectorized_backend._mental_contexts[:self.vectorized_backend.size]
        
        # Massive parallel similarity calculation
        similarities = torch.cosine_similarity(
            context_tensor.unsqueeze(0),  # [1, context_dim]
            all_contexts,  # [num_experiences, context_dim]
            dim=1
        )
        
        # Get top similar experiences
        top_k = min(max_candidates * 2, len(similarities))  # Get extra for filtering
        top_similarities, top_indices = torch.topk(similarities, top_k)
        
        # Filter by similarity threshold
        valid_mask = top_similarities >= self.similarity_threshold
        valid_indices = top_indices[valid_mask]
        valid_similarities = top_similarities[valid_mask]
        
        # Extract actions from similar experiences
        candidates = []
        for idx, similarity in zip(valid_indices, valid_similarities):
            idx = int(idx.cpu().item())
            similarity = float(similarity.cpu().item())
            
            # Get action from vectorized backend
            action_vector = self.vectorized_backend._action_vectors[idx]
            action = self._vector_to_action_dict(action_vector)
            
            # Get experience ID for tracking
            experience_id = self.vectorized_backend._index_to_node_id.get(idx, f"exp_{idx}")
            
            # Calculate additional metrics
            confidence = self._calculate_confidence(idx, similarity)
            success_history = self._get_success_history(idx)
            
            candidate = ActionCandidate(
                action=action,
                source_experience_id=experience_id,
                similarity_score=similarity,
                confidence=confidence,
                context_match=similarity,
                success_history=success_history
            )
            
            candidates.append(candidate)
            
            if len(candidates) >= max_candidates:
                break
        
        return candidates
    
    def _vector_to_action_dict(self, action_vector: torch.Tensor) -> Dict[str, float]:
        """Convert action vector to action dictionary."""
        # Standard action layout (should match _action_to_tensor in other modules)
        action_dict = {}
        
        # Move to CPU first to avoid MPS tensor conversion issues
        action_vector_cpu = action_vector.cpu()
        
        if len(action_vector_cpu) >= 3:
            action_dict['forward_motor'] = float(action_vector_cpu[0].item())
            action_dict['turn_motor'] = float(action_vector_cpu[1].item())
            action_dict['brake_motor'] = float(action_vector_cpu[2].item())
        
        if len(action_vector_cpu) >= 6:
            action_dict['forward'] = float(action_vector_cpu[3].item())
            action_dict['turn'] = float(action_vector_cpu[4].item())
            action_dict['brake'] = float(action_vector_cpu[5].item())
        
        return action_dict
    
    def _calculate_confidence(self, experience_index: int, similarity: float) -> float:
        """Calculate confidence for an action candidate."""
        # Base confidence on similarity
        base_confidence = similarity
        
        # Boost confidence for experiences with low prediction error
        if hasattr(self.vectorized_backend, '_prediction_errors'):
            if experience_index < len(self.vectorized_backend._prediction_errors):
                error = float(self.vectorized_backend._prediction_errors[experience_index].cpu().item())
                error_factor = max(0.1, 1.0 - error)
                base_confidence *= error_factor
        
        # Boost confidence for recent experiences
        if hasattr(self.vectorized_backend, '_timestamps'):
            if experience_index < len(self.vectorized_backend._timestamps):
                timestamp = float(self.vectorized_backend._timestamps[experience_index].cpu().item())
                recency_factor = self._calculate_recency_factor(timestamp)
                base_confidence *= recency_factor
        
        return max(0.0, min(1.0, base_confidence))
    
    def _get_success_history(self, experience_index: int) -> float:
        """Get success history for an experience."""
        # Check cache first
        if experience_index in self.action_success_cache:
            return self.action_success_cache[experience_index]
        
        # Calculate success based on available metrics
        success = 0.5  # Default neutral success
        
        # Use prediction error as success indicator (lower error = higher success)
        if hasattr(self.vectorized_backend, '_prediction_errors'):
            if experience_index < len(self.vectorized_backend._prediction_errors):
                error = float(self.vectorized_backend._prediction_errors[experience_index].cpu().item())
                success = max(0.1, 1.0 - error)
        
        # Use strength as success indicator
        if hasattr(self.vectorized_backend, '_strengths'):
            if experience_index < len(self.vectorized_backend._strengths):
                strength = float(self.vectorized_backend._strengths[experience_index].cpu().item())
                success = max(success, strength)
        
        # Cache the result
        self.action_success_cache[experience_index] = success
        return success
    
    def _calculate_recency_factor(self, timestamp: float) -> float:
        """Calculate recency factor for experience weighting."""
        current_time = time.time()
        age = current_time - timestamp
        
        # Decay factor (prefer recent experiences)
        decay_rate = 0.001  # Adjust based on typical experience timespan
        recency_factor = np.exp(-decay_rate * age)
        
        return max(0.1, min(1.0, recency_factor))
    
    def _enhance_candidates(self, candidates: List[ActionCandidate], 
                          current_context: List[float]) -> List[ActionCandidate]:
        """Enhance candidates with additional processing."""
        enhanced = []
        
        for candidate in candidates:
            # Normalize action values
            candidate.action = self._normalize_action(candidate.action)
            
            # Calculate context-specific adjustments
            candidate.context_match = self._refined_context_match(
                candidate, current_context
            )
            
            # Filter out invalid actions
            if self._is_valid_action(candidate.action):
                enhanced.append(candidate)
        
        return enhanced
    
    def _normalize_action(self, action: Dict[str, float]) -> Dict[str, float]:
        """Normalize action values to valid ranges."""
        normalized = {}
        
        for key, value in action.items():
            # Clamp to reasonable ranges
            if 'motor' in key or key in ['forward', 'turn', 'brake']:
                normalized[key] = max(-1.0, min(1.0, value))
            else:
                normalized[key] = value
        
        return normalized
    
    def _refined_context_match(self, candidate: ActionCandidate, 
                             current_context: List[float]) -> float:
        """Calculate refined context match score."""
        # This could be enhanced with more sophisticated matching
        return candidate.similarity_score
    
    def _is_valid_action(self, action: Dict[str, float]) -> bool:
        """Check if action is valid."""
        # Basic validation
        if not action:
            return False
        
        # Check for reasonable values
        for key, value in action.items():
            if not isinstance(value, (int, float)):
                return False
            if abs(value) > 10.0:  # Sanity check
                return False
        
        return True
    
    def _calculate_diversity(self, candidates: List[ActionCandidate]) -> float:
        """Calculate diversity score for candidate set."""
        if len(candidates) < 2:
            return 0.0
        
        # Calculate pairwise action differences
        total_diff = 0.0
        pairs = 0
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                action1 = candidates[i].action
                action2 = candidates[j].action
                
                # Calculate action difference
                keys = set(action1.keys()) | set(action2.keys())
                diff = sum(abs(action1.get(k, 0) - action2.get(k, 0)) for k in keys)
                total_diff += diff
                pairs += 1
        
        return total_diff / max(1, pairs)
    
    def _calculate_context_coverage(self, candidates: List[ActionCandidate], 
                                  current_context: List[float]) -> float:
        """Calculate how well candidates cover the context space."""
        if not candidates:
            return 0.0
        
        # Simple coverage based on similarity score distribution
        similarities = [c.similarity_score for c in candidates]
        coverage = np.std(similarities) if len(similarities) > 1 else 0.0
        
        return min(1.0, coverage)
    
    def _cpu_fallback_generation(self, current_context: List[float], 
                               max_candidates: int) -> ActionGenerationResult:
        """CPU fallback for action generation."""
        # Simple CPU-based generation
        candidates = []
        
        # Use existing experience-based action system as fallback
        try:
            from core.experience_based_actions import ExperienceBasedActionSystem
            exp_system = ExperienceBasedActionSystem(self.world_graph)
            
            # Get limited candidates from existing system
            basic_candidates = exp_system.generate_experience_based_actions(
                current_context, max_candidates=min(50, max_candidates)
            )
            
            # Convert to our format
            for i, action in enumerate(basic_candidates):
                candidate = ActionCandidate(
                    action=action,
                    source_experience_id=f"cpu_fallback_{i}",
                    similarity_score=0.5,
                    confidence=0.5,
                    context_match=0.5,
                    success_history=0.5
                )
                candidates.append(candidate)
        
        except Exception as e:
            print(f"CPU fallback also failed: {e}")
        
        return ActionGenerationResult(
            candidates=candidates,
            total_experiences_searched=len(self.world_graph.nodes),
            gpu_computation_time=0.0,
            candidate_diversity=0.0,
            context_coverage=0.0
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = self.total_gpu_time / max(1, self.generation_count)
        avg_candidates = self.total_candidates_generated / max(1, self.generation_count)
        
        return {
            'generation_count': self.generation_count,
            'total_gpu_time': self.total_gpu_time,
            'avg_generation_time': avg_time,
            'avg_candidates_per_generation': avg_candidates,
            'total_candidates_generated': self.total_candidates_generated,
            'device': self.device,
            'candidates_per_second': avg_candidates / max(0.001, avg_time)
        }
    
    def clear_caches(self):
        """Clear internal caches."""
        self.experience_cache.clear()
        self.action_success_cache.clear()
        self.cache_dirty = True
    
    def benchmark_generation_performance(self, num_tests: int = 10) -> Dict[str, Any]:
        """Benchmark action generation performance."""
        # Create test contexts
        context_dim = 8  # Standard context dimension
        test_contexts = []
        
        for _ in range(num_tests):
            context = [np.random.uniform(-1, 1) for _ in range(context_dim)]
            test_contexts.append(context)
        
        # Benchmark generation
        start_time = time.time()
        total_candidates = 0
        
        for context in test_contexts:
            result = self.generate_massive_action_candidates(context, max_candidates=100)
            total_candidates += len(result.candidates)
        
        total_time = time.time() - start_time
        
        return {
            'num_tests': num_tests,
            'total_time': total_time,
            'avg_time_per_generation': total_time / num_tests,
            'total_candidates': total_candidates,
            'avg_candidates_per_generation': total_candidates / num_tests,
            'candidates_per_second': total_candidates / total_time,
            'experiences_searched_per_test': self.vectorized_backend.size
        }