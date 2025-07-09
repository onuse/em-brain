"""
Vectorized Novelty Detection - GPU-accelerated novelty evaluation for massive speedup.

This system evaluates novelty across multiple dimensions using parallel tensor operations,
replacing sequential similarity calculations with batched GPU processing.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
import time
from dataclasses import dataclass
from enum import Enum

from core.novelty_detection import NoveltyDetector, NoveltyScore, ExperienceSignature, NoveltyDimension
from core.world_graph import WorldGraph
from core.adaptive_execution_engine import AdaptiveExecutionEngine, ExecutionMethod


@dataclass
class VectorizedNoveltyResult:
    """Result from vectorized novelty evaluation."""
    novelty_score: NoveltyScore
    dimension_scores: Dict[NoveltyDimension, float]
    candidate_similarities: torch.Tensor
    computation_time: float
    method_used: ExecutionMethod


class VectorizedNoveltyDetector(NoveltyDetector):
    """
    GPU-accelerated novelty detection for parallel similarity computation.
    
    This detector evaluates novelty across all dimensions simultaneously using
    tensor operations, delivering massive speedup for novelty evaluation.
    """
    
    def __init__(self, world_graph: WorldGraph, device: str = 'auto'):
        """
        Initialize vectorized novelty detector.
        
        Args:
            world_graph: World graph containing experiences
            device: 'auto', 'cuda', 'mps', or 'cpu'
        """
        super().__init__(world_graph)
        
        self.device = self._setup_device(device)
        
        # Adaptive execution engine - conservative thresholds for novelty detection
        # GPU overhead is significant for small datasets, so prefer CPU for small brains
        self.adaptive_engine = AdaptiveExecutionEngine(
            gpu_threshold_nodes=1000,  # Only use GPU for large datasets
            cpu_threshold_nodes=100,   # Prefer CPU for small datasets
            learning_rate=0.2
        )
        
        # Vectorized cache
        self.vectorized_cache = {}
        self.cache_valid = False
        
        # Performance tracking
        self.vectorized_stats = {
            'total_evaluations': 0,
            'gpu_evaluations': 0,
            'cpu_evaluations': 0,
            'adaptive_evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print(f"VectorizedNoveltyDetector initialized on {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device with automatic fallback."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        try:
            torch_device = torch.device(device)
            test_tensor = torch.zeros(1, device=torch_device)
            del test_tensor
            return torch_device
        except Exception as e:
            print(f"Warning: Could not use device {device}: {e}. Falling back to CPU.")
            return torch.device('cpu')
    
    def evaluate_experience_novelty_vectorized(self, 
                                             experience_signature: ExperienceSignature,
                                             search_radius: int = 50) -> VectorizedNoveltyResult:
        """
        Evaluate experience novelty using vectorized computation.
        
        This is the core method that delivers massive speedup by evaluating
        novelty across all dimensions in parallel.
        """
        start_time = time.time()
        
        # Fast path for very small datasets - always use CPU
        node_count = self.world_graph.node_count()
        if node_count <= 10:
            result = self._evaluate_novelty_cpu(experience_signature, search_radius)
            result.computation_time = time.time() - start_time
            self.vectorized_stats['total_evaluations'] += 1
            self.vectorized_stats['cpu_evaluations'] += 1
            return result
        
        # Choose execution method for larger datasets
        def cpu_evaluation():
            return self._evaluate_novelty_cpu(experience_signature, search_radius)
        
        def gpu_evaluation():
            return self._evaluate_novelty_gpu(experience_signature, search_radius)
        
        # Use adaptive engine
        result = self.adaptive_engine.execute_with_optimal_method(
            dataset_size=node_count,
            traversal_count=len(NoveltyDimension),
            cpu_function=cpu_evaluation,
            gpu_function=gpu_evaluation,
            complexity_hint="complex"
        )
        
        result.computation_time = time.time() - start_time
        self.vectorized_stats['total_evaluations'] += 1
        self.vectorized_stats['adaptive_evaluations'] += 1
        
        return result
    
    def _evaluate_novelty_gpu(self, experience_signature: ExperienceSignature,
                             search_radius: int) -> VectorizedNoveltyResult:
        """GPU-accelerated novelty evaluation using tensor operations."""
        # Update cache if needed
        if not self.cache_valid:
            self._update_vectorized_cache()
        
        if self.world_graph.node_count() == 0:
            return self._create_maximum_novelty_result(ExecutionMethod.GPU)
        
        # Convert experience to tensor
        experience_tensor = self._experience_to_tensor(experience_signature)
        
        # Get cached node tensors
        cached_tensors = self.vectorized_cache
        
        # Compute similarities across all dimensions simultaneously
        dimension_similarities = self._compute_dimensional_similarities_gpu(
            experience_tensor, cached_tensors, search_radius
        )
        
        # Calculate weighted novelty score
        novelty_score = self._calculate_novelty_from_similarities(dimension_similarities)
        
        return VectorizedNoveltyResult(
            novelty_score=novelty_score,
            dimension_scores=dimension_similarities,
            candidate_similarities=torch.tensor([0.0]),  # Placeholder
            computation_time=0.0,
            method_used=ExecutionMethod.GPU
        )
    
    def _evaluate_novelty_cpu(self, experience_signature: ExperienceSignature,
                             search_radius: int) -> VectorizedNoveltyResult:
        """CPU fallback for novelty evaluation."""
        # Use original method
        novelty_score = super().evaluate_experience_novelty(experience_signature, search_radius)
        
        # Extract dimension scores from the NoveltyScore
        dimension_scores = novelty_score.dimension_scores
        
        return VectorizedNoveltyResult(
            novelty_score=novelty_score,
            dimension_scores=dimension_scores,
            candidate_similarities=torch.tensor([0.0]),
            computation_time=0.0,
            method_used=ExecutionMethod.CPU
        )
    
    def _update_vectorized_cache(self):
        """Update vectorized cache with current world graph data."""
        if self.world_graph.node_count() == 0:
            return
        
        nodes = self.world_graph.all_nodes()
        
        # Extract tensors for each dimension
        mental_contexts = []
        motor_actions = []
        sensory_outcomes = []
        temporal_contexts = []
        predictive_accuracies = []
        
        for node in nodes:
            # Mental context
            mental_contexts.append(node.mental_context)
            
            # Motor action (convert to vector)
            action_vector = [
                node.action_taken.get('forward_motor', 0.0),
                node.action_taken.get('turn_motor', 0.0),
                node.action_taken.get('brake_motor', 0.0)
            ]
            motor_actions.append(action_vector)
            
            # Sensory outcome
            sensory_outcomes.append(node.actual_sensory[:8] if len(node.actual_sensory) >= 8 else node.actual_sensory)
            
            # Temporal context (last 5 elements of mental context)
            temporal_contexts.append(node.mental_context[-5:] if len(node.mental_context) > 5 else node.mental_context)
            
            # Predictive accuracy
            predictive_accuracies.append([1.0 - node.prediction_error])
        
        # Convert to tensors
        self.vectorized_cache = {
            'mental_contexts': torch.tensor(mental_contexts, dtype=torch.float32, device=self.device),
            'motor_actions': torch.tensor(motor_actions, dtype=torch.float32, device=self.device),
            'sensory_outcomes': torch.tensor(sensory_outcomes, dtype=torch.float32, device=self.device),
            'temporal_contexts': torch.tensor(temporal_contexts, dtype=torch.float32, device=self.device),
            'predictive_accuracies': torch.tensor(predictive_accuracies, dtype=torch.float32, device=self.device)
        }
        
        self.cache_valid = True
    
    def _experience_to_tensor(self, experience: ExperienceSignature) -> Dict[str, torch.Tensor]:
        """Convert experience signature to tensor representation."""
        # Mental context
        mental_context = torch.tensor(experience.mental_context, dtype=torch.float32, device=self.device)
        
        # Motor action
        motor_action = torch.tensor([
            experience.motor_action.get('forward_motor', 0.0),
            experience.motor_action.get('turn_motor', 0.0),
            experience.motor_action.get('brake_motor', 0.0)
        ], dtype=torch.float32, device=self.device)
        
        # Sensory outcome
        sensory_values = list(experience.sensory_outcome.values())[:8]
        sensory_outcome = torch.tensor(sensory_values, dtype=torch.float32, device=self.device)
        
        # Temporal context
        temporal_context = torch.tensor(experience.temporal_context, dtype=torch.float32, device=self.device)
        
        # Predictive accuracy
        predictive_accuracy = torch.tensor([experience.prediction_accuracy], dtype=torch.float32, device=self.device)
        
        return {
            'mental_context': mental_context,
            'motor_action': motor_action,
            'sensory_outcome': sensory_outcome,
            'temporal_context': temporal_context,
            'predictive_accuracy': predictive_accuracy
        }
    
    def _compute_dimensional_similarities_gpu(self, experience_tensor: Dict[str, torch.Tensor],
                                            cached_tensors: Dict[str, torch.Tensor],
                                            search_radius: int) -> Dict[NoveltyDimension, float]:
        """Compute similarities across all dimensions using GPU acceleration."""
        dimension_similarities = {}
        
        # Mental context similarity
        mental_sim = self._compute_cosine_similarity_batch(
            experience_tensor['mental_context'].unsqueeze(0),
            cached_tensors['mental_contexts']
        )
        dimension_similarities[NoveltyDimension.MENTAL_CONTEXT] = 1.0 - float(torch.max(mental_sim))
        
        # Motor action similarity
        motor_sim = self._compute_cosine_similarity_batch(
            experience_tensor['motor_action'].unsqueeze(0),
            cached_tensors['motor_actions']
        )
        dimension_similarities[NoveltyDimension.MOTOR_ACTION] = 1.0 - float(torch.max(motor_sim))
        
        # Sensory outcome similarity
        sensory_sim = self._compute_cosine_similarity_batch(
            experience_tensor['sensory_outcome'].unsqueeze(0),
            cached_tensors['sensory_outcomes']
        )
        dimension_similarities[NoveltyDimension.SENSORY_OUTCOME] = 1.0 - float(torch.max(sensory_sim))
        
        # Temporal pattern similarity
        temporal_sim = self._compute_cosine_similarity_batch(
            experience_tensor['temporal_context'].unsqueeze(0),
            cached_tensors['temporal_contexts']
        )
        dimension_similarities[NoveltyDimension.TEMPORAL_PATTERN] = 1.0 - float(torch.max(temporal_sim))
        
        # Predictive accuracy similarity
        pred_sim = self._compute_cosine_similarity_batch(
            experience_tensor['predictive_accuracy'].unsqueeze(0),
            cached_tensors['predictive_accuracies']
        )
        dimension_similarities[NoveltyDimension.PREDICTIVE_ACCURACY] = 1.0 - float(torch.max(pred_sim))
        
        return dimension_similarities
    
    def _compute_cosine_similarity_batch(self, query: torch.Tensor, 
                                       candidates: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and all candidates."""
        # Normalize vectors
        query_norm = torch.nn.functional.normalize(query, dim=1)
        candidates_norm = torch.nn.functional.normalize(candidates, dim=1)
        
        # Compute similarities
        similarities = torch.mm(query_norm, candidates_norm.t()).squeeze(0)
        
        return similarities
    
    def _calculate_novelty_from_similarities(self, dimension_similarities: Dict[NoveltyDimension, float]) -> NoveltyScore:
        """Calculate novelty score from dimensional similarities."""
        # Use original dimension weights
        overall_novelty = sum(
            dimension_similarities[dim] * self.dimension_weights[dim]
            for dim in NoveltyDimension
        )
        
        return NoveltyScore(
            overall_novelty=overall_novelty,
            dimension_scores=dimension_similarities,
            closest_existing_node=None,  # Would need to be determined separately
            similarity_score=1.0 - overall_novelty,
            consolidation_recommendation="create_new" if overall_novelty > 0.6 else "strengthen_existing",
            confidence=0.8
        )
    
    def _create_maximum_novelty_result(self, method: ExecutionMethod) -> VectorizedNoveltyResult:
        """Create maximum novelty result when no existing experiences."""
        dimension_scores = {dim: 1.0 for dim in NoveltyDimension}
        
        novelty_score = NoveltyScore(
            overall_novelty=1.0,
            dimension_scores=dimension_scores,
            closest_existing_node=None,
            similarity_score=0.0,
            consolidation_recommendation="create_new",
            confidence=1.0
        )
        
        return VectorizedNoveltyResult(
            novelty_score=novelty_score,
            dimension_scores=dimension_scores,
            candidate_similarities=torch.tensor([]),
            computation_time=0.0,
            method_used=method
        )
    
    def evaluate_experience_novelty(self, experience_signature: ExperienceSignature,
                                  search_radius: int = 50) -> NoveltyScore:
        """Maintain compatibility with base class interface."""
        result = self.evaluate_experience_novelty_vectorized(experience_signature, search_radius)
        return result.novelty_score
    
    def invalidate_cache(self):
        """Invalidate vectorized cache when world graph changes."""
        self.cache_valid = False
        self.vectorized_cache.clear()
    
    def get_vectorized_stats(self) -> Dict[str, Any]:
        """Get comprehensive vectorized performance statistics."""
        stats = {
            'total_evaluations': self.vectorized_stats['total_evaluations'],
            'gpu_evaluations': self.vectorized_stats['gpu_evaluations'],
            'cpu_evaluations': self.vectorized_stats['cpu_evaluations'],
            'adaptive_evaluations': self.vectorized_stats['adaptive_evaluations'],
            'cache_hits': self.vectorized_stats['cache_hits'],
            'cache_misses': self.vectorized_stats['cache_misses'],
            'device': str(self.device),
            'cache_size': len(self.vectorized_cache),
            'cache_valid': self.cache_valid
        }
        
        if hasattr(self, 'adaptive_engine'):
            stats['adaptive_engine_stats'] = self.adaptive_engine.get_performance_stats()
        
        return stats