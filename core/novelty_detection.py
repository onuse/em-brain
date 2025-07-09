"""
Multi-Dimensional Novelty Detection System.

This system evaluates whether a new experience is truly novel or should be consolidated
with existing similar experiences. It addresses the "memory bloat" problem by implementing
biological-style pattern recognition and selective encoding.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .experience_node import ExperienceNode
from .world_graph import WorldGraph
from .accelerated_similarity import AcceleratedSimilarityEngine


class NoveltyDimension(Enum):
    """Dimensions along which novelty can be evaluated."""
    MENTAL_CONTEXT = "mental_context"
    MOTOR_ACTION = "motor_action"
    SENSORY_OUTCOME = "sensory_outcome"
    TEMPORAL_PATTERN = "temporal_pattern"
    PREDICTIVE_ACCURACY = "predictive_accuracy"


@dataclass
class NoveltyScore:
    """Comprehensive novelty evaluation across multiple dimensions."""
    overall_novelty: float  # 0.0 = identical, 1.0 = completely novel
    dimension_scores: Dict[NoveltyDimension, float]
    closest_existing_node: Optional[ExperienceNode]
    similarity_score: float  # Inverse of novelty for closest match
    consolidation_recommendation: str  # "create_new", "strengthen_existing", "merge_partial"
    confidence: float  # Confidence in the novelty assessment


@dataclass
class ExperienceSignature:
    """Compact representation of an experience for novelty comparison."""
    mental_context: List[float]
    motor_action: Dict[str, float]
    sensory_outcome: Dict[str, float]
    prediction_accuracy: float
    temporal_context: List[float]  # Recent context history
    drive_states: Dict[str, float]


class NoveltyDetector:
    """
    Multi-dimensional novelty detection system that evaluates whether experiences
    are truly novel or should be consolidated with existing memories.
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        self.similarity_engine = AcceleratedSimilarityEngine()
        
        # Novelty thresholds for different dimensions (relaxed for better learning)
        self.dimension_thresholds = {
            NoveltyDimension.MENTAL_CONTEXT: 0.75,      # Relaxed - allow more context variations
            NoveltyDimension.MOTOR_ACTION: 0.60,        # Relaxed - actions can be more similar
            NoveltyDimension.SENSORY_OUTCOME: 0.65,     # Relaxed - outcomes can be more similar
            NoveltyDimension.TEMPORAL_PATTERN: 0.50,    # Relaxed - temporal patterns often repeat
            NoveltyDimension.PREDICTIVE_ACCURACY: 0.40  # Relaxed - accuracy variations are normal
        }
        
        # Weights for combining dimension scores
        self.dimension_weights = {
            NoveltyDimension.MENTAL_CONTEXT: 0.35,      # Most important - where we are mentally
            NoveltyDimension.MOTOR_ACTION: 0.25,        # Important - what we did
            NoveltyDimension.SENSORY_OUTCOME: 0.25,     # Important - what happened
            NoveltyDimension.TEMPORAL_PATTERN: 0.10,    # Less important - timing patterns
            NoveltyDimension.PREDICTIVE_ACCURACY: 0.05  # Least important - prediction quality
        }
        
        # Adaptive thresholds based on system state (relaxed for better learning)
        self.base_novelty_threshold = 0.45  # Lower threshold = more new nodes created (was 0.6)
        self.memory_pressure_factor = 1.0   # Increases threshold when memory is full
        
        # Maturity-based threshold scaling for infant brains
        self.enable_maturity_scaling = True
        self.infant_threshold_multiplier = 0.25  # Even lower threshold for infant brains (more liberal)
        self.mature_experience_count = 1000  # Brain is considered mature after 1000 experiences (was 200)
        
        # Performance tracking
        self.evaluations_performed = 0
        self.nodes_created = 0
        self.nodes_consolidated = 0
        self.false_positive_rate = 0.0
        
        # Recent context history for temporal pattern analysis
        self.recent_contexts = []
        self.max_context_history = 10
    
    def evaluate_experience_novelty(self, 
                                   experience_signature: ExperienceSignature,
                                   search_radius: int = 50) -> NoveltyScore:
        """
        Evaluate whether an experience is novel enough to warrant a new node.
        
        Args:
            experience_signature: The experience to evaluate
            search_radius: How many recent nodes to compare against
            
        Returns:
            NoveltyScore with comprehensive novelty assessment
        """
        self.evaluations_performed += 1
        
        # Get candidate nodes for comparison
        candidate_nodes = self._get_candidate_nodes(experience_signature, search_radius)
        
        # Debug logging (commented out for production)
        # if self.evaluations_performed <= 10:
        #     print(f"DEBUG: Found {len(candidate_nodes)} candidate nodes for comparison")
        
        if not candidate_nodes:
            # No existing nodes to compare against - definitely novel
            return NoveltyScore(
                overall_novelty=1.0,
                dimension_scores={dim: 1.0 for dim in NoveltyDimension},
                closest_existing_node=None,
                similarity_score=0.0,
                consolidation_recommendation="create_new",
                confidence=1.0
            )
        
        # Evaluate novelty across all dimensions
        best_match = None
        best_similarity = 0.0
        dimension_scores = {}
        
        for candidate in candidate_nodes:
            candidate_signature = self._extract_experience_signature(candidate)
            similarity_scores = self._calculate_dimensional_similarities(
                experience_signature, candidate_signature
            )
            
            # Calculate weighted overall similarity
            overall_similarity = sum(
                similarity_scores[dim] * self.dimension_weights[dim]
                for dim in NoveltyDimension
            )
            
            # Debug logging for first few candidates (commented out for production)
            # if self.evaluations_performed <= 3 and len(candidate_nodes) > 0:
            #     print(f"DEBUG: Candidate similarity = {overall_similarity:.4f}, Mental context sim = {similarity_scores[NoveltyDimension.MENTAL_CONTEXT]:.4f}")
            
            if overall_similarity > best_similarity:
                best_similarity = overall_similarity
                best_match = candidate
                dimension_scores = similarity_scores
        
        # Convert similarity to novelty (inverse relationship)
        overall_novelty = 1.0 - best_similarity
        novelty_dimension_scores = {
            dim: 1.0 - score for dim, score in dimension_scores.items()
        }
        
        # Determine consolidation recommendation
        recommendation = self._determine_consolidation_strategy(
            overall_novelty, novelty_dimension_scores, best_match
        )
        
        # Calculate confidence based on how clear the decision is
        confidence = self._calculate_confidence(overall_novelty, dimension_scores)
        
        # Debug logging for development (commented out for production)
        # if self.evaluations_performed <= 10:  # Only log first 10 evaluations
        #     print(f"DEBUG: Novelty={overall_novelty:.3f}, Threshold={self.base_novelty_threshold * self.memory_pressure_factor:.3f}, Recommendation={recommendation}")
        
        return NoveltyScore(
            overall_novelty=overall_novelty,
            dimension_scores=novelty_dimension_scores,
            closest_existing_node=best_match,
            similarity_score=best_similarity,
            consolidation_recommendation=recommendation,
            confidence=confidence
        )
    
    def _get_candidate_nodes(self, experience_signature: ExperienceSignature, 
                           search_radius: int) -> List[ExperienceNode]:
        """Get candidate nodes for novelty comparison."""
        # Strategy 1: Get nodes with similar mental contexts
        similar_context_nodes = self.world_graph.find_similar_nodes(
            experience_signature.mental_context,
            similarity_threshold=0.3,  # Lowered threshold for broader search
            max_results=search_radius // 2
        )
        
        # Debug logging (commented out for production)
        # if self.evaluations_performed <= 10:
        #     print(f"DEBUG: Mental context search found {len(similar_context_nodes)} nodes (threshold=0.3)")
        #     print(f"DEBUG: World graph has {self.world_graph.node_count()} nodes")
        #     print(f"DEBUG: Search radius // 2 = {search_radius // 2}")
        
        # Strategy 2: Get recent temporal nodes
        recent_nodes = self.world_graph.get_recent_nodes(search_radius // 2)
        
        # Debug logging (commented out for production)
        # if self.evaluations_performed <= 10:
        #     print(f"DEBUG: Recent nodes search found {len(recent_nodes)} nodes")
        
        # Strategy 3: Get nodes with similar drive states
        drive_similar_nodes = self._find_drive_similar_nodes(
            experience_signature.drive_states, search_radius // 4
        )
        
        # Combine and deduplicate
        all_candidates = similar_context_nodes + recent_nodes + drive_similar_nodes
        unique_candidates = []
        seen_ids = set()
        
        for node in all_candidates:
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                unique_candidates.append(node)
        
        return unique_candidates[:search_radius]
    
    def _find_drive_similar_nodes(self, drive_states: Dict[str, float], 
                                max_results: int) -> List[ExperienceNode]:
        """Find nodes with similar drive states."""
        # This would need to be implemented based on how drives are stored
        # For now, return empty list as drives aren't stored in experience nodes
        return []
    
    def _calculate_dimensional_similarities(self, 
                                         exp1: ExperienceSignature,
                                         exp2: ExperienceSignature) -> Dict[NoveltyDimension, float]:
        """Calculate similarity scores across all novelty dimensions."""
        similarities = {}
        
        # Mental context similarity - use simple calculation for compatibility
        context1 = exp1.mental_context
        context2 = exp2.mental_context
        
        if not context1 or not context2:
            similarities[NoveltyDimension.MENTAL_CONTEXT] = 0.0 if context1 != context2 else 1.0
        else:
            # Simple Euclidean distance calculation
            max_len = max(len(context1), len(context2))
            padded_context1 = context1 + [0.0] * (max_len - len(context1))
            padded_context2 = context2 + [0.0] * (max_len - len(context2))
            
            squared_diffs = [(a - b) ** 2 for a, b in zip(padded_context1, padded_context2)]
            distance = (sum(squared_diffs)) ** 0.5
            max_distance = (max_len * 4.0) ** 0.5
            
            similarities[NoveltyDimension.MENTAL_CONTEXT] = max(0.0, 1.0 - (distance / max_distance))
        
        # Motor action similarity
        similarities[NoveltyDimension.MOTOR_ACTION] = self._calculate_motor_similarity(
            exp1.motor_action, exp2.motor_action
        )
        
        # Sensory outcome similarity
        similarities[NoveltyDimension.SENSORY_OUTCOME] = self._calculate_sensory_similarity(
            exp1.sensory_outcome, exp2.sensory_outcome
        )
        
        # Temporal pattern similarity
        similarities[NoveltyDimension.TEMPORAL_PATTERN] = self._calculate_temporal_similarity(
            exp1.temporal_context, exp2.temporal_context
        )
        
        # Predictive accuracy similarity (with safeguards)
        try:
            acc1 = exp1.prediction_accuracy if exp1.prediction_accuracy is not None else 0.5
            acc2 = exp2.prediction_accuracy if exp2.prediction_accuracy is not None else 0.5
            
            # Debug logging (commented out for production)
            # if self.evaluations_performed <= 3:
            #     print(f"DEBUG: acc1={acc1}, acc2={acc2}, type1={type(acc1)}, type2={type(acc2)}")
            
            similarities[NoveltyDimension.PREDICTIVE_ACCURACY] = max(0.0, 1.0 - abs(acc1 - acc2))
        except (TypeError, ValueError) as e:
            # Fallback for any problematic values
            # if self.evaluations_performed <= 3:
            #     print(f"DEBUG: Exception in predictive accuracy: {e}")
            similarities[NoveltyDimension.PREDICTIVE_ACCURACY] = 0.5
        
        # Debug logging to find problematic dimension (commented out for production)
        # if self.evaluations_performed <= 3:
        #     for dim, sim in similarities.items():
        #         if sim < 0 or sim != sim:  # Check for negative or NaN values
        #             print(f"DEBUG: Dimension {dim} has problematic similarity: {sim}")
        #             if dim == NoveltyDimension.MOTOR_ACTION:
        #                 print(f"  Motor action 1: {exp1.motor_action}")
        #                 print(f"  Motor action 2: {exp2.motor_action}")
        #             elif dim == NoveltyDimension.SENSORY_OUTCOME:
        #                 print(f"  Sensory outcome 1: {exp1.sensory_outcome}")
        #                 print(f"  Sensory outcome 2: {exp2.sensory_outcome}")
        
        return similarities
    
    def _calculate_motor_similarity(self, action1: Dict[str, float], 
                                  action2: Dict[str, float]) -> float:
        """Calculate similarity between motor actions."""
        if not action1 or not action2:
            return 0.0 if action1 != action2 else 1.0
        
        # Get all motor keys
        all_keys = set(action1.keys()) | set(action2.keys())
        
        # Calculate component-wise differences
        total_diff = 0.0
        for key in all_keys:
            val1 = action1.get(key, 0.0)
            val2 = action2.get(key, 0.0)
            total_diff += abs(val1 - val2)
        
        # Convert to similarity (assuming max possible difference is 2.0 per component)
        max_possible_diff = len(all_keys) * 2.0
        similarity = 1.0 - (total_diff / max_possible_diff) if max_possible_diff > 0 else 1.0
        
        return max(0.0, similarity)
    
    def _calculate_sensory_similarity(self, sensory1: Dict[str, float], 
                                    sensory2: Dict[str, float]) -> float:
        """Calculate similarity between sensory outcomes."""
        if not sensory1 or not sensory2:
            return 0.0 if sensory1 != sensory2 else 1.0
        
        # Get all sensory keys
        all_keys = set(sensory1.keys()) | set(sensory2.keys())
        
        # Calculate component-wise differences
        total_diff = 0.0
        for key in all_keys:
            val1 = sensory1.get(key, 0.0)
            val2 = sensory2.get(key, 0.0)
            total_diff += abs(val1 - val2)
        
        # Convert to similarity (assuming max possible difference is 2.0 per component)
        max_possible_diff = len(all_keys) * 2.0
        similarity = 1.0 - (total_diff / max_possible_diff) if max_possible_diff > 0 else 1.0
        
        return max(0.0, similarity)
    
    def _calculate_temporal_similarity(self, context1: List[float], 
                                     context2: List[float]) -> float:
        """Calculate similarity between temporal context patterns."""
        if not context1 or not context2:
            return 0.0 if context1 != context2 else 1.0
        
        # Simple Euclidean distance calculation
        max_len = max(len(context1), len(context2))
        padded_context1 = context1 + [0.0] * (max_len - len(context1))
        padded_context2 = context2 + [0.0] * (max_len - len(context2))
        
        # Calculate Euclidean distance
        squared_diffs = [(a - b) ** 2 for a, b in zip(padded_context1, padded_context2)]
        distance = (sum(squared_diffs)) ** 0.5
        max_distance = (max_len * 4.0) ** 0.5  # Max possible distance
        
        # Convert to similarity
        similarity = max(0.0, 1.0 - (distance / max_distance))
        return similarity
    
    def _determine_consolidation_strategy(self, 
                                        overall_novelty: float,
                                        dimension_scores: Dict[NoveltyDimension, float],
                                        best_match: Optional[ExperienceNode]) -> str:
        """Determine the best consolidation strategy based on novelty analysis."""
        # Adjust threshold based on memory pressure
        effective_threshold = self.base_novelty_threshold * self.memory_pressure_factor
        
        # Apply maturity-based scaling for infant brains
        if self.enable_maturity_scaling:
            experience_count = self.world_graph.node_count()
            if experience_count < self.mature_experience_count:
                # Infant brain - use much lower threshold (more liberal about creating new experiences)
                maturity_ratio = experience_count / self.mature_experience_count
                infant_multiplier = self.infant_threshold_multiplier + (1.0 - self.infant_threshold_multiplier) * maturity_ratio
                effective_threshold *= infant_multiplier
                
                # Also relax the other thresholds for infant brains
                merge_threshold = 0.3 * infant_multiplier
                strengthen_threshold = 0.5 * infant_multiplier
                connection_threshold = 0.7 * infant_multiplier
            else:
                # Mature brain - use normal thresholds
                merge_threshold = 0.3
                strengthen_threshold = 0.5
                connection_threshold = 0.7
        else:
            # Normal thresholds when maturity scaling is disabled
            merge_threshold = 0.3
            strengthen_threshold = 0.5
            connection_threshold = 0.7
        
        if overall_novelty > effective_threshold:
            return "create_new"
        
        # Check if we should merge completely
        if overall_novelty < merge_threshold and best_match is not None:
            return "merge_complete"
        
        # Check if we should strengthen existing node
        if overall_novelty < strengthen_threshold and best_match is not None:
            return "strengthen_existing"
        
        # Check for partial consolidation (create connections but new node)
        if overall_novelty < connection_threshold and best_match is not None:
            return "create_with_connections"
        
        return "create_new"
    
    def _calculate_confidence(self, overall_novelty: float, 
                            dimension_scores: Dict[NoveltyDimension, float]) -> float:
        """Calculate confidence in the novelty assessment."""
        # Higher confidence when novelty is very high or very low
        extreme_factor = 1.0 - (2.0 * abs(overall_novelty - 0.5))
        
        # Higher confidence when dimension scores are consistent
        score_values = list(dimension_scores.values())
        if len(score_values) > 1:
            score_std = np.std(score_values)
            consistency_factor = 1.0 - min(1.0, score_std)
        else:
            consistency_factor = 1.0
        
        confidence = (extreme_factor * 0.6) + (consistency_factor * 0.4)
        return max(0.1, min(1.0, confidence))
    
    def _extract_experience_signature(self, node: ExperienceNode) -> ExperienceSignature:
        """Extract experience signature from an existing node."""
        # Extract motor action from node (stored in original_prediction)
        motor_action = {}
        if hasattr(node, 'original_prediction') and node.original_prediction:
            motor_action = node.original_prediction.motor_action or {}
        
        # Extract sensory outcome from node (stored in actual_sensory)
        sensory_outcome = {}
        if hasattr(node, 'actual_sensory') and node.actual_sensory:
            if isinstance(node.actual_sensory, list):
                sensory_outcome = dict(enumerate(node.actual_sensory))
            elif isinstance(node.actual_sensory, dict):
                sensory_outcome = node.actual_sensory
            else:
                sensory_outcome = {}
        
        # Calculate prediction accuracy if available
        prediction_accuracy = 0.5  # Default neutral accuracy
        if hasattr(node, 'prediction_accuracy'):
            prediction_accuracy = node.prediction_accuracy
        
        # Get temporal context (recent context history) - flatten to single list
        temporal_context = []
        if self.recent_contexts:
            # Take last few contexts and flatten them
            recent_flat = []
            for ctx in self.recent_contexts[-3:]:  # Last 3 contexts
                if isinstance(ctx, list):
                    recent_flat.extend(ctx[:3])  # Take first 3 values from each
            temporal_context = recent_flat[:8] if recent_flat else []  # Limit to 8 values
        
        # Drive states (not currently stored in nodes)
        drive_states = {}
        
        return ExperienceSignature(
            mental_context=node.mental_context,
            motor_action=motor_action,
            sensory_outcome=sensory_outcome,
            prediction_accuracy=prediction_accuracy,
            temporal_context=temporal_context,
            drive_states=drive_states
        )
    
    def update_memory_pressure(self, node_count: int, max_nodes: int):
        """Update memory pressure factor based on current memory usage."""
        if max_nodes > 0:
            usage_ratio = node_count / max_nodes
            # Increase novelty threshold when memory is getting full, but cap at reasonable level
            # This ensures the effective threshold never exceeds 1.0 (max possible novelty)
            self.memory_pressure_factor = 1.0 + min(0.3, usage_ratio * 0.3)
        else:
            self.memory_pressure_factor = 1.0
    
    def add_to_context_history(self, context: List[float]):
        """Add context to recent history for temporal pattern analysis."""
        self.recent_contexts.append(context)
        if len(self.recent_contexts) > self.max_context_history:
            self.recent_contexts.pop(0)
    
    def get_novelty_stats(self) -> Dict[str, Any]:
        """Get statistics about novelty detection performance."""
        total_decisions = self.nodes_created + self.nodes_consolidated
        
        return {
            "evaluations_performed": self.evaluations_performed,
            "nodes_created": self.nodes_created,
            "nodes_consolidated": self.nodes_consolidated,
            "consolidation_rate": self.nodes_consolidated / max(1, total_decisions),
            "memory_pressure_factor": self.memory_pressure_factor,
            "base_novelty_threshold": self.base_novelty_threshold,
            "dimension_thresholds": self.dimension_thresholds,
            "dimension_weights": self.dimension_weights
        }
    
    def reset_session_stats(self):
        """Reset per-session statistics."""
        self.evaluations_performed = 0
        self.nodes_created = 0
        self.nodes_consolidated = 0
        self.recent_contexts.clear()