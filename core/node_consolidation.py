"""
Node Consolidation Engine.

This system handles the consolidation of similar experiences into existing nodes
rather than creating new ones. It implements reinforcement learning through
selective strengthening and intelligent connection formation.
"""

import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .experience_node import ExperienceNode
from .world_graph import WorldGraph
from .novelty_detection import NoveltyScore, ExperienceSignature


class ConsolidationStrategy(Enum):
    """Different strategies for consolidating experiences."""
    CREATE_NEW = "create_new"
    STRENGTHEN_EXISTING = "strengthen_existing"
    MERGE_COMPLETE = "merge_complete"
    CREATE_WITH_CONNECTIONS = "create_with_connections"


@dataclass
class ConsolidationResult:
    """Result of a consolidation operation."""
    strategy_used: ConsolidationStrategy
    target_node: ExperienceNode  # The node that was created or strengthened
    secondary_nodes: List[ExperienceNode]  # Other nodes that were connected
    strength_adjustments: Dict[str, float]  # Node ID -> strength change
    connections_created: List[Tuple[str, str, float]]  # (from_id, to_id, weight)
    reinforcement_applied: bool
    consolidation_confidence: float


class NodeConsolidationEngine:
    """
    Engine for consolidating similar experiences through reinforcement learning.
    
    Instead of creating new nodes for every experience, this system:
    1. Strengthens existing similar nodes
    2. Creates new connections between related experiences
    3. Applies reinforcement learning through selective strengthening
    4. Maintains memory efficiency while preserving learning
    """
    
    def __init__(self, world_graph: WorldGraph):
        self.world_graph = world_graph
        
        # Consolidation parameters (further relaxed for richer memory diversity)
        self.min_strength_for_consolidation = 3.0  # Even lower threshold = more memory diversity
        self.max_strength_boost = 10.0  # Smaller boost per consolidation for more gradual learning
        self.connection_strength_multiplier = 0.4  # Weaker connections = more diverse memories
        self.reinforcement_decay_rate = 0.99  # Slower decay = longer memory retention
        
        # Reinforcement learning parameters
        self.positive_reinforcement_rate = 1.5  # Multiplier for successful predictions
        self.negative_reinforcement_rate = 0.7   # Multiplier for failed predictions
        self.temporal_reinforcement_window = 300.0  # 5 minutes for temporal associations
        
        # Performance tracking
        self.consolidations_performed = 0
        self.nodes_strengthened = 0
        self.connections_created = 0
        self.reinforcement_events = 0
        
        # Recent consolidation history for learning
        self.recent_consolidations = []
        self.max_consolidation_history = 100
    
    def consolidate_experience(self, 
                             experience_signature: ExperienceSignature,
                             novelty_score: NoveltyScore,
                             prediction_success: bool = True) -> ConsolidationResult:
        """
        Consolidate a new experience based on its novelty assessment.
        
        Args:
            experience_signature: The experience to consolidate
            novelty_score: Novelty assessment from novelty detector
            prediction_success: Whether the prediction was successful
            
        Returns:
            ConsolidationResult with details of the consolidation
        """
        self.consolidations_performed += 1
        
        # Determine consolidation strategy
        strategy = ConsolidationStrategy(novelty_score.consolidation_recommendation)
        
        # Execute consolidation based on strategy
        if strategy == ConsolidationStrategy.CREATE_NEW:
            result = self._create_new_node(experience_signature, novelty_score)
        elif strategy == ConsolidationStrategy.STRENGTHEN_EXISTING:
            result = self._strengthen_existing_node(
                experience_signature, novelty_score, prediction_success
            )
        elif strategy == ConsolidationStrategy.MERGE_COMPLETE:
            result = self._merge_with_existing_node(
                experience_signature, novelty_score, prediction_success
            )
        elif strategy == ConsolidationStrategy.CREATE_WITH_CONNECTIONS:
            result = self._create_with_connections(experience_signature, novelty_score)
        else:
            # Fallback to creating new node
            result = self._create_new_node(experience_signature, novelty_score)
        
        # Apply reinforcement learning
        self._apply_reinforcement_learning(result, prediction_success)
        
        # Update consolidation history
        self._update_consolidation_history(result, novelty_score)
        
        return result
    
    def _create_new_node(self, 
                        experience_signature: ExperienceSignature,
                        novelty_score: NoveltyScore) -> ConsolidationResult:
        """Create a completely new experience node."""
        # Convert sensory outcome dict back to list for ExperienceNode
        sensory_values = []
        if experience_signature.sensory_outcome:
            max_idx = max(experience_signature.sensory_outcome.keys()) if experience_signature.sensory_outcome else 0
            sensory_values = [experience_signature.sensory_outcome.get(i, 0.0) for i in range(max_idx + 1)]
        
        # Create the new node using proper ExperienceNode constructor
        new_node = ExperienceNode(
            mental_context=experience_signature.mental_context,
            action_taken=experience_signature.motor_action,
            predicted_sensory=sensory_values,  # Use sensory outcome as predicted for now
            actual_sensory=sensory_values,
            prediction_error=1.0 - experience_signature.prediction_accuracy
        )
        
        # Set initial strength
        new_node.strength = 15.0  # Start with moderate strength
        
        # Note: We don't add to world graph here - that's handled by the caller
        
        # Create connections to similar nodes if any exist
        connections_created = []
        if novelty_score.closest_existing_node:
            connection_weight = (1.0 - novelty_score.overall_novelty) * self.connection_strength_multiplier
            if connection_weight > 0.3:  # Only create meaningful connections
                new_node.connection_weights[novelty_score.closest_existing_node.node_id] = connection_weight
                novelty_score.closest_existing_node.connection_weights[new_node.node_id] = connection_weight
                connections_created.append((
                    new_node.node_id,
                    novelty_score.closest_existing_node.node_id,
                    connection_weight
                ))
        
        return ConsolidationResult(
            strategy_used=ConsolidationStrategy.CREATE_NEW,
            target_node=new_node,
            secondary_nodes=[novelty_score.closest_existing_node] if novelty_score.closest_existing_node else [],
            strength_adjustments={new_node.node_id: 15.0},
            connections_created=connections_created,
            reinforcement_applied=False,
            consolidation_confidence=novelty_score.confidence
        )
    
    def _strengthen_existing_node(self,
                                 experience_signature: ExperienceSignature,
                                 novelty_score: NoveltyScore,
                                 prediction_success: bool) -> ConsolidationResult:
        """Strengthen an existing similar node instead of creating a new one."""
        if not novelty_score.closest_existing_node:
            # Fallback to creating new node
            return self._create_new_node(experience_signature, novelty_score)
        
        target_node = novelty_score.closest_existing_node
        
        # Calculate strength boost based on similarity and success
        base_strength_boost = (1.0 - novelty_score.overall_novelty) * self.max_strength_boost
        
        # Apply reinforcement learning multiplier
        if prediction_success:
            strength_boost = base_strength_boost * self.positive_reinforcement_rate
        else:
            strength_boost = base_strength_boost * self.negative_reinforcement_rate
        
        # Apply strength boost
        original_strength = target_node.strength
        target_node.strength = min(100.0, target_node.strength + strength_boost)
        actual_strength_change = target_node.strength - original_strength
        
        # Update last activation time
        target_node.last_activation_time = time.time()
        
        # Create connections to other similar nodes
        connections_created = []
        similar_nodes = self.world_graph.find_similar_nodes(
            experience_signature.mental_context,
            similarity_threshold=0.7,
            max_results=3
        )
        
        for similar_node in similar_nodes:
            if similar_node.node_id != target_node.node_id:
                # Create or strengthen connection
                existing_weight = target_node.connection_weights.get(similar_node.node_id, 0.0)
                new_weight = min(1.0, existing_weight + 0.2)
                
                if new_weight > existing_weight:
                    target_node.connection_weights[similar_node.node_id] = new_weight
                    similar_node.connection_weights[target_node.node_id] = new_weight
                    connections_created.append((
                        target_node.node_id,
                        similar_node.node_id,
                        new_weight - existing_weight
                    ))
        
        self.nodes_strengthened += 1
        self.connections_created += len(connections_created)
        
        return ConsolidationResult(
            strategy_used=ConsolidationStrategy.STRENGTHEN_EXISTING,
            target_node=target_node,
            secondary_nodes=similar_nodes,
            strength_adjustments={target_node.node_id: actual_strength_change},
            connections_created=connections_created,
            reinforcement_applied=True,
            consolidation_confidence=novelty_score.confidence
        )
    
    def _merge_with_existing_node(self,
                                 experience_signature: ExperienceSignature,
                                 novelty_score: NoveltyScore,
                                 prediction_success: bool) -> ConsolidationResult:
        """Merge experience with an existing very similar node."""
        if not novelty_score.closest_existing_node:
            return self._create_new_node(experience_signature, novelty_score)
        
        target_node = novelty_score.closest_existing_node
        
        # Merge mental contexts (weighted average)
        similarity_weight = 1.0 - novelty_score.overall_novelty
        existing_weight = 1.0 - similarity_weight
        
        # Update mental context as weighted average
        new_context = []
        for i, (existing_val, new_val) in enumerate(zip(target_node.mental_context, experience_signature.mental_context)):
            merged_val = (existing_val * existing_weight) + (new_val * similarity_weight)
            new_context.append(merged_val)
        
        target_node.mental_context = new_context
        
        # Strengthen the node significantly
        strength_boost = self.max_strength_boost * (1.5 if prediction_success else 0.8)
        original_strength = target_node.strength
        target_node.strength = min(100.0, target_node.strength + strength_boost)
        actual_strength_change = target_node.strength - original_strength
        
        # Update activation time
        target_node.last_activation_time = time.time()
        
        # Merge sensory outcomes if available
        if hasattr(target_node, 'actual_sensory') and experience_signature.sensory_outcome:
            # Could implement sensory outcome merging here
            pass
        
        return ConsolidationResult(
            strategy_used=ConsolidationStrategy.MERGE_COMPLETE,
            target_node=target_node,
            secondary_nodes=[],
            strength_adjustments={target_node.node_id: actual_strength_change},
            connections_created=[],
            reinforcement_applied=True,
            consolidation_confidence=novelty_score.confidence
        )
    
    def _create_with_connections(self,
                               experience_signature: ExperienceSignature,
                               novelty_score: NoveltyScore) -> ConsolidationResult:
        """Create new node but with strong connections to similar experiences."""
        # First create the new node
        new_node_result = self._create_new_node(experience_signature, novelty_score)
        new_node = new_node_result.target_node
        
        # Find additional similar nodes for connections
        similar_nodes = self.world_graph.find_similar_nodes(
            experience_signature.mental_context,
            similarity_threshold=0.6,
            max_results=5
        )
        
        connections_created = list(new_node_result.connections_created)
        secondary_nodes = list(new_node_result.secondary_nodes)
        
        # Create connections to multiple similar nodes
        for similar_node in similar_nodes:
            if similar_node.node_id != new_node.node_id:
                # Calculate connection strength based on similarity
                similarity = self.world_graph._calculate_context_similarity(
                    new_node.mental_context, similar_node.mental_context
                )
                connection_weight = similarity * self.connection_strength_multiplier
                
                if connection_weight > 0.4:  # Only create meaningful connections
                    new_node.connection_weights[similar_node.node_id] = connection_weight
                    similar_node.connection_weights[new_node.node_id] = connection_weight
                    connections_created.append((
                        new_node.node_id,
                        similar_node.node_id,
                        connection_weight
                    ))
                    secondary_nodes.append(similar_node)
        
        self.connections_created += len(connections_created) - len(new_node_result.connections_created)
        
        return ConsolidationResult(
            strategy_used=ConsolidationStrategy.CREATE_WITH_CONNECTIONS,
            target_node=new_node,
            secondary_nodes=secondary_nodes,
            strength_adjustments=new_node_result.strength_adjustments,
            connections_created=connections_created,
            reinforcement_applied=False,
            consolidation_confidence=novelty_score.confidence
        )
    
    def _apply_reinforcement_learning(self, 
                                    result: ConsolidationResult,
                                    prediction_success: bool):
        """Apply reinforcement learning based on prediction success."""
        if not result.reinforcement_applied:
            return
        
        self.reinforcement_events += 1
        
        # Reinforce connections based on success
        for from_id, to_id, weight in result.connections_created:
            from_node = self.world_graph.get_node(from_id)
            to_node = self.world_graph.get_node(to_id)
            
            if from_node and to_node:
                # Adjust connection strength based on prediction success
                current_weight = from_node.connection_weights.get(to_id, 0.0)
                
                if prediction_success:
                    # Strengthen successful connections
                    new_weight = min(1.0, current_weight * self.positive_reinforcement_rate)
                else:
                    # Weaken unsuccessful connections
                    new_weight = max(0.1, current_weight * self.negative_reinforcement_rate)
                
                from_node.connection_weights[to_id] = new_weight
                to_node.connection_weights[from_id] = new_weight
        
        # Apply temporal reinforcement to recent similar experiences
        self._apply_temporal_reinforcement(result, prediction_success)
    
    def _apply_temporal_reinforcement(self, 
                                    result: ConsolidationResult,
                                    prediction_success: bool):
        """Apply reinforcement to temporally related experiences."""
        current_time = time.time()
        
        # Find nodes that were activated recently
        recent_nodes = []
        for node in self.world_graph.all_nodes():
            if (current_time - node.last_activation_time) < self.temporal_reinforcement_window:
                recent_nodes.append(node)
        
        # Apply reinforcement to recent nodes
        for node in recent_nodes:
            if node.node_id != result.target_node.node_id:
                # Calculate temporal proximity weight
                time_diff = current_time - node.last_activation_time
                proximity_weight = 1.0 - (time_diff / self.temporal_reinforcement_window)
                
                # Apply reinforcement
                if prediction_success:
                    strength_boost = proximity_weight * 2.0
                    node.strength = min(100.0, node.strength + strength_boost)
                else:
                    strength_penalty = proximity_weight * 1.0
                    node.strength = max(1.0, node.strength - strength_penalty)
    
    def _update_consolidation_history(self, 
                                    result: ConsolidationResult,
                                    novelty_score: NoveltyScore):
        """Update consolidation history for learning."""
        history_entry = {
            'timestamp': time.time(),
            'strategy': result.strategy_used,
            'novelty_score': novelty_score.overall_novelty,
            'confidence': result.consolidation_confidence,
            'strength_changes': sum(result.strength_adjustments.values()),
            'connections_created': len(result.connections_created)
        }
        
        self.recent_consolidations.append(history_entry)
        if len(self.recent_consolidations) > self.max_consolidation_history:
            self.recent_consolidations.pop(0)
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get statistics about consolidation performance."""
        if not self.recent_consolidations:
            return {
                "consolidations_performed": self.consolidations_performed,
                "nodes_strengthened": self.nodes_strengthened,
                "connections_created": self.connections_created,
                "reinforcement_events": self.reinforcement_events,
                "average_novelty": 0.0,
                "average_confidence": 0.0,
                "strategy_distribution": {}
            }
        
        # Calculate averages
        avg_novelty = sum(c['novelty_score'] for c in self.recent_consolidations) / len(self.recent_consolidations)
        avg_confidence = sum(c['confidence'] for c in self.recent_consolidations) / len(self.recent_consolidations)
        
        # Calculate strategy distribution
        strategy_counts = {}
        for consolidation in self.recent_consolidations:
            strategy = consolidation['strategy'].value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "consolidations_performed": self.consolidations_performed,
            "nodes_strengthened": self.nodes_strengthened,
            "connections_created": self.connections_created,
            "reinforcement_events": self.reinforcement_events,
            "average_novelty": avg_novelty,
            "average_confidence": avg_confidence,
            "strategy_distribution": strategy_counts,
            "consolidation_rate": 1.0 - (strategy_counts.get("create_new", 0) / max(1, len(self.recent_consolidations)))
        }
    
    def optimize_consolidation_parameters(self):
        """Optimize consolidation parameters based on recent performance."""
        if len(self.recent_consolidations) < 20:
            return  # Need more data
        
        # Analyze recent performance
        recent_20 = self.recent_consolidations[-20:]
        
        # Adjust novelty threshold based on memory efficiency
        create_new_rate = sum(1 for c in recent_20 if c['strategy'] == ConsolidationStrategy.CREATE_NEW) / 20
        
        if create_new_rate > 0.8:  # Too many new nodes
            # Increase consolidation aggressiveness
            self.min_strength_for_consolidation *= 0.9
            self.max_strength_boost *= 1.1
        elif create_new_rate < 0.3:  # Too much consolidation
            # Decrease consolidation aggressiveness
            self.min_strength_for_consolidation *= 1.1
            self.max_strength_boost *= 0.9
        
        # Adjust reinforcement rates based on confidence
        avg_confidence = sum(c['confidence'] for c in recent_20) / 20
        
        if avg_confidence > 0.8:
            # High confidence - can be more aggressive with reinforcement
            self.positive_reinforcement_rate = min(2.0, self.positive_reinforcement_rate * 1.05)
        elif avg_confidence < 0.5:
            # Low confidence - be more conservative
            self.positive_reinforcement_rate = max(1.1, self.positive_reinforcement_rate * 0.95)
    
    def reset_session_stats(self):
        """Reset per-session statistics."""
        self.consolidations_performed = 0
        self.nodes_strengthened = 0
        self.connections_created = 0
        self.reinforcement_events = 0
        self.recent_consolidations.clear()