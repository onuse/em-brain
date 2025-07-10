"""
Single graph traversal implementation.
Walks through the experience graph to find predictions based on similarity and strength.
"""

import random
from typing import List, Optional, Dict, Any, Tuple
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.communication import PredictionPacket
from .curiosity_driven_predictor import CuriosityDrivenPredictor
from core.gpu_similarity_engine import get_gpu_similarity_engine
from datetime import datetime


class TraversalResult:
    """Result of a single graph traversal."""
    
    def __init__(self, prediction: Optional[PredictionPacket], path: List[str], 
                 terminal_node: Optional[ExperienceNode], depth_reached: int):
        self.prediction = prediction
        self.path = path  # List of node IDs traversed
        self.terminal_node = terminal_node
        self.depth_reached = depth_reached
        self.terminal_strength = terminal_node.strength if terminal_node else 0.0


class SingleTraversal:
    """
    Implements single graph traversal with depth limiting and weighted selection.
    This is where the robot "thinks" by walking through its memories.
    """
    
    def __init__(self, max_depth: int = 5, randomness_factor: float = 0.3, 
                 exploration_rate: float = 0.3):
        """
        Initialize single traversal engine.
        
        Args:
            max_depth: Maximum depth to traverse in the graph
            randomness_factor: Amount of randomness in node selection (0.0 = pure strength, 1.0 = pure random)
            exploration_rate: Curiosity-driven exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
        """
        self.max_depth = max_depth
        self.randomness_factor = randomness_factor
        self.curiosity_predictor = CuriosityDrivenPredictor(exploration_rate=exploration_rate)
        
        # Initialize GPU similarity engine for vectorized similarity calculations
        self.gpu_similarity_engine = get_gpu_similarity_engine()
    
    def traverse(self, start_context: List[float], world_graph: WorldGraph, 
                random_seed: int = None) -> TraversalResult:
        """
        Perform a single traversal through the experience graph.
        
        Args:
            start_context: Current mental context to start traversal from
            world_graph: The experience graph to traverse
            random_seed: Random seed for reproducible traversals
            
        Returns:
            TraversalResult containing prediction and traversal information
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        # Find starting node using similarity
        start_node = self._find_most_similar_node(start_context, world_graph)
        if start_node is None:
            return TraversalResult(None, [], None, 0)
        
        # Perform bounded depth traversal
        traversal_path = [start_node.node_id]
        current_node = start_node
        
        for depth in range(self.max_depth):
            # Get neighboring nodes
            neighbors = self._get_all_neighbors(current_node, world_graph)
            
            if not neighbors:
                break  # Dead end reached
            
            # Select next node using weighted random selection
            next_node = self._weighted_random_selection(neighbors)
            if next_node is None:
                break
            
            traversal_path.append(next_node.node_id)
            current_node = next_node
        
        # Strengthen all nodes in the traversal path
        self._strengthen_path(traversal_path, world_graph)
        
        # Extract prediction from terminal node
        prediction = self._extract_prediction(
            current_node, len(traversal_path), start_context, world_graph
        )
        
        return TraversalResult(
            prediction=prediction,
            path=traversal_path,
            terminal_node=current_node,
            depth_reached=len(traversal_path)
        )
    
    def _find_most_similar_node(self, target_context: List[float], 
                               world_graph: WorldGraph) -> Optional[ExperienceNode]:
        """Find the node with the most similar mental context using GPU vectorization."""
        if not world_graph.has_nodes():
            return None
        
        # Get all nodes and their contexts
        all_nodes = list(world_graph.all_nodes())
        if not all_nodes:
            return None
        
        # Extract contexts and strengths for batch processing
        candidate_contexts = [node.mental_context for node in all_nodes]
        candidate_strengths = [node.strength for node in all_nodes]
        
        # Use GPU vectorized similarity search
        best_idx, best_similarity = self.gpu_similarity_engine.find_most_similar_batch(
            target_context, candidate_contexts, candidate_strengths
        )
        
        if best_idx >= 0 and best_idx < len(all_nodes):
            return all_nodes[best_idx]
        
        return None
    
    def _get_all_neighbors(self, node: ExperienceNode, 
                          world_graph: WorldGraph) -> List[ExperienceNode]:
        """Get all connected nodes (temporal, prediction sources, similar contexts)."""
        neighbors = []
        
        # Add temporally connected nodes
        if node.temporal_predecessor:
            pred_node = world_graph.get_node(node.temporal_predecessor)
            if pred_node:
                neighbors.append(pred_node)
        
        if node.temporal_successor:
            succ_node = world_graph.get_node(node.temporal_successor)
            if succ_node:
                neighbors.append(succ_node)
        
        # Add prediction source nodes
        for source_id in node.prediction_sources:
            source_node = world_graph.get_node(source_id)
            if source_node:
                neighbors.append(source_node)
        
        # Add similar context nodes (most important for learning)
        for similar_id in node.similar_contexts:
            similar_node = world_graph.get_node(similar_id)
            if similar_node:
                neighbors.append(similar_node)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_neighbors = []
        for neighbor in neighbors:
            if neighbor.node_id not in seen:
                seen.add(neighbor.node_id)
                unique_neighbors.append(neighbor)
        
        return unique_neighbors
    
    def _weighted_random_selection(self, neighbors: List[ExperienceNode]) -> Optional[ExperienceNode]:
        """Select next node based on strength with randomness."""
        if not neighbors:
            return None
        
        # Create weights based on node strength
        weights = []
        for node in neighbors:
            # Base weight is node strength
            base_weight = max(0.1, node.strength)  # Minimum weight to avoid zero
            
            # Add randomness factor
            random_component = random.random() * self.randomness_factor
            final_weight = base_weight * (1.0 + random_component)
            
            weights.append(final_weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(neighbors)
        
        # Generate random value and find corresponding node
        random_value = random.random() * total_weight
        cumulative_weight = 0.0
        
        for neighbor, weight in zip(neighbors, weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return neighbor
        
        # Fallback (shouldn't happen)
        return neighbors[-1]
    
    def _strengthen_path(self, traversal_path: List[str], world_graph: WorldGraph):
        """Increase strength of all nodes in the traversal path."""
        for node_id in traversal_path:
            # Strengthen the node itself
            world_graph.strengthen_node(node_id, 1.0)
            
            # Spillover effect: slightly strengthen similar nodes
            node = world_graph.get_node(node_id)
            if node:
                for similar_id in node.similar_contexts:
                    world_graph.strengthen_node(similar_id, 0.1)
    
    def _extract_prediction(self, terminal_node: ExperienceNode, 
                           thinking_depth: int, current_context: List[float] = None, 
                           world_graph: WorldGraph = None) -> PredictionPacket:
        """Generate curiosity-driven prediction from terminal node."""
        # Use curiosity-driven predictor to generate intelligent action
        prediction = self.curiosity_predictor.generate_curiosity_driven_prediction(
            current_context=current_context or [],
            world_graph=world_graph,
            terminal_node=terminal_node,
            sequence_id=0  # Will be set by caller
        )
        
        # Update prediction metadata
        prediction.thinking_depth = thinking_depth
        prediction.add_traversal_path([terminal_node.node_id])
        prediction.consensus_strength = "curiosity_driven"
        
        return prediction
    
    def get_traversal_stats(self) -> Dict[str, Any]:
        """Get statistics about traversal configuration."""
        return {
            "max_depth": self.max_depth,
            "randomness_factor": self.randomness_factor,
            "traversal_type": "single"
        }