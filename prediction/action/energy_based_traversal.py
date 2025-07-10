"""
Energy-based graph traversal implementation.
Bio-inspired approach where search continues based on "energy" rather than fixed depth.
Good discoveries boost energy and trigger branching, poor results drain energy.
"""

import random
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from core.world_graph import WorldGraph
from core.experience_node import ExperienceNode
from core.communication import PredictionPacket
from .curiosity_driven_predictor import CuriosityDrivenPredictor
from .sophisticated_branching import SophisticatedBranchingManager, BranchingStrategy
from datetime import datetime


@dataclass
class ExplorationResult:
    """Result of exploring a single node."""
    node: ExperienceNode
    quality: float  # 0.0 to 1.0, how "interesting" this node is
    energy_delta: float  # Change in energy from this exploration
    promising_neighbors: List[ExperienceNode]
    reasoning: str  # Why this was interesting/uninteresting


@dataclass
class EnergyTraversalResult:
    """Result of energy-based graph traversal."""
    prediction: Optional[PredictionPacket]
    path: List[str]  # Node IDs traversed
    terminal_node: Optional[ExperienceNode]
    steps_taken: int
    final_energy: float
    branches_spawned: int
    exploration_results: List[ExplorationResult]
    termination_reason: str  # "energy_depleted", "time_limit", "natural_stopping"


class EnergyBasedTraversal:
    """
    Bio-inspired traversal that uses "energy" instead of fixed depth limits.
    
    Key principles:
    - Interesting discoveries boost energy and enable deeper exploration
    - Poor results drain energy, leading to natural stopping points
    - High-energy states trigger branching/parallel exploration
    - Time-bounded with organic termination conditions
    """
    
    def __init__(self, 
                 initial_energy: float = 100.0,
                 time_budget: float = 0.1,
                 interest_threshold: float = 0.6,
                 energy_boost_rate: float = 25.0,
                 energy_drain_rate: float = 15.0,
                 branching_threshold: float = 80.0,
                 minimum_energy: float = 5.0,
                 curiosity_momentum: float = 0.2,
                 randomness_factor: float = 0.3,
                 exploration_rate: float = 0.3):
        """
        Initialize energy-based traversal system.
        
        Args:
            initial_energy: Starting energy for search
            time_budget: Maximum time allowed for traversal
            interest_threshold: Quality threshold for "interesting" discoveries
            energy_boost_rate: Energy gained from interesting discoveries
            energy_drain_rate: Energy lost from poor discoveries
            branching_threshold: Energy level that triggers branching
            minimum_energy: Energy level below which search terminates
            curiosity_momentum: Boost multiplier for consecutive good finds
            randomness_factor: Amount of randomness in node selection
            exploration_rate: Curiosity-driven exploration rate
        """
        self.initial_energy = initial_energy
        self.time_budget = time_budget
        self.interest_threshold = interest_threshold
        self.energy_boost_rate = energy_boost_rate
        self.energy_drain_rate = energy_drain_rate
        self.branching_threshold = branching_threshold
        self.minimum_energy = minimum_energy
        self.curiosity_momentum = curiosity_momentum
        self.randomness_factor = randomness_factor
        
        self.curiosity_predictor = CuriosityDrivenPredictor(exploration_rate=exploration_rate)
        
        # Sophisticated branching system
        self.branching_manager = SophisticatedBranchingManager(
            max_concurrent_branches=4,
            total_energy_budget=initial_energy * 2  # Allow for energy accumulation
        )
        
        # Tracking variables
        self.consecutive_good_finds = 0
        self.recently_explored = set()
        self.context_patterns = []
    
    def traverse(self, start_context: List[float], world_graph: WorldGraph, 
                random_seed: int = None) -> EnergyTraversalResult:
        """
        Perform energy-based traversal through the experience graph.
        
        Args:
            start_context: Current mental context to start traversal from
            world_graph: The experience graph to traverse
            random_seed: Random seed for reproducible traversals
            
        Returns:
            EnergyTraversalResult containing prediction and traversal information
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        start_time = time.time()
        energy = self.initial_energy
        
        # Find starting node using similarity
        start_node = self._find_most_similar_node(start_context, world_graph)
        if start_node is None:
            return EnergyTraversalResult(
                prediction=None, path=[], terminal_node=None, steps_taken=0,
                final_energy=energy, branches_spawned=0, exploration_results=[],
                termination_reason="no_starting_node"
            )
        
        # Initialize traversal state
        traversal_path = [start_node.node_id]
        current_nodes = [start_node]  # Can have multiple nodes for branching
        exploration_results = []
        branches_spawned = 0
        steps_taken = 0
        
        # Main exploration loop
        while (energy > self.minimum_energy and 
               time.time() - start_time < self.time_budget and
               current_nodes):
            
            # Explore current frontier
            next_nodes = []
            
            for node in current_nodes:
                if time.time() - start_time >= self.time_budget:
                    break
                
                # Evaluate current node
                result = self._evaluate_node(node, world_graph, start_context)
                exploration_results.append(result)
                steps_taken += 1
                
                # Update energy based on result quality
                if result.quality > self.interest_threshold:
                    # Good find - boost energy
                    self.consecutive_good_finds += 1
                    momentum_multiplier = 1.0 + (self.consecutive_good_finds * self.curiosity_momentum)
                    energy_gain = self.energy_boost_rate * momentum_multiplier
                    energy += energy_gain
                    result.energy_delta = energy_gain
                    
                    # Store successful pattern
                    self.context_patterns.append(node.mental_context)
                    
                    # Consider sophisticated branching for very interesting finds
                    context = {
                        'threat_level': 'normal',  # Could be derived from sensory input
                        'cognitive_load': min(1.0, steps_taken / 20.0),  # Increases with steps
                        'time_pressure': (time.time() - start_time) / self.time_budget
                    }
                    
                    should_branch, strategy = self.branching_manager.evaluate_branching_opportunity(
                        node, energy, world_graph, context
                    )
                    
                    if should_branch:
                        new_branches = self.branching_manager.spawn_branches(
                            node, energy, strategy, world_graph, context
                        )
                        # Add branch nodes to exploration frontier
                        for branch in new_branches:
                            next_nodes.append(branch.current_node)
                        branches_spawned += len(new_branches)
                        # Update energy to account for branch allocation
                        total_branch_energy = sum(b.energy_allocated for b in new_branches)
                        energy = max(self.minimum_energy, energy - total_branch_energy * 0.3)  # Partial energy cost
                    
                    # Continue exploration from this node
                    neighbors = self._get_filtered_neighbors(node, world_graph)
                    if neighbors:
                        next_node = self._weighted_random_selection(neighbors)
                        if next_node:
                            next_nodes.append(next_node)
                            traversal_path.append(next_node.node_id)
                
                else:
                    # Poor result - drain energy
                    self.consecutive_good_finds = 0  # Reset momentum
                    energy_loss = self._calculate_energy_drain(node)
                    energy -= energy_loss
                    result.energy_delta = -energy_loss
                    
                    # Less likely to continue from poor results
                    if energy > self.minimum_energy and random.random() < 0.3:
                        neighbors = self._get_filtered_neighbors(node, world_graph)
                        if neighbors:
                            next_node = self._weighted_random_selection(neighbors)
                            if next_node:
                                next_nodes.append(next_node)
                                traversal_path.append(next_node.node_id)
                
                # Mark as recently explored
                self.recently_explored.add(node.node_id)
            
            # Update current frontier
            current_nodes = next_nodes
            
            # Update branching manager state
            self.branching_manager.update_inhibited_nodes()
            
            # Update active branches and collect any completed explorations
            active_branch_results = self._update_active_branches(world_graph, start_context)
            exploration_results.extend(active_branch_results)
            
            # Natural stopping condition - if energy is very low
            if energy < self.minimum_energy:
                break
        
        # Determine termination reason
        if energy <= self.minimum_energy:
            termination_reason = "energy_depleted"
        elif time.time() - start_time >= self.time_budget:
            termination_reason = "time_limit"
        else:
            termination_reason = "natural_stopping"
        
        # Get terminal node (last explored node)
        terminal_node = exploration_results[-1].node if exploration_results else start_node
        
        # Strengthen successful traversal path
        self._strengthen_path(traversal_path, world_graph)
        
        # Extract prediction from terminal node
        prediction = self._extract_prediction(
            terminal_node, steps_taken, start_context, world_graph
        )
        
        return EnergyTraversalResult(
            prediction=prediction,
            path=traversal_path,
            terminal_node=terminal_node,
            steps_taken=steps_taken,
            final_energy=energy,
            branches_spawned=branches_spawned,
            exploration_results=exploration_results,
            termination_reason=termination_reason
        )
    
    def _evaluate_node(self, node: ExperienceNode, world_graph: WorldGraph, 
                      start_context: List[float]) -> ExplorationResult:
        """Evaluate how interesting/promising a node is."""
        quality_factors = []
        reasoning_parts = []
        
        # Factor 1: Node strength (stronger = more important)
        strength_factor = min(1.0, node.strength / 100.0)
        quality_factors.append(strength_factor * 0.3)
        reasoning_parts.append(f"strength={node.strength:.1f}")
        
        # Factor 2: Context relevance to current situation
        context_similarity = world_graph._calculate_context_similarity(
            start_context, node.mental_context
        )
        quality_factors.append(context_similarity * 0.4)
        reasoning_parts.append(f"context_sim={context_similarity:.3f}")
        
        # Factor 3: Novelty (less recently accessed = more interesting)
        time_since_access = time.time() - node.last_activation_time
        novelty_factor = min(1.0, time_since_access / 300.0)  # 5 minutes for full novelty
        quality_factors.append(novelty_factor * 0.2)
        reasoning_parts.append(f"novelty={novelty_factor:.3f}")
        
        # Factor 4: Connection richness (more connections = more promising)
        connection_factor = min(1.0, len(node.connection_weights) / 20.0)
        quality_factors.append(connection_factor * 0.1)
        reasoning_parts.append(f"connections={len(node.connection_weights)}")
        
        # Calculate overall quality
        quality = sum(quality_factors)
        
        # Get promising neighbors for potential branching
        neighbors = self._get_filtered_neighbors(node, world_graph)
        promising_neighbors = [n for n in neighbors if n.strength > 50.0][:3]  # Top 3
        
        return ExplorationResult(
            node=node,
            quality=quality,
            energy_delta=0.0,  # Will be set by caller
            promising_neighbors=promising_neighbors,
            reasoning=" | ".join(reasoning_parts)
        )
    
    def _calculate_energy_drain(self, node: ExperienceNode) -> float:
        """Calculate energy drain based on various factors."""
        base_drain = self.energy_drain_rate
        
        # Recently explored nodes cost more energy (inhibition of return)
        if node.node_id in self.recently_explored:
            base_drain *= 1.5
        
        # Very weak nodes drain more energy
        if node.strength < 10.0:
            base_drain *= 1.3
        
        return base_drain
    
    def _update_active_branches(self, world_graph: WorldGraph, 
                               start_context: List[float]) -> List[ExplorationResult]:
        """Update and manage active branches, returning any completed explorations."""
        branch_results = []
        
        # Process each active branch
        for branch_id, branch in list(self.branching_manager.active_branches.items()):
            if branch.energy_remaining > 0:
                # Continue exploration from this branch
                result = self._evaluate_node(branch.current_node, world_graph, start_context)
                branch_results.append(result)
                
                # Update branch state
                branch.steps_taken += 1
                branch.value_accumulated += result.quality
                branch.exploration_path.append(branch.current_node.node_id)
                branch.performance_history.append(result.quality)
                
                # Update branch energy based on result
                if result.quality > self.interest_threshold:
                    branch.energy_remaining += self.energy_boost_rate * 0.5  # Branches get partial boost
                    branch.last_interesting_find = time.time()
                else:
                    branch.energy_remaining -= self.energy_drain_rate * 0.8  # Branches drain slower
                
                # Continue branch exploration if it should continue
                current_time = time.time()
                time_since_interesting = current_time - branch.last_interesting_find
                
                if branch.should_continue(time_since_interesting):
                    # Move to next node in this branch
                    neighbors = self._get_filtered_neighbors(branch.current_node, world_graph)
                    if neighbors:
                        next_node = self._weighted_random_selection(neighbors)
                        if next_node:
                            branch.current_node = next_node
                
        return branch_results
    
    def _find_most_similar_node(self, target_context: List[float], 
                               world_graph: WorldGraph) -> Optional[ExperienceNode]:
        """Find the node with the most similar mental context."""
        if not world_graph.has_nodes():
            return None
        
        # Use the existing similarity search from world_graph
        similar_nodes = world_graph.find_similar_nodes(
            target_context, similarity_threshold=0.3, max_results=1
        )
        
        if similar_nodes:
            return similar_nodes[0]
        
        # Fallback to any strong node
        strong_nodes = world_graph.get_strongest_nodes(1)
        return strong_nodes[0] if strong_nodes else None
    
    def _get_filtered_neighbors(self, node: ExperienceNode, 
                              world_graph: WorldGraph) -> List[ExperienceNode]:
        """Get neighboring nodes, filtered for relevance."""
        neighbors = []
        
        # Temporal connections
        if node.temporal_predecessor:
            pred_node = world_graph.get_node(node.temporal_predecessor)
            if pred_node:
                neighbors.append(pred_node)
        
        if node.temporal_successor:
            succ_node = world_graph.get_node(node.temporal_successor)
            if succ_node:
                neighbors.append(succ_node)
        
        # Connection-based neighbors (from connection_weights)
        for neighbor_id, weight in node.connection_weights.items():
            if weight > 0.3:  # Only follow strong connections
                neighbor = world_graph.get_node(neighbor_id)
                if neighbor:
                    neighbors.append(neighbor)
        
        # Remove duplicates and recently explored
        filtered_neighbors = []
        seen = set()
        for neighbor in neighbors:
            if (neighbor.node_id not in seen and 
                neighbor.node_id not in self.recently_explored):
                seen.add(neighbor.node_id)
                filtered_neighbors.append(neighbor)
        
        return filtered_neighbors
    
    def _weighted_random_selection(self, neighbors: List[ExperienceNode]) -> Optional[ExperienceNode]:
        """Select next node based on strength with randomness."""
        if not neighbors:
            return None
        
        # Create weights based on node strength and other factors
        weights = []
        for node in neighbors:
            # Base weight is node strength
            base_weight = max(0.1, node.strength)
            
            # Bonus for high-quality nodes
            if node.strength > 75.0:
                base_weight *= 1.5
            
            # Add randomness factor
            random_component = random.random() * self.randomness_factor
            final_weight = base_weight * (1.0 + random_component)
            
            weights.append(final_weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(neighbors)
        
        random_value = random.random() * total_weight
        cumulative_weight = 0.0
        
        for neighbor, weight in zip(neighbors, weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return neighbor
        
        return neighbors[-1]
    
    def _strengthen_path(self, traversal_path: List[str], world_graph: WorldGraph):
        """Strengthen nodes in the traversal path."""
        for i, node_id in enumerate(traversal_path):
            # More recent nodes in path get more strengthening
            strength_boost = 1.0 + (i * 0.2)
            world_graph.strengthen_node(node_id, strength_boost)
    
    def _extract_prediction(self, terminal_node: ExperienceNode, 
                          steps_taken: int, current_context: List[float],
                          world_graph: WorldGraph) -> PredictionPacket:
        """Generate prediction from terminal node."""
        # Use curiosity-driven predictor to generate intelligent action
        prediction = self.curiosity_predictor.generate_curiosity_driven_prediction(
            current_context=current_context,
            world_graph=world_graph,
            terminal_node=terminal_node,
            sequence_id=0  # Will be set by caller
        )
        
        # Update prediction metadata
        prediction.thinking_depth = steps_taken
        prediction.add_traversal_path([terminal_node.node_id])
        prediction.consensus_strength = f"energy_based_{steps_taken}_steps"
        
        return prediction
    
    def get_traversal_stats(self) -> Dict[str, Any]:
        """Get statistics about energy-based traversal configuration."""
        base_stats = {
            "initial_energy": self.initial_energy,
            "time_budget": self.time_budget,
            "interest_threshold": self.interest_threshold,
            "energy_boost_rate": self.energy_boost_rate,
            "energy_drain_rate": self.energy_drain_rate,
            "branching_threshold": self.branching_threshold,
            "minimum_energy": self.minimum_energy,
            "traversal_type": "energy_based_with_sophisticated_branching"
        }
        
        # Add branching statistics
        branching_stats = self.branching_manager.get_branching_stats()
        base_stats.update({
            "branching_stats": branching_stats
        })
        
        return base_stats
    
    def reset_session_state(self):
        """Reset per-session tracking variables."""
        self.consecutive_good_finds = 0
        self.recently_explored.clear()
        self.context_patterns.clear()
        # Reset branching manager state
        self.branching_manager.active_branches.clear()
        self.branching_manager.inhibited_nodes.clear()
        self.branching_manager.node_inhibition_times.clear()