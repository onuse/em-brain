"""
Sophisticated Branching Mechanism for Energy-Based Traversal.
Implements biologically-inspired attention allocation and competitive exploration.

Current simple branching:
- Fixed threshold (energy > 80) 
- Spawn all promising neighbors
- No resource competition
- No strategic allocation

Sophisticated branching:
- Competitive attention allocation
- Dynamic resource management  
- Predictive value estimation
- Context-dependent strategies
- Inhibition mechanisms
"""

import time
import random
import heapq
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from core.experience_node import ExperienceNode
from core.world_graph import WorldGraph


class BranchingStrategy(Enum):
    """Different branching strategies based on context."""
    CONSERVATIVE = "conservative"      # Few, high-confidence branches
    AGGRESSIVE = "aggressive"          # Many parallel explorations  
    FOCUSED = "focused"               # Single deep exploration
    CURIOSITY_DRIVEN = "curiosity"    # Novel/unexpected directions
    EMERGENCY = "emergency"           # Rapid, shallow search


@dataclass
class BranchCandidate:
    """A potential exploration branch with predicted value."""
    node: ExperienceNode
    predicted_value: float        # Expected utility of exploring this branch
    confidence: float            # Confidence in the prediction
    resource_cost: float         # Estimated computational cost
    exploration_type: str        # Type of exploration (temporal, similarity, etc.)
    priority_score: float = 0.0  # Final priority score
    
    def calculate_priority(self, available_energy: float, context: Dict) -> float:
        """Calculate branch priority based on value, cost, and context."""
        # Base priority from predicted value and confidence
        base_priority = self.predicted_value * self.confidence
        
        # Cost-benefit ratio
        if self.resource_cost > 0:
            efficiency = base_priority / self.resource_cost
        else:
            efficiency = base_priority
        
        # Context modifiers
        threat_level = context.get('threat_level', 'normal')
        if threat_level in ['danger', 'critical']:
            # In danger, prioritize high-confidence, low-cost options
            urgency_modifier = self.confidence * (2.0 - self.resource_cost)
        else:
            # In safety, can afford more expensive exploration
            urgency_modifier = 1.0
        
        # Energy availability modifier
        energy_modifier = min(1.0, available_energy / 50.0)  # Full modifier at 50+ energy
        
        self.priority_score = efficiency * urgency_modifier * energy_modifier
        return self.priority_score


@dataclass
class ActiveBranch:
    """An actively exploring branch with resource tracking."""
    branch_id: str
    current_node: ExperienceNode
    energy_allocated: float
    energy_remaining: float
    steps_taken: int
    value_accumulated: float
    last_interesting_find: float  # Timestamp
    exploration_path: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    
    def get_performance_trend(self) -> float:
        """Calculate if this branch is improving or declining."""
        if len(self.performance_history) < 2:
            return 0.0
        
        recent = self.performance_history[-3:]
        if len(recent) < 2:
            return 0.0
        
        # Simple trend calculation
        return recent[-1] - recent[0]
    
    def should_continue(self, time_since_interesting: float) -> bool:
        """Determine if this branch should continue exploring."""
        # Stop if out of energy
        if self.energy_remaining <= 0:
            return False
        
        # Stop if no interesting finds recently
        if time_since_interesting > 30.0:  # 30 seconds without interesting find
            return False
        
        # Stop if performance is declining consistently
        trend = self.get_performance_trend()
        if len(self.performance_history) >= 3 and trend < -0.1:
            return False
        
        return True


class SophisticatedBranchingManager:
    """
    Advanced branching system that implements competitive attention allocation
    and strategic exploration management.
    """
    
    def __init__(self, max_concurrent_branches: int = 4, 
                 total_energy_budget: float = 200.0):
        self.max_concurrent_branches = max_concurrent_branches
        self.total_energy_budget = total_energy_budget
        
        # Active branch management
        self.active_branches: Dict[str, ActiveBranch] = {}
        self.branch_counter = 0
        
        # Resource allocation
        self.energy_allocation_strategy = "dynamic"  # or "equal", "priority_based"
        self.min_energy_per_branch = 10.0
        
        # Performance tracking
        self.successful_strategies: Dict[BranchingStrategy, int] = {}
        self.failed_strategies: Dict[BranchingStrategy, int] = {}
        
        # Inhibition mechanisms
        self.inhibited_nodes: Set[str] = set()
        self.inhibition_decay_time = 60.0  # Seconds before inhibition wears off
        self.node_inhibition_times: Dict[str, float] = {}
    
    def evaluate_branching_opportunity(self, current_node: ExperienceNode,
                                     available_energy: float,
                                     world_graph: WorldGraph,
                                     context: Dict) -> Tuple[bool, BranchingStrategy]:
        """
        Sophisticated decision about whether and how to branch.
        
        Returns:
            (should_branch, strategy_to_use)
        """
        # 1. Context-based strategy selection
        strategy = self._select_branching_strategy(context, available_energy)
        
        # 2. Resource availability check
        if available_energy < self.min_energy_per_branch * 2:  # Need energy for at least 2 branches
            return False, strategy
        
        # 3. Current branch load check
        if len(self.active_branches) >= self.max_concurrent_branches:
            # Consider terminating weak branches to make room
            self._prune_weak_branches()
            if len(self.active_branches) >= self.max_concurrent_branches:
                return False, strategy
        
        # 4. Opportunity assessment
        branch_candidates = self._generate_branch_candidates(current_node, world_graph, context)
        if not branch_candidates:
            return False, strategy
        
        # 5. Value threshold check
        best_candidate = max(branch_candidates, key=lambda c: c.predicted_value)
        min_value_threshold = self._get_value_threshold(strategy, context)
        
        if best_candidate.predicted_value < min_value_threshold:
            return False, strategy
        
        return True, strategy
    
    def spawn_branches(self, current_node: ExperienceNode,
                      available_energy: float,
                      strategy: BranchingStrategy,
                      world_graph: WorldGraph,
                      context: Dict) -> List[ActiveBranch]:
        """
        Spawn new exploration branches using the specified strategy.
        """
        # Generate and evaluate candidates
        candidates = self._generate_branch_candidates(current_node, world_graph, context)
        
        # Score and prioritize candidates
        for candidate in candidates:
            candidate.calculate_priority(available_energy, context)
        
        # Select branches based on strategy
        selected_branches = self._select_branches_by_strategy(candidates, strategy, available_energy)
        
        # Allocate energy to selected branches
        active_branches = self._create_active_branches(selected_branches, available_energy, strategy)
        
        # Update tracking
        self.successful_strategies[strategy] = self.successful_strategies.get(strategy, 0) + 1
        
        return active_branches
    
    def _select_branching_strategy(self, context: Dict, available_energy: float) -> BranchingStrategy:
        """Select the most appropriate branching strategy for the current context."""
        threat_level = context.get('threat_level', 'normal')
        cognitive_load = context.get('cognitive_load', 0.5)
        time_pressure = context.get('time_pressure', 0.5)
        
        # Emergency situations: rapid, shallow exploration
        if threat_level in ['danger', 'critical'] or time_pressure > 0.8:
            return BranchingStrategy.EMERGENCY
        
        # High cognitive load: conservative approach
        if cognitive_load > 0.8:
            return BranchingStrategy.CONSERVATIVE
        
        # Safe environment with good energy: curiosity-driven exploration
        if threat_level == 'safe' and available_energy > 100:
            return BranchingStrategy.CURIOSITY_DRIVEN
        
        # Limited energy: focused exploration
        if available_energy < 50:
            return BranchingStrategy.FOCUSED
        
        # Default: aggressive exploration
        return BranchingStrategy.AGGRESSIVE
    
    def _generate_branch_candidates(self, current_node: ExperienceNode,
                                  world_graph: WorldGraph,
                                  context: Dict) -> List[BranchCandidate]:
        """Generate potential exploration branches from current node."""
        candidates = []
        
        # 1. Temporal exploration (predecessor/successor)
        if current_node.temporal_predecessor:
            pred_node = world_graph.get_node(current_node.temporal_predecessor)
            if pred_node and pred_node.node_id not in self.inhibited_nodes:
                value = self._predict_branch_value(pred_node, "temporal_backward", context)
                candidates.append(BranchCandidate(
                    node=pred_node,
                    predicted_value=value,
                    confidence=0.8,  # Temporal connections are reliable
                    resource_cost=5.0,
                    exploration_type="temporal_backward"
                ))
        
        if current_node.temporal_successor:
            succ_node = world_graph.get_node(current_node.temporal_successor)
            if succ_node and succ_node.node_id not in self.inhibited_nodes:
                value = self._predict_branch_value(succ_node, "temporal_forward", context)
                candidates.append(BranchCandidate(
                    node=succ_node,
                    predicted_value=value,
                    confidence=0.8,
                    resource_cost=5.0,
                    exploration_type="temporal_forward"
                ))
        
        # 2. Similarity-based exploration (strongest connections)
        connection_items = list(current_node.connection_weights.items())
        connection_items.sort(key=lambda x: x[1], reverse=True)  # Sort by weight
        
        for neighbor_id, weight in connection_items[:5]:  # Top 5 connections
            if neighbor_id not in self.inhibited_nodes:
                neighbor = world_graph.get_node(neighbor_id)
                if neighbor:
                    value = self._predict_branch_value(neighbor, "similarity", context)
                    confidence = min(0.9, weight)  # Higher weight = higher confidence
                    cost = 10.0 / max(0.1, weight)  # Stronger connections cost less
                    
                    candidates.append(BranchCandidate(
                        node=neighbor,
                        predicted_value=value * weight,  # Weight by connection strength
                        confidence=confidence,
                        resource_cost=cost,
                        exploration_type="similarity"
                    ))
        
        # 3. Novelty-based exploration (least recently accessed)
        similar_nodes = world_graph.find_similar_nodes(
            current_node.mental_context, 
            similarity_threshold=0.5, 
            max_results=10
        )
        
        for node in similar_nodes:
            if node.node_id not in self.inhibited_nodes:
                # Novelty based on time since last access
                time_since_access = time.time() - node.last_activation_time
                novelty_factor = min(1.0, time_since_access / 300.0)  # 5 minutes for full novelty
                
                if novelty_factor > 0.3:  # Only consider reasonably novel nodes
                    value = self._predict_branch_value(node, "novelty", context)
                    candidates.append(BranchCandidate(
                        node=node,
                        predicted_value=value * novelty_factor,
                        confidence=0.6,  # Lower confidence for novel explorations
                        resource_cost=15.0,  # Novelty exploration is more expensive
                        exploration_type="novelty"
                    ))
        
        return candidates
    
    def _predict_branch_value(self, node: ExperienceNode, exploration_type: str, 
                            context: Dict) -> float:
        """Predict the value of exploring a particular branch."""
        base_value = 0.0
        
        # Node strength factor
        strength_factor = min(1.0, node.strength / 100.0)
        base_value += strength_factor * 0.4
        
        # Context relevance
        # (This would need access to current sensory context)
        context_relevance = 0.5  # Placeholder
        base_value += context_relevance * 0.3
        
        # Exploration type bonuses
        if exploration_type == "temporal_forward":
            base_value += 0.2  # Future predictions are valuable
        elif exploration_type == "similarity":
            base_value += 0.15  # Similar contexts often useful
        elif exploration_type == "novelty":
            base_value += 0.25  # Novelty can lead to breakthroughs
        
        # Connection richness (nodes with many connections are hubs)
        connection_factor = min(1.0, len(node.connection_weights) / 20.0)
        base_value += connection_factor * 0.1
        
        return min(1.0, base_value)
    
    def _select_branches_by_strategy(self, candidates: List[BranchCandidate],
                                   strategy: BranchingStrategy,
                                   available_energy: float) -> List[BranchCandidate]:
        """Select which candidates to explore based on strategy."""
        if not candidates:
            return []
        
        # Sort by priority
        candidates.sort(key=lambda c: c.priority_score, reverse=True)
        
        if strategy == BranchingStrategy.CONSERVATIVE:
            # Take only the best 1-2 candidates
            return candidates[:min(2, len(candidates))]
        
        elif strategy == BranchingStrategy.AGGRESSIVE:
            # Take as many as energy allows
            selected = []
            energy_used = 0
            for candidate in candidates:
                if energy_used + candidate.resource_cost <= available_energy * 0.8:  # Use 80% of energy
                    selected.append(candidate)
                    energy_used += candidate.resource_cost
                    if len(selected) >= self.max_concurrent_branches:
                        break
            return selected
        
        elif strategy == BranchingStrategy.FOCUSED:
            # Single best candidate
            return candidates[:1]
        
        elif strategy == BranchingStrategy.CURIOSITY_DRIVEN:
            # Prefer novelty and high-value explorations
            novelty_candidates = [c for c in candidates if c.exploration_type == "novelty"]
            high_value_candidates = [c for c in candidates if c.predicted_value > 0.7]
            
            # Mix of novelty and high-value
            selected = []
            selected.extend(novelty_candidates[:2])
            selected.extend([c for c in high_value_candidates if c not in selected][:2])
            return selected[:self.max_concurrent_branches]
        
        elif strategy == BranchingStrategy.EMERGENCY:
            # Quick, low-cost options only
            quick_candidates = [c for c in candidates if c.resource_cost < 8.0]
            return quick_candidates[:3]
        
        return candidates[:2]  # Default fallback
    
    def _create_active_branches(self, selected_candidates: List[BranchCandidate],
                              available_energy: float,
                              strategy: BranchingStrategy) -> List[ActiveBranch]:
        """Create active branch objects with energy allocation."""
        active_branches = []
        
        if not selected_candidates:
            return active_branches
        
        # Calculate energy allocation
        if self.energy_allocation_strategy == "equal":
            energy_per_branch = available_energy / len(selected_candidates)
        elif self.energy_allocation_strategy == "priority_based":
            total_priority = sum(c.priority_score for c in selected_candidates)
            energy_allocations = [
                (c.priority_score / total_priority) * available_energy 
                for c in selected_candidates
            ]
        else:  # dynamic
            energy_allocations = [
                max(self.min_energy_per_branch, c.resource_cost * 2.0)
                for c in selected_candidates
            ]
        
        # Create active branches
        for i, candidate in enumerate(selected_candidates):
            self.branch_counter += 1
            branch_id = f"branch_{self.branch_counter}"
            
            energy_allocated = energy_allocations[i] if self.energy_allocation_strategy == "priority_based" else energy_allocations[i]
            
            active_branch = ActiveBranch(
                branch_id=branch_id,
                current_node=candidate.node,
                energy_allocated=energy_allocated,
                energy_remaining=energy_allocated,
                steps_taken=0,
                value_accumulated=0.0,
                last_interesting_find=time.time(),
                exploration_path=[candidate.node.node_id]
            )
            
            self.active_branches[branch_id] = active_branch
            active_branches.append(active_branch)
        
        return active_branches
    
    def _prune_weak_branches(self):
        """Remove underperforming branches to free up resources."""
        current_time = time.time()
        branches_to_remove = []
        
        for branch_id, branch in self.active_branches.items():
            time_since_interesting = current_time - branch.last_interesting_find
            
            if not branch.should_continue(time_since_interesting):
                branches_to_remove.append(branch_id)
                # Add nodes to inhibition list to avoid immediate re-exploration
                for node_id in branch.exploration_path[-3:]:  # Last 3 nodes
                    self.inhibited_nodes.add(node_id)
                    self.node_inhibition_times[node_id] = current_time
        
        for branch_id in branches_to_remove:
            del self.active_branches[branch_id]
    
    def _get_value_threshold(self, strategy: BranchingStrategy, context: Dict) -> float:
        """Get minimum value threshold for branching based on strategy."""
        base_threshold = 0.5
        
        if strategy == BranchingStrategy.CONSERVATIVE:
            return base_threshold + 0.2
        elif strategy == BranchingStrategy.AGGRESSIVE:
            return base_threshold - 0.1
        elif strategy == BranchingStrategy.EMERGENCY:
            return base_threshold - 0.2
        elif strategy == BranchingStrategy.CURIOSITY_DRIVEN:
            return base_threshold - 0.15
        
        return base_threshold
    
    def update_inhibited_nodes(self):
        """Remove nodes from inhibition list after decay time."""
        current_time = time.time()
        nodes_to_remove = []
        
        for node_id, inhibition_time in self.node_inhibition_times.items():
            if current_time - inhibition_time > self.inhibition_decay_time:
                nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            self.inhibited_nodes.discard(node_id)
            del self.node_inhibition_times[node_id]
    
    def get_branching_stats(self) -> Dict[str, any]:
        """Get current branching system statistics."""
        total_strategies_tried = sum(self.successful_strategies.values()) + sum(self.failed_strategies.values())
        
        return {
            "active_branches": len(self.active_branches),
            "max_concurrent": self.max_concurrent_branches,
            "inhibited_nodes": len(self.inhibited_nodes),
            "total_strategies_tried": total_strategies_tried,
            "successful_strategies": dict(self.successful_strategies),
            "failed_strategies": dict(self.failed_strategies),
            "energy_allocation_strategy": self.energy_allocation_strategy
        }