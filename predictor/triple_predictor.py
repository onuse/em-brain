"""
Triple traversal predictor system.
The main prediction engine that runs 3 parallel graph traversals and finds consensus.
"""

import time
from typing import List, Dict, Any, Optional
from core.world_graph import WorldGraph
from core.communication import PredictionPacket
from .single_traversal import SingleTraversal, TraversalResult
from .consensus_resolver import ConsensusResolver, ConsensusResult


class TriplePredictor:
    """
    Main prediction engine implementing the time-budgeted traversal algorithm.
    
    This is where the robot "thinks" by running as many parallel thought processes
    as time allows, finding consensus between them - the core of emergent intelligence.
    Hardware scaling: faster computers naturally get more thinking time.
    """
    
    def __init__(self, max_depth: int = 5, randomness_factor: float = 0.3,
                 action_similarity_threshold: float = 0.1, base_time_budget: float = 0.1,
                 exploration_rate: float = 0.3):
        """
        Initialize the time-budgeted predictor system.
        
        Args:
            max_depth: Maximum depth for each traversal
            randomness_factor: Amount of randomness in node selection
            action_similarity_threshold: How similar actions need to be for consensus
            base_time_budget: Base thinking time budget in seconds (default 100ms)
            exploration_rate: Curiosity-driven exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)
        """
        self.base_time_budget = base_time_budget
        self.exploration_rate = exploration_rate
        self.single_traversal = SingleTraversal(max_depth, randomness_factor, exploration_rate)
        self.consensus_resolver = ConsensusResolver(action_similarity_threshold)
        
        # Threat-responsive time scaling
        self.threat_multipliers = {
            "safe": 2.0,      # 200ms - can think longer when safe
            "normal": 1.0,    # 100ms - baseline
            "alert": 0.5,     # 50ms - need quick decisions  
            "danger": 0.2,    # 20ms - immediate action required
            "critical": 0.05  # 5ms - emergency reflexes
        }
        
        # Statistics tracking
        self.total_predictions = 0
        self.consensus_stats = {
            'perfect': 0,
            'strong': 0, 
            'weak': 0,
            'single': 0,
            'no_consensus': 0
        }
        self.average_thinking_time = 0.0
        self.traversal_count_history = []  # Track how many traversals per prediction
    
    def generate_prediction(self, current_context: List[float], 
                           world_graph: WorldGraph,
                           sequence_id: int = 0, threat_level: str = "normal") -> ConsensusResult:
        """
        Generate a prediction using time-budgeted traversal consensus.
        
        Args:
            current_context: Current mental context to predict from
            world_graph: Experience graph to traverse
            sequence_id: Sequence ID for the prediction packet
            threat_level: Threat level affecting time budget ("safe", "normal", "alert", "danger", "critical")
            
        Returns:
            ConsensusResult containing the chosen prediction and consensus information
        """
        start_time = time.time()
        
        # Handle empty graph (bootstrap case)
        if not world_graph.has_nodes():
            return self._bootstrap_prediction(sequence_id, threat_level)
        
        # Calculate time budget based on threat level
        time_budget = self._calculate_time_budget(threat_level)
        
        # Run time-budgeted traversals
        traversal_results = []
        traversal_count = 0
        
        # First traversal (always do at least one)
        first_seed = int(time.time() * 1000)
        first_result = self.single_traversal.traverse(
            start_context=current_context,
            world_graph=world_graph,
            random_seed=first_seed
        )
        traversal_results.append(first_result)
        traversal_count = 1
        first_duration = time.time() - start_time
        
        # Continue thinking if we have time budget remaining
        while self._should_continue_thinking(start_time, time_budget, first_duration):
            random_seed = int(time.time() * 1000) + traversal_count
            
            result = self.single_traversal.traverse(
                start_context=current_context,
                world_graph=world_graph,
                random_seed=random_seed
            )
            
            traversal_results.append(result)
            traversal_count += 1
            
            # Safety brake for runaway thinking
            if traversal_count >= 20:  # Hardware safety limit
                break
        
        # Track how many traversals we managed
        self.traversal_count_history.append(traversal_count)
        
        # Resolve consensus between traversals
        consensus_result = self.consensus_resolver.resolve_consensus(traversal_results)
        
        # Update prediction packet with metadata
        if consensus_result.prediction:
            consensus_result.prediction.sequence_id = sequence_id
            consensus_result.prediction.consensus_strength = consensus_result.consensus_strength
            
            # Add all traversal paths to prediction
            all_paths = [result.path for result in traversal_results if result.prediction]
            consensus_result.prediction.traversal_paths = all_paths
            
            # Add thinking metadata
            consensus_result.prediction.traversal_count = traversal_count
            consensus_result.prediction.time_budget_used = time.time() - start_time
            consensus_result.prediction.threat_level = threat_level
        
        # Update statistics
        self._update_statistics(consensus_result, time.time() - start_time)
        
        return consensus_result
    
    def _calculate_time_budget(self, threat_level: str) -> float:
        """Calculate thinking time budget based on threat level."""
        multiplier = self.threat_multipliers.get(threat_level, 1.0)
        return self.base_time_budget * multiplier
    
    def _should_continue_thinking(self, start_time: float, time_budget: float, first_duration: float) -> bool:
        """Decide whether to start another traversal based on time remaining."""
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        
        # Conservative estimate: need 1.2x first duration for next traversal
        # This accounts for potential variance in traversal times
        estimated_next = first_duration * 1.2
        
        return remaining > estimated_next
    
    def _bootstrap_prediction(self, sequence_id: int, threat_level: str = "normal") -> ConsensusResult:
        """Generate a bootstrap prediction when the graph is empty."""
        from datetime import datetime
        from .curiosity_driven_predictor import CuriosityDrivenPredictor
        
        # Use curiosity-driven predictor for bootstrap
        curiosity_predictor = CuriosityDrivenPredictor(exploration_rate=self.exploration_rate)
        bootstrap_prediction = curiosity_predictor._create_bootstrap_prediction(sequence_id)
        
        # Update bootstrap metadata
        bootstrap_prediction.consensus_strength = "bootstrap"
        bootstrap_prediction.traversal_paths = []
        bootstrap_prediction.threat_level = threat_level
        bootstrap_prediction.traversal_count = 0
        bootstrap_prediction.time_budget_used = 0.0
        
        return ConsensusResult(
            prediction=bootstrap_prediction,
            consensus_strength="bootstrap",
            agreement_count=0,
            total_traversals=0,
            reasoning="Empty graph - using curiosity-driven bootstrap exploration"
        )
    
    def _update_statistics(self, consensus_result: ConsensusResult, thinking_time: float):
        """Update internal statistics tracking."""
        self.total_predictions += 1
        
        # Update consensus statistics
        consensus_type = consensus_result.consensus_strength
        if consensus_type in self.consensus_stats:
            self.consensus_stats[consensus_type] += 1
        
        # Update average thinking time
        self.average_thinking_time = (
            (self.average_thinking_time * (self.total_predictions - 1) + thinking_time) 
            / self.total_predictions
        )
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about prediction performance."""
        if self.total_predictions == 0:
            return {"total_predictions": 0}
        
        # Calculate consensus percentages
        consensus_percentages = {}
        for consensus_type, count in self.consensus_stats.items():
            percentage = (count / self.total_predictions) * 100
            consensus_percentages[f"{consensus_type}_percentage"] = percentage
        
        return {
            "total_predictions": self.total_predictions,
            "consensus_counts": self.consensus_stats.copy(),
            "consensus_percentages": consensus_percentages,
            "average_thinking_time": self.average_thinking_time,
            "strong_consensus_rate": (
                (self.consensus_stats['perfect'] + self.consensus_stats['strong']) 
                / self.total_predictions * 100
            ) if self.total_predictions > 0 else 0,
            "traversal_statistics": self._get_traversal_stats(),
            "configuration": {
                "base_time_budget": self.base_time_budget,
                "max_depth": self.single_traversal.max_depth,
                "randomness_factor": self.single_traversal.randomness_factor,
                "action_similarity_threshold": self.consensus_resolver.action_similarity_threshold,
                "threat_multipliers": self.threat_multipliers
            }
        }
    
    def get_recent_consensus_quality(self, recent_window: int = 10) -> float:
        """
        Get the quality of recent consensus decisions.
        Returns a score from 0.0 to 1.0 based on consensus strength.
        """
        if self.total_predictions == 0:
            return 0.0
        
        # Simple quality scoring based on consensus types
        quality_scores = {
            'perfect': 1.0,
            'strong': 0.8,
            'weak': 0.4,
            'single': 0.2,
            'no_consensus': 0.0,
            'bootstrap': 0.1
        }
        
        # Calculate weighted average (recent predictions weighted more heavily)
        total_quality = 0.0
        total_weight = 0.0
        
        for consensus_type, count in self.consensus_stats.items():
            if count > 0:
                quality = quality_scores.get(consensus_type, 0.0)
                # Simple weighting - could be made more sophisticated
                weight = count
                total_quality += quality * weight
                total_weight += weight
        
        return total_quality / total_weight if total_weight > 0 else 0.0
    
    def _get_traversal_stats(self) -> Dict[str, Any]:
        """Get statistics about traversal counts and hardware scaling."""
        if not self.traversal_count_history:
            return {"no_data": True}
        
        counts = self.traversal_count_history
        return {
            "average_traversals_per_prediction": sum(counts) / len(counts),
            "min_traversals": min(counts),
            "max_traversals": max(counts),
            "total_traversals": sum(counts),
            "hardware_scaling_factor": sum(counts) / len(counts) / 3.0,  # Relative to old "triple" system
            "recent_traversals": counts[-10:] if len(counts) >= 10 else counts
        }
    
    def set_adaptive_parameters(self, new_depth: int = None, new_randomness: float = None,
                               new_similarity_threshold: float = None, new_time_budget: float = None):
        """
        Adjust predictor parameters based on learning performance.
        This allows the system to adapt its thinking patterns.
        """
        if new_depth is not None:
            self.single_traversal.max_depth = max(1, min(10, new_depth))
        
        if new_randomness is not None:
            self.single_traversal.randomness_factor = max(0.0, min(1.0, new_randomness))
        
        if new_similarity_threshold is not None:
            self.consensus_resolver.action_similarity_threshold = max(0.01, min(0.5, new_similarity_threshold))
        
        if new_time_budget is not None:
            self.base_time_budget = max(0.001, min(1.0, new_time_budget))  # 1ms to 1s range
    
    def reset_statistics(self):
        """Reset all prediction statistics."""
        self.total_predictions = 0
        self.consensus_stats = {
            'perfect': 0,
            'strong': 0,
            'weak': 0, 
            'single': 0,
            'no_consensus': 0
        }
        self.average_thinking_time = 0.0
        self.traversal_count_history = []