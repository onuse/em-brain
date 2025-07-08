"""
Consensus resolution for multiple graph traversals.
Implements the "2 out of 3 agreement" logic and fallback strategies.
"""

from typing import List, Tuple, Dict, Any
from core.communication import PredictionPacket
from .single_traversal import TraversalResult


class ConsensusResult:
    """Result of consensus resolution between multiple traversals."""
    
    def __init__(self, prediction: PredictionPacket, consensus_strength: str, 
                 agreement_count: int, total_traversals: int, reasoning: str):
        self.prediction = prediction
        self.consensus_strength = consensus_strength  # 'perfect', 'strong', 'weak', 'single'
        self.agreement_count = agreement_count
        self.total_traversals = total_traversals
        self.reasoning = reasoning  # Human-readable explanation of decision


class ConsensusResolver:
    """
    Resolves consensus between multiple graph traversals.
    Implements embarrassingly simple voting logic with fallbacks.
    """
    
    def __init__(self, action_similarity_threshold: float = 0.1):
        """
        Initialize consensus resolver.
        
        Args:
            action_similarity_threshold: How similar actions need to be to count as "same"
        """
        self.action_similarity_threshold = action_similarity_threshold
    
    def resolve_consensus(self, traversal_results: List[TraversalResult]) -> ConsensusResult:
        """
        Resolve consensus between multiple traversal results.
        
        Args:
            traversal_results: List of traversal results to find consensus among
            
        Returns:
            ConsensusResult with chosen prediction and reasoning
        """
        # Filter out failed traversals
        valid_results = [r for r in traversal_results if r.prediction is not None]
        
        if not valid_results:
            return self._create_empty_result(len(traversal_results))
        
        if len(valid_results) == 1:
            return self._create_single_result(valid_results[0], len(traversal_results))
        
        # Group similar predictions
        prediction_groups = self._group_similar_predictions(valid_results)
        
        # Find the largest group (most agreement)
        # If groups are same size, pick the one with highest strength
        def group_priority(group):
            group_size = len(group)
            max_strength = max(result.terminal_strength for result in group)
            return (group_size, max_strength)
        
        best_group = max(prediction_groups, key=group_priority)
        agreement_count = len(best_group)
        
        # Determine consensus strength and select best prediction from group
        if len(valid_results) >= 3:
            return self._resolve_triple_consensus(best_group, agreement_count, len(traversal_results))
        else:
            return self._resolve_dual_consensus(best_group, agreement_count, len(traversal_results))
    
    def _group_similar_predictions(self, results: List[TraversalResult]) -> List[List[TraversalResult]]:
        """Group traversal results by similar predictions."""
        groups = []
        
        for result in results:
            # Try to find an existing group this result belongs to
            assigned_to_group = False
            
            for group in groups:
                if self._predictions_similar(result.prediction, group[0].prediction):
                    group.append(result)
                    assigned_to_group = True
                    break
            
            # If no existing group found, create new group
            if not assigned_to_group:
                groups.append([result])
        
        return groups
    
    def _predictions_similar(self, pred1: PredictionPacket, pred2: PredictionPacket) -> bool:
        """Check if two predictions are similar enough to be considered the same."""
        # Compare motor actions (most important)
        if not self._actions_similar(pred1.motor_action, pred2.motor_action):
            return False
        
        # If motor actions are similar, consider them the same prediction
        # (sensory predictions can vary slightly)
        return True
    
    def _actions_similar(self, action1: Dict[str, float], action2: Dict[str, float]) -> bool:
        """Check if two motor actions are similar within threshold."""
        # Check that both have the same keys
        if set(action1.keys()) != set(action2.keys()):
            return False
        
        # Check that all values are within threshold
        for key in action1:
            if abs(action1[key] - action2[key]) > self.action_similarity_threshold:
                return False
        
        return True
    
    def _resolve_triple_consensus(self, best_group: List[TraversalResult], 
                                 agreement_count: int, total_traversals: int) -> ConsensusResult:
        """Resolve consensus for 3+ traversals."""
        chosen_result = self._select_best_from_group(best_group)
        
        # Preserve curiosity-driven consensus type if present
        original_consensus = chosen_result.prediction.consensus_strength
        if original_consensus == "curiosity_driven":
            consensus_strength = "curiosity_driven"
            reasoning = f"Curiosity-driven decision: {agreement_count} out of {total_traversals} traversals"
        elif agreement_count == total_traversals:
            # Perfect consensus - all agree
            consensus_strength = "perfect"
            reasoning = f"All {total_traversals} traversals agreed on the same action"
        elif agreement_count >= 2:
            # Strong consensus - majority agrees
            consensus_strength = "strong"
            reasoning = f"{agreement_count} out of {total_traversals} traversals agreed"
        else:
            # Weak consensus - no clear majority, use strongest traversal
            chosen_result = max(best_group, key=lambda r: r.terminal_strength)
            consensus_strength = "weak"
            reasoning = f"No consensus, chose strongest traversal (strength: {chosen_result.terminal_strength:.2f})"
        
        return ConsensusResult(
            prediction=chosen_result.prediction,
            consensus_strength=consensus_strength,
            agreement_count=agreement_count,
            total_traversals=total_traversals,
            reasoning=reasoning
        )
    
    def _resolve_dual_consensus(self, best_group: List[TraversalResult], 
                               agreement_count: int, total_traversals: int) -> ConsensusResult:
        """Resolve consensus for 2 traversals."""
        chosen_result = self._select_best_from_group(best_group)
        
        # Preserve curiosity-driven consensus type if present
        original_consensus = chosen_result.prediction.consensus_strength
        if original_consensus == "curiosity_driven":
            consensus_strength = "curiosity_driven"
            reasoning = f"Curiosity-driven decision: {agreement_count} out of {total_traversals} traversals"
        elif agreement_count == 2:
            # Strong consensus - both agree
            consensus_strength = "strong"
            reasoning = "Both traversals agreed on the same action"
        else:
            # Weak consensus - choose stronger traversal
            chosen_result = max(best_group, key=lambda r: r.terminal_strength)
            consensus_strength = "weak"
            reasoning = f"No consensus, chose stronger traversal (strength: {chosen_result.terminal_strength:.2f})"
        
        return ConsensusResult(
            prediction=chosen_result.prediction,
            consensus_strength=consensus_strength,
            agreement_count=agreement_count,
            total_traversals=total_traversals,
            reasoning=reasoning
        )
    
    def _select_best_from_group(self, group: List[TraversalResult]) -> TraversalResult:
        """Select the best traversal result from a group of similar predictions."""
        # Select based on terminal node strength and thinking depth
        def selection_score(result):
            strength_score = result.terminal_strength
            depth_score = result.depth_reached * 0.1  # Slight bonus for deeper thinking
            return strength_score + depth_score
        
        return max(group, key=selection_score)
    
    def _create_empty_result(self, total_traversals: int) -> ConsensusResult:
        """Create result for when no valid traversals were found."""
        # Create a default "do nothing" prediction
        from datetime import datetime
        
        default_prediction = PredictionPacket(
            expected_sensory=[],  # Empty - brain doesn't assume sensory length
            motor_action={"forward_motor": 0.0, "turn_motor": 0.0, "brake_motor": 1.0},  # Stop
            confidence=0.0,
            timestamp=datetime.now(),
            sequence_id=0
        )
        default_prediction.consensus_strength = "no_consensus"
        
        return ConsensusResult(
            prediction=default_prediction,
            consensus_strength="no_consensus",
            agreement_count=0,
            total_traversals=total_traversals,
            reasoning="No valid traversals found, defaulting to stop action"
        )
    
    def _create_single_result(self, result: TraversalResult, total_traversals: int) -> ConsensusResult:
        """Create result for when only one valid traversal was found."""
        # Preserve curiosity-driven consensus type if present
        original_consensus = result.prediction.consensus_strength
        if original_consensus == "curiosity_driven":
            consensus_strength = "curiosity_driven"
            reasoning = f"Single curiosity-driven decision (1 out of {total_traversals} traversals)"
        else:
            consensus_strength = "single"
            reasoning = f"Only 1 out of {total_traversals} traversals succeeded"
        
        return ConsensusResult(
            prediction=result.prediction,
            consensus_strength=consensus_strength,
            agreement_count=1,
            total_traversals=total_traversals,
            reasoning=reasoning
        )
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get statistics about consensus configuration."""
        return {
            "action_similarity_threshold": self.action_similarity_threshold,
            "consensus_type": "majority_voting"
        }