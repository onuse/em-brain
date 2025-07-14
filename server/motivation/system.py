"""
Motivation System - Competitive Action Selection

Orchestrates competition between motivations to select actions.
"""

import time
from typing import List, Dict, Any, Optional
from .base import BaseMotivation, ActionProposal


class MotivationSystem:
    """
    Orchestrates competitive action selection between motivations.
    
    The core brain handles pattern recognition and prediction.
    This system handles which predictions to pursue.
    """
    
    def __init__(self, brain):
        """
        Initialize motivation system.
        
        Args:
            brain: The 4-system minimal brain (provides prediction services)
        """
        self.brain = brain
        self.motivations: List[BaseMotivation] = []
        
        # Decision history for analysis
        self.decision_history = []
        self.competition_log = []
        
        # Performance tracking
        self.total_decisions = 0
        self.decision_times = []
        
    def add_motivation(self, motivation: BaseMotivation):
        """
        Add motivation to the competitive system.
        
        Args:
            motivation: Motivation instance to add
        """
        motivation.connect_to_brain(self.brain)
        self.motivations.append(motivation)
        print(f"🧠 Added motivation: {motivation}")
    
    def remove_motivation(self, motivation_name: str):
        """Remove motivation by name."""
        self.motivations = [m for m in self.motivations if m.name != motivation_name]
        print(f"🧠 Removed motivation: {motivation_name}")
    
    def select_action(self, current_state: Any) -> Any:
        """
        Select action through motivation competition.
        
        Args:
            current_state: Current sensory/robot state
            
        Returns:
            Winning action to execute
        """
        start_time = time.time()
        
        if not self.motivations:
            # No motivations - brain can't decide
            print("⚠️ No motivations available - cannot select action")
            return None
        
        # Each motivation proposes an action
        proposals = []
        for motivation in self.motivations:
            try:
                proposal = motivation.propose_action(current_state)
                proposal.motivation_value *= motivation.weight  # Apply global weight
                motivation._record_proposal(proposal)
                proposals.append(proposal)
            except Exception as e:
                print(f"❌ Error from {motivation.name}: {e}")
                continue
        
        if not proposals:
            print("❌ No valid proposals generated")
            return None
        
        # Competition: highest combined score wins
        winning_proposal = max(proposals, key=lambda p: p.combined_score)
        
        # Record winner
        for motivation in self.motivations:
            if motivation.name == winning_proposal.motivation_name:
                motivation._record_win()
                break
        
        # Log competition results
        self._log_competition(current_state, proposals, winning_proposal)
        
        # Track performance
        decision_time = time.time() - start_time
        self.decision_times.append(decision_time)
        self.total_decisions += 1
        
        return winning_proposal.action
    
    def _log_competition(self, state: Any, proposals: List[ActionProposal], winner: ActionProposal):
        """Log competition results for analysis."""
        
        competition_record = {
            'timestamp': time.time(),
            'state_summary': self._summarize_state(state),
            'proposals': [
                {
                    'motivation': p.motivation_name,
                    'action': str(p.action),
                    'value': p.motivation_value,
                    'confidence': p.confidence,
                    'score': p.combined_score,
                    'reasoning': p.reasoning
                }
                for p in proposals
            ],
            'winner': {
                'motivation': winner.motivation_name,
                'action': str(winner.action),
                'score': winner.combined_score,
                'reasoning': winner.reasoning
            }
        }
        
        self.competition_log.append(competition_record)
        
        # Keep log manageable
        if len(self.competition_log) > 1000:
            self.competition_log = self.competition_log[-500:]
        
        # Print competition summary (optional verbose mode)
        if hasattr(self, 'verbose') and self.verbose:
            self._print_competition_summary(proposals, winner)
    
    def _print_competition_summary(self, proposals: List[ActionProposal], winner: ActionProposal):
        """Print detailed competition results."""
        
        print(f"\n🏆 MOTIVATION COMPETITION")
        print(f"   Proposals: {len(proposals)}")
        
        for proposal in sorted(proposals, key=lambda p: p.combined_score, reverse=True):
            status = "🥇 WINNER" if proposal == winner else f"   Score: {proposal.combined_score:.3f}"
            print(f"   {proposal.motivation_name:12} {status:15} - {proposal.reasoning}")
        
        print(f"   → Action: {winner.action}")
    
    def _summarize_state(self, state: Any) -> str:
        """Create brief state summary for logging."""
        # This would be customized based on state format
        if hasattr(state, '__dict__'):
            return f"State({len(state.__dict__)} fields)"
        elif isinstance(state, dict):
            return f"State({len(state)} keys)"
        else:
            return f"State({type(state).__name__})"
    
    def get_motivation_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for all motivations."""
        
        stats = {
            'total_decisions': self.total_decisions,
            'average_decision_time': sum(self.decision_times) / max(1, len(self.decision_times)),
            'motivations': [m.get_statistics() for m in self.motivations],
            'competition_history_size': len(self.competition_log)
        }
        
        return stats
    
    def print_motivation_report(self):
        """Print comprehensive motivation system report."""
        
        stats = self.get_motivation_statistics()
        
        print(f"\n📊 MOTIVATION SYSTEM REPORT")
        print(f"   Total decisions: {stats['total_decisions']}")
        print(f"   Avg decision time: {stats['average_decision_time']*1000:.1f}ms")
        print(f"   Active motivations: {len(self.motivations)}")
        
        print(f"\n🏆 MOTIVATION PERFORMANCE:")
        for motivation_stats in stats['motivations']:
            name = motivation_stats['name']
            win_rate = motivation_stats['win_rate']
            proposals = motivation_stats['proposals_made']
            print(f"   {name:15} {win_rate:6.1%} win rate ({proposals} proposals)")
        
        # Show recent competition pattern
        if len(self.competition_log) >= 5:
            print(f"\n🔄 RECENT DECISIONS:")
            for record in self.competition_log[-5:]:
                winner = record['winner']
                print(f"   {winner['motivation']:12} → {winner['action']}")
    
    def set_verbose(self, verbose: bool):
        """Enable/disable verbose competition logging."""
        self.verbose = verbose
        print(f"🔊 Motivation competition {'verbose' if verbose else 'quiet'} mode")
    
    def clear_history(self):
        """Clear decision and competition history."""
        self.decision_history.clear()
        self.competition_log.clear()
        self.decision_times.clear()
        self.total_decisions = 0
        
        for motivation in self.motivations:
            motivation.proposals_made = 0
            motivation.proposals_won = 0
            motivation.total_value_generated = 0.0
        
        print("🧹 Motivation system history cleared")