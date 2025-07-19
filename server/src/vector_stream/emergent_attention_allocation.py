#!/usr/bin/env python3
"""
Emergent Attention Allocation System

Implements constraint-based attention allocation where streams compete for limited
attention resources based on their current state, urgency, and processing needs.

Key principles:
- Attention emerges from resource competition, not explicit programming
- Streams bid for attention based on activation strength and constraint pressures
- Winner-take-all dynamics with biological realism
- Dynamic allocation based on real-time system state
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from .stream_types import StreamType, ConstraintType


@dataclass
class AttentionBid:
    """A bid for attention resources from a stream."""
    stream_type: StreamType
    base_activation: float       # Current activation strength (0.0-1.0)
    urgency_pressure: float      # Urgency constraint pressure (0.0-1.0+)
    processing_load: float       # Current processing load (0.0-1.0+)
    resource_need: float         # How much attention needed (0.0-1.0)
    bid_strength: float          # Final computed bid strength
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttentionAllocation:
    """Result of attention allocation for a stream."""
    stream_type: StreamType
    allocated_attention: float   # Amount of attention allocated (0.0-1.0)
    allocation_ratio: float      # Fraction of requested amount (0.0-1.0)
    bid_rank: int               # Ranking in competition (1 = highest)
    competition_pressure: float  # How competitive this allocation was (0.0-1.0)


class AttentionCompetitionMode(Enum):
    """Different modes of attention competition."""
    WINNER_TAKE_ALL = "winner_take_all"          # One stream gets most attention
    PROPORTIONAL = "proportional"                # Attention divided by bid strength
    THRESHOLD_GATING = "threshold_gating"        # Only bids above threshold get attention
    BIOLOGICAL_INHIBITION = "biological_inhibition"  # Winner inhibits others


class EmergentAttentionAllocator:
    """
    Manages emergent attention allocation through competitive resource dynamics.
    
    Implements biologically-inspired attention competition where streams naturally
    compete for limited attention resources based on their processing needs.
    """
    
    def __init__(self, total_attention_budget: float = 1.0, 
                 competition_mode: AttentionCompetitionMode = AttentionCompetitionMode.BIOLOGICAL_INHIBITION,
                 quiet_mode: bool = False):
        """
        Initialize emergent attention allocator.
        
        Args:
            total_attention_budget: Total attention available per cycle
            competition_mode: How streams compete for attention
            quiet_mode: Suppress debug output
        """
        self.total_attention_budget = total_attention_budget
        self.competition_mode = competition_mode
        self.quiet_mode = quiet_mode
        
        # Attention allocation history
        self.allocation_history = []
        self.competition_stats = {
            'total_competitions': 0,
            'winner_changes': 0,
            'avg_competition_pressure': 0.0,
            'attention_distribution_entropy': 0.0
        }
        
        # Biological parameters
        self.inhibition_strength = 0.7      # How much winner inhibits others
        self.attention_persistence = 0.8    # How much previous allocation affects current
        self.activation_threshold = 0.3     # Minimum activation to compete
        self.urgency_boost_factor = 0.5     # How much urgency increases bid strength
        
        # Current allocations (persists across cycles for momentum)
        self.current_allocations = {}
        self.last_winner = None
        
        if not quiet_mode:
            print(f"🧠 EmergentAttentionAllocator initialized")
            print(f"   Budget: {total_attention_budget:.2f}")
            print(f"   Mode: {competition_mode.value}")
            print(f"   Inhibition: {self.inhibition_strength:.2f}")
    
    def compute_attention_bids(self, stream_states: Dict[StreamType, Any], 
                             constraint_pressures: Dict[StreamType, Dict[ConstraintType, float]]) -> List[AttentionBid]:
        """
        Compute attention bids from all streams based on their current state.
        
        Args:
            stream_states: Current state of each stream
            constraint_pressures: Constraint pressures affecting each stream
            
        Returns:
            List of attention bids from competing streams
        """
        bids = []
        
        for stream_type in StreamType:
            if stream_type not in stream_states:
                continue
            
            stream_state = stream_states[stream_type]
            constraints = constraint_pressures.get(stream_type, {})
            
            # Extract stream activation and processing state
            base_activation = getattr(stream_state, 'activation_strength', 0.0)
            processing_phase = getattr(stream_state, 'processing_phase', 'inactive')
            active_patterns = getattr(stream_state, 'active_patterns', [])
            
            # Skip streams below activation threshold
            if base_activation < self.activation_threshold:
                continue
            
            # Calculate constraint-based modifiers
            urgency_pressure = constraints.get(ConstraintType.URGENCY_SIGNAL, 0.0)
            processing_load = constraints.get(ConstraintType.PROCESSING_LOAD, 0.0)
            resource_scarcity = constraints.get(ConstraintType.RESOURCE_SCARCITY, 0.0)
            
            # Compute resource need based on processing state
            resource_need = self._compute_resource_need(
                base_activation, processing_phase, len(active_patterns), constraints
            )
            
            # Calculate bid strength through competitive dynamics
            bid_strength = self._compute_bid_strength(
                base_activation, urgency_pressure, processing_load, 
                resource_scarcity, resource_need, stream_type
            )
            
            # Create attention bid
            bid = AttentionBid(
                stream_type=stream_type,
                base_activation=base_activation,
                urgency_pressure=urgency_pressure,
                processing_load=processing_load,
                resource_need=resource_need,
                bid_strength=bid_strength,
                metadata={
                    'processing_phase': processing_phase,
                    'pattern_count': len(active_patterns),
                    'constraint_pressures': constraints
                }
            )
            
            bids.append(bid)
        
        return bids
    
    def _compute_resource_need(self, base_activation: float, processing_phase: str, 
                             pattern_count: int, constraints: Dict[ConstraintType, float]) -> float:
        """Compute how much attention this stream actually needs."""
        # Base need from activation level
        base_need = base_activation * 0.5
        
        # Higher need during active processing
        if processing_phase == "active":
            base_need *= 1.5
        elif processing_phase == "completed":
            base_need *= 0.5
        
        # More patterns = more attention needed
        pattern_factor = min(1.0, pattern_count / 10.0)  # Saturate at 10 patterns
        base_need += pattern_factor * 0.3
        
        # Constraint-based need modulation
        interference = constraints.get(ConstraintType.INTERFERENCE, 0.0)
        coherence_pressure = constraints.get(ConstraintType.COHERENCE_PRESSURE, 0.0)
        
        # Interference increases attention need
        base_need += interference * 0.4
        
        # Coherence pressure increases attention need
        base_need += coherence_pressure * 0.3
        
        return min(1.0, base_need)
    
    def _compute_bid_strength(self, base_activation: float, urgency_pressure: float,
                            processing_load: float, resource_scarcity: float,
                            resource_need: float, stream_type: StreamType) -> float:
        """Compute competitive bid strength for attention resources."""
        # Start with base activation
        bid_strength = base_activation
        
        # Urgency significantly boosts bid strength
        bid_strength += urgency_pressure * self.urgency_boost_factor
        
        # High processing load can either increase or decrease bid strength
        # - Light load: stream is efficient, deserves more attention
        # - Heavy load: stream is struggling, needs attention
        # - Extreme load: stream is overwhelmed, should get less attention
        if processing_load < 0.3:
            bid_strength += (0.3 - processing_load) * 0.2  # Efficiency bonus
        elif processing_load > 0.8:
            bid_strength -= (processing_load - 0.8) * 0.5  # Overload penalty
        else:
            bid_strength += processing_load * 0.1  # Moderate load bonus
        
        # Resource scarcity increases bid (survival mode)
        bid_strength += resource_scarcity * 0.3
        
        # Resource need directly affects bid strength
        bid_strength *= (0.5 + 0.5 * resource_need)
        
        # Stream-specific modulations
        stream_modifiers = {
            StreamType.SENSORY: 1.1,      # Sensory gets slight priority (external world)
            StreamType.MOTOR: 1.0,        # Motor baseline
            StreamType.TEMPORAL: 0.9,     # Temporal slightly lower (internal)
            StreamType.CONFIDENCE: 0.8,   # Confidence lower priority
            StreamType.ATTENTION: 0.7     # Attention doesn't compete with itself
        }
        
        bid_strength *= stream_modifiers.get(stream_type, 1.0)
        
        # Attention persistence - previous winner gets momentum
        if stream_type == self.last_winner:
            bid_strength *= (1.0 + self.attention_persistence * 0.2)
        
        return max(0.0, bid_strength)
    
    def allocate_attention(self, bids: List[AttentionBid]) -> List[AttentionAllocation]:
        """
        Allocate attention among competing streams based on their bids.
        
        Args:
            bids: List of attention bids from streams
            
        Returns:
            List of attention allocations
        """
        if not bids:
            return []
        
        # Sort bids by strength (highest first)
        sorted_bids = sorted(bids, key=lambda b: b.bid_strength, reverse=True)
        
        # Apply competition mode
        allocations = self._apply_competition_mode(sorted_bids)
        
        # Update statistics
        self._update_competition_stats(allocations)
        
        # Track allocation history
        self.allocation_history.append({
            'timestamp': time.time(),
            'allocations': allocations,
            'total_bids': len(bids),
            'winner': allocations[0].stream_type if allocations else None
        })
        
        # Update current state
        self.current_allocations = {alloc.stream_type: alloc.allocated_attention for alloc in allocations}
        if allocations:
            self.last_winner = allocations[0].stream_type
        
        if not self.quiet_mode and allocations:
            winner = allocations[0]
            print(f"🧠 Attention winner: {winner.stream_type.value} "
                  f"(allocated: {winner.allocated_attention:.3f}, "
                  f"bid: {sorted_bids[0].bid_strength:.3f})")
        
        return allocations
    
    def _apply_competition_mode(self, sorted_bids: List[AttentionBid]) -> List[AttentionAllocation]:
        """Apply the configured competition mode to determine allocations."""
        if self.competition_mode == AttentionCompetitionMode.WINNER_TAKE_ALL:
            return self._winner_take_all_allocation(sorted_bids)
        elif self.competition_mode == AttentionCompetitionMode.PROPORTIONAL:
            return self._proportional_allocation(sorted_bids)
        elif self.competition_mode == AttentionCompetitionMode.THRESHOLD_GATING:
            return self._threshold_gating_allocation(sorted_bids)
        elif self.competition_mode == AttentionCompetitionMode.BIOLOGICAL_INHIBITION:
            return self._biological_inhibition_allocation(sorted_bids)
        else:
            return self._proportional_allocation(sorted_bids)  # Default
    
    def _biological_inhibition_allocation(self, sorted_bids: List[AttentionBid]) -> List[AttentionAllocation]:
        """
        Biological attention allocation with winner inhibition.
        
        Winner gets most attention and inhibits others based on strength difference.
        """
        if not sorted_bids:
            return []
        
        allocations = []
        remaining_budget = self.total_attention_budget
        
        # Winner gets primary allocation
        winner = sorted_bids[0]
        winner_allocation = min(winner.resource_need, remaining_budget * 0.8)  # Up to 80% of budget
        remaining_budget -= winner_allocation
        
        # Calculate inhibition effect
        winner_strength = winner.bid_strength
        
        allocations.append(AttentionAllocation(
            stream_type=winner.stream_type,
            allocated_attention=winner_allocation,
            allocation_ratio=winner_allocation / winner.resource_need if winner.resource_need > 0 else 1.0,
            bid_rank=1,
            competition_pressure=1.0
        ))
        
        # Other streams get inhibited allocations
        for i, bid in enumerate(sorted_bids[1:], 2):
            if remaining_budget <= 0:
                break
            
            # Calculate inhibition based on strength difference
            strength_ratio = bid.bid_strength / winner_strength if winner_strength > 0 else 0.0
            inhibition_factor = 1.0 - (self.inhibition_strength * (1.0 - strength_ratio))
            inhibition_factor = max(0.1, inhibition_factor)  # Minimum 10% gets through
            
            # Allocate remaining budget proportionally with inhibition
            max_allocation = remaining_budget / (len(sorted_bids) - i + 1)
            allocation = min(bid.resource_need * inhibition_factor, max_allocation)
            
            remaining_budget -= allocation
            
            competition_pressure = bid.bid_strength / winner_strength if winner_strength > 0 else 0.0
            
            allocations.append(AttentionAllocation(
                stream_type=bid.stream_type,
                allocated_attention=allocation,
                allocation_ratio=allocation / bid.resource_need if bid.resource_need > 0 else 1.0,
                bid_rank=i,
                competition_pressure=competition_pressure
            ))
        
        return allocations
    
    def _proportional_allocation(self, sorted_bids: List[AttentionBid]) -> List[AttentionAllocation]:
        """Allocate attention proportionally to bid strengths."""
        if not sorted_bids:
            return []
        
        total_bid_strength = sum(bid.bid_strength for bid in sorted_bids)
        if total_bid_strength == 0:
            return []
        
        allocations = []
        
        for i, bid in enumerate(sorted_bids):
            proportion = bid.bid_strength / total_bid_strength
            allocation = min(bid.resource_need, self.total_attention_budget * proportion)
            
            allocations.append(AttentionAllocation(
                stream_type=bid.stream_type,
                allocated_attention=allocation,
                allocation_ratio=allocation / bid.resource_need if bid.resource_need > 0 else 1.0,
                bid_rank=i + 1,
                competition_pressure=proportion
            ))
        
        return allocations
    
    def _winner_take_all_allocation(self, sorted_bids: List[AttentionBid]) -> List[AttentionAllocation]:
        """Winner takes all available attention."""
        if not sorted_bids:
            return []
        
        winner = sorted_bids[0]
        allocation = min(winner.resource_need, self.total_attention_budget)
        
        allocations = [AttentionAllocation(
            stream_type=winner.stream_type,
            allocated_attention=allocation,
            allocation_ratio=allocation / winner.resource_need if winner.resource_need > 0 else 1.0,
            bid_rank=1,
            competition_pressure=1.0
        )]
        
        # Others get nothing
        for i, bid in enumerate(sorted_bids[1:], 2):
            allocations.append(AttentionAllocation(
                stream_type=bid.stream_type,
                allocated_attention=0.0,
                allocation_ratio=0.0,
                bid_rank=i,
                competition_pressure=0.0
            ))
        
        return allocations
    
    def _threshold_gating_allocation(self, sorted_bids: List[AttentionBid]) -> List[AttentionAllocation]:
        """Only bids above threshold get attention."""
        if not sorted_bids:
            return []
        
        threshold = 0.6  # Minimum bid strength to get attention
        qualified_bids = [bid for bid in sorted_bids if bid.bid_strength >= threshold]
        
        if not qualified_bids:
            # If no one qualifies, give winner minimal attention
            winner = sorted_bids[0]
            return [AttentionAllocation(
                stream_type=winner.stream_type,
                allocated_attention=0.1,
                allocation_ratio=0.1 / winner.resource_need if winner.resource_need > 0 else 1.0,
                bid_rank=1,
                competition_pressure=0.1
            )]
        
        # Allocate proportionally among qualified bids
        return self._proportional_allocation(qualified_bids)
    
    def _update_competition_stats(self, allocations: List[AttentionAllocation]):
        """Update competition statistics."""
        if not allocations:
            return
        
        self.competition_stats['total_competitions'] += 1
        
        # Check for winner changes
        current_winner = allocations[0].stream_type
        if self.last_winner and current_winner != self.last_winner:
            self.competition_stats['winner_changes'] += 1
        
        # Calculate average competition pressure
        avg_pressure = np.mean([alloc.competition_pressure for alloc in allocations])
        total_comps = self.competition_stats['total_competitions']
        prev_avg = self.competition_stats['avg_competition_pressure']
        self.competition_stats['avg_competition_pressure'] = (
            (prev_avg * (total_comps - 1) + avg_pressure) / total_comps
        )
        
        # Calculate attention distribution entropy (measure of concentration)
        attention_dist = [alloc.allocated_attention for alloc in allocations]
        total_attention = sum(attention_dist)
        if total_attention > 0:
            probs = [a / total_attention for a in attention_dist if a > 0]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            prev_entropy = self.competition_stats['attention_distribution_entropy']
            self.competition_stats['attention_distribution_entropy'] = (
                (prev_entropy * (total_comps - 1) + entropy) / total_comps
            )
    
    def get_current_allocations(self) -> Dict[StreamType, float]:
        """Get current attention allocations for each stream."""
        return self.current_allocations.copy()
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """Get competition statistics."""
        stats = self.competition_stats.copy()
        
        # Add derived metrics
        if stats['total_competitions'] > 0:
            stats['winner_stability'] = 1.0 - (stats['winner_changes'] / stats['total_competitions'])
        else:
            stats['winner_stability'] = 1.0
        
        stats['attention_concentration'] = 1.0 / (1.0 + stats['attention_distribution_entropy'])
        
        # Recent allocation summary
        if self.allocation_history:
            recent_winners = [entry['winner'] for entry in self.allocation_history[-10:] if entry['winner']]
            if recent_winners:
                winner_counts = {}
                for winner in recent_winners:
                    winner_counts[winner] = winner_counts.get(winner, 0) + 1
                stats['recent_winner_distribution'] = {w.value: c for w, c in winner_counts.items()}
        
        return stats


# Factory function
def create_attention_allocator(budget: float = 1.0, 
                             mode: AttentionCompetitionMode = AttentionCompetitionMode.BIOLOGICAL_INHIBITION,
                             quiet_mode: bool = False) -> EmergentAttentionAllocator:
    """Create an emergent attention allocator with specified parameters."""
    return EmergentAttentionAllocator(budget, mode, quiet_mode)


if __name__ == "__main__":
    # Example usage
    print("🧠 Testing Emergent Attention Allocation")
    
    allocator = create_attention_allocator(budget=1.0, quiet_mode=False)
    
    # Mock stream states and constraints for testing
    from .stream_types import StreamType, ConstraintType
    
    class MockStreamState:
        def __init__(self, activation_strength=0.5, processing_phase="active", active_patterns=None):
            self.activation_strength = activation_strength
            self.processing_phase = processing_phase
            self.active_patterns = active_patterns or [1, 2, 3]
    
    stream_states = {
        StreamType.SENSORY: MockStreamState(0.8, "active", [1, 2, 3, 4]),
        StreamType.MOTOR: MockStreamState(0.6, "completed", [1, 2]),
        StreamType.TEMPORAL: MockStreamState(0.4, "active", [1])
    }
    
    constraint_pressures = {
        StreamType.SENSORY: {
            ConstraintType.URGENCY_SIGNAL: 0.7,
            ConstraintType.PROCESSING_LOAD: 0.3
        },
        StreamType.MOTOR: {
            ConstraintType.PROCESSING_LOAD: 0.8,
            ConstraintType.RESOURCE_SCARCITY: 0.4
        },
        StreamType.TEMPORAL: {
            ConstraintType.INTERFERENCE: 0.5
        }
    }
    
    # Compute bids and allocate attention
    bids = allocator.compute_attention_bids(stream_states, constraint_pressures)
    allocations = allocator.allocate_attention(bids)
    
    print(f"\n📊 Attention Allocation Results:")
    for allocation in allocations:
        print(f"  {allocation.stream_type.value}: {allocation.allocated_attention:.3f} "
              f"(rank: {allocation.bid_rank}, pressure: {allocation.competition_pressure:.3f})")
    
    print(f"\n📈 Competition Stats:")
    stats = allocator.get_competition_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")