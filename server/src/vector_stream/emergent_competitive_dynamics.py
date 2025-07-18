#!/usr/bin/env python3
"""
Emergent Competitive Dynamics Through Resource Constraints

Instead of explicit winner-take-all algorithms, competitive behaviors emerge from:
1. Storage capacity constraints (limited pattern slots force competition)
2. Computational resource constraints (processing budget creates selection pressure)
3. Activation energy constraints (patterns need sufficient energy to remain active)
4. Access frequency constraints (unused patterns get evicted naturally)

This mimics biological resource competition where neurons compete for:
- Metabolic resources (glucose, oxygen)
- Growth factors and connectivity
- Synaptic strength and maintenance
- Attention and processing bandwidth

The key insight: Competition emerges from resource scarcity, not explicit algorithms.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque, defaultdict
import heapq

try:
    from .sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage
except ImportError:
    from sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage


@dataclass
class ResourceBudget:
    """Resource budget constraints that create competitive pressure."""
    name: str
    max_active_patterns: int     # Maximum patterns that can be active simultaneously
    max_storage_slots: int       # Maximum total patterns in storage
    activation_threshold: float  # Minimum activation energy to remain active
    decay_rate: float           # How quickly unused patterns lose energy
    competition_pressure: float  # How aggressively patterns compete (0-1)


@dataclass 
class PatternResource:
    """Resource allocation for individual patterns."""
    pattern_id: str
    activation_energy: float     # Current activation level
    access_frequency: float      # How often this pattern is accessed
    last_access_time: float     # When pattern was last accessed
    storage_priority: float     # Priority for remaining in storage
    prediction_confidence: float = 0.5  # How predictable this pattern is (0.0-1.0)
    active: bool = False        # Whether pattern is currently active


class ResourceConstrainedStorage:
    """
    Pattern storage with resource constraints that create competitive dynamics.
    
    Competition emerges from:
    - Limited storage slots (patterns compete for persistence)
    - Limited active slots (patterns compete for current activation)
    - Energy decay (unused patterns lose resources)
    - Access-based priorities (frequently used patterns get advantages)
    """
    
    def __init__(self, budget: ResourceBudget, pattern_storage: SparsePatternStorage, quiet_mode: bool = False):
        self.budget = budget
        self.storage = pattern_storage
        self.quiet_mode = quiet_mode
        
        # Resource tracking for each pattern
        self.pattern_resources: Dict[str, PatternResource] = {}
        
        # Currently active patterns (limited by budget)
        self.active_patterns: Set[str] = set()
        
        # Competition tracking
        self.competition_events = deque(maxlen=1000)
        self.eviction_history = deque(maxlen=100)
        
        # Resource allocation statistics
        self.total_competitions = 0
        self.total_evictions = 0
        self.resource_pressure = 0.0  # 0-1, how much resource competition is occurring
        
        if not quiet_mode:
            print(f"🏆 Resource-constrained storage initialized: {budget.name}")
            print(f"   Max active: {budget.max_active_patterns}, Max storage: {budget.max_storage_slots}")
            print(f"   Competition pressure: {budget.competition_pressure:.2f}")
    
    def store_pattern_with_competition(self, pattern: SparsePattern, current_time: float) -> bool:
        """
        Store pattern with resource competition - may evict existing patterns.
        
        Returns True if pattern was successfully stored (may have evicted others).
        """
        pattern_id = pattern.pattern_id
        
        # Check if storage is at capacity
        if len(self.pattern_resources) >= self.budget.max_storage_slots:
            # Resource competition: decide if new pattern should evict existing one
            if not self._compete_for_storage_slot(pattern, current_time):
                self.competition_events.append({
                    'type': 'storage_rejected',
                    'pattern_id': pattern_id,
                    'time': current_time,
                    'reason': 'insufficient_priority'
                })
                return False
        
        # Store the pattern in underlying storage
        try:
            storage_index = self.storage.store_pattern(pattern)
        except Exception:
            # Storage failed (maybe internal limits reached)
            return False
        
        # Initialize resource allocation for this pattern
        initial_energy = 1.0  # New patterns start with full energy
        self.pattern_resources[pattern_id] = PatternResource(
            pattern_id=pattern_id,
            activation_energy=initial_energy,
            access_frequency=1.0,
            last_access_time=current_time,
            storage_priority=initial_energy,
            active=False
        )
        
        if not self.quiet_mode and len(self.pattern_resources) % 100 == 0:
            print(f"   Stored {len(self.pattern_resources)} patterns (competition pressure: {self.resource_pressure:.2f})")
        
        return True
    
    def activate_pattern_with_competition(self, pattern_id: str, current_time: float) -> bool:
        """
        Activate pattern with resource competition - may deactivate others.
        
        Returns True if pattern was successfully activated.
        """
        if pattern_id not in self.pattern_resources:
            return False
        
        # Check if activation budget is exceeded
        if len(self.active_patterns) >= self.budget.max_active_patterns:
            # Resource competition: decide which patterns remain active
            if not self._compete_for_activation_slot(pattern_id, current_time):
                self.competition_events.append({
                    'type': 'activation_rejected',
                    'pattern_id': pattern_id,
                    'time': current_time,
                    'reason': 'insufficient_activation_energy'
                })
                return False
        
        # Activate the pattern
        self.active_patterns.add(pattern_id)
        resource = self.pattern_resources[pattern_id]
        resource.active = True
        resource.last_access_time = current_time
        resource.access_frequency += 1.0
        
        # Boost activation energy for accessed patterns
        resource.activation_energy = min(1.0, resource.activation_energy + 0.1)
        
        self.total_competitions += 1
        return True
    
    def update_pattern_confidence(self, pattern_id: str, prediction_confidence: float):
        """
        Update prediction confidence for a pattern.
        
        This creates the dual motivation system:
        - High confidence (>0.9) creates "boredom cost" (restlessness)
        - Low confidence creates competitive advantage (anxiety for accuracy)
        """
        if pattern_id in self.pattern_resources:
            # Smooth update to prevent oscillation
            current = self.pattern_resources[pattern_id].prediction_confidence
            # Exponential moving average with 0.1 learning rate
            self.pattern_resources[pattern_id].prediction_confidence = \
                0.9 * current + 0.1 * prediction_confidence
    
    def _compete_for_storage_slot(self, new_pattern: SparsePattern, current_time: float) -> bool:
        """
        Compete for storage slot - may evict weakest existing pattern.
        
        Returns True if new pattern wins and gets stored.
        """
        if not self.pattern_resources:
            return True
        
        # Find pattern with lowest storage priority
        weakest_pattern_id = min(
            self.pattern_resources.keys(),
            key=lambda pid: self._calculate_storage_priority(pid, current_time)
        )
        
        weakest_priority = self._calculate_storage_priority(weakest_pattern_id, current_time)
        new_pattern_priority = 1.0  # New patterns start with high priority
        
        # Competition pressure determines how aggressive competition is
        competition_threshold = 1.0 - self.budget.competition_pressure
        
        if new_pattern_priority > weakest_priority + competition_threshold:
            # New pattern wins - evict the weakest
            self._evict_pattern(weakest_pattern_id, 'resource_competition', current_time)
            return True
        else:
            # Existing pattern keeps its slot
            return False
    
    def _compete_for_activation_slot(self, pattern_id: str, current_time: float) -> bool:
        """
        Compete for activation slot - may deactivate weakest active pattern.
        
        Returns True if pattern wins activation slot.
        """
        if len(self.active_patterns) < self.budget.max_active_patterns:
            return True
        
        # Find weakest active pattern
        active_priorities = {
            pid: self._calculate_activation_priority(pid, current_time) 
            for pid in self.active_patterns
        }
        
        weakest_active_id = min(active_priorities.keys(), key=lambda pid: active_priorities[pid])
        weakest_priority = active_priorities[weakest_active_id]
        
        new_pattern_priority = self._calculate_activation_priority(pattern_id, current_time)
        
        if new_pattern_priority > weakest_priority:
            # New pattern wins - deactivate weakest
            self._deactivate_pattern(weakest_active_id, 'resource_competition', current_time)
            return True
        else:
            return False
    
    def _calculate_storage_priority(self, pattern_id: str, current_time: float) -> float:
        """Calculate storage priority for pattern (higher = more likely to persist)."""
        if pattern_id not in self.pattern_resources:
            return 0.0
        
        resource = self.pattern_resources[pattern_id]
        
        # Age penalty (older patterns get lower priority unless frequently accessed)
        age = current_time - resource.last_access_time
        age_penalty = np.exp(-age * self.budget.decay_rate)
        
        # Frequency bonus (frequently accessed patterns get higher priority)
        frequency_bonus = np.log1p(resource.access_frequency) / 10.0
        
        # Activation energy (current energy level)
        energy_bonus = resource.activation_energy
        
        # DUAL MOTIVATION: Boredom cost based on pattern sophistication
        # Smart brains tolerate higher confidence before becoming restless
        sophistication_threshold = self._calculate_sophistication_threshold()
        boredom_cost = 0.0
        if resource.prediction_confidence > sophistication_threshold:
            # Excess confidence creates restlessness pressure
            excess_confidence = resource.prediction_confidence - sophistication_threshold
            boredom_cost = (excess_confidence * 10.0) ** 2  # Quadratic scaling
        
        base_priority = (0.4 * age_penalty + 0.4 * frequency_bonus + 0.2 * energy_bonus)
        
        # Subtract boredom cost - highly predictable patterns lose priority
        priority = base_priority - boredom_cost
        return max(0.0, priority)  # Ensure non-negative
    
    def _calculate_activation_priority(self, pattern_id: str, current_time: float) -> float:
        """Calculate activation priority for pattern (higher = more likely to be active)."""
        if pattern_id not in self.pattern_resources:
            return 0.0
        
        resource = self.pattern_resources[pattern_id]
        
        # Recent access bonus
        recency = current_time - resource.last_access_time
        recency_bonus = np.exp(-recency / 5.0)  # 5 second half-life
        
        # Current activation energy
        energy = resource.activation_energy
        
        # Frequency of recent accesses
        frequency = resource.access_frequency
        
        # DUAL MOTIVATION: Boredom cost based on pattern sophistication
        # Restlessness emerges when patterns exceed sophistication-appropriate confidence
        sophistication_threshold = self._calculate_sophistication_threshold()
        boredom_cost = 0.0
        if resource.prediction_confidence > sophistication_threshold:
            excess_confidence = resource.prediction_confidence - sophistication_threshold
            boredom_cost = (excess_confidence * 5.0) ** 2  # Lighter cost for activation than storage
        
        base_priority = (0.5 * recency_bonus + 0.3 * energy + 0.2 * np.log1p(frequency) / 10.0)
        
        # Subtract boredom cost - brain becomes "restless" with predictable patterns
        priority = base_priority - boredom_cost
        return max(0.0, priority)  # Ensure non-negative
    
    def _calculate_sophistication_threshold(self) -> float:
        """
        Calculate confidence threshold based on pattern sophistication.
        
        Elegant principle: Smarter brains tolerate higher confidence before restlessness.
        - Simple patterns (child brain): Restless at low confidence → constant exploration
        - Complex patterns (mature brain): Only restless when genuinely stagnating
        """
        if not self.pattern_resources:
            return 0.3  # Child brain: restless at 30% confidence
        
        # Pattern sophistication metrics
        total_patterns = len(self.pattern_resources)
        avg_access_frequency = np.mean([r.access_frequency for r in self.pattern_resources.values()])
        competition_density = self.total_competitions / max(1, total_patterns)
        
        # Sophistication: How complex/interconnected the pattern space is
        pattern_density = min(1.0, np.log10(max(1, total_patterns)) / 5.0)  # 0-1
        interaction_complexity = min(1.0, avg_access_frequency / 10.0)        # 0-1  
        competitive_maturity = min(1.0, competition_density / 5.0)            # 0-1
        
        sophistication = (pattern_density + interaction_complexity + competitive_maturity) / 3.0
        
        # Threshold scaling: 
        # Naive brain (0.0 sophistication): 30% confidence threshold → constant exploration
        # Sophisticated brain (1.0 sophistication): 90% confidence threshold → conservative but anti-stagnation
        min_threshold = 0.3  # Child brain
        max_threshold = 0.9  # Mature brain
        
        return min_threshold + sophistication * (max_threshold - min_threshold)
    
    def _evict_pattern(self, pattern_id: str, reason: str, current_time: float):
        """Evict pattern from storage due to resource competition."""
        if pattern_id in self.pattern_resources:
            # Remove from active patterns if active
            self.active_patterns.discard(pattern_id)
            
            # Remove from storage
            if pattern_id in self.storage.pattern_index:
                del self.storage.pattern_index[pattern_id]
            
            # Remove resource tracking
            del self.pattern_resources[pattern_id]
            
            # Record eviction
            self.eviction_history.append({
                'pattern_id': pattern_id,
                'reason': reason,
                'time': current_time
            })
            
            self.total_evictions += 1
    
    def _deactivate_pattern(self, pattern_id: str, reason: str, current_time: float):
        """Deactivate pattern due to resource competition."""
        if pattern_id in self.active_patterns:
            self.active_patterns.remove(pattern_id)
            
            if pattern_id in self.pattern_resources:
                self.pattern_resources[pattern_id].active = False
            
            self.competition_events.append({
                'type': 'deactivation',
                'pattern_id': pattern_id,
                'reason': reason,
                'time': current_time
            })
    
    def update_resource_dynamics(self, current_time: float):
        """Update resource dynamics - energy decay, pressure calculation."""
        # Decay energy for all patterns
        for resource in self.pattern_resources.values():
            age = current_time - resource.last_access_time
            decay = np.exp(-age * self.budget.decay_rate)
            resource.activation_energy *= decay
            
            # Deactivate patterns below threshold
            if (resource.active and 
                resource.activation_energy < self.budget.activation_threshold):
                self._deactivate_pattern(resource.pattern_id, 'energy_decay', current_time)
        
        # Calculate current resource pressure
        storage_pressure = len(self.pattern_resources) / self.budget.max_storage_slots
        activation_pressure = len(self.active_patterns) / self.budget.max_active_patterns
        self.resource_pressure = max(storage_pressure, activation_pressure)
    
    def get_active_pattern_ids(self, k: int = None) -> List[str]:
        """Get currently active pattern IDs, sorted by activation priority."""
        if k is None:
            k = len(self.active_patterns)
        
        # Sort active patterns by activation priority
        current_time = time.time()
        active_with_priority = [
            (pid, self._calculate_activation_priority(pid, current_time))
            for pid in self.active_patterns
        ]
        
        active_with_priority.sort(key=lambda x: x[1], reverse=True)
        return [pid for pid, _ in active_with_priority[:k]]
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """Get competitive dynamics statistics."""
        recent_competitions = len([
            event for event in self.competition_events 
            if time.time() - event['time'] < 60.0  # Last minute
        ])
        
        recent_evictions = len([
            eviction for eviction in self.eviction_history
            if time.time() - eviction['time'] < 60.0  # Last minute
        ])
        
        # Analyze competition types
        competition_types = defaultdict(int)
        for event in self.competition_events:
            competition_types[event['type']] += 1
        
        return {
            'total_competitions': self.total_competitions,
            'total_evictions': self.total_evictions,
            'recent_competitions_per_minute': recent_competitions,
            'recent_evictions_per_minute': recent_evictions,
            'resource_pressure': self.resource_pressure,
            'active_patterns': len(self.active_patterns),
            'stored_patterns': len(self.pattern_resources),
            'storage_utilization': len(self.pattern_resources) / self.budget.max_storage_slots,
            'activation_utilization': len(self.active_patterns) / self.budget.max_active_patterns,
            'competition_types': dict(competition_types)
        }


class EmergentCompetitiveDynamics:
    """
    Competitive dynamics that emerge from resource constraints.
    
    Instead of explicit winner-take-all algorithms, competition emerges from:
    - Storage capacity limits forcing pattern competition
    - Processing bandwidth limits creating activation competition
    - Energy constraints creating natural pattern decay
    - Access patterns creating natural clustering
    """
    
    def __init__(self, unified_storage, quiet_mode: bool = False):
        self.unified_storage = unified_storage
        self.quiet_mode = quiet_mode
        
        # Access unified storage for compatibility
        if hasattr(unified_storage, 'base_storage'):
            self.storage = unified_storage.base_storage
        else:
            self.storage = unified_storage
        
        # Derive resource budgets from actual physical constraints
        # Base limits from pattern storage capacity and processing constraints
        base_storage_capacity = self.storage.max_patterns
        base_pattern_dim = self.storage.pattern_dim
        
        # Physical constraint: processing bandwidth limits concurrent active patterns
        # Constraint: ~1 active pattern per 100 dimensions of processing capacity
        max_concurrent_processing = max(5, base_pattern_dim // 100)  # Minimum 5 for meaningful competition
        
        # Physical constraint: memory pressure creates storage competition
        # Constraint: competitive dynamics emerge when storage >50% full
        competitive_storage_limit = max(100, base_storage_capacity // 2)  # Minimum 100 for meaningful storage
        
        # Physical constraint: energy budget limits activation thresholds
        # Constraint: activation energy follows power law with pattern complexity
        base_activation_threshold = 0.1 * (base_pattern_dim / 1000) ** 0.5
        
        self.budgets = {
            'aggressive': ResourceBudget(
                name='aggressive',
                max_active_patterns=max(1, max_concurrent_processing // 4),  # 25% of processing capacity
                max_storage_slots=competitive_storage_limit // 5,            # 10% of base storage
                activation_threshold=base_activation_threshold * 3,          # 3x energy requirement
                decay_rate=0.1,                                             # Fast decay under pressure
                competition_pressure=0.8                                    # High competition
            ),
            'moderate': ResourceBudget(
                name='moderate',
                max_active_patterns=max_concurrent_processing,               # Full processing capacity
                max_storage_slots=competitive_storage_limit,                 # 50% of base storage
                activation_threshold=base_activation_threshold,              # Base energy requirement
                decay_rate=0.05,                                            # Moderate decay
                competition_pressure=0.5                                    # Moderate competition
            ),
            'relaxed': ResourceBudget(
                name='relaxed',
                max_active_patterns=max_concurrent_processing * 4,          # 4x processing (burst capacity)
                max_storage_slots=base_storage_capacity,                     # Full storage capacity
                activation_threshold=base_activation_threshold / 2,          # Half energy requirement
                decay_rate=0.01,                                            # Slow decay
                competition_pressure=0.2                                    # Low competition
            )
        }
        
        # Start with moderate competition
        self.current_budget = self.budgets['moderate']
        self.resource_storage = ResourceConstrainedStorage(
            self.current_budget, self.storage, quiet_mode
        )
        
        # Adaptation tracking
        self.competition_history = deque(maxlen=1000)
        self.adaptation_cycles = 0
        
        if not quiet_mode:
            print(f"\n🏆 EMERGENT COMPETITIVE DYNAMICS INITIALIZED")
            print(f"   🎯 Competition emerges from resource constraints")
            print(f"   Storage slots: {self.current_budget.max_storage_slots}")
            print(f"   Active slots: {self.current_budget.max_active_patterns}")
            print(f"   Competition pressure: {self.current_budget.competition_pressure:.2f}")
    
    def process_with_competition(self, pattern: SparsePattern, current_time: float) -> Dict[str, Any]:
        """
        Process pattern through competitive resource allocation.
        
        Returns information about competitive dynamics that occurred.
        """
        # Store pattern with resource competition
        stored = self.resource_storage.store_pattern_with_competition(pattern, current_time)
        
        # Attempt to activate pattern
        activated = False
        if stored:
            activated = self.resource_storage.activate_pattern_with_competition(
                pattern.pattern_id, current_time
            )
        
        # Update resource dynamics
        self.resource_storage.update_resource_dynamics(current_time)
        
        # Record competition event
        competition_info = {
            'pattern_stored': stored,
            'pattern_activated': activated,
            'resource_pressure': self.resource_storage.resource_pressure,
            'active_count': len(self.resource_storage.active_patterns),
            'total_stored': len(self.resource_storage.pattern_resources)
        }
        
        self.competition_history.append(competition_info)
        
        # Adaptive competition pressure based on system state
        if len(self.competition_history) > 50:
            self._adapt_competition_pressure()
        
        return competition_info
    
    def update_pattern_prediction_confidence(self, pattern_id: str, prediction_confidence: float):
        """
        Update prediction confidence for a pattern in the competitive system.
        
        This is the core of the dual motivation system:
        - High confidence patterns (>0.9) become cognitively expensive (restlessness emerges)
        - Low confidence patterns remain competitive (anxiety for accuracy emerges)
        - Resource competition naturally balances these motivations
        """
        self.resource_storage.update_pattern_confidence(pattern_id, prediction_confidence)
    
    def get_competitive_clusters(self, k: int = 10) -> List[List[str]]:
        """
        Get emergent pattern clusters from competitive dynamics.
        
        Clusters emerge from which patterns successfully compete together.
        """
        # Get most successful competing patterns
        active_patterns = self.resource_storage.get_active_pattern_ids(k * 2)
        
        if len(active_patterns) < 2:
            return [active_patterns] if active_patterns else []
        
        # Simple clustering based on competition success patterns
        # Patterns that are often active together form clusters
        clusters = []
        used_patterns = set()
        
        for pattern_id in active_patterns:
            if pattern_id in used_patterns:
                continue
                
            # Start new cluster
            cluster = [pattern_id]
            used_patterns.add(pattern_id)
            
            # Find patterns with similar competition profiles
            for other_id in active_patterns:
                if (other_id not in used_patterns and 
                    self._patterns_compete_similarly(pattern_id, other_id)):
                    cluster.append(other_id)
                    used_patterns.add(other_id)
            
            clusters.append(cluster)
            
            if len(clusters) >= k:
                break
        
        return clusters
    
    def _patterns_compete_similarly(self, pattern1_id: str, pattern2_id: str) -> bool:
        """
        Check if two patterns have similar competitive profiles.
        
        Patterns that compete similarly often represent related concepts.
        """
        # Simple heuristic: patterns accessed around the same time compete similarly
        if (pattern1_id not in self.resource_storage.pattern_resources or
            pattern2_id not in self.resource_storage.pattern_resources):
            return False
        
        resource1 = self.resource_storage.pattern_resources[pattern1_id]
        resource2 = self.resource_storage.pattern_resources[pattern2_id]
        
        # Time difference in access
        time_diff = abs(resource1.last_access_time - resource2.last_access_time)
        
        # Similar access frequencies
        freq_ratio = min(resource1.access_frequency, resource2.access_frequency) / max(resource1.access_frequency, resource2.access_frequency)
        
        return time_diff < 10.0 and freq_ratio > 0.5
    
    def _adapt_competition_pressure(self):
        """Adapt competition pressure based on system performance."""
        recent_history = list(self.competition_history)[-50:]
        
        # Calculate recent performance metrics
        storage_success_rate = np.mean([h['pattern_stored'] for h in recent_history])
        activation_success_rate = np.mean([h['pattern_activated'] for h in recent_history])
        avg_pressure = np.mean([h['resource_pressure'] for h in recent_history])
        
        # Adapt budget based on performance
        if storage_success_rate < 0.3:  # Too much competition
            if self.current_budget.name != 'relaxed':
                self.current_budget = self.budgets['relaxed']
                self._update_resource_storage()
        elif storage_success_rate > 0.8 and avg_pressure < 0.3:  # Too little competition
            if self.current_budget.name != 'aggressive':
                self.current_budget = self.budgets['aggressive']
                self._update_resource_storage()
        
        self.adaptation_cycles += 1
    
    def _update_resource_storage(self):
        """Update resource storage with new budget."""
        if not self.quiet_mode:
            print(f"   🔄 Adapting competition: {self.current_budget.name} mode")
        
        # Create new resource storage with updated budget
        self.resource_storage = ResourceConstrainedStorage(
            self.current_budget, self.storage, self.quiet_mode
        )
    
    def get_competition_stats(self) -> Dict[str, Any]:
        """Get comprehensive competitive dynamics statistics."""
        base_stats = self.resource_storage.get_competition_stats()
        
        # Add emergent behavior analysis
        clusters = self.get_competitive_clusters()
        
        recent_history = list(self.competition_history)[-100:] if self.competition_history else []
        recent_storage_rate = np.mean([h['pattern_stored'] for h in recent_history]) if recent_history else 0.0
        recent_activation_rate = np.mean([h['pattern_activated'] for h in recent_history]) if recent_history else 0.0
        
        return {
            **base_stats,
            'current_budget': self.current_budget.name,
            'adaptation_cycles': self.adaptation_cycles,
            'emergent_clusters': len(clusters),
            'recent_storage_success_rate': recent_storage_rate,
            'recent_activation_success_rate': recent_activation_rate,
            'competitive_emergence': 'active' if base_stats['resource_pressure'] > 0.3 else 'minimal'
        }


def demonstrate_emergent_competition():
    """Demonstrate emergent competitive dynamics from resource constraints."""
    print("🏆 EMERGENT COMPETITIVE DYNAMICS DEMONSTRATION")
    print("=" * 60)
    
    # Create sparse storage and competitive dynamics
    storage = SparsePatternStorage(pattern_dim=16, max_patterns=100, quiet_mode=True)
    encoder = SparsePatternEncoder(pattern_dim=16, sparsity=0.02, quiet_mode=True)
    competition = EmergentCompetitiveDynamics(storage, quiet_mode=True)
    
    print("Testing competitive resource allocation...")
    
    # Create many patterns to force competition
    patterns = []
    for i in range(50):
        pattern_vec = torch.randn(16)
        pattern = encoder.encode_top_k(pattern_vec, f"pattern_{i}")
        patterns.append(pattern)
    
    # Process patterns and observe competition
    competition_results = []
    for i, pattern in enumerate(patterns):
        result = competition.process_with_competition(pattern, time.time())
        competition_results.append(result)
        
        if i % 10 == 0:
            stats = competition.get_competition_stats()
            print(f"   Processed {i} patterns: {stats['active_patterns']} active, "
                  f"pressure: {stats['resource_pressure']:.2f}")
    
    # Analyze emergent competitive behaviors
    final_stats = competition.get_competition_stats()
    clusters = competition.get_competitive_clusters()
    
    print(f"\nEmergent competitive behaviors:")
    print(f"  Total competitions: {final_stats['total_competitions']}")
    print(f"  Total evictions: {final_stats['total_evictions']}")
    print(f"  Resource pressure: {final_stats['resource_pressure']:.2f}")
    print(f"  Active patterns: {final_stats['active_patterns']}")
    print(f"  Emergent clusters: {final_stats['emergent_clusters']}")
    print(f"  Storage success rate: {final_stats['recent_storage_success_rate']:.2f}")
    print(f"  Competitive emergence: {final_stats['competitive_emergence']}")
    
    print(f"\nPattern clusters from competition:")
    for i, cluster in enumerate(clusters[:5]):
        print(f"  Cluster {i+1}: {len(cluster)} patterns")
    
    print(f"\n✅ EMERGENT COMPETITIVE DYNAMICS DEMONSTRATION COMPLETE")
    print(f"Competition emerged from resource constraints, not explicit algorithms!")


if __name__ == "__main__":
    demonstrate_emergent_competition()