#!/usr/bin/env python3
"""
Enhanced Pattern Memory System for Vector Streams

This system replaces the simple 50-pattern limit with a sophisticated
hierarchical pattern storage that can scale to thousands of patterns
while maintaining real-time performance.

Key features:
- Hierarchical storage: fast buffers + persistent memory
- Dynamic promotion/demotion based on utility
- Cross-stream pattern linking for episodic capabilities
- Intelligent forgetting based on importance scores
- Memory pressure management
"""

import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class PatternTier(Enum):
    """Pattern storage tiers with different access speeds and persistence."""
    ACTIVE = "active"           # Fast rolling buffer, always accessible
    WORKING = "working"         # Medium-term storage, frequently accessed
    CONSOLIDATED = "consolidated"  # Long-term storage, infrequently accessed


@dataclass
class EnhancedVectorPattern:
    """
    Enhanced pattern with richer context and cross-stream relationships.
    
    This replaces the simple VectorPattern with sophisticated tracking
    of pattern importance, relationships, and usage history.
    """
    # Core pattern data
    activation_pattern: torch.Tensor
    pattern_id: str = field(default_factory=lambda: f"pat_{int(time.time() * 1000000)}")
    
    # Temporal context
    temporal_context: float = 0.0
    creation_time: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_activated: float = 0.0
    
    # Usage statistics
    frequency: int = 1
    prediction_successes: int = 0
    prediction_attempts: int = 0
    activation_strength: float = 1.0
    
    # Importance scoring
    novelty_score: float = 1.0      # How unique this pattern is
    utility_score: float = 0.0      # How useful for predictions
    recency_score: float = 1.0      # How recently seen
    importance_score: float = 0.0   # Combined importance
    
    # Cross-stream relationships
    linked_patterns: Dict[str, Set[str]] = field(default_factory=dict)  # stream_name -> pattern_ids
    episode_contexts: List[str] = field(default_factory=list)  # Episode IDs this pattern belongs to
    
    # Storage tier
    storage_tier: PatternTier = PatternTier.ACTIVE
    tier_promotion_count: int = 0
    tier_demotion_count: int = 0
    
    def update_importance_score(self):
        """Calculate combined importance score for storage tier decisions."""
        # Prediction utility (most important)
        prediction_rate = self.prediction_successes / max(1, self.prediction_attempts)
        utility_weight = 0.5
        
        # Frequency and recency
        frequency_weight = 0.3
        recency_weight = 0.2
        
        # Normalize frequency (assume max frequency of 1000)
        normalized_frequency = min(1.0, self.frequency / 1000.0)
        
        # Recency based on last_seen (decay over 24 hours)
        hours_since_seen = (time.time() - self.last_seen) / 3600.0
        recency_factor = np.exp(-hours_since_seen / 24.0)
        
        self.importance_score = (
            utility_weight * prediction_rate +
            frequency_weight * normalized_frequency +
            recency_weight * recency_factor
        )
    
    def add_cross_stream_link(self, stream_name: str, pattern_id: str):
        """Add a link to a pattern in another stream."""
        if stream_name not in self.linked_patterns:
            self.linked_patterns[stream_name] = set()
        self.linked_patterns[stream_name].add(pattern_id)
    
    def get_prediction_accuracy(self) -> float:
        """Get prediction accuracy rate."""
        if self.prediction_attempts == 0:
            return 0.0
        return self.prediction_successes / self.prediction_attempts
    
    def __eq__(self, other):
        """Override equality to compare by pattern_id instead of tensor values."""
        if not isinstance(other, EnhancedVectorPattern):
            return False
        return self.pattern_id == other.pattern_id
    
    def __hash__(self):
        """Make pattern hashable by pattern_id."""
        return hash(self.pattern_id)


class HierarchicalPatternMemory:
    """
    Hierarchical pattern storage system that can scale to thousands of patterns.
    
    Uses three tiers:
    - ACTIVE: Fast access, always in memory (100-200 patterns)
    - WORKING: Medium access, frequently used (500-1000 patterns) 
    - CONSOLIDATED: Slow access, long-term storage (unlimited)
    """
    
    def __init__(self, stream_name: str, max_active: int = 200, max_working: int = 1000):
        self.stream_name = stream_name
        self.max_active = max_active
        self.max_working = max_working
        
        # Three-tier storage
        self.active_patterns: List[EnhancedVectorPattern] = []
        self.working_patterns: List[EnhancedVectorPattern] = []
        self.consolidated_patterns: List[EnhancedVectorPattern] = []
        
        # Fast lookup by pattern ID
        self.pattern_index: Dict[str, EnhancedVectorPattern] = {}
        
        # Memory pressure tracking
        self.total_patterns = 0
        self.memory_pressure = 0.0
        
        # Pattern relationship tracking
        self.cross_stream_links: Dict[str, Dict[str, Set[str]]] = {}  # stream -> pattern_id -> linked_ids
        
        print(f"🧠 HierarchicalPatternMemory '{stream_name}' initialized")
        print(f"   Active tier: {max_active} patterns (fast access)")
        print(f"   Working tier: {max_working} patterns (medium access)")
        print(f"   Consolidated tier: unlimited patterns (slow access)")
    
    def store_pattern(self, pattern: EnhancedVectorPattern) -> str:
        """
        Store a new pattern in the appropriate tier.
        
        New patterns start in ACTIVE tier and may be promoted/demoted
        based on usage and importance.
        """
        pattern.storage_tier = PatternTier.ACTIVE
        self.active_patterns.append(pattern)
        self.pattern_index[pattern.pattern_id] = pattern
        self.total_patterns += 1
        
        # Check if we need to manage memory pressure
        if len(self.active_patterns) > self.max_active:
            self._manage_memory_pressure()
        
        return pattern.pattern_id
    
    def find_similar_patterns(self, activation: torch.Tensor, 
                            threshold: float = 0.8, 
                            max_results: int = 10) -> List[Tuple[EnhancedVectorPattern, float]]:
        """
        Find patterns similar to the given activation.
        
        Searches active tier first (fast), then working tier (medium),
        then consolidated tier (slow) if needed.
        """
        results = []
        
        # Search active tier first (highest priority)
        for pattern in self.active_patterns:
            similarity = self._calculate_similarity(activation, pattern.activation_pattern)
            if similarity > threshold:
                results.append((pattern, similarity))
        
        # If we need more results, search working tier
        if len(results) < max_results and len(self.working_patterns) > 0:
            for pattern in self.working_patterns:
                similarity = self._calculate_similarity(activation, pattern.activation_pattern)
                if similarity > threshold:
                    results.append((pattern, similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Promote accessed patterns (they're being used)
        for pattern, _ in results[:3]:  # Promote top 3 accessed patterns
            self._consider_promotion(pattern)
        
        return results[:max_results]
    
    def get_pattern(self, pattern_id: str) -> Optional[EnhancedVectorPattern]:
        """Get a pattern by ID, promoting it if accessed."""
        pattern = self.pattern_index.get(pattern_id)
        if pattern:
            pattern.last_activated = time.time()
            self._consider_promotion(pattern)
        return pattern
    
    def update_pattern_usage(self, pattern_id: str, prediction_success: bool):
        """Update pattern usage statistics for importance scoring."""
        pattern = self.pattern_index.get(pattern_id)
        if pattern:
            pattern.prediction_attempts += 1
            if prediction_success:
                pattern.prediction_successes += 1
            pattern.update_importance_score()
    
    def link_patterns_across_streams(self, pattern_id: str, other_stream: str, other_pattern_id: str):
        """Create a link between patterns in different streams."""
        pattern = self.pattern_index.get(pattern_id)
        if pattern:
            pattern.add_cross_stream_link(other_stream, other_pattern_id)
            
            # Update cross-stream index
            if other_stream not in self.cross_stream_links:
                self.cross_stream_links[other_stream] = {}
            if pattern_id not in self.cross_stream_links[other_stream]:
                self.cross_stream_links[other_stream][pattern_id] = set()
            self.cross_stream_links[other_stream][pattern_id].add(other_pattern_id)
    
    def get_linked_patterns(self, pattern_id: str, other_stream: str) -> Set[str]:
        """Get patterns linked to this pattern in another stream."""
        pattern = self.pattern_index.get(pattern_id)
        if pattern and other_stream in pattern.linked_patterns:
            return pattern.linked_patterns[other_stream]
        return set()
    
    def _calculate_similarity(self, activation1: torch.Tensor, activation2: torch.Tensor) -> float:
        """Calculate similarity between two activations (cosine similarity for now)."""
        # Handle zero vectors
        if torch.norm(activation1) < 1e-8 or torch.norm(activation2) < 1e-8:
            return 0.0
        
        return torch.cosine_similarity(activation1, activation2, dim=0).item()
    
    def _consider_promotion(self, pattern: EnhancedVectorPattern):
        """Consider promoting a pattern to a higher tier based on usage."""
        pattern.update_importance_score()
        
        # Promote from CONSOLIDATED to WORKING if important enough
        if (pattern.storage_tier == PatternTier.CONSOLIDATED and 
            pattern.importance_score > 0.6 and 
            len(self.working_patterns) < self.max_working):
            
            self._promote_pattern(pattern, PatternTier.WORKING)
        
        # Promote from WORKING to ACTIVE if very important
        elif (pattern.storage_tier == PatternTier.WORKING and 
              pattern.importance_score > 0.8 and 
              len(self.active_patterns) < self.max_active):
            
            self._promote_pattern(pattern, PatternTier.ACTIVE)
    
    def _promote_pattern(self, pattern: EnhancedVectorPattern, target_tier: PatternTier):
        """Promote a pattern to a higher tier."""
        # Remove from current tier (search by pattern_id to avoid tensor comparison issues)
        if pattern.storage_tier == PatternTier.CONSOLIDATED:
            self.consolidated_patterns = [p for p in self.consolidated_patterns if p.pattern_id != pattern.pattern_id]
        elif pattern.storage_tier == PatternTier.WORKING:
            self.working_patterns = [p for p in self.working_patterns if p.pattern_id != pattern.pattern_id]
        elif pattern.storage_tier == PatternTier.ACTIVE:
            self.active_patterns = [p for p in self.active_patterns if p.pattern_id != pattern.pattern_id]
        
        # Add to target tier
        pattern.storage_tier = target_tier
        pattern.tier_promotion_count += 1
        
        if target_tier == PatternTier.ACTIVE:
            self.active_patterns.append(pattern)
        elif target_tier == PatternTier.WORKING:
            self.working_patterns.append(pattern)
        elif target_tier == PatternTier.CONSOLIDATED:
            self.consolidated_patterns.append(pattern)
    
    def _manage_memory_pressure(self):
        """Manage memory pressure by demoting less important patterns."""
        self.memory_pressure = len(self.active_patterns) / self.max_active
        
        if len(self.active_patterns) > self.max_active:
            # Sort active patterns by importance (lowest first) and update scores
            for pattern in self.active_patterns:
                pattern.update_importance_score()
            self.active_patterns.sort(key=lambda p: p.importance_score)
            
            # Demote least important patterns
            patterns_to_demote = len(self.active_patterns) - self.max_active + 10  # Extra buffer
            patterns_to_demote = min(patterns_to_demote, len(self.active_patterns))
            
            # Create list of patterns to demote to avoid index issues
            demote_patterns = self.active_patterns[:patterns_to_demote]
            
            for pattern in demote_patterns:
                if len(self.working_patterns) < self.max_working:
                    self._promote_pattern(pattern, PatternTier.WORKING)
                else:
                    self._promote_pattern(pattern, PatternTier.CONSOLIDATED)
        
        # Also manage working tier
        if len(self.working_patterns) > self.max_working:
            # Update importance scores and sort
            for pattern in self.working_patterns:
                pattern.update_importance_score()
            self.working_patterns.sort(key=lambda p: p.importance_score)
            
            patterns_to_demote = len(self.working_patterns) - self.max_working + 20
            patterns_to_demote = min(patterns_to_demote, len(self.working_patterns))
            
            # Create list of patterns to demote
            demote_patterns = self.working_patterns[:patterns_to_demote]
            
            for pattern in demote_patterns:
                self._promote_pattern(pattern, PatternTier.CONSOLIDATED)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            'stream_name': self.stream_name,
            'total_patterns': self.total_patterns,
            'active_patterns': len(self.active_patterns),
            'working_patterns': len(self.working_patterns),
            'consolidated_patterns': len(self.consolidated_patterns),
            'memory_pressure': self.memory_pressure,
            'cross_stream_links': sum(len(links) for links in self.cross_stream_links.values())
        }
    
    def cleanup_old_patterns(self, max_age_hours: float = 168):  # 1 week default
        """Remove very old patterns that haven't been used recently."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)
        
        patterns_removed = 0
        
        # Only clean up consolidated patterns (preserve active/working)
        patterns_to_remove = [
            p for p in self.consolidated_patterns 
            if p.last_seen <= cutoff_time and p.importance_score <= 0.3
        ]
        
        self.consolidated_patterns = [
            p for p in self.consolidated_patterns 
            if p.last_seen > cutoff_time or p.importance_score > 0.3
        ]
        
        patterns_removed = len(patterns_to_remove)
        
        # Update pattern index
        for pattern in patterns_to_remove:
            if pattern.pattern_id in self.pattern_index:
                del self.pattern_index[pattern.pattern_id]
        
        self.total_patterns -= patterns_removed
        
        if patterns_removed > 0:
            print(f"🧹 Cleaned up {patterns_removed} old patterns from {self.stream_name}")
        
        return patterns_removed


# Episode management for linking patterns across streams at the same time
@dataclass
class Episode:
    """
    An episode represents patterns that occurred together across streams.
    This enables episodic memory capabilities.
    """
    episode_id: str = field(default_factory=lambda: f"ep_{int(time.time() * 1000000)}")
    timestamp: float = field(default_factory=time.time)
    stream_patterns: Dict[str, str] = field(default_factory=dict)  # stream_name -> pattern_id
    context_description: str = ""
    importance_score: float = 0.0
    
    def add_stream_pattern(self, stream_name: str, pattern_id: str):
        """Add a pattern from a specific stream to this episode."""
        self.stream_patterns[stream_name] = pattern_id
    
    def get_involved_streams(self) -> Set[str]:
        """Get all streams involved in this episode."""
        return set(self.stream_patterns.keys())


class EpisodeManager:
    """Manages episodes for cross-stream pattern linking."""
    
    def __init__(self, max_episodes: int = 10000):
        self.episodes: List[Episode] = []
        self.episode_index: Dict[str, Episode] = {}
        self.max_episodes = max_episodes
    
    def create_episode(self, stream_patterns: Dict[str, str], context: str = "") -> str:
        """Create a new episode linking patterns across streams."""
        episode = Episode(context_description=context)
        
        for stream_name, pattern_id in stream_patterns.items():
            episode.add_stream_pattern(stream_name, pattern_id)
        
        self.episodes.append(episode)
        self.episode_index[episode.episode_id] = episode
        
        # Manage episode memory
        if len(self.episodes) > self.max_episodes:
            # Remove oldest episodes with low importance
            self.episodes.sort(key=lambda e: (e.importance_score, e.timestamp))
            episodes_to_remove = len(self.episodes) - self.max_episodes + 100
            episodes_to_remove = min(episodes_to_remove, len(self.episodes))
            
            # Remove from index first
            for i in range(episodes_to_remove):
                old_episode = self.episodes[i]
                if old_episode.episode_id in self.episode_index:
                    del self.episode_index[old_episode.episode_id]
            
            # Then update episodes list
            self.episodes = self.episodes[episodes_to_remove:]
        
        return episode.episode_id
    
    def find_episodes_with_pattern(self, stream_name: str, pattern_id: str) -> List[Episode]:
        """Find all episodes containing a specific pattern."""
        return [
            episode for episode in self.episodes 
            if episode.stream_patterns.get(stream_name) == pattern_id
        ]
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get episode management statistics."""
        return {
            'total_episodes': len(self.episodes),
            'avg_streams_per_episode': np.mean([len(e.stream_patterns) for e in self.episodes]) if self.episodes else 0,
            'max_episodes': self.max_episodes
        }