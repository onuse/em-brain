#!/usr/bin/env python3
"""
Persistent Cross-Session Memory System - Phase 6.5

Enables brain state persistence across sessions for genuine long-term learning.
Core principle: Critical patterns compete for limited persistent storage slots.

Architecture:
- Constraint-based pattern selection (only important patterns persist)
- Background thread for incremental saves (never block brain)
- Inline loading at startup (brain needs state before first cycle)
- Bounded memory growth (biological storage constraints)
"""

import json
import time
import threading
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np


@dataclass
class PersistentPattern:
    """A pattern worthy of cross-session persistence."""
    pattern_id: str
    pattern_data: Dict[str, Any]
    activation_count: int
    last_accessed: float
    success_rate: float
    energy_level: float
    stream_source: str  # sensory, motor, temporal
    consolidation_score: float


@dataclass
class PersistentConfidenceState:
    """Core confidence dynamics worth preserving."""
    confidence_history: List[float]
    pattern_progression: List[str]
    dunning_kruger_phases: List[Dict[str, Any]]
    meta_accuracy_trend: List[float]
    total_updates: int
    session_count: int


@dataclass
class PersistentHardwareAdaptation:
    """Stable hardware adaptations learned over time."""
    working_memory_limit: int
    similarity_search_limit: int
    cognitive_energy_budget: int
    adaptation_history: List[Dict[str, Any]]
    stable_performance_profile: Dict[str, float]


@dataclass
class PersistentBrainState:
    """Complete persistent brain state for cross-session continuity."""
    version: str
    session_count: int
    total_lifetime_experiences: int
    creation_time: float
    last_save_time: float
    
    # Core state components
    patterns: List[PersistentPattern]
    confidence_state: PersistentConfidenceState
    hardware_adaptations: PersistentHardwareAdaptation
    learning_trajectories: Dict[str, List[float]]
    
    # Metadata
    brain_architecture: str
    evolutionary_phases: List[str]


class PatternConsolidationEngine:
    """Biological consolidation - preserve almost everything, strengthen important patterns."""
    
    def __init__(self, consolidation_threshold: float = 0.01):
        # Much lower threshold - preserve almost everything like real sleep
        self.consolidation_threshold = consolidation_threshold
        self.strengthening_factor = 1.1  # Strengthen frequently accessed patterns
        self.natural_decay_rate = 0.99  # Very slow natural forgetting
        
    def consolidate_patterns_for_persistence(self, all_patterns: Dict[str, List[Dict]]) -> List[PersistentPattern]:
        """Consolidate patterns like biological sleep - preserve almost everything, strengthen important ones."""
        consolidated_patterns = []
        
        for stream_name, patterns in all_patterns.items():
            # Convert and consolidate all patterns (like real sleep)
            for i, pattern in enumerate(patterns):
                persistent_pattern = PersistentPattern(
                    pattern_id=f"{stream_name}_{i}_{int(time.time())}",
                    pattern_data=pattern,
                    activation_count=pattern.get('activation_count', 1),
                    last_accessed=pattern.get('last_accessed', time.time()),
                    success_rate=pattern.get('success_rate', 0.5),
                    energy_level=self._apply_consolidation(pattern),
                    stream_source=stream_name,
                    consolidation_score=self._calculate_consolidation_score(pattern)
                )
                
                # Preserve almost everything (like real brains)
                if persistent_pattern.energy_level > self.consolidation_threshold:
                    consolidated_patterns.append(persistent_pattern)
        
        return consolidated_patterns
    
    def _apply_consolidation(self, pattern: Dict[str, Any]) -> float:
        """Apply sleep-like consolidation - strengthen or naturally decay."""
        current_energy = pattern.get('energy_level', 0.1)
        activation_count = pattern.get('activation_count', 1)
        
        # Strengthen frequently used patterns (like sleep consolidation)
        if activation_count > 3:
            return min(1.0, current_energy * self.strengthening_factor)
        else:
            # Natural decay for unused patterns
            return current_energy * self.natural_decay_rate
    
    def _calculate_consolidation_score(self, pattern: Dict[str, Any]) -> float:
        """Calculate how important this pattern is for persistence."""
        activation_score = min(1.0, pattern.get('activation_count', 1) / 10.0)
        recency_score = self._calculate_recency_score(pattern.get('last_accessed', time.time()))
        success_score = pattern.get('success_rate', 0.5)
        energy_score = min(1.0, pattern.get('energy_level', 0.1) / 1.0)
        
        # Weighted combination (biological priorities)
        return (0.3 * activation_score + 
                0.2 * recency_score + 
                0.3 * success_score + 
                0.2 * energy_score)
    
    def _calculate_recency_score(self, last_accessed: float) -> float:
        """Recent patterns get higher scores (biological recency bias)."""
        hours_ago = (time.time() - last_accessed) / 3600.0
        return max(0.0, 1.0 - (hours_ago / 24.0))  # Decay over 24 hours
    
    def _meets_persistence_criteria(self, pattern: PersistentPattern) -> bool:
        """Check if pattern meets minimum criteria for persistence (very permissive like biology)."""
        return pattern.energy_level > self.consolidation_threshold


class PersistentMemoryManager:
    """Manages brain state persistence with background saves and startup loading."""
    
    def __init__(self, memory_path: str, enable_incremental_saves: bool = True):
        self.memory_path = Path(memory_path)
        self.enable_incremental_saves = enable_incremental_saves
        
        # Background saving
        self.save_queue = []
        self.save_thread = None
        self.shutdown_event = threading.Event()
        self.save_lock = threading.Lock()
        
        # Pattern consolidation (biological sleep-like)
        self.pattern_consolidator = PatternConsolidationEngine()
        
        # Ensure directory exists
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # State files
        self.full_state_file = self.memory_path / "brain_state.json"
        self.incremental_state_file = self.memory_path / "incremental_state.json"
        self.metadata_file = self.memory_path / "session_metadata.json"
        
        # Performance tracking
        self.save_stats = {
            'total_saves': 0,
            'incremental_saves': 0,
            'full_saves': 0,
            'avg_save_time_ms': 0.0
        }
        
        if enable_incremental_saves:
            self._start_background_saver()
    
    def _start_background_saver(self):
        """Start background thread for non-blocking saves."""
        self.save_thread = threading.Thread(
            target=self._background_save_worker,
            name="PersistentMemorySaver",
            daemon=True
        )
        self.save_thread.start()
    
    def _background_save_worker(self):
        """Background worker thread for processing save requests."""
        while not self.shutdown_event.is_set():
            try:
                # Check for save requests
                with self.save_lock:
                    if self.save_queue:
                        save_request = self.save_queue.pop(0)
                    else:
                        save_request = None
                
                if save_request:
                    self._execute_save_request(save_request)
                else:
                    time.sleep(0.1)  # Brief sleep if no work
                    
            except Exception as e:
                print(f"⚠️ Background save error: {e}")
    
    def _execute_save_request(self, save_request: Dict[str, Any]):
        """Execute a save request in background thread."""
        start_time = time.perf_counter()
        
        try:
            if save_request['type'] == 'incremental':
                self._save_incremental_state(save_request['data'])
                self.save_stats['incremental_saves'] += 1
            elif save_request['type'] == 'full':
                self._save_full_state(save_request['data'])
                self.save_stats['full_saves'] += 1
            
            # Update performance stats
            save_time_ms = (time.perf_counter() - start_time) * 1000
            self.save_stats['total_saves'] += 1
            self.save_stats['avg_save_time_ms'] = (
                (self.save_stats['avg_save_time_ms'] * (self.save_stats['total_saves'] - 1) + save_time_ms) 
                / self.save_stats['total_saves']
            )
            
        except Exception as e:
            print(f"⚠️ Failed to execute save request: {e}")
    
    def load_brain_state(self) -> Optional[PersistentBrainState]:
        """Load brain state at startup (inline, blocking)."""
        if not self.full_state_file.exists():
            print("📂 No persistent brain state found - starting fresh")
            return None
        
        try:
            start_time = time.perf_counter()
            
            with open(self.full_state_file, 'r') as f:
                state_data = json.load(f)
            
            # Convert back to PersistentBrainState
            brain_state = self._deserialize_brain_state(state_data)
            
            load_time_ms = (time.perf_counter() - start_time) * 1000
            
            print(f"🧠 Loaded persistent brain state:")
            print(f"   Sessions: {brain_state.session_count}")
            print(f"   Patterns: {len(brain_state.patterns)}")
            print(f"   Lifetime experiences: {brain_state.total_lifetime_experiences}")
            print(f"   Load time: {load_time_ms:.1f}ms")
            
            return brain_state
            
        except Exception as e:
            print(f"⚠️ Failed to load brain state: {e}")
            print("📂 Starting with fresh brain state")
            return None
    
    def queue_incremental_save(self, brain_data: Dict[str, Any]):
        """Queue incremental save (non-blocking)."""
        if not self.enable_incremental_saves:
            return
        
        with self.save_lock:
            # Keep only most recent incremental save request
            self.save_queue = [req for req in self.save_queue if req['type'] != 'incremental']
            self.save_queue.append({
                'type': 'incremental',
                'data': brain_data,
                'timestamp': time.time()
            })
    
    def queue_full_save(self, brain_state: PersistentBrainState):
        """Queue full brain state save (non-blocking)."""
        with self.save_lock:
            self.save_queue.append({
                'type': 'full', 
                'data': brain_state,
                'timestamp': time.time()
            })
    
    def save_brain_state_blocking(self, brain_state: PersistentBrainState):
        """Save brain state immediately (blocking - for shutdown)."""
        self._save_full_state(brain_state)
        self.save_stats['full_saves'] += 1
        print(f"💾 Brain state saved successfully ({len(brain_state.patterns)} patterns)")
    
    def _save_full_state(self, brain_state: PersistentBrainState):
        """Save complete brain state to disk."""
        brain_state.last_save_time = time.time()
        
        # Convert to JSON-serializable format
        state_data = self._serialize_brain_state(brain_state)
        
        # Atomic write (write to temp file, then rename)
        temp_file = self.full_state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        temp_file.rename(self.full_state_file)
    
    def _save_incremental_state(self, brain_data: Dict[str, Any]):
        """Save incremental brain data."""
        incremental_data = {
            'timestamp': time.time(),
            'data': brain_data
        }
        
        with open(self.incremental_state_file, 'w') as f:
            json.dump(incremental_data, f, indent=2)
    
    def _serialize_brain_state(self, brain_state: PersistentBrainState) -> Dict[str, Any]:
        """Convert PersistentBrainState to JSON-serializable format."""
        return {
            'version': brain_state.version,
            'session_count': brain_state.session_count,
            'total_lifetime_experiences': brain_state.total_lifetime_experiences,
            'creation_time': brain_state.creation_time,
            'last_save_time': brain_state.last_save_time,
            'patterns': [asdict(pattern) for pattern in brain_state.patterns],
            'confidence_state': asdict(brain_state.confidence_state),
            'hardware_adaptations': asdict(brain_state.hardware_adaptations),
            'learning_trajectories': brain_state.learning_trajectories,
            'brain_architecture': brain_state.brain_architecture,
            'evolutionary_phases': brain_state.evolutionary_phases
        }
    
    def _deserialize_brain_state(self, state_data: Dict[str, Any]) -> PersistentBrainState:
        """Convert JSON data back to PersistentBrainState."""
        return PersistentBrainState(
            version=state_data['version'],
            session_count=state_data['session_count'],
            total_lifetime_experiences=state_data['total_lifetime_experiences'],
            creation_time=state_data['creation_time'],
            last_save_time=state_data['last_save_time'],
            patterns=[PersistentPattern(**p) for p in state_data['patterns']],
            confidence_state=PersistentConfidenceState(**state_data['confidence_state']),
            hardware_adaptations=PersistentHardwareAdaptation(**state_data['hardware_adaptations']),
            learning_trajectories=state_data['learning_trajectories'],
            brain_architecture=state_data['brain_architecture'],
            evolutionary_phases=state_data['evolutionary_phases']
        )
    
    def get_save_stats(self) -> Dict[str, Any]:
        """Get save performance statistics."""
        return self.save_stats.copy()
    
    def shutdown(self, timeout: float = 5.0):
        """Shutdown persistent memory manager gracefully."""
        if self.save_thread:
            # Process any remaining saves
            remaining_saves = len(self.save_queue)
            if remaining_saves > 0:
                print(f"💾 Processing {remaining_saves} remaining saves...")
                
            self.shutdown_event.set()
            self.save_thread.join(timeout=timeout)
            
        print(f"📊 Persistent Memory Stats: {self.save_stats}")


# Global persistent memory manager instance
_global_memory_manager: Optional[PersistentMemoryManager] = None


def initialize_persistent_memory(memory_path: str, enable_incremental_saves: bool = True):
    """Initialize global persistent memory manager."""
    global _global_memory_manager
    _global_memory_manager = PersistentMemoryManager(memory_path, enable_incremental_saves)


def get_persistent_memory_manager() -> Optional[PersistentMemoryManager]:
    """Get global persistent memory manager instance."""
    return _global_memory_manager