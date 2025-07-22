#!/usr/bin/env python3
"""
Field-Native Memory System: Phase B3

Revolutionary memory system based on persistent field topology rather than discrete patterns.
Memory emerges from stable field configurations that influence future field dynamics.

Key Principles:
- Memory IS field topology persistence, not pattern storage
- Biological forgetting curves with decay over time
- Sparse storage of significant field regions only
- Sleep consolidation strengthens important memories
- Memory retrieval through field resonance, not lookup

This represents a fundamental shift from discrete memory to continuous field memory.
"""

import torch
import numpy as np
import time
import math
import pickle
import gzip
import os
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import asyncio

# Import field-native brain components
try:
    from .field_native_brain import UnifiedFieldBrain, FieldDynamicsFamily
    from .field_native_robot_interface import FieldNativeRobotInterface
except ImportError:
    from field_native_brain import UnifiedFieldBrain, FieldDynamicsFamily
    from field_native_robot_interface import FieldNativeRobotInterface


@dataclass
class FieldMemoryTrace:
    """A memory trace representing persistent field topology."""
    timestamp: float
    field_coordinates: torch.Tensor  # 37D field position
    importance: float = 1.0          # Memory strength/importance
    access_count: int = 0            # How often this memory is accessed
    last_accessed: float = 0.0       # Last access timestamp
    decay_rate: float = 0.1          # How fast this memory decays
    consolidation_level: int = 0     # How many consolidation cycles it survived
    memory_type: str = "experience"  # Type: experience, skill, concept
    associated_traces: List[int] = field(default_factory=list)  # Connected memories


@dataclass
class FieldTopologyRegion:
    """A stable region in field topology that persists as memory."""
    center: torch.Tensor             # Center position in 37D field
    radius: float                    # Influence radius
    strength: float                  # Field strength at center
    stability: float                 # How stable this region is
    formation_time: float            # When this region formed
    activation_count: int = 0        # How many times activated
    last_activation: float = 0.0     # Last activation time


class FieldMemoryType(Enum):
    """Types of field memories with different persistence characteristics."""
    EXPERIENCE = "experience"        # Episodic memories from robot experiences
    SKILL = "skill"                  # Motor skills and learned behaviors  
    CONCEPT = "concept"              # Abstract concepts and relationships
    REFLEX = "reflex"               # Fast responses and habits
    WORKING = "working"             # Temporary working memory


class FieldNativeMemorySystem:
    """
    Field-Native Memory System
    
    Implements memory as persistent field topology rather than discrete storage.
    Memory emerges from stable field configurations that influence future dynamics.
    """
    
    def __init__(self,
                 field_brain: UnifiedFieldBrain,
                 memory_capacity: int = 10000,
                 consolidation_threshold: float = 0.5,
                 forgetting_rate: float = 0.001,
                 sleep_consolidation_strength: float = 0.8):
        
        self.field_brain = field_brain
        self.memory_capacity = memory_capacity
        self.consolidation_threshold = consolidation_threshold
        self.forgetting_rate = forgetting_rate
        self.sleep_consolidation_strength = sleep_consolidation_strength
        
        # Memory storage
        self.memory_traces: Dict[int, FieldMemoryTrace] = {}
        self.topology_regions: Dict[int, FieldTopologyRegion] = {}
        self.next_memory_id = 0
        self.next_region_id = 0
        
        # Memory dynamics
        self.current_field_state = torch.zeros(self.field_brain.total_dimensions)
        self.memory_influence_field = torch.zeros(self.field_brain.total_dimensions)
        self.working_memory = deque(maxlen=5)  # Short-term memory buffer
        
        # Biological timing
        self.last_consolidation = time.time()
        self.consolidation_interval = 300.0  # 5 minutes between consolidations
        self.sleep_mode = False
        self.sleep_cycles_completed = 0
        
        # Statistics
        self.memories_formed = 0
        self.memories_forgotten = 0
        self.consolidations_performed = 0
        self.retrieval_count = 0
        
        print(f"🧠 Field-Native Memory System initialized")
        print(f"   Memory capacity: {memory_capacity} traces")
        print(f"   Consolidation threshold: {consolidation_threshold}")
        print(f"   Forgetting rate: {forgetting_rate}/hour")
        print(f"   Field dimensions: {self.field_brain.total_dimensions}")
    
    def form_memory(self, 
                   field_coordinates: torch.Tensor, 
                   memory_type: FieldMemoryType = FieldMemoryType.EXPERIENCE,
                   importance: float = 1.0) -> int:
        """
        Form a new memory from current field state.
        
        In field-native memory, memories are stable field topology configurations
        that influence future field dynamics through resonance.
        """
        current_time = time.time()
        
        # Check if this field state is novel enough to store
        novelty = self._calculate_memory_novelty(field_coordinates)
        if novelty < 0.1:  # Too similar to existing memories
            return -1
        
        # Create memory trace
        memory_id = self.next_memory_id
        self.next_memory_id += 1
        
        # Calculate decay rate based on memory type
        decay_rates = {
            FieldMemoryType.EXPERIENCE: 0.1,   # Episodic memories decay moderately
            FieldMemoryType.SKILL: 0.02,       # Skills decay slowly
            FieldMemoryType.CONCEPT: 0.05,     # Concepts decay slowly
            FieldMemoryType.REFLEX: 0.001,     # Reflexes very persistent
            FieldMemoryType.WORKING: 0.5       # Working memory decays quickly
        }
        
        memory_trace = FieldMemoryTrace(
            timestamp=current_time,
            field_coordinates=field_coordinates.clone(),
            importance=importance * novelty,  # Scale by novelty
            decay_rate=decay_rates.get(memory_type, 0.1),
            memory_type=memory_type.value,
            last_accessed=current_time
        )
        
        self.memory_traces[memory_id] = memory_trace
        self.memories_formed += 1
        
        # Add to working memory if recent
        if memory_type == FieldMemoryType.WORKING:
            self.working_memory.append(memory_id)
        
        # Check for memory capacity and prune if needed
        if len(self.memory_traces) > self.memory_capacity:
            self._prune_weak_memories()
        
        # Check if this creates a stable topology region
        self._update_topology_regions(field_coordinates, memory_id)
        
        return memory_id
    
    def retrieve_memories(self, 
                         query_field: torch.Tensor,
                         max_memories: int = 5,
                         similarity_threshold: float = 0.3) -> List[Tuple[int, float]]:
        """
        Retrieve memories similar to query field through field resonance.
        
        Returns list of (memory_id, similarity_score) tuples.
        """
        current_time = time.time()
        self.retrieval_count += 1
        
        retrieved_memories = []
        
        for memory_id, trace in self.memory_traces.items():
            # Calculate field similarity (resonance)
            similarity = self._calculate_field_similarity(query_field, trace.field_coordinates)
            
            # Apply decay factor
            time_decay = math.exp(-trace.decay_rate * (current_time - trace.timestamp) / 3600.0)
            effective_similarity = similarity * time_decay * trace.importance
            
            if effective_similarity > similarity_threshold:
                retrieved_memories.append((memory_id, effective_similarity))
                
                # Update access statistics
                trace.access_count += 1
                trace.last_accessed = current_time
                
                # Strengthen frequently accessed memories
                if trace.access_count > 5:
                    trace.importance = min(2.0, trace.importance * 1.05)
        
        # Sort by similarity and return top matches
        retrieved_memories.sort(key=lambda x: x[1], reverse=True)
        return retrieved_memories[:max_memories]
    
    def update_memory_influence(self, current_field: torch.Tensor):
        """
        Update memory influence on current field dynamics.
        
        Memories influence current field through resonance and attraction.
        """
        self.current_field_state = current_field.clone()
        
        # Clear previous influence
        self.memory_influence_field.zero_()
        
        # Calculate influence from relevant memories
        relevant_memories = self.retrieve_memories(current_field, max_memories=10, similarity_threshold=0.2)
        
        for memory_id, similarity in relevant_memories:
            trace = self.memory_traces[memory_id]
            
            # Memory attracts field toward stored configuration
            memory_influence = (trace.field_coordinates - current_field) * similarity * trace.importance * 0.1
            self.memory_influence_field += memory_influence
        
        # Add topology region influence
        for region in self.topology_regions.values():
            distance = torch.norm(current_field - region.center)
            if distance < region.radius:
                # Inside topology region - strengthen the region pattern
                influence_strength = (1.0 - distance / region.radius) * region.strength * 0.05
                region_influence = (region.center - current_field) * influence_strength
                self.memory_influence_field += region_influence
                
                # Update region activation
                region.activation_count += 1
                region.last_activation = time.time()
    
    def get_memory_influenced_field(self, base_field: torch.Tensor) -> torch.Tensor:
        """
        Get field state influenced by memory.
        
        This is how memory affects perception and decision-making.
        """
        return base_field + self.memory_influence_field
    
    def consolidate_memories(self, force_consolidation: bool = False):
        """
        Consolidate memories by strengthening important ones and forgetting weak ones.
        
        This is the field-native equivalent of sleep consolidation.
        """
        current_time = time.time()
        
        if not force_consolidation and (current_time - self.last_consolidation) < self.consolidation_interval:
            return
        
        print(f"🧠 Starting memory consolidation cycle {self.consolidations_performed + 1}")
        
        consolidated_count = 0
        forgotten_count = 0
        
        # Process each memory for consolidation
        memories_to_remove = []
        
        for memory_id, trace in self.memory_traces.items():
            # Calculate memory strength
            age_hours = (current_time - trace.timestamp) / 3600.0
            access_factor = math.log(1 + trace.access_count)
            decay_factor = math.exp(-trace.decay_rate * age_hours)
            
            current_strength = trace.importance * decay_factor * (1 + access_factor * 0.1)
            
            if current_strength < 0.1:  # Very weak memory
                memories_to_remove.append(memory_id)
                forgotten_count += 1
            elif current_strength > self.consolidation_threshold:
                # Consolidate strong memory
                trace.importance = min(2.0, trace.importance * self.sleep_consolidation_strength)
                trace.consolidation_level += 1
                consolidated_count += 1
            
            # Update importance with decay
            trace.importance = current_strength
        
        # Remove forgotten memories
        for memory_id in memories_to_remove:
            del self.memory_traces[memory_id]
            self.memories_forgotten += 1
        
        # Consolidate topology regions
        self._consolidate_topology_regions()
        
        self.last_consolidation = current_time
        self.consolidations_performed += 1
        
        print(f"   Consolidated: {consolidated_count} memories")
        print(f"   Forgotten: {forgotten_count} memories")
        print(f"   Total memories: {len(self.memory_traces)}")
        print(f"   Topology regions: {len(self.topology_regions)}")
    
    def enter_sleep_mode(self):
        """Enter sleep mode for intensive memory consolidation."""
        self.sleep_mode = True
        print(f"🌙 Entering sleep mode for memory consolidation")
        
        # Perform multiple consolidation cycles during sleep
        for cycle in range(3):
            self.consolidate_memories(force_consolidation=True)
            self.sleep_cycles_completed += 1
            time.sleep(0.1)  # Brief pause between cycles
        
        print(f"🌅 Sleep consolidation complete: {self.sleep_cycles_completed} total cycles")
    
    def exit_sleep_mode(self):
        """Exit sleep mode and return to normal operation."""
        self.sleep_mode = False
        print(f"☀️ Exiting sleep mode")
    
    def save_memory_state(self, filepath: str):
        """Save memory state to file with compression."""
        memory_data = {
            'memory_traces': self.memory_traces,
            'topology_regions': self.topology_regions,
            'next_memory_id': self.next_memory_id,
            'next_region_id': self.next_region_id,
            'statistics': {
                'memories_formed': self.memories_formed,
                'memories_forgotten': self.memories_forgotten,
                'consolidations_performed': self.consolidations_performed,
                'retrieval_count': self.retrieval_count,
                'sleep_cycles_completed': self.sleep_cycles_completed
            },
            'timestamp': time.time()
        }
        
        # Compress and save
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(memory_data, f)
        
        print(f"💾 Memory state saved: {len(self.memory_traces)} memories, {len(self.topology_regions)} regions")
    
    def load_memory_state(self, filepath: str) -> bool:
        """Load memory state from file."""
        if not os.path.exists(filepath):
            print(f"⚠️ Memory file not found: {filepath}")
            return False
        
        try:
            with gzip.open(filepath, 'rb') as f:
                memory_data = pickle.load(f)
            
            self.memory_traces = memory_data['memory_traces']
            self.topology_regions = memory_data['topology_regions']
            self.next_memory_id = memory_data['next_memory_id']
            self.next_region_id = memory_data['next_region_id']
            
            # Restore statistics
            stats = memory_data['statistics']
            self.memories_formed = stats['memories_formed']
            self.memories_forgotten = stats['memories_forgotten']
            self.consolidations_performed = stats['consolidations_performed']
            self.retrieval_count = stats['retrieval_count']
            self.sleep_cycles_completed = stats['sleep_cycles_completed']
            
            load_time = memory_data['timestamp']
            age_hours = (time.time() - load_time) / 3600.0
            
            print(f"📂 Memory state loaded: {len(self.memory_traces)} memories, {len(self.topology_regions)} regions")
            print(f"   Memory age: {age_hours:.2f} hours")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load memory state: {e}")
            return False
    
    def _calculate_memory_novelty(self, field_coordinates: torch.Tensor) -> float:
        """Calculate how novel a field state is compared to existing memories."""
        if not self.memory_traces:
            return 1.0
        
        max_similarity = 0.0
        for trace in self.memory_traces.values():
            similarity = self._calculate_field_similarity(field_coordinates, trace.field_coordinates)
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _calculate_field_similarity(self, field1: torch.Tensor, field2: torch.Tensor) -> float:
        """Calculate similarity between two field states."""
        if field1.shape != field2.shape:
            return 0.0
        
        # Normalized dot product similarity
        norm1 = torch.norm(field1)
        norm2 = torch.norm(field2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        similarity = torch.dot(field1, field2) / (norm1 * norm2)
        return max(0.0, similarity.item())
    
    def _update_topology_regions(self, field_coordinates: torch.Tensor, memory_id: int):
        """Update topology regions based on new field coordinates."""
        # Check if this coordinate is near an existing region
        for region in self.topology_regions.values():
            distance = torch.norm(field_coordinates - region.center)
            if distance < region.radius:
                # Strengthen existing region
                region.strength = min(2.0, region.strength * 1.1)
                region.stability += 0.1
                return
        
        # Create new topology region if field strength is significant
        field_strength = torch.norm(field_coordinates)
        if field_strength > 0.5:
            region_id = self.next_region_id
            self.next_region_id += 1
            
            self.topology_regions[region_id] = FieldTopologyRegion(
                center=field_coordinates.clone(),
                radius=0.5,  # Initial radius
                strength=field_strength,
                stability=0.1,
                formation_time=time.time()
            )
    
    def _consolidate_topology_regions(self):
        """Consolidate topology regions by merging close ones and removing weak ones."""
        regions_to_remove = []
        
        # Remove weak regions
        for region_id, region in self.topology_regions.items():
            age_hours = (time.time() - region.formation_time) / 3600.0
            if region.strength < 0.2 and age_hours > 24:  # Remove weak old regions
                regions_to_remove.append(region_id)
        
        for region_id in regions_to_remove:
            del self.topology_regions[region_id]
        
        # TODO: Merge nearby regions in future enhancement
    
    def _prune_weak_memories(self):
        """Remove weakest memories when capacity is exceeded."""
        if len(self.memory_traces) <= self.memory_capacity:
            return
        
        # Sort memories by strength (importance * decay factor)
        current_time = time.time()
        memory_strengths = []
        
        for memory_id, trace in self.memory_traces.items():
            age_hours = (current_time - trace.timestamp) / 3600.0
            decay_factor = math.exp(-trace.decay_rate * age_hours)
            strength = trace.importance * decay_factor
            memory_strengths.append((memory_id, strength))
        
        # Sort by strength and remove weakest
        memory_strengths.sort(key=lambda x: x[1])
        
        # Remove bottom 10% when pruning
        prune_count = max(1, len(self.memory_traces) // 10)
        for i in range(prune_count):
            memory_id = memory_strengths[i][0]
            del self.memory_traces[memory_id]
            self.memories_forgotten += 1
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        current_time = time.time()
        
        # Analyze memory types
        type_counts = defaultdict(int)
        total_importance = 0.0
        avg_age_hours = 0.0
        
        for trace in self.memory_traces.values():
            type_counts[trace.memory_type] += 1
            total_importance += trace.importance
            age_hours = (current_time - trace.timestamp) / 3600.0
            avg_age_hours += age_hours
        
        if self.memory_traces:
            avg_age_hours /= len(self.memory_traces)
            avg_importance = total_importance / len(self.memory_traces)
        else:
            avg_importance = 0.0
        
        return {
            'total_memories': len(self.memory_traces),
            'topology_regions': len(self.topology_regions),
            'memory_types': dict(type_counts),
            'average_importance': avg_importance,
            'average_age_hours': avg_age_hours,
            'memories_formed': self.memories_formed,
            'memories_forgotten': self.memories_forgotten,
            'consolidations_performed': self.consolidations_performed,
            'retrieval_count': self.retrieval_count,
            'sleep_cycles_completed': self.sleep_cycles_completed,
            'capacity_utilization': len(self.memory_traces) / self.memory_capacity,
            'working_memory_size': len(self.working_memory),
            'sleep_mode': self.sleep_mode
        }


def create_field_native_memory_system(field_brain: UnifiedFieldBrain,
                                     memory_capacity: int = 10000,
                                     forgetting_rate: float = 0.001) -> FieldNativeMemorySystem:
    """
    Create a field-native memory system integrated with the field brain.
    
    Args:
        field_brain: The unified field brain to attach memory to
        memory_capacity: Maximum number of memory traces to store
        forgetting_rate: Rate at which memories decay (per hour)
    
    Returns:
        Configured field-native memory system
    """
    return FieldNativeMemorySystem(
        field_brain=field_brain,
        memory_capacity=memory_capacity,
        forgetting_rate=forgetting_rate
    )