"""
Memory Consolidation Loop

A separate thread that consolidates experiences from working memory to long-term storage.
This decouples immediate action generation from memory formation, allowing the brain
to respond quickly while memories are formed asynchronously.

Biological inspiration:
- Hippocampal consolidation happens during quiet periods and sleep
- Immediate responses don't wait for memory formation
- Consolidation can be selective based on importance
"""

import time
import threading
from typing import Optional, Dict, Any, List
from collections import deque

from .working_memory import WorkingMemoryBuffer, WorkingMemoryItem
from .storage import ExperienceStorage


class MemoryConsolidationLoop:
    """
    Asynchronous memory consolidation from working memory to long-term storage.
    
    Features:
    - Runs independently of action generation
    - Adaptive consolidation rate based on cognitive load
    - Selective consolidation based on importance/novelty
    - Batch processing for efficiency
    """
    
    def __init__(self,
                 working_memory: WorkingMemoryBuffer,
                 experience_storage: ExperienceStorage,
                 base_interval_ms: float = 100.0,
                 batch_size: int = 5):
        """
        Initialize memory consolidation loop.
        
        Args:
            working_memory: The working memory buffer
            experience_storage: Long-term experience storage
            base_interval_ms: Base consolidation interval in milliseconds
            batch_size: Number of experiences to consolidate per cycle
        """
        self.working_memory = working_memory
        self.experience_storage = experience_storage
        self.base_interval_s = base_interval_ms / 1000.0
        self.batch_size = batch_size
        
        # Loop state
        self.running = False
        self.thread = None
        
        # Adaptive timing based on cognitive state
        self.current_interval_s = self.base_interval_s
        self.cognitive_load_factor = 1.0
        
        # Statistics
        self.total_consolidated = 0
        self.consolidation_cycles = 0
        self.failed_consolidations = 0
        self.consolidation_times = deque(maxlen=100)
        
        # Importance thresholds
        self.min_activation_for_storage = 0.3  # Don't store if activation too low
        self.novelty_boost_threshold = 0.8     # Boost novel experiences
        
        print(f"💾 MemoryConsolidationLoop initialized")
        print(f"   Base interval: {base_interval_ms}ms (reality verification window)")
        print(f"   Batch size: {batch_size} experiences/cycle")
        print(f"   Decoupled from action generation")
    
    def start(self):
        """Start the consolidation loop in a separate thread."""
        if self.running:
            print("⚠️ Consolidation loop already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._consolidation_loop, daemon=True)
        self.thread.start()
        
        print("🔄 Memory consolidation loop started")
    
    def stop(self):
        """Stop the consolidation loop."""
        if not self.running:
            return
        
        print("🛑 Stopping memory consolidation loop...")
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None
        
        print("✅ Memory consolidation loop stopped")
    
    def set_cognitive_load(self, load_factor: float):
        """
        Adjust consolidation rate based on cognitive load.
        
        Args:
            load_factor: 0.5 = low load (faster consolidation)
                        1.0 = normal load
                        2.0 = high load (slower consolidation)
        """
        self.cognitive_load_factor = max(0.1, min(5.0, load_factor))
        self.current_interval_s = self.base_interval_s * self.cognitive_load_factor
    
    def _consolidation_loop(self):
        """Main consolidation loop running independently."""
        print("💭 Consolidation loop started - memories form asynchronously")
        
        while self.running:
            cycle_start = time.time()
            
            try:
                # Get experiences from working memory
                experiences = self._select_experiences_for_consolidation()
                
                if experiences:
                    # Consolidate to long-term storage
                    consolidated_count = self._consolidate_experiences(experiences)
                    
                    self.total_consolidated += consolidated_count
                    self.consolidation_cycles += 1
                    
                    # Track timing
                    consolidation_time = time.time() - cycle_start
                    self.consolidation_times.append(consolidation_time)
                
                # Adaptive sleep based on cognitive load
                remaining_time = self.current_interval_s - (time.time() - cycle_start)
                if remaining_time > 0:
                    time.sleep(remaining_time)
                    
            except Exception as e:
                print(f"❌ Consolidation error: {e}")
                self.failed_consolidations += 1
                time.sleep(self.base_interval_s)  # Brief pause on error
    
    def _select_experiences_for_consolidation(self) -> List[WorkingMemoryItem]:
        """
        Select which experiences to consolidate from working memory.
        
        Uses importance criteria:
        - Activation level (how much it was accessed)
        - Recency (newer might be more relevant)
        - Novelty (different from existing memories)
        - Completeness (has outcome)
        """
        # Get all experiences with sufficient activation
        candidates = []
        current_time = time.time()
        
        for item, weight in self.working_memory.get_experiences_for_matching():
            if item.activation_level >= self.min_activation_for_storage:
                # Check if this experience is eligible for consolidation
                if item.consolidation_eligible_time is not None:
                    # Individual experience lifetime - must wait for its personal timer
                    if current_time < item.consolidation_eligible_time:
                        continue  # Not ready yet, still in verification window
                
                # Prefer complete experiences (with outcomes)
                if item.outcome is not None:
                    candidates.append((item, weight * 1.5))  # Boost complete experiences
                else:
                    candidates.append((item, weight))
        
        # Sort by importance (weight)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top batch_size
        selected = []
        for item, _ in candidates[:self.batch_size]:
            selected.append(item)
            
        return selected
    
    def _consolidate_experiences(self, experiences: List[WorkingMemoryItem]) -> int:
        """
        Consolidate experiences to long-term storage.
        
        Returns number successfully consolidated.
        """
        consolidated = 0
        
        for exp in experiences:
            try:
                # Only consolidate if we have outcome (complete experience)
                if exp.outcome is not None:
                    # Create Experience object for storage
                    from .models import Experience
                    experience = Experience(
                        sensory_input=exp.sensory_input,
                        action_taken=exp.action_taken,
                        outcome=exp.outcome,
                        prediction_error=0.1,  # Mock prediction error for now
                        timestamp=exp.timestamp
                    )
                    
                    # Store in long-term memory
                    exp_id = self.experience_storage.add_experience(experience)
                    
                    if exp_id:
                        consolidated += 1
                        
                        # Mark as consolidated in working memory
                        self.working_memory.mark_for_consolidation([exp.experience_id])
                        
            except Exception as e:
                print(f"⚠️ Failed to consolidate experience: {e}")
                self.failed_consolidations += 1
        
        return consolidated
    
    def force_consolidate_all(self):
        """Force immediate consolidation of all working memory (e.g., before shutdown)."""
        print("🔄 Force consolidating all working memory...")
        
        all_experiences = self.working_memory.get_recent_experiences(self.working_memory.capacity)
        consolidated = self._consolidate_experiences(all_experiences)
        
        print(f"✅ Force consolidated {consolidated} experiences")
        return consolidated
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consolidation loop statistics."""
        avg_time = 0.0
        if self.consolidation_times:
            avg_time = sum(self.consolidation_times) / len(self.consolidation_times)
        
        success_rate = 0.0
        total_attempts = self.total_consolidated + self.failed_consolidations
        if total_attempts > 0:
            success_rate = self.total_consolidated / total_attempts
        
        return {
            'running': self.running,
            'total_consolidated': self.total_consolidated,
            'consolidation_cycles': self.consolidation_cycles,
            'failed_consolidations': self.failed_consolidations,
            'success_rate': success_rate,
            'avg_consolidation_time_ms': avg_time * 1000,
            'current_interval_ms': self.current_interval_s * 1000,
            'cognitive_load_factor': self.cognitive_load_factor,
            'batch_size': self.batch_size
        }
    
    def print_consolidation_report(self):
        """Print consolidation performance report."""
        stats = self.get_statistics()
        
        print(f"\n💾 MEMORY CONSOLIDATION REPORT")
        print(f"=" * 40)
        print(f"🔄 Status: {'Running' if stats['running'] else 'Stopped'}")
        print(f"📊 Total consolidated: {stats['total_consolidated']:,}")
        print(f"🔁 Consolidation cycles: {stats['consolidation_cycles']:,}")
        print(f"❌ Failed attempts: {stats['failed_consolidations']}")
        print(f"✅ Success rate: {stats['success_rate']:.1%}")
        print(f"⏱️  Avg consolidation time: {stats['avg_consolidation_time_ms']:.1f}ms")
        print(f"🧠 Cognitive load factor: {stats['cognitive_load_factor']:.1f}x")
        print(f"📦 Batch size: {stats['batch_size']} experiences/cycle")
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        if stats['running']:
            return f"MemoryConsolidation(running, {stats['total_consolidated']} consolidated)"
        else:
            return f"MemoryConsolidation(stopped, {stats['total_consolidated']} consolidated)"