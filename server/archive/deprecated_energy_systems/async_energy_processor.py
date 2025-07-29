"""
Asynchronous Energy Processing

Handles expensive energy operations in background thread to keep main cycle fast.
"""

import torch
import threading
import queue
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .unified_energy_system import UnifiedEnergySystem, UnifiedEnergyConfig


@dataclass
class EnergyTask:
    """Task for background energy processing."""
    field_snapshot: torch.Tensor
    sensory_pattern: Optional[torch.Tensor]
    prediction_error: float
    reward: float
    timestamp: float


class AsyncEnergyProcessor:
    """
    Processes energy calculations in background thread.
    
    Main thread gets immediate recommendations based on last known state,
    while background thread updates the energy model.
    """
    
    def __init__(self, energy_system: UnifiedEnergySystem, quiet_mode: bool = False):
        """Initialize async energy processor."""
        self.energy_system = energy_system
        self.quiet_mode = quiet_mode
        
        # Task queue
        self._task_queue = queue.Queue(maxsize=10)
        
        # Cache for fast access
        self._cached_recommendations = {
            'sensory_amplification': 1.0,
            'motor_noise': 0.2,
            'spontaneous_weight': 0.5,
            'decay_rate': 0.999,
            'attention_bias': 'balanced'
        }
        self._cached_energy_state = {
            'mode': 'BALANCED',
            'current_energy': 0.5,
            'smoothed_energy': 0.5
        }
        
        # Background thread
        self._shutdown = threading.Event()
        self._worker_thread = threading.Thread(target=self._energy_worker, daemon=True)
        self._worker_thread.start()
        
        # Performance tracking
        self._last_update_time = 0
        self._update_count = 0
        
    def update_energy_async(self,
                          field: torch.Tensor,
                          sensory_pattern: Optional[torch.Tensor] = None,
                          prediction_error: float = 0.5,
                          reward: float = 0.0) -> Dict[str, Any]:
        """
        Queue energy update for background processing.
        
        Returns immediately with cached recommendations.
        """
        # Create task (lightweight - just tensor reference)
        task = EnergyTask(
            field_snapshot=field,  # Reference, not copy
            sensory_pattern=sensory_pattern,
            prediction_error=prediction_error,
            reward=reward,
            timestamp=time.time()
        )
        
        # Try to queue task (drop if queue full - not critical)
        try:
            self._task_queue.put_nowait(task)
        except queue.Full:
            # Skip this update - not critical
            pass
            
        # Return cached state immediately
        return {
            'recommendations': self._cached_recommendations.copy(),
            'energy_state': self._cached_energy_state.copy()
        }
        
    def apply_energy_modulation_fast(self, field: torch.Tensor, 
                                   activation_threshold: float = 0.001) -> torch.Tensor:
        """
        Fast energy modulation using cached state.
        
        Only does critical operations inline.
        """
        # Apply decay from cache
        decay = self._cached_recommendations['decay_rate']
        if decay < 1.0:
            field *= decay
            
        # Only do pruning if definitely needed (cached mode)
        if self._cached_energy_state['mode'] == 'SATIATED':
            # Defer heavy pruning to background thread
            # Just mark for later processing
            pass
            
        return field
        
    def _energy_worker(self):
        """Background worker for energy calculations."""
        while not self._shutdown.is_set():
            try:
                # Get task with timeout
                task = self._task_queue.get(timeout=0.1)
                
                # Process energy update
                with torch.no_grad():
                    # Full energy calculation
                    energy_update = self.energy_system.update_energy(
                        field=task.field_snapshot,
                        sensory_pattern=task.sensory_pattern,
                        prediction_error=task.prediction_error,
                        reward=task.reward
                    )
                    
                    # Update cache
                    self._cached_recommendations = energy_update['recommendations'].copy()
                    self._cached_energy_state = {
                        'mode': energy_update['mode'],
                        'current_energy': energy_update['current_energy'],
                        'smoothed_energy': energy_update['smoothed_energy']
                    }
                    
                    self._last_update_time = time.time()
                    self._update_count += 1
                    
                    # Log mode changes
                    if not self.quiet_mode and self._update_count % 100 == 0:
                        mode = self._cached_energy_state['mode']
                        energy = self._cached_energy_state['smoothed_energy']
                        print(f"🔋 Energy: {mode} ({energy:.3f})")
                        
            except queue.Empty:
                continue
            except Exception as e:
                if not self.quiet_mode:
                    print(f"⚠️ Energy worker error: {e}")
                    
    def get_cached_state(self) -> Dict[str, Any]:
        """Get current cached state."""
        return {
            'recommendations': self._cached_recommendations.copy(),
            'energy_state': self._cached_energy_state.copy(),
            'last_update': time.time() - self._last_update_time,
            'update_count': self._update_count
        }
        
    def shutdown(self):
        """Shutdown background thread."""
        self._shutdown.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)