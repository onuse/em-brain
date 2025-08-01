#!/usr/bin/env python3
"""
Cached Plan System - Decoupled Planning Phase 1

This system allows the brain to remain responsive while GPU simulates futures 
in the background. It maintains a cache of plans and their execution status.
"""

import time
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import torch
from concurrent.futures import Future

from .action_prediction_system import PredictiveAction
from .gpu_future_simulator import SimulatedAction


@dataclass
class CachedPlan:
    """A cached plan with metadata for execution tracking."""
    actions: List[Union[PredictiveAction, SimulatedAction]]
    creation_time: float
    confidence: float
    context_hash: int  # Hash of the field state when plan was created
    execution_step: int = 0
    execution_success: float = 1.0  # Running average of success
    total_steps: int = 0
    
    @property
    def age(self) -> float:
        """Age of the plan in seconds."""
        return time.time() - self.creation_time
    
    @property
    def is_stale(self) -> bool:
        """Check if plan is too old or has failed too much."""
        # Plans decay in usefulness over time
        if self.age > 10.0:  # 10 seconds max
            return True
        if self.execution_success < 0.5:  # Less than 50% success
            return True
        return False
    
    @property
    def adjusted_confidence(self) -> float:
        """Confidence adjusted for age and execution success."""
        age_factor = max(0.0, 1.0 - self.age / 10.0)
        return self.confidence * self.execution_success * age_factor
    
    def get_current_action(self) -> Optional[Union[PredictiveAction, SimulatedAction]]:
        """Get the current action in the plan sequence."""
        if self.execution_step < len(self.actions):
            return self.actions[self.execution_step]
        return None
    
    def advance(self, success: bool = True):
        """Advance to next step and update success tracking."""
        self.total_steps += 1
        if success:
            self.execution_success = (self.execution_success * (self.total_steps - 1) + 1.0) / self.total_steps
        else:
            self.execution_success = (self.execution_success * (self.total_steps - 1)) / self.total_steps
        self.execution_step += 1


class CachedPlanSystem:
    """
    Manages cached plans and background planning for responsive behavior.
    
    This is the key to decoupling: the brain can act immediately using cached
    plans while new plans are computed in the background.
    """
    
    def __init__(self, 
                 field_shape: tuple,
                 confidence_threshold: float = 0.3,
                 max_cache_size: int = 5):
        """
        Initialize the cached plan system.
        
        Args:
            field_shape: Shape of the unified field for context hashing
            confidence_threshold: Minimum confidence to use a cached plan
            max_cache_size: Maximum number of plans to cache
        """
        self.field_shape = field_shape
        self.confidence_threshold = confidence_threshold
        self.max_cache_size = max_cache_size
        
        # Current cached plans (sorted by confidence)
        self.cached_plans: List[CachedPlan] = []
        
        # Background planning future
        self.background_future: Optional[Future] = None
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.plan_executions = 0
        self.plan_completions = 0
        
    def _compute_context_hash(self, field: torch.Tensor) -> int:
        """
        Compute a hash of the current field state for context matching.
        
        This is a simplified version - in practice we'd want a more
        sophisticated similarity metric.
        """
        # Simple approach: sample a few key points from the field
        # This avoids the MPS avg_pool3d issue
        key_points = []
        
        # Sample 8 corners of the field
        for i in [0, -1]:
            for j in [0, -1]:
                for k in [0, -1]:
                    # Get mean of feature channels at this point
                    point_value = field[i, j, k].mean().item()
                    key_points.append(round(point_value, 2))
        
        # Also sample center point
        mid = field.shape[0] // 2
        center_value = field[mid, mid, mid].mean().item()
        key_points.append(round(center_value, 2))
        
        # Convert to hash
        return hash(tuple(key_points))
    
    def get_cached_action(self, 
                         current_field: torch.Tensor,
                         min_confidence: Optional[float] = None) -> Optional[Union[PredictiveAction, SimulatedAction]]:
        """
        Get an action from cache if available and suitable.
        
        Args:
            current_field: Current brain field state
            min_confidence: Override confidence threshold
            
        Returns:
            Cached action if available, None otherwise
        """
        if not self.cached_plans:
            self.cache_misses += 1
            return None
        
        min_conf = min_confidence or self.confidence_threshold
        current_context = self._compute_context_hash(current_field)
        
        # Check for background planning completion
        if self.background_future and self.background_future.done():
            try:
                new_plan = self.background_future.result(timeout=0.01)
                self._add_plan_to_cache(new_plan)
                self.background_future = None
            except:
                # Background planning failed, continue with cache
                self.background_future = None
        
        # Find best matching plan
        best_plan = None
        best_score = 0.0
        
        for plan in self.cached_plans:
            if plan.is_stale:
                continue
                
            # Simple context matching (could be much more sophisticated)
            context_similarity = 1.0 if plan.context_hash == current_context else 0.7
            score = plan.adjusted_confidence * context_similarity
            
            if score > best_score and score >= min_conf:
                best_score = score
                best_plan = plan
        
        if best_plan:
            self.cache_hits += 1
            action = best_plan.get_current_action()
            if action:
                return action
            else:
                # Plan completed, remove it
                self.cached_plans.remove(best_plan)
                self.plan_completions += 1
        
        self.cache_misses += 1
        return None
    
    def execute_cached_action(self, 
                            action: Union[PredictiveAction, SimulatedAction],
                            success: bool = True):
        """
        Record execution of a cached action.
        
        Args:
            action: The action that was executed
            success: Whether execution was successful
        """
        self.plan_executions += 1
        
        # Find which plan this action came from
        for plan in self.cached_plans:
            if plan.execution_step < len(plan.actions):
                if plan.actions[plan.execution_step] == action:
                    plan.advance(success)
                    break
    
    def _add_plan_to_cache(self, plan_data: Dict[str, Any]):
        """
        Add a new plan to the cache.
        
        Args:
            plan_data: Dictionary containing 'actions', 'confidence', 'context_hash'
        """
        new_plan = CachedPlan(
            actions=plan_data['actions'],
            creation_time=time.time(),
            confidence=plan_data['confidence'],
            context_hash=plan_data['context_hash']
        )
        
        self.cached_plans.append(new_plan)
        
        # Sort by confidence and trim to max size
        self.cached_plans.sort(key=lambda p: p.adjusted_confidence, reverse=True)
        if len(self.cached_plans) > self.max_cache_size:
            self.cached_plans = self.cached_plans[:self.max_cache_size]
    
    def update_background_planning(self, future: Future):
        """
        Update the background planning future.
        
        Args:
            future: New background planning task
        """
        # Cancel old future if still running
        if self.background_future and not self.background_future.done():
            self.background_future.cancel()
        
        self.background_future = future
    
    def cleanup_stale_plans(self):
        """Remove stale plans from cache."""
        self.cached_plans = [p for p in self.cached_plans if not p.is_stale]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        return {
            'cache_size': len(self.cached_plans),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'plan_executions': self.plan_executions,
            'plan_completions': self.plan_completions,
            'background_planning': self.background_future is not None
        }
    
    def reset_statistics(self):
        """Reset performance counters."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.plan_executions = 0
        self.plan_completions = 0