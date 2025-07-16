"""
Tensor Optimization Integration Guide

This module provides easy integration of tensor optimizations
into existing brain systems without breaking existing functionality.
"""

from typing import Dict, List, Any, Optional, Type
import time
from contextlib import contextmanager

from .tensor_optimization import TensorOptimizationCoordinator, TensorOptimizationConfig
from .batch_processor import BatchExperienceProcessor
# Activation dynamics archived with experience-based architecture
# from ..activation.optimized_dynamics import OptimizedActivationDynamics


class OptimizedBrainMixin:
    """
    Mixin class that adds tensor optimization capabilities to MinimalBrain.
    
    Usage:
        # Option 1: Apply at initialization
        brain = MinimalBrain()
        brain = apply_tensor_optimizations(brain)
        
        # Option 2: Use optimized brain class
        brain = OptimizedMinimalBrain()
    """
    
    def __init__(self, *args, enable_tensor_optimization: bool = True, 
                 optimization_config: Optional[TensorOptimizationConfig] = None, **kwargs):
        """Initialize with tensor optimization support."""
        super().__init__(*args, **kwargs)
        
        if enable_tensor_optimization:
            self._apply_tensor_optimizations(optimization_config)
    
    def _apply_tensor_optimizations(self, config: Optional[TensorOptimizationConfig] = None):
        """Apply tensor optimizations to the brain."""
        print("⚡ Applying tensor optimizations...")
        
        # Create optimization coordinator
        self.tensor_optimizer = TensorOptimizationCoordinator(config)
        
        # Replace activation dynamics with optimized version if using traditional activation
        if not self.use_utility_based_activation:
            print("   Upgrading activation dynamics to optimized version")
            self.activation_dynamics = OptimizedActivationDynamics(
                use_gpu=True,
                use_mixed_precision=True,
                initial_capacity=1000
            )
        
        # Wrap store_experience for batching
        self._original_store_experience = self.store_experience
        self.store_experience = self._optimized_store_experience
        
        # Create batched similarity search
        self._batched_similarity = self.tensor_optimizer.create_batched_similarity_search(
            self.similarity_engine
        )
        
        print("   ✅ Tensor optimizations applied")
    
    def _optimized_store_experience(self, sensory_input: List[float], 
                                   action_taken: List[float], 
                                   outcome: List[float], 
                                   predicted_action: List[float] = None) -> str:
        """Optimized experience storage with batching."""
        
        # Check if we should use batching
        if hasattr(self, 'tensor_optimizer') and self.tensor_optimizer.config.enable_batching:
            return self._batched_store_experience(
                sensory_input, action_taken, outcome, predicted_action
            )
        else:
            return self._original_store_experience(
                sensory_input, action_taken, outcome, predicted_action
            )
    
    def _batched_store_experience(self, sensory_input: List[float],
                                 action_taken: List[float],
                                 outcome: List[float],
                                 predicted_action: List[float] = None) -> str:
        """Store experience with batch optimization."""
        
        # Create experience data
        experience_data = {
            'sensory_input': sensory_input,
            'action_taken': action_taken,
            'outcome': outcome,
            'predicted_action': predicted_action,
            'timestamp': time.time()
        }
        
        # Add to batch
        self.tensor_optimizer.add_experience_to_batch(experience_data)
        
        # Check if we should process batch
        if self.tensor_optimizer.should_process_batch():
            batch_start = time.time()
            batch = self.tensor_optimizer.get_experience_batch()
            
            # Process batch
            experience_ids = []
            for batch_exp in batch:
                exp_id = self._original_store_experience(
                    batch_exp['sensory_input'],
                    batch_exp['action_taken'],
                    batch_exp['outcome'],
                    batch_exp.get('predicted_action')
                )
                experience_ids.append(exp_id)
            
            batch_time = time.time() - batch_start
            
            # Record performance
            self.tensor_optimizer.record_batch_processing(len(batch), batch_time)
            
            # Return the ID for the current experience (last in batch)
            return experience_ids[-1]
        else:
            # Generate experience ID for consistency but don't store yet
            # This is a limitation - we need to store immediately for API consistency
            return self._original_store_experience(sensory_input, action_taken, outcome, predicted_action)
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get tensor optimization statistics."""
        if hasattr(self, 'tensor_optimizer'):
            return self.tensor_optimizer.get_optimization_stats()
        else:
            return {'error': 'Tensor optimization not enabled'}
    
    def suggest_optimizations(self) -> List[str]:
        """Get optimization suggestions."""
        if hasattr(self, 'tensor_optimizer'):
            return self.tensor_optimizer.suggest_optimizations()
        else:
            return ['Enable tensor optimization with apply_tensor_optimizations()']


def apply_tensor_optimizations(brain_instance, 
                              config: Optional[TensorOptimizationConfig] = None) -> 'MinimalBrain':
    """
    Apply tensor optimizations to an existing brain instance.
    
    Args:
        brain_instance: Existing MinimalBrain instance
        config: Optional optimization configuration
        
    Returns:
        Brain instance with optimizations applied
    """
    print("⚡ Applying tensor optimizations to existing brain...")
    
    # Add optimization methods to the brain instance
    brain_instance.tensor_optimizer = TensorOptimizationCoordinator(config)
    
    # Replace activation dynamics if needed
    if not brain_instance.use_utility_based_activation:
        print("   Upgrading activation dynamics...")
        brain_instance.activation_dynamics = OptimizedActivationDynamics(
            use_gpu=True,
            use_mixed_precision=True,
            initial_capacity=len(brain_instance.experience_storage._experiences) + 500
        )
    
    # Wrap methods for optimization
    brain_instance._original_store_experience = brain_instance.store_experience
    
    def optimized_store_experience(sensory_input, action_taken, outcome, predicted_action=None):
        return _optimized_store_wrapper(
            brain_instance, sensory_input, action_taken, outcome, predicted_action
        )
    
    brain_instance.store_experience = optimized_store_experience
    
    # Add optimization methods
    brain_instance.get_optimization_statistics = lambda: brain_instance.tensor_optimizer.get_optimization_stats()
    brain_instance.suggest_optimizations = lambda: brain_instance.tensor_optimizer.suggest_optimizations()
    
    print("   ✅ Optimizations applied to existing brain")
    return brain_instance


def _optimized_store_wrapper(brain, sensory_input, action_taken, outcome, predicted_action):
    """Wrapper for optimized experience storage."""
    
    # For now, use direct storage to maintain API compatibility
    # In a full implementation, this would implement intelligent batching
    single_start = time.time()
    exp_id = brain._original_store_experience(sensory_input, action_taken, outcome, predicted_action)
    single_time = time.time() - single_start
    
    # Record performance for optimization analysis
    brain.tensor_optimizer.record_single_processing(single_time)
    
    return exp_id


@contextmanager
def optimized_experience_batch(brain_instance, max_batch_size: int = 20):
    """
    Context manager for manual batch processing.
    
    Usage:
        with optimized_experience_batch(brain) as batch:
            for experience in experiences:
                batch.add_experience(experience)
        # Batch is automatically processed when exiting context
    """
    
    class BatchContext:
        def __init__(self, brain, max_size):
            self.brain = brain
            self.max_size = max_size
            self.batch = []
        
        def add_experience(self, sensory_input, action_taken, outcome, predicted_action=None):
            """Add experience to batch."""
            self.batch.append({
                'sensory_input': sensory_input,
                'action_taken': action_taken,
                'outcome': outcome,
                'predicted_action': predicted_action
            })
            
            # Auto-flush if batch is full
            if len(self.batch) >= self.max_size:
                self.flush()
        
        def flush(self):
            """Process current batch."""
            if not self.batch:
                return
            
            batch_start = time.time()
            
            for exp in self.batch:
                if hasattr(self.brain, '_original_store_experience'):
                    self.brain._original_store_experience(
                        exp['sensory_input'],
                        exp['action_taken'],
                        exp['outcome'],
                        exp.get('predicted_action')
                    )
                else:
                    self.brain.store_experience(
                        exp['sensory_input'],
                        exp['action_taken'],
                        exp['outcome'],
                        exp.get('predicted_action')
                    )
            
            batch_time = time.time() - batch_start
            
            print(f"📦 Processed batch of {len(self.batch)} experiences in {batch_time*1000:.1f}ms "
                  f"({batch_time/len(self.batch)*1000:.1f}ms each)")
            
            self.batch.clear()
    
    context = BatchContext(brain_instance, max_batch_size)
    try:
        yield context
    finally:
        context.flush()  # Process any remaining experiences


class OptimizationProfiler:
    """Profiles brain operations to identify optimization opportunities."""
    
    def __init__(self):
        self.operation_times = {}
        self.call_counts = {}
        self.tensor_events = []
    
    def profile_operation(self, operation_name: str):
        """Decorator to profile an operation."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                if operation_name not in self.operation_times:
                    self.operation_times[operation_name] = []
                    self.call_counts[operation_name] = 0
                
                self.operation_times[operation_name].append(elapsed)
                self.call_counts[operation_name] += 1
                
                return result
            return wrapper
        return decorator
    
    def record_tensor_event(self, event_type: str, details: Dict[str, Any]):
        """Record a tensor-related event."""
        self.tensor_events.append({
            'type': event_type,
            'timestamp': time.time(),
            'details': details
        })
    
    def get_profile_report(self) -> Dict[str, Any]:
        """Generate profiling report."""
        report = {
            'operations': {},
            'tensor_events': len(self.tensor_events),
            'total_operations': sum(self.call_counts.values())
        }
        
        for op_name, times in self.operation_times.items():
            report['operations'][op_name] = {
                'call_count': self.call_counts[op_name],
                'total_time': sum(times),
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'total_time_percent': sum(times) / sum(sum(t) for t in self.operation_times.values()) * 100
            }
        
        return report
    
    def suggest_optimizations_from_profile(self) -> List[str]:
        """Suggest optimizations based on profiling data."""
        suggestions = []
        report = self.get_profile_report()
        
        # Find operations that take the most time
        operations = report['operations']
        if operations:
            slowest_op = max(operations.items(), key=lambda x: x[1]['total_time'])
            suggestions.append(f"Optimize '{slowest_op[0]}' - takes {slowest_op[1]['total_time_percent']:.1f}% of total time")
        
        # Check for frequent operations
        if operations:
            most_frequent = max(operations.items(), key=lambda x: x[1]['call_count'])
            if most_frequent[1]['call_count'] > 100:
                suggestions.append(f"Consider batching '{most_frequent[0]}' - called {most_frequent[1]['call_count']} times")
        
        # Check tensor events
        if len(self.tensor_events) > 20:
            suggestions.append(f"High tensor activity ({len(self.tensor_events)} events) - consider incremental updates")
        
        return suggestions


# Convenience function for quick optimization
def quick_optimize_brain(brain_instance, profile: bool = False) -> 'MinimalBrain':
    """
    Quick optimization application with sensible defaults.
    
    Args:
        brain_instance: Brain to optimize
        profile: Whether to enable profiling
        
    Returns:
        Optimized brain instance
    """
    config = TensorOptimizationConfig(
        enable_batching=True,
        min_batch_size=5,
        max_batch_size=15,
        max_batch_delay_ms=50.0,
        adaptive_batch_sizing=True
    )
    
    optimized_brain = apply_tensor_optimizations(brain_instance, config)
    
    if profile:
        optimized_brain.profiler = OptimizationProfiler()
        print("🔍 Profiling enabled for optimization analysis")
    
    return optimized_brain