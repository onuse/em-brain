"""
Tensor Optimization Integration

Coordinates batch processing and incremental updates across all brain systems
to minimize GPU overhead and tensor rebuilding.
"""

from typing import Dict, List, Any, Optional
import time
from dataclasses import dataclass
from collections import deque

from ..utils.batch_processor import BatchExperienceProcessor, BatchedSimilaritySearch


@dataclass
class TensorOptimizationConfig:
    """Configuration for tensor optimization."""
    enable_batching: bool = True
    min_batch_size: int = 5
    max_batch_size: int = 20
    max_batch_delay_ms: float = 100.0
    enable_incremental_updates: bool = True
    tensor_sync_interval_ms: float = 100.0
    adaptive_batch_sizing: bool = True
    gpu_warmup_size: int = 100  # Pre-allocate for this many experiences


class TensorOptimizationCoordinator:
    """
    Coordinates tensor optimization across all brain systems.
    
    Responsibilities:
    1. Batch experience processing
    2. Coordinate GPU upgrades
    3. Manage tensor lifecycle
    4. Monitor performance
    5. Adapt optimization parameters
    """
    
    def __init__(self, config: Optional[TensorOptimizationConfig] = None):
        """Initialize optimization coordinator."""
        self.config = config or TensorOptimizationConfig()
        
        # Batch processors for each system
        self.experience_batcher = BatchExperienceProcessor(
            min_batch_size=self.config.min_batch_size,
            max_batch_size=self.config.max_batch_size,
            max_delay_ms=self.config.max_batch_delay_ms,
            adaptive=self.config.adaptive_batch_sizing
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=100)
        self.optimization_events = []
        self.total_tensor_rebuilds = 0
        self.total_incremental_updates = 0
        
        # System state
        self.systems_on_gpu = {
            'similarity': False,
            'activation': False,
            'pattern': False
        }
        
        # Optimization state
        self.last_optimization_check = time.time()
        self.optimization_check_interval = 5.0  # Check every 5 seconds
        
        print(f"🎯 TensorOptimizationCoordinator initialized")
        print(f"   Batching: {self.config.enable_batching} "
              f"(size: {self.config.min_batch_size}-{self.config.max_batch_size})")
        print(f"   Incremental updates: {self.config.enable_incremental_updates}")
        print(f"   Adaptive optimization: {self.config.adaptive_batch_sizing}")
    
    def should_process_batch(self) -> bool:
        """Check if we should process the current batch."""
        if not self.config.enable_batching:
            return False
        return self.experience_batcher.should_process_batch()
    
    def add_experience_to_batch(self, experience_data: Dict[str, Any]):
        """Add experience to batch queue."""
        if self.config.enable_batching:
            self.experience_batcher.add_experience(experience_data)
    
    def get_experience_batch(self) -> List[Dict[str, Any]]:
        """Get current batch of experiences."""
        return self.experience_batcher.get_batch()
    
    def record_batch_processing(self, batch_size: int, processing_time: float,
                               tensor_rebuilds: int = 0):
        """Record batch processing performance."""
        self.experience_batcher.record_batch_performance(batch_size, processing_time)
        
        # Track tensor rebuilds
        if tensor_rebuilds > 0:
            self.total_tensor_rebuilds += tensor_rebuilds
            self.optimization_events.append({
                'type': 'tensor_rebuild',
                'count': tensor_rebuilds,
                'batch_size': batch_size,
                'timestamp': time.time()
            })
        
        # Record performance
        self.performance_history.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'time_per_experience': processing_time / batch_size if batch_size > 0 else 0,
            'tensor_rebuilds': tensor_rebuilds,
            'timestamp': time.time()
        })
        
        # Check if we need to adapt optimization
        current_time = time.time()
        if current_time - self.last_optimization_check > self.optimization_check_interval:
            self._adapt_optimization_parameters()
            self.last_optimization_check = current_time
    
    def record_single_processing(self, processing_time: float):
        """Record single experience processing for comparison."""
        self.experience_batcher.record_single_performance(processing_time)
    
    def record_incremental_update(self, system: str, update_count: int):
        """Record incremental tensor update."""
        self.total_incremental_updates += update_count
        self.optimization_events.append({
            'type': 'incremental_update',
            'system': system,
            'count': update_count,
            'timestamp': time.time()
        })
    
    def record_gpu_upgrade(self, system: str):
        """Record when a system upgrades to GPU."""
        self.systems_on_gpu[system] = True
        self.optimization_events.append({
            'type': 'gpu_upgrade',
            'system': system,
            'timestamp': time.time()
        })
        print(f"🚀 {system} system upgraded to GPU")
    
    def _adapt_optimization_parameters(self):
        """Adapt optimization parameters based on performance."""
        if len(self.performance_history) < 10:
            return
        
        # Analyze recent performance
        recent_perf = list(self.performance_history)[-20:]
        avg_time_per_exp = sum(p['time_per_experience'] for p in recent_perf) / len(recent_perf)
        avg_batch_size = sum(p['batch_size'] for p in recent_perf) / len(recent_perf)
        recent_rebuilds = sum(p['tensor_rebuilds'] for p in recent_perf)
        
        # Check if we're having too many rebuilds
        if recent_rebuilds > 5:
            # Increase batch size to reduce rebuild frequency
            new_min = min(self.config.min_batch_size + 2, 15)
            new_max = min(self.config.max_batch_size + 5, 50)
            
            if new_min != self.config.min_batch_size:
                print(f"🎯 Adapting batch size to reduce rebuilds: "
                      f"{self.config.min_batch_size}-{self.config.max_batch_size} → "
                      f"{new_min}-{new_max}")
                self.config.min_batch_size = new_min
                self.config.max_batch_size = new_max
                self.experience_batcher.min_batch_size = new_min
                self.experience_batcher.max_batch_size = new_max
        
        # Check if processing is too slow
        if avg_time_per_exp > 0.01:  # More than 10ms per experience
            # Increase batching to improve throughput
            new_delay = max(self.config.max_batch_delay_ms * 0.8, 50)
            if new_delay != self.config.max_batch_delay_ms:
                print(f"🎯 Reducing batch delay for better throughput: "
                      f"{self.config.max_batch_delay_ms}ms → {new_delay}ms")
                self.config.max_batch_delay_ms = new_delay
                self.experience_batcher.max_delay_ms = new_delay
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        batch_stats = self.experience_batcher.get_statistics()
        
        # Calculate optimization effectiveness
        total_updates = self.total_tensor_rebuilds + self.total_incremental_updates
        incremental_ratio = (self.total_incremental_updates / total_updates 
                           if total_updates > 0 else 0)
        
        # Performance trends
        if len(self.performance_history) > 10:
            recent = list(self.performance_history)[-10:]
            older = list(self.performance_history)[-20:-10]
            
            recent_avg_time = sum(p['time_per_experience'] for p in recent) / len(recent)
            older_avg_time = sum(p['time_per_experience'] for p in older) / len(older)
            performance_trend = (older_avg_time - recent_avg_time) / older_avg_time if older_avg_time > 0 else 0
        else:
            performance_trend = 0
        
        return {
            'batch_processing': batch_stats,
            'tensor_optimization': {
                'total_rebuilds': self.total_tensor_rebuilds,
                'total_incremental_updates': self.total_incremental_updates,
                'incremental_ratio': incremental_ratio,
                'systems_on_gpu': sum(self.systems_on_gpu.values()),
                'gpu_systems': [s for s, on_gpu in self.systems_on_gpu.items() if on_gpu]
            },
            'performance': {
                'performance_trend': performance_trend,  # Positive = improving
                'avg_time_per_experience_ms': (sum(p['time_per_experience'] for p in self.performance_history) / 
                                              len(self.performance_history) * 1000 
                                              if self.performance_history else 0),
                'optimization_events': len(self.optimization_events)
            },
            'current_config': {
                'batching_enabled': self.config.enable_batching,
                'batch_size_range': f"{self.config.min_batch_size}-{self.config.max_batch_size}",
                'max_delay_ms': self.config.max_batch_delay_ms,
                'incremental_updates': self.config.enable_incremental_updates
            }
        }
    
    def create_batched_similarity_search(self, similarity_engine):
        """Create a batched similarity search wrapper."""
        return BatchedSimilaritySearch(similarity_engine)
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations based on current performance."""
        suggestions = []
        
        stats = self.get_optimization_stats()
        
        # Check tensor rebuild ratio
        if stats['tensor_optimization']['incremental_ratio'] < 0.8:
            suggestions.append("Enable incremental tensor updates to reduce rebuilds")
        
        # Check GPU utilization
        if stats['tensor_optimization']['systems_on_gpu'] < 3:
            gpu_missing = [s for s, on_gpu in self.systems_on_gpu.items() if not on_gpu]
            suggestions.append(f"Systems not using GPU: {', '.join(gpu_missing)}")
        
        # Check batch efficiency
        batch_efficiency = stats['batch_processing'].get('efficiency_gain', 1.0)
        if batch_efficiency < 1.5 and self.config.enable_batching:
            suggestions.append("Batch processing efficiency is low, consider larger batch sizes")
        
        # Check performance trend
        perf_trend = stats['performance']['performance_trend']
        if perf_trend < -0.1:  # Performance getting worse
            suggestions.append("Performance degrading - check for memory leaks or inefficient operations")
        
        return suggestions


class TensorLifecycleManager:
    """
    Manages tensor lifecycle to minimize rebuilds and optimize memory usage.
    """
    
    def __init__(self):
        """Initialize tensor lifecycle manager."""
        self.tensor_registry = {}  # Track all managed tensors
        self.tensor_versions = {}  # Track tensor versions
        self.rebuild_history = deque(maxlen=100)
        
        print("🔄 TensorLifecycleManager initialized")
    
    def register_tensor(self, name: str, initial_capacity: int, 
                       growth_factor: float = 1.5):
        """Register a tensor for lifecycle management."""
        self.tensor_registry[name] = {
            'capacity': initial_capacity,
            'size': 0,
            'growth_factor': growth_factor,
            'rebuilds': 0,
            'last_rebuild': None,
            'created': time.time()
        }
        self.tensor_versions[name] = 0
        print(f"   Registered tensor '{name}' with capacity {initial_capacity}")
    
    def check_capacity(self, name: str, required_size: int) -> bool:
        """Check if tensor has sufficient capacity."""
        if name not in self.tensor_registry:
            return False
        
        info = self.tensor_registry[name]
        return required_size <= info['capacity']
    
    def calculate_new_capacity(self, name: str, required_size: int) -> int:
        """Calculate new capacity for tensor resize."""
        if name not in self.tensor_registry:
            return required_size
        
        info = self.tensor_registry[name]
        new_capacity = info['capacity']
        
        while new_capacity < required_size:
            new_capacity = int(new_capacity * info['growth_factor'])
        
        return new_capacity
    
    def record_rebuild(self, name: str, old_capacity: int, new_capacity: int,
                      rebuild_time: float):
        """Record a tensor rebuild event."""
        if name in self.tensor_registry:
            info = self.tensor_registry[name]
            info['capacity'] = new_capacity
            info['rebuilds'] += 1
            info['last_rebuild'] = time.time()
            self.tensor_versions[name] += 1
        
        self.rebuild_history.append({
            'tensor': name,
            'old_capacity': old_capacity,
            'new_capacity': new_capacity,
            'rebuild_time': rebuild_time,
            'timestamp': time.time()
        })
        
        print(f"🔄 Tensor '{name}' rebuilt: {old_capacity} → {new_capacity} "
              f"({rebuild_time*1000:.1f}ms)")
    
    def get_lifecycle_stats(self) -> Dict[str, Any]:
        """Get tensor lifecycle statistics."""
        total_rebuilds = sum(info['rebuilds'] for info in self.tensor_registry.values())
        
        recent_rebuilds = [r for r in self.rebuild_history 
                          if time.time() - r['timestamp'] < 60]  # Last minute
        
        return {
            'registered_tensors': len(self.tensor_registry),
            'total_rebuilds': total_rebuilds,
            'recent_rebuilds': len(recent_rebuilds),
            'tensors': {
                name: {
                    'capacity': info['capacity'],
                    'rebuilds': info['rebuilds'],
                    'age_seconds': time.time() - info['created'],
                    'version': self.tensor_versions.get(name, 0)
                }
                for name, info in self.tensor_registry.items()
            }
        }