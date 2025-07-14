"""
Hardware Discovery & Adaptation System

Discovers hardware capabilities through minimal benchmarking and adapts
cognitive constants dynamically based on real performance. This enables
the same brain architecture to run optimally on any hardware from
Raspberry Pi to high-end GPU servers.

Biological inspiration: Real brains adapt their processing strategies
based on their actual computational capabilities during development.
"""

import time
import psutil
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# GPU detection
try:
    import torch
    GPU_AVAILABLE = torch.backends.mps.is_available() or torch.cuda.is_available()
    if GPU_AVAILABLE:
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
    else:
        device = 'cpu'
except ImportError:
    GPU_AVAILABLE = False
    device = 'cpu'


@dataclass
class HardwareProfile:
    """Discovered hardware capabilities and current performance state."""
    
    # Static capabilities (discovered once)
    cpu_cores: int
    total_memory_gb: float
    gpu_available: bool
    gpu_memory_gb: Optional[float]
    
    # Dynamic performance state (updated continuously)
    avg_cycle_time_ms: float
    memory_pressure: float  # 0.0-1.0
    cpu_utilization: float  # 0.0-1.0
    
    # Adaptive cognitive limits (computed from capabilities + performance)
    working_memory_limit: int
    similarity_search_limit: int
    batch_processing_threshold: int
    cognitive_energy_budget: int
    max_experiences_per_cycle: int
    
    # Performance tracking
    adaptation_count: int = 0
    last_adaptation_time: float = 0.0


class HardwareAdaptation:
    """
    Lightweight hardware discovery and dynamic cognitive adaptation.
    
    Strategy:
    1. Minimal startup benchmarking (< 1 second)
    2. Conservative initial limits based on discovered hardware
    3. Dynamic adaptation based on actual performance during operation
    4. Graceful degradation under pressure, scaling up when possible
    """
    
    def __init__(self):
        self.hardware_profile: Optional[HardwareProfile] = None
        self.performance_history = []  # Recent cycle times for adaptation
        self.adaptation_rate = 0.1  # How aggressively to adapt (conservative)
        self.min_samples_for_adaptation = 10  # Minimum cycles before adapting
        
        # Performance targets (biological constraints)
        self.target_cycle_time_ms = 50.0  # Ideal: 50ms cycles
        self.max_acceptable_cycle_time_ms = 100.0  # Upper bound: 100ms
        self.memory_pressure_threshold = 0.8  # Start adapting at 80% memory usage
        
        print("🔧 Hardware adaptation system initializing...")
        self._discover_hardware()
        print(f"🎯 Target: {self.target_cycle_time_ms}ms cycles, adapting dynamically")
    
    def _discover_hardware(self):
        """Minimal hardware benchmarking for ballpark capabilities."""
        print("🔍 Discovering hardware capabilities...")
        start_time = time.time()
        
        # Basic system info
        cpu_cores = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        total_memory_gb = memory_info.total / (1024**3)
        
        # Quick GPU check
        gpu_memory_gb = None
        if GPU_AVAILABLE:
            try:
                if device == 'mps':
                    # MPS doesn't expose memory info easily, estimate based on system
                    gpu_memory_gb = min(total_memory_gb * 0.6, 16.0)  # Rough estimate
                elif device == 'cuda':
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except Exception:
                gpu_memory_gb = None
        
        # Minimal computational benchmark (< 500ms)
        cycle_time_ms = self._quick_computation_benchmark()
        
        # Calculate initial conservative limits
        initial_limits = self._calculate_initial_limits(
            cpu_cores, total_memory_gb, GPU_AVAILABLE, gpu_memory_gb, cycle_time_ms
        )
        
        # Create hardware profile
        self.hardware_profile = HardwareProfile(
            cpu_cores=cpu_cores,
            total_memory_gb=total_memory_gb,
            gpu_available=GPU_AVAILABLE,
            gpu_memory_gb=gpu_memory_gb,
            avg_cycle_time_ms=cycle_time_ms,
            memory_pressure=0.0,
            cpu_utilization=0.0,
            **initial_limits
        )
        
        discovery_time = time.time() - start_time
        print(f"✅ Hardware discovered in {discovery_time:.3f}s:")
        print(f"   CPU: {cpu_cores} cores, RAM: {total_memory_gb:.1f}GB")
        print(f"   GPU: {'Yes' if GPU_AVAILABLE else 'No'}")
        print(f"   Initial cycle time: {cycle_time_ms:.1f}ms")
        print(f"   Working memory limit: {self.hardware_profile.working_memory_limit}")
        print(f"   Similarity search limit: {self.hardware_profile.similarity_search_limit}")
    
    def _quick_computation_benchmark(self) -> float:
        """Quick computational benchmark to estimate processing speed."""
        # Simple matrix operations similar to brain processing
        sizes_to_test = [100, 200, 500]  # Start small, scale up if fast
        
        for size in sizes_to_test:
            start_time = time.time()
            
            # Simulate similarity computation workload
            matrix_a = np.random.rand(size, 20)  # Experiences x features
            matrix_b = np.random.rand(20)        # Query vector
            
            # Similarity computation (cosine similarity)
            norms_a = np.linalg.norm(matrix_a, axis=1)
            norm_b = np.linalg.norm(matrix_b)
            similarities = np.dot(matrix_a, matrix_b) / (norms_a * norm_b)
            
            # Sort and select top results (typical brain operation)
            top_indices = np.argsort(similarities)[-10:]
            
            cycle_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Scale estimate to brain-sized workload (approximately)
            estimated_brain_cycle_time = cycle_time * (50 / size)  # Scale to ~50 experiences
            
            # If this size completes quickly, try larger size
            if cycle_time < 10.0 and size < sizes_to_test[-1]:
                continue
            else:
                return max(10.0, estimated_brain_cycle_time)  # Minimum 10ms estimate
        
        return 50.0  # Conservative fallback
    
    def _calculate_initial_limits(self, cpu_cores: int, memory_gb: float, 
                                gpu_available: bool, gpu_memory_gb: Optional[float], 
                                cycle_time_ms: float) -> Dict[str, int]:
        """Calculate conservative initial cognitive limits based on hardware."""
        
        # Memory-based limits (conservative: use 50% of available memory)
        available_memory_mb = memory_gb * 1024 * 0.5
        bytes_per_experience = 1024  # Rough estimate: 1KB per experience
        max_experiences_by_memory = int(available_memory_mb * 1024 / bytes_per_experience)
        
        # CPU-based limits
        cpu_multiplier = min(4.0, cpu_cores / 2.0)  # Conservative CPU utilization
        
        # Cycle time based limits (if slow, reduce computational load)
        if cycle_time_ms <= 30.0:
            performance_tier = "high"
            perf_multiplier = 2.0
        elif cycle_time_ms <= 60.0:
            performance_tier = "medium"
            perf_multiplier = 1.0
        else:
            performance_tier = "low"
            perf_multiplier = 0.5
        
        # GPU acceleration bonus
        gpu_multiplier = 2.0 if gpu_available else 1.0
        
        # Calculate limits
        base_working_memory = 15  # Human-like baseline
        base_similarity_search = 500
        base_batch_threshold = 50
        base_energy_budget = 20
        
        working_memory_limit = int(base_working_memory * cpu_multiplier * perf_multiplier)
        similarity_search_limit = int(base_similarity_search * gpu_multiplier * perf_multiplier)
        batch_threshold = int(base_batch_threshold * perf_multiplier)
        energy_budget = int(base_energy_budget * cpu_multiplier * perf_multiplier)
        
        # Memory constraints
        working_memory_limit = min(working_memory_limit, max_experiences_by_memory // 100)
        similarity_search_limit = min(similarity_search_limit, max_experiences_by_memory // 5)
        
        # Sanity bounds
        working_memory_limit = max(5, min(100, working_memory_limit))
        similarity_search_limit = max(100, min(10000, similarity_search_limit))
        batch_threshold = max(10, min(200, batch_threshold))
        energy_budget = max(5, min(100, energy_budget))
        
        max_experiences_per_cycle = working_memory_limit * 2  # Allow some headroom
        
        print(f"📊 Hardware tier: {performance_tier} (cycle: {cycle_time_ms:.1f}ms)")
        print(f"🧠 Initial cognitive limits: WM={working_memory_limit}, Search={similarity_search_limit}")
        
        return {
            'working_memory_limit': working_memory_limit,
            'similarity_search_limit': similarity_search_limit,
            'batch_processing_threshold': batch_threshold,
            'cognitive_energy_budget': energy_budget,
            'max_experiences_per_cycle': max_experiences_per_cycle
        }
    
    def record_cycle_performance(self, cycle_time_ms: float, memory_usage_mb: float = 0.0):
        """Record actual performance for dynamic adaptation."""
        if self.hardware_profile is None:
            return
        
        # Update performance history
        self.performance_history.append({
            'cycle_time_ms': cycle_time_ms,
            'memory_usage_mb': memory_usage_mb,
            'timestamp': time.time()
        })
        
        # Keep recent history only
        if len(self.performance_history) > 50:
            self.performance_history = self.performance_history[-25:]
        
        # Update current state
        self.hardware_profile.avg_cycle_time_ms = cycle_time_ms
        self.hardware_profile.memory_pressure = min(1.0, memory_usage_mb / (self.hardware_profile.total_memory_gb * 1024))
        self.hardware_profile.cpu_utilization = min(1.0, psutil.cpu_percent(interval=None) / 100.0)
        
        # Check if adaptation is needed
        if len(self.performance_history) >= self.min_samples_for_adaptation:
            self._consider_adaptation()
    
    def _consider_adaptation(self):
        """Consider adapting cognitive limits based on recent performance."""
        if self.hardware_profile is None:
            return
        
        current_time = time.time()
        
        # Don't adapt too frequently
        if current_time - self.hardware_profile.last_adaptation_time < 30.0:  # 30 second minimum
            return
        
        recent_cycles = [p['cycle_time_ms'] for p in self.performance_history[-10:]]
        avg_recent_cycle_time = np.mean(recent_cycles)
        
        # Determine if adaptation is needed
        performance_pressure = self._calculate_performance_pressure(avg_recent_cycle_time)
        
        if abs(performance_pressure) > 0.2:  # Significant pressure
            self._adapt_cognitive_limits(performance_pressure)
            self.hardware_profile.last_adaptation_time = current_time
            self.hardware_profile.adaptation_count += 1
    
    def _calculate_performance_pressure(self, avg_cycle_time_ms: float) -> float:
        """Calculate performance pressure (-1.0 to 1.0). Negative = under pressure, positive = room to grow."""
        target = self.target_cycle_time_ms
        max_acceptable = self.max_acceptable_cycle_time_ms
        
        if avg_cycle_time_ms > max_acceptable:
            # Severe pressure - reduce limits aggressively
            return -1.0
        elif avg_cycle_time_ms > target:
            # Moderate pressure - reduce limits proportionally
            pressure_ratio = (avg_cycle_time_ms - target) / (max_acceptable - target)
            return -pressure_ratio
        else:
            # Under target - room to increase limits
            # Be more conservative about increasing than decreasing
            improvement_ratio = (target - avg_cycle_time_ms) / target
            return min(0.5, improvement_ratio)  # Cap at 0.5 for conservative growth
    
    def _adapt_cognitive_limits(self, performance_pressure: float):
        """Adapt cognitive limits based on performance pressure."""
        if self.hardware_profile is None:
            return
        
        # Calculate adaptation factor
        adaptation_factor = 1.0 + (performance_pressure * self.adaptation_rate)
        
        # Apply adaptation to limits
        old_limits = {
            'working_memory': self.hardware_profile.working_memory_limit,
            'similarity_search': self.hardware_profile.similarity_search_limit,
            'cognitive_energy': self.hardware_profile.cognitive_energy_budget
        }
        
        # Adapt each limit with bounds checking
        self.hardware_profile.working_memory_limit = self._adapt_limit(
            self.hardware_profile.working_memory_limit, adaptation_factor, 5, 100
        )
        self.hardware_profile.similarity_search_limit = self._adapt_limit(
            self.hardware_profile.similarity_search_limit, adaptation_factor, 100, 10000
        )
        self.hardware_profile.cognitive_energy_budget = self._adapt_limit(
            self.hardware_profile.cognitive_energy_budget, adaptation_factor, 5, 100
        )
        self.hardware_profile.batch_processing_threshold = self._adapt_limit(
            self.hardware_profile.batch_processing_threshold, adaptation_factor, 10, 200
        )
        
        # Update derived limits
        self.hardware_profile.max_experiences_per_cycle = self.hardware_profile.working_memory_limit * 2
        
        # Log adaptation
        direction = "Scaling up" if performance_pressure > 0 else "Scaling down"
        print(f"🔄 {direction} cognitive limits (pressure: {performance_pressure:.2f}):")
        print(f"   Working memory: {old_limits['working_memory']} → {self.hardware_profile.working_memory_limit}")
        print(f"   Similarity search: {old_limits['similarity_search']} → {self.hardware_profile.similarity_search_limit}")
        print(f"   Cognitive energy: {old_limits['cognitive_energy']} → {self.hardware_profile.cognitive_energy_budget}")
    
    def _adapt_limit(self, current_value: int, adaptation_factor: float, min_val: int, max_val: int) -> int:
        """Adapt a single limit with bounds checking."""
        new_value = int(current_value * adaptation_factor)
        return max(min_val, min(max_val, new_value))
    
    def get_cognitive_limits(self) -> Dict[str, int]:
        """Get current adaptive cognitive limits for the brain systems."""
        if self.hardware_profile is None:
            # Fallback defaults
            return {
                'working_memory_limit': 15,
                'similarity_search_limit': 500,
                'batch_processing_threshold': 50,
                'cognitive_energy_budget': 20,
                'max_experiences_per_cycle': 30
            }
        
        return {
            'working_memory_limit': self.hardware_profile.working_memory_limit,
            'similarity_search_limit': self.hardware_profile.similarity_search_limit,
            'batch_processing_threshold': self.hardware_profile.batch_processing_threshold,
            'cognitive_energy_budget': self.hardware_profile.cognitive_energy_budget,
            'max_experiences_per_cycle': self.hardware_profile.max_experiences_per_cycle
        }
    
    def get_hardware_profile(self) -> Dict[str, Any]:
        """Get complete hardware profile for monitoring and debugging."""
        if self.hardware_profile is None:
            return {'status': 'not_initialized'}
        
        profile_dict = asdict(self.hardware_profile)
        
        # Add performance statistics
        if self.performance_history:
            recent_cycles = [p['cycle_time_ms'] for p in self.performance_history[-10:]]
            profile_dict['performance_stats'] = {
                'recent_avg_cycle_time_ms': np.mean(recent_cycles),
                'recent_min_cycle_time_ms': np.min(recent_cycles),
                'recent_max_cycle_time_ms': np.max(recent_cycles),
                'cycle_time_std_ms': np.std(recent_cycles),
                'samples_collected': len(self.performance_history)
            }
        
        return profile_dict
    
    def should_use_gpu_for_operation(self, operation_size: int) -> bool:
        """Determine if GPU should be used for a given operation size."""
        if not self.hardware_profile or not self.hardware_profile.gpu_available:
            return False
        
        # Use GPU if operation size is above the adaptive threshold
        return operation_size >= self.hardware_profile.batch_processing_threshold
    
    def get_optimal_batch_size(self, total_items: int) -> int:
        """Get optimal batch size for processing given total items."""
        if self.hardware_profile is None:
            return min(50, total_items)
        
        batch_size = self.hardware_profile.batch_processing_threshold
        
        # Adjust based on current performance pressure
        if self.hardware_profile.memory_pressure > 0.8:
            batch_size = int(batch_size * 0.7)  # Reduce batch size under memory pressure
        elif self.hardware_profile.avg_cycle_time_ms < self.target_cycle_time_ms * 0.7:
            batch_size = int(batch_size * 1.3)  # Increase if performing well
        
        return max(10, min(batch_size, total_items))


# Global hardware adaptation instance
_hardware_adaptation = None

def get_hardware_adaptation() -> HardwareAdaptation:
    """Get the global hardware adaptation instance."""
    global _hardware_adaptation
    if _hardware_adaptation is None:
        _hardware_adaptation = HardwareAdaptation()
    return _hardware_adaptation

def get_adaptive_cognitive_limits() -> Dict[str, int]:
    """Get current adaptive cognitive limits."""
    return get_hardware_adaptation().get_cognitive_limits()

def record_brain_cycle_performance(cycle_time_ms: float, memory_usage_mb: float = 0.0):
    """Record a brain cycle's performance for adaptation."""
    get_hardware_adaptation().record_cycle_performance(cycle_time_ms, memory_usage_mb)

def should_use_gpu_for_similarity_search(num_experiences: int) -> bool:
    """Check if GPU should be used for similarity search of given size."""
    return get_hardware_adaptation().should_use_gpu_for_operation(num_experiences)