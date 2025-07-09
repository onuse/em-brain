"""
Adaptive Execution Engine - Intelligent CPU/GPU switching for optimal performance.

This engine automatically decides whether to use CPU or GPU based on:
- Dataset size
- Historical performance
- Current system load
- Prediction complexity

The goal is to get the best performance without jarring handovers.
"""

import time
import torch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue


class ExecutionMethod(Enum):
    """Available execution methods."""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"


@dataclass
class PerformanceMetric:
    """Performance measurement for a specific workload."""
    method: ExecutionMethod
    dataset_size: int
    traversal_count: int
    execution_time: float
    success: bool
    timestamp: float


class AdaptiveExecutionEngine:
    """
    Intelligent CPU/GPU switching engine for optimal performance.
    
    This engine learns from historical performance to make smart decisions
    about when to use CPU vs GPU execution without jarring handovers.
    """
    
    def __init__(self, 
                 gpu_threshold_nodes: int = 50,
                 cpu_threshold_nodes: int = 10,
                 learning_rate: float = 0.1,
                 performance_history_size: int = 100,
                 maturity_based_scaling: bool = True):
        """
        Initialize adaptive execution engine.
        
        Args:
            gpu_threshold_nodes: Minimum nodes to consider GPU (for mature brain)
            cpu_threshold_nodes: Maximum nodes to force CPU (for mature brain)
            learning_rate: How quickly to adapt thresholds
            performance_history_size: How many measurements to remember
            maturity_based_scaling: Whether to scale thresholds based on brain maturity
        """
        # Base thresholds for mature brain
        self.base_gpu_threshold = gpu_threshold_nodes
        self.base_cpu_threshold = cpu_threshold_nodes
        
        # Current adaptive thresholds
        self.gpu_threshold_nodes = gpu_threshold_nodes
        self.cpu_threshold_nodes = cpu_threshold_nodes
        
        self.learning_rate = learning_rate
        self.performance_history_size = performance_history_size
        self.maturity_based_scaling = maturity_based_scaling
        
        # Performance tracking
        self.performance_history: List[PerformanceMetric] = []
        self.method_stats = {
            ExecutionMethod.CPU: {
                'total_time': 0.0,
                'total_calls': 0,
                'avg_time': 0.0,
                'success_rate': 1.0
            },
            ExecutionMethod.GPU: {
                'total_time': 0.0,
                'total_calls': 0,
                'avg_time': 0.0,
                'success_rate': 1.0
            }
        }
        
        # Decision caching to avoid constant switching
        self.decision_cache = {}
        self.cache_timeout = 1.0  # seconds
        
        # Thread-safe performance logging
        self.performance_lock = threading.Lock()
        
        print(f"AdaptiveExecutionEngine initialized:")
        print(f"  CPU preferred: ≤{cpu_threshold_nodes} nodes")
        print(f"  GPU preferred: ≥{gpu_threshold_nodes} nodes")
        print(f"  Adaptive zone: {cpu_threshold_nodes}-{gpu_threshold_nodes} nodes")
    
    def choose_execution_method(self, 
                              dataset_size: int,
                              traversal_count: int = 1,
                              complexity_hint: str = "normal") -> ExecutionMethod:
        """
        Choose the optimal execution method based on workload characteristics.
        
        Args:
            dataset_size: Number of nodes in dataset
            traversal_count: Number of traversals to perform
            complexity_hint: "simple", "normal", "complex"
            
        Returns:
            ExecutionMethod to use
        """
        # Create cache key
        cache_key = (dataset_size, traversal_count, complexity_hint)
        
        # Check cache first
        if cache_key in self.decision_cache:
            cached_decision, timestamp = self.decision_cache[cache_key]
            if time.time() - timestamp < self.cache_timeout:
                return cached_decision
        
        # Make decision based on current rules
        decision = self._make_decision(dataset_size, traversal_count, complexity_hint)
        
        # Cache decision
        self.decision_cache[cache_key] = (decision, time.time())
        
        return decision
    
    def _make_decision(self, 
                      dataset_size: int,
                      traversal_count: int,
                      complexity_hint: str) -> ExecutionMethod:
        """
        Make intelligent decision about execution method.
        
        This is where the adaptive logic lives.
        """
        # Clear rules for extreme cases
        if dataset_size <= self.cpu_threshold_nodes:
            return ExecutionMethod.CPU
        
        if dataset_size >= self.gpu_threshold_nodes:
            return ExecutionMethod.GPU
        
        # Adaptive zone - use historical performance
        total_workload = dataset_size * traversal_count
        
        # Get historical performance for similar workloads
        cpu_performance = self._get_historical_performance(
            ExecutionMethod.CPU, dataset_size, traversal_count
        )
        gpu_performance = self._get_historical_performance(
            ExecutionMethod.GPU, dataset_size, traversal_count
        )
        
        # If we have good historical data, use it
        if cpu_performance and gpu_performance:
            if cpu_performance < gpu_performance * 0.8:  # 20% buffer to avoid oscillation
                return ExecutionMethod.CPU
            else:
                return ExecutionMethod.GPU
        
        # Fallback rules based on workload characteristics
        if complexity_hint == "simple" and total_workload < 5000:
            return ExecutionMethod.CPU
        elif complexity_hint == "complex" or total_workload > 10000:
            return ExecutionMethod.GPU
        else:
            # For medium workloads, prefer GPU (it's the future)
            return ExecutionMethod.GPU
    
    def _get_historical_performance(self, 
                                  method: ExecutionMethod,
                                  dataset_size: int,
                                  traversal_count: int) -> Optional[float]:
        """
        Get historical performance for similar workloads.
        
        Returns average execution time for similar workloads, or None if no data.
        """
        similar_metrics = []
        
        for metric in self.performance_history:
            if (metric.method == method and 
                metric.success and
                abs(metric.dataset_size - dataset_size) < dataset_size * 0.3 and
                abs(metric.traversal_count - traversal_count) < max(1, traversal_count * 0.5)):
                similar_metrics.append(metric.execution_time)
        
        if similar_metrics:
            return sum(similar_metrics) / len(similar_metrics)
        return None
    
    def record_performance(self, 
                          method: ExecutionMethod,
                          dataset_size: int,
                          traversal_count: int,
                          execution_time: float,
                          success: bool = True):
        """
        Record performance measurement for future decisions.
        
        This is how the engine learns and adapts.
        """
        with self.performance_lock:
            # Add to history
            metric = PerformanceMetric(
                method=method,
                dataset_size=dataset_size,
                traversal_count=traversal_count,
                execution_time=execution_time,
                success=success,
                timestamp=time.time()
            )
            
            self.performance_history.append(metric)
            
            # Keep history size manageable
            if len(self.performance_history) > self.performance_history_size:
                self.performance_history.pop(0)
            
            # Update method statistics
            stats = self.method_stats[method]
            stats['total_time'] += execution_time
            stats['total_calls'] += 1
            stats['avg_time'] = stats['total_time'] / stats['total_calls']
            
            # Update success rate with exponential moving average
            if success:
                stats['success_rate'] = stats['success_rate'] * 0.9 + 0.1
            else:
                stats['success_rate'] = stats['success_rate'] * 0.9
            
            # Adaptive threshold adjustment
            self._adapt_thresholds(metric)
    
    def _adapt_thresholds(self, metric: PerformanceMetric):
        """
        Adaptively adjust thresholds based on performance data.
        
        This is the learning mechanism.
        """
        if not metric.success:
            return
        
        # Get comparison performance
        other_method = ExecutionMethod.CPU if metric.method == ExecutionMethod.GPU else ExecutionMethod.GPU
        other_performance = self._get_historical_performance(
            other_method, metric.dataset_size, metric.traversal_count
        )
        
        if other_performance is None:
            return
        
        # Calculate performance ratio
        if metric.method == ExecutionMethod.GPU:
            gpu_time = metric.execution_time
            cpu_time = other_performance
        else:
            cpu_time = metric.execution_time
            gpu_time = other_performance
        
        # If GPU is significantly faster, lower GPU threshold
        if gpu_time < cpu_time * 0.5 and metric.dataset_size < self.gpu_threshold_nodes:
            adjustment = int(self.learning_rate * (self.gpu_threshold_nodes - metric.dataset_size))
            self.gpu_threshold_nodes = max(100, self.gpu_threshold_nodes - adjustment)
        
        # If CPU is significantly faster, raise GPU threshold
        elif cpu_time < gpu_time * 0.5 and metric.dataset_size > self.cpu_threshold_nodes:
            adjustment = int(self.learning_rate * (metric.dataset_size - self.cpu_threshold_nodes))
            self.gpu_threshold_nodes = min(10000, self.gpu_threshold_nodes + adjustment)
    
    def execute_with_optimal_method(self, 
                                  dataset_size: int,
                                  traversal_count: int,
                                  cpu_function: Callable,
                                  gpu_function: Callable,
                                  complexity_hint: str = "normal") -> Any:
        """
        Execute function with optimal method and record performance.
        
        This is the main interface for adaptive execution.
        """
        # Choose method
        method = self.choose_execution_method(dataset_size, traversal_count, complexity_hint)
        
        # Execute with chosen method
        start_time = time.time()
        success = True
        result = None
        
        try:
            if method == ExecutionMethod.CPU:
                result = cpu_function()
            else:
                result = gpu_function()
        except Exception as e:
            success = False
            # Fallback to other method
            try:
                if method == ExecutionMethod.CPU:
                    result = gpu_function()
                    method = ExecutionMethod.GPU
                else:
                    result = cpu_function()
                    method = ExecutionMethod.CPU
                success = True
            except Exception as fallback_e:
                raise fallback_e
        
        execution_time = time.time() - start_time
        
        # Record performance
        self.record_performance(method, dataset_size, traversal_count, execution_time, success)
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self.performance_lock:
            total_cpu_time = self.method_stats[ExecutionMethod.CPU]['total_time']
            total_gpu_time = self.method_stats[ExecutionMethod.GPU]['total_time']
            total_time = total_cpu_time + total_gpu_time
            
            return {
                'current_thresholds': {
                    'cpu_threshold_nodes': self.cpu_threshold_nodes,
                    'gpu_threshold_nodes': self.gpu_threshold_nodes,
                    'adaptive_zone_size': self.gpu_threshold_nodes - self.cpu_threshold_nodes
                },
                'method_stats': self.method_stats.copy(),
                'performance_history_size': len(self.performance_history),
                'cache_size': len(self.decision_cache),
                'utilization': {
                    'cpu_percentage': (total_cpu_time / max(0.001, total_time)) * 100,
                    'gpu_percentage': (total_gpu_time / max(0.001, total_time)) * 100
                },
                'recent_decisions': self._get_recent_decisions()
            }
    
    def _get_recent_decisions(self) -> List[Dict[str, Any]]:
        """Get recent execution decisions for analysis."""
        recent = []
        for metric in self.performance_history[-10:]:
            recent.append({
                'method': metric.method.value,
                'dataset_size': metric.dataset_size,
                'traversal_count': metric.traversal_count,
                'execution_time_ms': metric.execution_time * 1000,
                'success': metric.success
            })
        return recent
    
    def optimize_thresholds(self) -> Dict[str, Any]:
        """
        Optimize thresholds based on all historical data.
        
        This can be called periodically to tune performance.
        """
        if len(self.performance_history) < 10:
            return {"message": "Insufficient data for optimization"}
        
        # Find optimal crossover point
        optimal_gpu_threshold = self._find_optimal_crossover()
        
        old_threshold = self.gpu_threshold_nodes
        if optimal_gpu_threshold != old_threshold:
            self.gpu_threshold_nodes = optimal_gpu_threshold
            self.cpu_threshold_nodes = min(self.cpu_threshold_nodes, optimal_gpu_threshold // 2)
            
            return {
                "optimization_applied": True,
                "old_gpu_threshold": old_threshold,
                "new_gpu_threshold": optimal_gpu_threshold,
                "improvement_expected": True
            }
        
        return {"optimization_applied": False, "message": "Current thresholds are optimal"}
    
    def _find_optimal_crossover(self) -> int:
        """Find the optimal crossover point between CPU and GPU."""
        # Group performance data by dataset size
        size_buckets = {}
        for metric in self.performance_history:
            if metric.success:
                size_bucket = (metric.dataset_size // 100) * 100  # Round to nearest 100
                if size_bucket not in size_buckets:
                    size_buckets[size_bucket] = {'cpu': [], 'gpu': []}
                size_buckets[size_bucket][metric.method.value].append(metric.execution_time)
        
        # Find crossover point
        best_threshold = self.gpu_threshold_nodes
        
        for size, times in size_buckets.items():
            if times['cpu'] and times['gpu']:
                avg_cpu = sum(times['cpu']) / len(times['cpu'])
                avg_gpu = sum(times['gpu']) / len(times['gpu'])
                
                # If GPU becomes faster at this size, update threshold
                if avg_gpu < avg_cpu * 0.9 and size < best_threshold:
                    best_threshold = size
        
        return max(100, best_threshold)
    
    def clear_cache(self):
        """Clear decision cache to force fresh decisions."""
        self.decision_cache.clear()
    
    def reset_learning(self):
        """Reset all learning data (use with caution)."""
        with self.performance_lock:
            self.performance_history.clear()
            self.method_stats = {
                ExecutionMethod.CPU: {
                    'total_time': 0.0,
                    'total_calls': 0,
                    'avg_time': 0.0,
                    'success_rate': 1.0
                },
                ExecutionMethod.GPU: {
                    'total_time': 0.0,
                    'total_calls': 0,
                    'avg_time': 0.0,
                    'success_rate': 1.0
                }
            }
            self.decision_cache.clear()
        
        print("🧠 Adaptive execution engine learning data reset")