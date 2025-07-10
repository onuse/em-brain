#!/usr/bin/env python3
"""
Asynchronous Graph Maintenance - Non-blocking graph optimization and consolidation.

This module handles background maintenance of the graph structure to prevent
blocking the main prediction pipeline. Key operations include:
- Tensor consolidation and optimization
- Connection matrix cleanup
- Memory defragmentation
- Statistical cache updates
"""

import asyncio
import threading
import time
import torch
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime
import weakref
import logging

from core.world_graph import WorldGraph
from core.vectorized_backend import VectorizedBackend


@dataclass
class MaintenanceTask:
    """Represents a maintenance task to be performed."""
    task_type: str
    priority: int  # 1=high, 2=medium, 3=low
    target_graph: str  # Graph ID or reference
    parameters: Dict[str, Any]
    created_at: float
    estimated_duration: float  # seconds


class AsyncGraphMaintenance:
    """
    Asynchronous graph maintenance system.
    
    Runs background tasks to optimize graph structure without blocking
    the main prediction pipeline. Uses threading to ensure compatibility
    with synchronous code.
    """
    
    def __init__(self, max_workers: int = 2):
        """
        Initialize asynchronous graph maintenance.
        
        Args:
            max_workers: Maximum number of maintenance threads
        """
        self.max_workers = max_workers
        self.running = False
        self.maintenance_thread = None
        self.task_queue = asyncio.Queue()
        self.graph_references = weakref.WeakSet()
        
        # Performance tracking
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_maintenance_time = 0.0
        self.last_maintenance_time = 0.0
        
        # Maintenance intervals (seconds)
        self.tensor_consolidation_interval = 60.0  # Every 1 minute
        self.connection_cleanup_interval = 300.0   # Every 5 minutes
        self.memory_defrag_interval = 600.0        # Every 10 minutes
        
        # Last maintenance timestamps
        self.last_tensor_consolidation = 0.0
        self.last_connection_cleanup = 0.0
        self.last_memory_defrag = 0.0
        
        # Statistics
        self.maintenance_stats = {
            'tensor_consolidations': 0,
            'connection_cleanups': 0,
            'memory_defrags': 0,
            'cache_updates': 0,
            'total_time_saved': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the asynchronous maintenance system."""
        if self.running:
            return
        
        self.running = True
        self.maintenance_thread = threading.Thread(
            target=self._run_maintenance_loop,
            name="AsyncGraphMaintenance",
            daemon=True
        )
        self.maintenance_thread.start()
        self.logger.info("Async graph maintenance started")
    
    def stop(self):
        """Stop the asynchronous maintenance system."""
        if not self.running:
            return
        
        self.running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5.0)
        
        self.logger.info("Async graph maintenance stopped")
    
    def register_graph(self, graph: WorldGraph):
        """Register a graph for maintenance."""
        self.graph_references.add(graph)
        self.logger.info(f"Graph registered for maintenance: {type(graph).__name__}")
    
    def schedule_maintenance(self, task: MaintenanceTask):
        """Schedule a maintenance task."""
        # Use thread-safe approach since we're mixing threading and asyncio
        if self.running and self.maintenance_thread and self.maintenance_thread.is_alive():
            # Put task in queue using thread-safe method
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.task_queue.put(task))
                loop.close()
            except Exception as e:
                self.logger.error(f"Failed to schedule maintenance task: {e}")
        else:
            self.logger.warning("Maintenance system not running - task not scheduled")
    
    def _run_maintenance_loop(self):
        """Main maintenance loop running in separate thread."""
        # Create event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._async_maintenance_loop())
        except Exception as e:
            self.logger.error(f"Maintenance loop error: {e}")
        finally:
            loop.close()
    
    async def _async_maintenance_loop(self):
        """Async maintenance loop."""
        self.logger.info("Starting async maintenance loop")
        
        while self.running:
            try:
                # Check for scheduled tasks
                await self._process_queued_tasks()
                
                # Perform periodic maintenance
                await self._perform_periodic_maintenance()
                
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Maintenance loop iteration error: {e}")
                await asyncio.sleep(1.0)  # Longer sleep on error
    
    async def _process_queued_tasks(self):
        """Process tasks from the queue."""
        try:
            # Non-blocking check for queued tasks
            while True:
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=0.01)
                    await self._execute_maintenance_task(task)
                    self.task_queue.task_done()
                except asyncio.TimeoutError:
                    break  # No more tasks
                    
        except Exception as e:
            self.logger.error(f"Error processing queued tasks: {e}")
    
    async def _perform_periodic_maintenance(self):
        """Perform periodic maintenance tasks."""
        current_time = time.time()
        
        # Tensor consolidation
        if current_time - self.last_tensor_consolidation > self.tensor_consolidation_interval:
            await self._consolidate_tensors()
            self.last_tensor_consolidation = current_time
        
        # Connection cleanup
        if current_time - self.last_connection_cleanup > self.connection_cleanup_interval:
            await self._cleanup_connections()
            self.last_connection_cleanup = current_time
        
        # Memory defragmentation
        if current_time - self.last_memory_defrag > self.memory_defrag_interval:
            await self._defragment_memory()
            self.last_memory_defrag = current_time
    
    async def _execute_maintenance_task(self, task: MaintenanceTask):
        """Execute a specific maintenance task."""
        start_time = time.time()
        
        try:
            if task.task_type == "tensor_consolidation":
                await self._consolidate_tensors()
            elif task.task_type == "connection_cleanup":
                await self._cleanup_connections()
            elif task.task_type == "memory_defrag":
                await self._defragment_memory()
            elif task.task_type == "cache_update":
                await self._update_caches()
            elif task.task_type == "similarity_rebuild":
                await self._rebuild_similarity_indices()
            else:
                self.logger.warning(f"Unknown maintenance task: {task.task_type}")
                self.tasks_failed += 1
                return
            
            self.tasks_completed += 1
            execution_time = time.time() - start_time
            self.total_maintenance_time += execution_time
            
            self.logger.debug(f"Completed {task.task_type} in {execution_time:.3f}s")
            
        except Exception as e:
            self.tasks_failed += 1
            self.logger.error(f"Failed to execute {task.task_type}: {e}")
    
    async def _consolidate_tensors(self):
        """Consolidate tensors to optimize memory usage."""
        consolidation_count = 0
        
        for graph in list(self.graph_references):
            if hasattr(graph, 'vectorized_backend'):
                backend = graph.vectorized_backend
                
                # Consolidate connection matrices
                if backend._connection_count > 0:
                    await self._consolidate_connection_matrix(backend)
                    consolidation_count += 1
                
                # Consolidate experience tensors
                if backend.size > 0:
                    await self._consolidate_experience_tensors(backend)
                    consolidation_count += 1
        
        if consolidation_count > 0:
            self.maintenance_stats['tensor_consolidations'] += consolidation_count
            self.logger.info(f"Consolidated {consolidation_count} tensor groups")
    
    async def _consolidate_connection_matrix(self, backend: VectorizedBackend):
        """Consolidate the sparse connection matrix."""
        if backend._connection_count == 0:
            return
        
        # Remove gaps in connection matrix
        valid_indices = backend._connection_indices[:, :backend._connection_count]
        valid_values = backend._connection_values[:backend._connection_count]
        
        # Remove duplicate connections
        unique_connections = {}
        for i in range(backend._connection_count):
            source = valid_indices[0, i].item()
            target = valid_indices[1, i].item()
            weight = valid_values[i].item()
            
            # Keep the strongest connection for duplicates
            key = (source, target)
            if key not in unique_connections or weight > unique_connections[key]:
                unique_connections[key] = weight
        
        # Rebuild connection matrix
        if len(unique_connections) < backend._connection_count:
            new_count = len(unique_connections)
            new_indices = torch.zeros((2, new_count), dtype=torch.long, device=backend.device)
            new_values = torch.zeros(new_count, dtype=torch.float32, device=backend.device)
            
            for i, ((source, target), weight) in enumerate(unique_connections.items()):
                new_indices[0, i] = source
                new_indices[1, i] = target
                new_values[i] = weight
            
            # Update the backend
            backend._connection_indices[:, :new_count] = new_indices
            backend._connection_values[:new_count] = new_values
            backend._connection_count = new_count
            
            self.logger.debug(f"Consolidated connections: {backend._connection_count} -> {new_count}")
    
    async def _consolidate_experience_tensors(self, backend: VectorizedBackend):
        """Consolidate experience tensors to remove unused capacity."""
        if backend.size == 0:
            return
        
        # Check if we have significant unused capacity
        usage_ratio = backend.size / backend.capacity
        if usage_ratio > 0.8:  # Only consolidate if less than 80% used
            return
        
        # Create new tensors with optimal size
        new_capacity = max(backend.size * 2, 1000)  # Leave some growth room
        
        # We'll consolidate in the background without blocking
        # This is a complex operation that would need careful implementation
        # For now, we'll just log the opportunity
        self.logger.debug(f"Tensor consolidation opportunity: {usage_ratio:.1%} usage")
    
    async def _cleanup_connections(self):
        """Clean up dead or weak connections."""
        cleanup_count = 0
        
        for graph in list(self.graph_references):
            if hasattr(graph, 'vectorized_backend'):
                backend = graph.vectorized_backend
                
                # Remove weak connections
                if backend._connection_count > 0:
                    weak_threshold = 0.05  # Remove connections below 5% strength
                    
                    # Find strong connections
                    strong_mask = backend._connection_values[:backend._connection_count] > weak_threshold
                    strong_count = strong_mask.sum().item()
                    
                    if strong_count < backend._connection_count:
                        # Keep only strong connections
                        strong_indices = backend._connection_indices[:, :backend._connection_count][:, strong_mask]
                        strong_values = backend._connection_values[:backend._connection_count][strong_mask]
                        
                        # Update backend
                        backend._connection_indices[:, :strong_count] = strong_indices
                        backend._connection_values[:strong_count] = strong_values
                        backend._connection_count = strong_count
                        
                        cleanup_count += 1
                        self.logger.debug(f"Cleaned up weak connections: {strong_count} remaining")
        
        if cleanup_count > 0:
            self.maintenance_stats['connection_cleanups'] += cleanup_count
    
    async def _defragment_memory(self):
        """Defragment GPU memory."""
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # For MPS, we don't have direct cache control
        # but we can trigger garbage collection
        import gc
        gc.collect()
        
        self.maintenance_stats['memory_defrags'] += 1
        self.logger.debug("Performed memory defragmentation")
    
    async def _update_caches(self):
        """Update statistical caches."""
        for graph in list(self.graph_references):
            if hasattr(graph, '_invalidate_caches'):
                # Trigger cache updates in background
                graph._invalidate_caches()
            
            if hasattr(graph, '_update_statistics'):
                graph._update_statistics()
        
        self.maintenance_stats['cache_updates'] += 1
    
    async def _rebuild_similarity_indices(self):
        """Rebuild similarity indices for better performance."""
        for graph in list(self.graph_references):
            if hasattr(graph, 'vectorized_backend'):
                backend = graph.vectorized_backend
                
                # Rebuild similarity caches
                if hasattr(backend, 'similarity_engine'):
                    backend.similarity_engine.clear_cache()
                    self.logger.debug("Rebuilt similarity indices")
    
    def get_maintenance_stats(self) -> Dict[str, Any]:
        """Get maintenance statistics."""
        return {
            'running': self.running,
            'tasks_completed': self.tasks_completed,
            'tasks_failed': self.tasks_failed,
            'total_maintenance_time': self.total_maintenance_time,
            'registered_graphs': len(self.graph_references),
            'maintenance_stats': self.maintenance_stats.copy(),
            'avg_task_time': self.total_maintenance_time / max(1, self.tasks_completed)
        }
    
    def schedule_tensor_consolidation(self, graph_id: str = "default"):
        """Schedule tensor consolidation task."""
        task = MaintenanceTask(
            task_type="tensor_consolidation",
            priority=2,
            target_graph=graph_id,
            parameters={},
            created_at=time.time(),
            estimated_duration=0.5
        )
        self.schedule_maintenance(task)
    
    def schedule_connection_cleanup(self, graph_id: str = "default"):
        """Schedule connection cleanup task."""
        task = MaintenanceTask(
            task_type="connection_cleanup",
            priority=3,
            target_graph=graph_id,
            parameters={},
            created_at=time.time(),
            estimated_duration=0.2
        )
        self.schedule_maintenance(task)
    
    def schedule_memory_defrag(self, graph_id: str = "default"):
        """Schedule memory defragmentation task."""
        task = MaintenanceTask(
            task_type="memory_defrag",
            priority=1,
            target_graph=graph_id,
            parameters={},
            created_at=time.time(),
            estimated_duration=0.1
        )
        self.schedule_maintenance(task)


# Global maintenance instance
_global_maintenance = None

def get_global_maintenance() -> AsyncGraphMaintenance:
    """Get the global maintenance instance."""
    global _global_maintenance
    if _global_maintenance is None:
        _global_maintenance = AsyncGraphMaintenance()
    return _global_maintenance

def start_global_maintenance():
    """Start the global maintenance system."""
    maintenance = get_global_maintenance()
    maintenance.start()

def stop_global_maintenance():
    """Stop the global maintenance system."""
    maintenance = get_global_maintenance()
    maintenance.stop()