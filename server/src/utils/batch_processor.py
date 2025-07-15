"""
Batch Experience Processor

Optimizes tensor operations by batching experience processing
to reduce GPU overhead and tensor rebuilding frequency.
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
from collections import deque


class BatchExperienceProcessor:
    """
    Batches experience processing to reduce tensor rebuild overhead.
    
    Key optimizations:
    1. Accumulate experiences before processing
    2. Batch similarity searches
    3. Batch activation updates
    4. Batch pattern analysis
    5. Intelligent flushing based on batch size and time
    """
    
    def __init__(self, 
                 min_batch_size: int = 5,
                 max_batch_size: int = 20,
                 max_delay_ms: float = 100.0,
                 adaptive: bool = True):
        """
        Initialize batch processor.
        
        Args:
            min_batch_size: Minimum experiences before processing
            max_batch_size: Maximum batch size to prevent memory issues
            max_delay_ms: Maximum delay before forced flush
            adaptive: Whether to adapt batch size based on performance
        """
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.max_delay_ms = max_delay_ms
        self.adaptive = adaptive
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=max_batch_size)
        self.buffer_start_time = None
        
        # Performance tracking
        self.batch_processing_times = deque(maxlen=100)
        self.single_processing_times = deque(maxlen=100)
        self.current_batch_size = min_batch_size
        
        # Batch statistics
        self.total_batches_processed = 0
        self.total_experiences_batched = 0
        self.forced_flushes = 0
        
        print(f"📦 BatchExperienceProcessor initialized - batch size: {min_batch_size}-{max_batch_size}, "
              f"max delay: {max_delay_ms}ms")
    
    def should_process_batch(self) -> bool:
        """Determine if batch should be processed now."""
        if not self.experience_buffer:
            return False
        
        # Check batch size threshold
        if len(self.experience_buffer) >= self.current_batch_size:
            return True
        
        # Check time threshold
        if self.buffer_start_time:
            elapsed_ms = (time.time() - self.buffer_start_time) * 1000
            if elapsed_ms >= self.max_delay_ms:
                self.forced_flushes += 1
                return True
        
        return False
    
    def add_experience(self, experience_data: Dict[str, Any]):
        """Add experience to batch buffer."""
        if not self.buffer_start_time:
            self.buffer_start_time = time.time()
        
        self.experience_buffer.append(experience_data)
    
    def get_batch(self) -> List[Dict[str, Any]]:
        """Get current batch and clear buffer."""
        batch = list(self.experience_buffer)
        self.experience_buffer.clear()
        self.buffer_start_time = None
        return batch
    
    def adapt_batch_size(self):
        """Adapt batch size based on performance metrics."""
        if not self.adaptive or len(self.batch_processing_times) < 10:
            return
        
        # Calculate average processing time per experience
        avg_batch_time = np.mean(self.batch_processing_times)
        avg_single_time = np.mean(self.single_processing_times) if self.single_processing_times else avg_batch_time
        
        # Calculate efficiency ratio
        efficiency_ratio = avg_single_time / (avg_batch_time / self.current_batch_size)
        
        # Adapt batch size based on efficiency
        if efficiency_ratio > 1.5:  # Batching is significantly more efficient
            # Increase batch size
            self.current_batch_size = min(
                self.current_batch_size + 2,
                self.max_batch_size
            )
        elif efficiency_ratio < 1.1:  # Batching is not much better
            # Decrease batch size
            self.current_batch_size = max(
                self.current_batch_size - 1,
                self.min_batch_size
            )
    
    def record_batch_performance(self, batch_size: int, processing_time: float):
        """Record batch processing performance."""
        self.batch_processing_times.append(processing_time)
        self.total_batches_processed += 1
        self.total_experiences_batched += batch_size
        
        # Adapt batch size periodically
        if self.total_batches_processed % 10 == 0:
            self.adapt_batch_size()
    
    def record_single_performance(self, processing_time: float):
        """Record single experience processing performance."""
        self.single_processing_times.append(processing_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        avg_batch_time = np.mean(self.batch_processing_times) if self.batch_processing_times else 0
        avg_batch_size = (self.total_experiences_batched / self.total_batches_processed 
                         if self.total_batches_processed > 0 else 0)
        
        return {
            'total_batches': self.total_batches_processed,
            'total_experiences_batched': self.total_experiences_batched,
            'current_batch_size': self.current_batch_size,
            'avg_batch_time_ms': avg_batch_time * 1000,
            'avg_batch_size': avg_batch_size,
            'forced_flushes': self.forced_flushes,
            'efficiency_gain': self._calculate_efficiency_gain()
        }
    
    def _calculate_efficiency_gain(self) -> float:
        """Calculate efficiency gain from batching."""
        if not self.batch_processing_times or not self.single_processing_times:
            return 1.0
        
        avg_batch_time = np.mean(self.batch_processing_times)
        avg_single_time = np.mean(self.single_processing_times)
        avg_batch_size = (self.total_experiences_batched / self.total_batches_processed 
                         if self.total_batches_processed > 0 else 1)
        
        # Time per experience in batch vs single
        batch_time_per_exp = avg_batch_time / avg_batch_size
        
        return avg_single_time / batch_time_per_exp if batch_time_per_exp > 0 else 1.0


class BatchedSimilaritySearch:
    """Optimizes similarity searches by batching multiple queries."""
    
    def __init__(self, similarity_engine):
        """Initialize with reference to similarity engine."""
        self.similarity_engine = similarity_engine
        self.batch_cache = {}
        self.cache_hits = 0
        self.total_searches = 0
    
    def batch_find_similar(self, 
                          target_vectors: List[List[float]],
                          experience_vectors: List[List[float]],
                          experience_ids: List[str],
                          max_results: int = 10,
                          min_similarity: float = 0.3) -> List[List[Tuple[str, float]]]:
        """
        Batch similarity search for multiple target vectors.
        
        Returns list of results for each target vector.
        """
        self.total_searches += len(target_vectors)
        
        # Check if we can use GPU batch processing
        use_gpu = (self.similarity_engine.use_gpu and 
                  len(experience_vectors) > 50 and 
                  len(target_vectors) > 1)
        
        if use_gpu:
            return self._gpu_batch_search(
                target_vectors, experience_vectors, experience_ids,
                max_results, min_similarity
            )
        else:
            # Fall back to sequential processing
            results = []
            for target in target_vectors:
                result = self.similarity_engine.find_similar_experiences(
                    target, experience_vectors, experience_ids,
                    max_results, min_similarity
                )
                results.append(result)
            return results
    
    def _gpu_batch_search(self,
                         target_vectors: List[List[float]],
                         experience_vectors: List[List[float]],
                         experience_ids: List[str],
                         max_results: int,
                         min_similarity: float) -> List[List[Tuple[str, float]]]:
        """GPU-optimized batch similarity search."""
        try:
            import torch
            
            # Convert to tensors
            targets_tensor = torch.tensor(target_vectors, dtype=torch.float32, device='mps')
            experiences_tensor = torch.tensor(experience_vectors, dtype=torch.float32, device='mps')
            
            # Normalize for cosine similarity
            targets_norm = torch.nn.functional.normalize(targets_tensor, dim=1)
            experiences_norm = torch.nn.functional.normalize(experiences_tensor, dim=1)
            
            # Batch matrix multiplication for all similarities at once
            similarities = torch.mm(targets_norm, experiences_norm.t())
            
            # Process results for each target
            results = []
            for i in range(len(target_vectors)):
                target_sims = similarities[i].cpu().numpy()
                
                # Find best matches
                target_results = []
                for j, sim in enumerate(target_sims):
                    if sim >= min_similarity:
                        target_results.append((experience_ids[j], float(sim)))
                
                # Sort and limit
                target_results.sort(key=lambda x: x[1], reverse=True)
                results.append(target_results[:max_results])
            
            return results
            
        except Exception as e:
            print(f"GPU batch search failed: {e}, falling back to CPU")
            # Fall back to sequential
            results = []
            for target in target_vectors:
                result = self.similarity_engine.find_similar_experiences(
                    target, experience_vectors, experience_ids,
                    max_results, min_similarity
                )
                results.append(result)
            return results


class IncrementalTensorUpdater:
    """
    Updates tensors incrementally instead of rebuilding from scratch.
    """
    
    def __init__(self, initial_capacity: int = 1000):
        """Initialize with initial tensor capacity."""
        self.capacity = initial_capacity
        self.growth_factor = 1.5
        self.current_size = 0
        
        # Track tensor versions for consistency
        self.tensor_version = 0
        self.last_sync_version = 0
        
        print(f"🔄 IncrementalTensorUpdater initialized - capacity: {initial_capacity}")
    
    def should_resize(self, new_size: int) -> bool:
        """Check if tensor needs resizing."""
        return new_size > self.capacity
    
    def get_new_capacity(self, required_size: int) -> int:
        """Calculate new capacity with growth factor."""
        new_capacity = self.capacity
        while new_capacity < required_size:
            new_capacity = int(new_capacity * self.growth_factor)
        return new_capacity
    
    def create_resized_tensor(self, old_tensor, new_capacity: int, device: str):
        """Create resized tensor preserving old data."""
        import torch
        
        # Create new tensor with expanded capacity
        old_shape = old_tensor.shape
        new_shape = (new_capacity,) + old_shape[1:]
        
        new_tensor = torch.zeros(new_shape, dtype=old_tensor.dtype, device=device)
        
        # Copy old data
        if old_shape[0] > 0:
            new_tensor[:old_shape[0]] = old_tensor
        
        return new_tensor
    
    def increment_version(self):
        """Increment tensor version for tracking changes."""
        self.tensor_version += 1
    
    def needs_sync(self) -> bool:
        """Check if tensors need synchronization."""
        return self.tensor_version != self.last_sync_version
    
    def mark_synced(self):
        """Mark tensors as synchronized."""
        self.last_sync_version = self.tensor_version