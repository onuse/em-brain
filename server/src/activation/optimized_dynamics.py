"""
Optimized Activation Dynamics Engine

Enhanced version with incremental tensor updates and batch processing
to reduce GPU overhead and eliminate excessive tensor rebuilding.
"""

from typing import Dict, List, Set, Optional, Tuple
import time
import numpy as np
from collections import defaultdict, deque

from ..experience import Experience
from ..utils.batch_processor import IncrementalTensorUpdater

# GPU acceleration
try:
    import torch
    TORCH_AVAILABLE = True
    MPS_AVAILABLE = torch.backends.mps.is_available()
    
    # Test MPS functionality
    MPS_FUNCTIONAL = False
    if MPS_AVAILABLE:
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('mps')
            _ = test_tensor + 1
            MPS_FUNCTIONAL = True
        except Exception:
            MPS_FUNCTIONAL = False
            
except ImportError:
    TORCH_AVAILABLE = False
    MPS_AVAILABLE = False
    MPS_FUNCTIONAL = False


class OptimizedActivationDynamics:
    """
    Optimized activation dynamics with incremental updates and batching.
    
    Key optimizations:
    1. Incremental tensor updates instead of full rebuilds
    2. Batch processing of activation updates
    3. Lazy tensor synchronization
    4. Memory pooling for tensor allocations
    5. Intelligent resizing strategy
    """
    
    def __init__(self, use_gpu: bool = True, use_mixed_precision: bool = True,
                 initial_capacity: int = 1000):
        """
        Initialize optimized activation dynamics.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            use_mixed_precision: Whether to use FP16 for memory efficiency
            initial_capacity: Initial tensor capacity to avoid early rebuilds
        """
        # GPU configuration - lazy initialization
        self.gpu_capable = use_gpu and MPS_FUNCTIONAL
        self.use_gpu = False  # Start with CPU, upgrade when dataset is large enough
        self.device = 'cpu'  # Start with CPU
        self.use_mixed_precision = use_mixed_precision
        self.gpu_device = 'mps' if self.gpu_capable else 'cpu'
        
        # Precision configuration
        self.compute_dtype = torch.float16 if self.use_mixed_precision else torch.float32
        self.storage_dtype = torch.float32
        
        # Adaptive parameters
        self.base_decay_rate = 0.02
        self.spread_strength = 0.1
        self.min_activation = 0.01
        
        # Performance tracking
        self.recent_prediction_errors = []
        self.adaptation_rate = 0.1
        
        # Activation tracking
        self._last_update = time.time()
        self._activation_history = defaultdict(list)
        
        # Spreading activation state
        self._spread_queue = deque()  # Use deque for efficient queue operations
        self._similarity_cache = {}
        
        # Optimized GPU tensor management
        self.tensor_updater = IncrementalTensorUpdater(initial_capacity)
        self._gpu_activation_tensor = None  # Pre-allocated tensor
        self._gpu_similarity_matrix = None  # Sparse similarity matrix
        self._experience_id_to_index = {}
        self._index_to_experience_id = {}
        self._free_indices = set()  # Track available indices for reuse
        
        # Batch update state
        self._pending_updates = deque()
        self._update_batch_size = 10
        self._last_tensor_sync = time.time()
        self._sync_interval = 0.1  # Sync every 100ms
        
        # Performance metrics
        self.tensor_rebuilds = 0
        self.incremental_updates = 0
        self.batch_updates = 0
        
        precision_info = f"FP16 compute, FP32 storage" if self.use_mixed_precision else "FP32"
        gpu_status = f"GPU capable: {self.gpu_capable} (optimized lazy initialization)"
        print(f"⚡ OptimizedActivationDynamics initialized - {gpu_status}, {precision_info}")
        print(f"   Initial capacity: {initial_capacity}, batch size: {self._update_batch_size}")
    
    def _check_and_upgrade_to_gpu(self, num_experiences: int):
        """Check if we should upgrade to GPU based on number of experiences."""
        if not self.gpu_capable or self.use_gpu:
            return
        
        # Check with hardware adaptation system
        try:
            from ..utils.hardware_adaptation import should_use_gpu_for_activation_dynamics
            if should_use_gpu_for_activation_dynamics(num_experiences):
                self._upgrade_to_gpu(num_experiences)
        except ImportError:
            # Fallback to simple threshold
            if num_experiences >= 20:
                self._upgrade_to_gpu(num_experiences)
    
    def _upgrade_to_gpu(self, num_experiences: int):
        """Upgrade from CPU to GPU processing with pre-allocation."""
        if not self.gpu_capable or self.use_gpu:
            return
        
        print(f"⚡ Upgrading activation dynamics to GPU ({self.gpu_device}) - "
              f"pre-allocating for {self.tensor_updater.capacity} experiences")
        
        self.use_gpu = True
        self.device = self.gpu_device
        
        # Pre-allocate GPU tensors with capacity
        self._initialize_gpu_tensors(self.tensor_updater.capacity)
    
    def _initialize_gpu_tensors(self, capacity: int):
        """Initialize GPU tensors with given capacity."""
        if not self.use_gpu:
            return
        
        try:
            # Pre-allocate activation tensor
            self._gpu_activation_tensor = torch.zeros(
                capacity, dtype=self.storage_dtype, device=self.device
            )
            
            # Initialize sparse similarity matrix (we'll build it incrementally)
            # Using a dictionary for now, can optimize to sparse tensor later
            self._gpu_similarity_cache = {}
            
            print(f"   ✅ GPU tensors initialized with capacity {capacity}")
            
        except Exception as e:
            print(f"GPU tensor initialization failed: {e}, disabling GPU")
            self.use_gpu = False
            self.device = 'cpu'
    
    def _get_or_allocate_index(self, experience_id: str) -> int:
        """Get index for experience, allocating new one if needed."""
        if experience_id in self._experience_id_to_index:
            return self._experience_id_to_index[experience_id]
        
        # Try to reuse a free index
        if self._free_indices:
            index = self._free_indices.pop()
        else:
            # Allocate new index
            index = len(self._experience_id_to_index)
            
            # Check if we need to resize tensors
            if self.use_gpu and index >= self.tensor_updater.capacity:
                self._resize_gpu_tensors(index + 1)
        
        self._experience_id_to_index[experience_id] = index
        self._index_to_experience_id[index] = experience_id
        return index
    
    def _resize_gpu_tensors(self, required_size: int):
        """Resize GPU tensors when capacity is exceeded."""
        if not self.use_gpu or not self._gpu_activation_tensor:
            return
        
        new_capacity = self.tensor_updater.get_new_capacity(required_size)
        print(f"⚡ Resizing GPU tensors: {self.tensor_updater.capacity} → {new_capacity}")
        
        try:
            # Resize activation tensor
            self._gpu_activation_tensor = self.tensor_updater.create_resized_tensor(
                self._gpu_activation_tensor, new_capacity, self.device
            )
            
            self.tensor_updater.capacity = new_capacity
            self.tensor_rebuilds += 1
            
        except Exception as e:
            print(f"GPU tensor resize failed: {e}")
    
    def activate_experience(self, experience: Experience, strength: float = 1.0):
        """Activate an experience with batched update."""
        # Update the experience activation directly
        experience.activate(strength)
        
        # Add to batch update queue
        self._pending_updates.append({
            'experience_id': experience.experience_id,
            'activation': experience.activation_level,
            'type': 'direct'
        })
        
        # Add to spreading queue
        self._spread_queue.append((experience.experience_id, strength))
        
        # Track activation history
        current_time = time.time()
        self._activation_history[experience.experience_id].append((current_time, strength))
        
        # Process batch if large enough
        if len(self._pending_updates) >= self._update_batch_size:
            self._process_pending_updates()
    
    def update_all_activations(self, all_experiences: Dict[str, Experience]):
        """Update activation levels with optimized batch processing."""
        current_time = time.time()
        time_delta = current_time - self._last_update
        
        if time_delta < 0.05:  # Don't update too frequently
            return
        
        # Check if we should upgrade to GPU
        self._check_and_upgrade_to_gpu(len(all_experiences))
        
        # Process any pending updates first
        if self._pending_updates:
            self._process_pending_updates()
        
        # Batch decay and spreading updates
        if self.use_gpu:
            self._gpu_batch_update_activations(all_experiences, time_delta)
        else:
            self._cpu_update_activations(all_experiences, time_delta)
        
        self._last_update = current_time
        
        # Periodic tensor sync
        if current_time - self._last_tensor_sync > self._sync_interval:
            self._sync_tensors_to_experiences(all_experiences)
            self._last_tensor_sync = current_time
    
    def _process_pending_updates(self):
        """Process pending activation updates in batch."""
        if not self._pending_updates:
            return
        
        batch_start = time.time()
        
        if self.use_gpu and self._gpu_activation_tensor is not None:
            # Batch GPU updates
            indices = []
            values = []
            
            for update in self._pending_updates:
                idx = self._get_or_allocate_index(update['experience_id'])
                indices.append(idx)
                values.append(update['activation'])
            
            # Apply batch update
            if indices:
                indices_tensor = torch.tensor(indices, device=self.device)
                values_tensor = torch.tensor(values, dtype=self.storage_dtype, device=self.device)
                self._gpu_activation_tensor[indices_tensor] = values_tensor
                self.incremental_updates += len(indices)
        
        self._pending_updates.clear()
        self.batch_updates += 1
        
        batch_time = time.time() - batch_start
        if batch_time > 0.01:  # Log slow batches
            print(f"⚡ Batch update: {len(indices)} activations in {batch_time*1000:.1f}ms")
    
    def _gpu_batch_update_activations(self, all_experiences: Dict[str, Experience], time_delta: float):
        """Optimized GPU batch activation updates."""
        if not self._gpu_activation_tensor:
            return
        
        try:
            # Get active indices (experiences that exist)
            active_indices = []
            for exp_id in all_experiences:
                if exp_id in self._experience_id_to_index:
                    active_indices.append(self._experience_id_to_index[exp_id])
            
            if not active_indices:
                return
            
            active_indices_tensor = torch.tensor(active_indices, device=self.device)
            
            # Batch decay application
            decay_amount = self.base_decay_rate * time_delta
            decay_tensor = torch.tensor(decay_amount, dtype=self.compute_dtype, device=self.device)
            
            # Apply decay only to active experiences
            active_activations = self._gpu_activation_tensor[active_indices_tensor]
            active_activations = torch.clamp(active_activations - decay_tensor, min=0.0)
            self._gpu_activation_tensor[active_indices_tensor] = active_activations
            
            # Process spreading activation in batches
            self._gpu_batch_spreading_activation()
            
            # Apply minimum threshold
            mask = (self._gpu_activation_tensor > 0) & (self._gpu_activation_tensor < self.min_activation)
            self._gpu_activation_tensor[mask] = 0.0
            
            self.tensor_updater.increment_version()
            
        except Exception as e:
            print(f"GPU batch update failed: {e}, falling back to CPU")
            self._cpu_update_activations(all_experiences, time_delta)
    
    def _gpu_batch_spreading_activation(self):
        """Batch process spreading activation on GPU."""
        if not self._spread_queue:
            return
        
        # Process spread queue in batches
        batch_size = min(len(self._spread_queue), 50)
        batch = [self._spread_queue.popleft() for _ in range(batch_size)]
        
        for exp_id, source_strength in batch:
            if exp_id not in self._experience_id_to_index:
                continue
            
            source_idx = self._experience_id_to_index[exp_id]
            
            # Get cached similarities for this experience
            if source_idx in self._gpu_similarity_cache:
                similar_indices = self._gpu_similarity_cache[source_idx]['indices']
                similarities = self._gpu_similarity_cache[source_idx]['values']
                
                # Batch spread calculation
                spread_amount = source_strength * self.spread_strength
                spread_values = similarities * spread_amount
                
                # Apply spread with threshold
                mask = spread_values > 0.01
                if mask.any():
                    self._gpu_activation_tensor[similar_indices[mask]] += spread_values[mask]
    
    def _sync_tensors_to_experiences(self, all_experiences: Dict[str, Experience]):
        """Sync GPU tensors back to experience objects."""
        if not self.use_gpu or not self._gpu_activation_tensor or not self.tensor_updater.needs_sync():
            return
        
        try:
            # Batch read from GPU
            active_indices = []
            exp_ids = []
            
            for exp_id, experience in all_experiences.items():
                if exp_id in self._experience_id_to_index:
                    active_indices.append(self._experience_id_to_index[exp_id])
                    exp_ids.append(exp_id)
            
            if active_indices:
                indices_tensor = torch.tensor(active_indices, device=self.device)
                activations = self._gpu_activation_tensor[indices_tensor].cpu().numpy()
                
                # Batch update experiences
                for i, exp_id in enumerate(exp_ids):
                    all_experiences[exp_id].activation_level = float(activations[i])
            
            self.tensor_updater.mark_synced()
            
        except Exception as e:
            print(f"Tensor sync failed: {e}")
    
    def update_similarity_cache(self, experience_id: str, similar_experiences: List[Tuple[str, float]]):
        """Update similarity cache for efficient spreading."""
        if not self.use_gpu or experience_id not in self._experience_id_to_index:
            return
        
        source_idx = self._experience_id_to_index[experience_id]
        
        # Build similarity data for GPU
        indices = []
        values = []
        
        for sim_id, similarity in similar_experiences:
            if sim_id in self._experience_id_to_index and similarity > 0.3:
                indices.append(self._experience_id_to_index[sim_id])
                values.append(similarity)
        
        if indices:
            self._gpu_similarity_cache[source_idx] = {
                'indices': torch.tensor(indices, device=self.device),
                'values': torch.tensor(values, dtype=self.storage_dtype, device=self.device)
            }
    
    def get_activation_statistics(self, all_experiences: Dict[str, Experience]) -> Dict:
        """Get comprehensive activation statistics."""
        # Ensure tensors are synced
        self._sync_tensors_to_experiences(all_experiences)
        
        activations = [exp.activation_level for exp in all_experiences.values()]
        
        if not activations:
            return {
                'total_experiences': 0,
                'activated_count': 0,
                'avg_activation': 0.0,
                'max_activation': 0.0,
                'working_memory_size': 0,
                'optimization_stats': self.get_optimization_stats()
            }
        
        activated_count = sum(1 for a in activations if a >= self.min_activation)
        working_memory_count = sum(1 for a in activations if a >= 0.1)
        
        return {
            'total_experiences': len(activations),
            'activated_count': activated_count,
            'working_memory_size': working_memory_count,
            'avg_activation': np.mean(activations),
            'max_activation': np.max(activations),
            'activation_distribution': {
                'very_high': sum(1 for a in activations if a >= 0.8),
                'high': sum(1 for a in activations if 0.5 <= a < 0.8),
                'medium': sum(1 for a in activations if 0.2 <= a < 0.5),
                'low': sum(1 for a in activations if 0.1 <= a < 0.2),
                'very_low': sum(1 for a in activations if 0.0 < a < 0.1),
                'inactive': sum(1 for a in activations if a == 0.0)
            },
            'optimization_stats': self.get_optimization_stats()
        }
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization performance statistics."""
        return {
            'tensor_rebuilds': self.tensor_rebuilds,
            'incremental_updates': self.incremental_updates,
            'batch_updates': self.batch_updates,
            'tensor_capacity': self.tensor_updater.capacity,
            'used_capacity': len(self._experience_id_to_index),
            'free_indices': len(self._free_indices),
            'gpu_enabled': self.use_gpu,
            'update_batch_size': self._update_batch_size
        }
    
    def _cpu_update_activations(self, all_experiences: Dict[str, Experience], time_delta: float):
        """CPU fallback with batch optimization."""
        # Apply decay in batch
        decay_amount = self.base_decay_rate * time_delta
        
        for experience in all_experiences.values():
            experience.decay_activation(decay_amount)
        
        # Process spreading activation
        self._process_spreading_activation(all_experiences)
        
        # Cleanup weak activations
        self._cleanup_weak_activations(all_experiences)
    
    def _process_spreading_activation(self, all_experiences: Dict[str, Experience]):
        """Process spreading activation queue."""
        if not self._spread_queue:
            return
        
        # Process in batches
        batch_size = min(len(self._spread_queue), 20)
        
        for _ in range(batch_size):
            if not self._spread_queue:
                break
                
            exp_id, source_strength = self._spread_queue.popleft()
            source_exp = all_experiences.get(exp_id)
            
            if not source_exp:
                continue
            
            # Spread to similar experiences
            for similar_id, similarity in source_exp.similar_experiences.items():
                target_exp = all_experiences.get(similar_id)
                if target_exp and similarity > 0.3:
                    spread_amount = source_strength * similarity * self.spread_strength
                    if spread_amount > 0.01:
                        target_exp.activate(spread_amount)
    
    def _cleanup_weak_activations(self, all_experiences: Dict[str, Experience]):
        """Clean up very weak activations."""
        for experience in all_experiences.values():
            if 0 < experience.activation_level < self.min_activation:
                experience.activation_level = 0.0
    
    # Compatibility methods
    def get_activated_experiences(self, all_experiences: Dict[str, Experience], 
                                min_activation: float = 0.1) -> List[Experience]:
        """Get currently activated experiences."""
        self._sync_tensors_to_experiences(all_experiences)
        
        activated = []
        for experience in all_experiences.values():
            if experience.activation_level >= min_activation:
                activated.append(experience)
        
        activated.sort(key=lambda exp: exp.activation_level, reverse=True)
        return activated
    
    def spread_activation_from_similar(self, source_experience: Experience,
                                     similar_experiences: List[Tuple[Experience, float]]):
        """Spread activation to similar experiences."""
        source_activation = source_experience.activation_level
        
        if source_activation < self.min_activation:
            return
        
        # Add to batch updates
        for target_experience, similarity in similar_experiences:
            spread_amount = (source_activation * similarity * self.spread_strength)
            
            if spread_amount > 0.01:
                target_experience.activate(spread_amount)
                self._pending_updates.append({
                    'experience_id': target_experience.experience_id,
                    'activation': target_experience.activation_level,
                    'type': 'spread'
                })
    
    def clear_all_activations(self, all_experiences: Dict[str, Experience]):
        """Clear all activations."""
        for experience in all_experiences.values():
            experience.activation_level = 0.0
        
        self._spread_queue.clear()
        self._pending_updates.clear()
        self._activation_history.clear()
        
        # Clear GPU tensors
        if self.use_gpu and self._gpu_activation_tensor is not None:
            self._gpu_activation_tensor.zero_()
            self._gpu_similarity_cache.clear()
        
        print("⚡ All activations cleared (optimized)")