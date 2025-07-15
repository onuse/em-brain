"""
Activation Dynamics Engine

Neural-like spreading activation that creates working memory effects.
Recently accessed and related experiences stay "hot" and influence decisions more.

GPU-accelerated for fast spreading activation across large experience sets.
"""

from typing import Dict, List, Set, Optional, Tuple
import time
import numpy as np
from collections import defaultdict

from ..experience import Experience
from ..utils.cache_adapters import ActivationCacheAdapter

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


class ActivationDynamics:
    """
    Manages activation levels and spreading activation through experience memory.
    
    This creates working memory effects - recently accessed and related experiences
    stay activated and influence decisions more than inactive memories.
    """
    
    def __init__(self, use_gpu: bool = True, use_mixed_precision: bool = True):
        """
        Initialize activation dynamics with adaptive parameters.
        
        Args:
            use_gpu: Whether to use GPU acceleration for spreading activation
            use_mixed_precision: Whether to use FP16 for memory efficiency
        """
        # GPU configuration - lazy initialization
        self.gpu_capable = use_gpu and MPS_FUNCTIONAL
        self.use_gpu = False  # Start with CPU, upgrade when dataset is large enough
        self.device = 'cpu'  # Start with CPU
        self.use_mixed_precision = use_mixed_precision
        self.gpu_device = 'mps' if self.gpu_capable else 'cpu'
        
        # Precision configuration - biological neural noise simulation
        self.compute_dtype = torch.float16 if self.use_mixed_precision else torch.float32
        self.storage_dtype = torch.float32  # Critical activations stored in FP32
        
        # Adaptive parameters that adjust based on prediction performance
        self.base_decay_rate = 0.02  # Will adapt based on system performance
        self.spread_strength = 0.1   # Will adapt based on learning success
        self.min_activation = 0.01   # Will adapt based on memory effectiveness
        
        # Performance tracking for adaptation
        self.recent_prediction_errors = []
        self.adaptation_rate = 0.1
        
        # Activation tracking
        self._last_update = time.time()
        self._activation_history = defaultdict(list)  # Track activation over time
        
        # Spreading activation state
        self._spread_queue = []  # Experiences to spread activation from
        
        # Memory-managed cache for similarity lookups and GPU tensors
        self._cache_adapter = ActivationCacheAdapter(
            max_entries=2000,
            max_size_mb=100.0,
            eviction_policy="utility_based"  # Activation benefits from utility-based eviction
        )
        
        # GPU tensors for efficient computation
        self._gpu_activation_levels = None  # Tensor of all activation levels
        self._gpu_similarity_matrix = None  # Cached similarity connections
        self._experience_id_to_index = {}   # Map experience IDs to tensor indices
        self._index_to_experience_id = {}   # Map tensor indices back to experience IDs
        
        precision_info = f"FP16 compute, FP32 storage" if self.use_mixed_precision else "FP32"
        gpu_status = f"GPU capable: {self.gpu_capable} (lazy initialization enabled)"
        print(f"🧠 ActivationDynamics initialized - adaptive parameters ({gpu_status}, {precision_info})")
    
    def _check_and_upgrade_to_gpu(self, num_experiences: int):
        """Check if we should upgrade to GPU based on number of experiences."""
        if not self.gpu_capable or self.use_gpu:
            return  # Already using GPU or not capable
        
        # Check with hardware adaptation system
        try:
            from ..utils.hardware_adaptation import should_use_gpu_for_activation_dynamics
            if should_use_gpu_for_activation_dynamics(num_experiences):
                self._upgrade_to_gpu()
        except ImportError:
            # Fallback to simple threshold
            if num_experiences >= 20:
                self._upgrade_to_gpu()
    
    def _upgrade_to_gpu(self):
        """Upgrade from CPU to GPU processing."""
        if not self.gpu_capable or self.use_gpu:
            return
        
        print(f"🚀 Upgrading activation dynamics to GPU ({self.gpu_device}) - experience set large enough to benefit")
        
        self.use_gpu = True
        self.device = self.gpu_device
    
    def _rebuild_gpu_tensors(self, all_experiences: Dict[str, Experience]):
        """Rebuild GPU tensors when experience set changes significantly."""
        if not self.use_gpu:
            return
            
        try:
            # Build mapping from experience IDs to tensor indices
            experience_ids = list(all_experiences.keys())
            self._experience_id_to_index = {exp_id: i for i, exp_id in enumerate(experience_ids)}
            self._index_to_experience_id = {i: exp_id for exp_id, i in self._experience_id_to_index.items()}
            
            num_experiences = len(experience_ids)
            if num_experiences == 0:
                self._gpu_activation_levels = None
                self._gpu_similarity_matrix = None
                return
            
            # Create activation levels tensor with storage precision
            activation_levels = []
            for exp_id in experience_ids:
                activation_levels.append(all_experiences[exp_id].activation_level)
            
            self._gpu_activation_levels = torch.tensor(
                activation_levels, dtype=self.storage_dtype, device=self.device
            )
            
            # Build similarity matrix from cached connections with storage precision
            similarity_matrix = torch.zeros((num_experiences, num_experiences), 
                                           dtype=self.storage_dtype, device=self.device)
            
            for i, exp_id_1 in enumerate(experience_ids):
                experience = all_experiences[exp_id_1]
                for exp_id_2, similarity in experience.similar_experiences.items():
                    if exp_id_2 in self._experience_id_to_index:
                        j = self._experience_id_to_index[exp_id_2]
                        similarity_matrix[i, j] = similarity
            
            self._gpu_similarity_matrix = similarity_matrix
            
        except Exception as e:
            print(f"GPU tensor rebuild failed: {e}, disabling GPU acceleration")
            self.use_gpu = False
            self.device = 'cpu'
    
    def _sync_gpu_activations_to_experiences(self, all_experiences: Dict[str, Experience]):
        """Sync GPU activation levels back to experience objects."""
        if not self.use_gpu or self._gpu_activation_levels is None:
            return
            
        try:
            activation_levels = self._gpu_activation_levels.cpu().numpy()
            for i, activation in enumerate(activation_levels):
                if i in self._index_to_experience_id:
                    exp_id = self._index_to_experience_id[i]
                    if exp_id in all_experiences:
                        all_experiences[exp_id].activation_level = float(activation)
        except Exception as e:
            print(f"GPU activation sync failed: {e}")
    
    def adapt_parameters(self, recent_prediction_errors: List[float]):
        """
        Adapt activation parameters based on recent prediction performance.
        
        This implements the core principle: parameters adjust to optimize prediction error.
        Args:
            recent_prediction_errors: Recent prediction errors to learn from
        """
        if len(recent_prediction_errors) < 5:
            return  # Need some data to adapt
        
        # Store for trend analysis
        self.recent_prediction_errors.extend(recent_prediction_errors[-10:])
        if len(self.recent_prediction_errors) > 50:
            self.recent_prediction_errors = self.recent_prediction_errors[-25:]
        
        # Compute performance metrics
        avg_error = sum(self.recent_prediction_errors) / len(self.recent_prediction_errors)
        recent_avg = sum(self.recent_prediction_errors[-10:]) / min(10, len(self.recent_prediction_errors))
        
        # Adapt decay rate based on prediction performance
        # If predictions are getting worse, try faster decay (fresher memory)
        # If predictions are getting better, try slower decay (longer memory)
        error_trend = recent_avg - avg_error
        
        if error_trend > 0.1:  # Getting worse - try faster decay
            self.base_decay_rate = min(0.1, self.base_decay_rate * (1 + self.adaptation_rate))
        elif error_trend < -0.1:  # Getting better - try slower decay  
            self.base_decay_rate = max(0.001, self.base_decay_rate * (1 - self.adaptation_rate))
        
        # Adapt spread strength based on how well similar experiences predict
        # This is more complex and would require tracking similarity prediction success
        # For now, adapt based on error variance (high variance = need more spreading)
        error_variance = sum((e - avg_error) ** 2 for e in self.recent_prediction_errors[-10:]) / min(10, len(self.recent_prediction_errors))
        
        if error_variance > 0.1:  # High variance - try more spreading
            self.spread_strength = min(0.3, self.spread_strength * (1 + self.adaptation_rate * 0.5))
        else:  # Low variance - try less spreading
            self.spread_strength = max(0.01, self.spread_strength * (1 - self.adaptation_rate * 0.5))
    
    def activate_experience(self, experience: Experience, strength: float = 1.0):
        """
        Activate an experience (bring into working memory).
        
        Args:
            experience: The experience to activate
            strength: Activation strength (0.0-1.0)
        """
        # Update the experience activation directly
        experience.activate(strength)
        
        # Add to spreading queue for later propagation
        self._spread_queue.append((experience.experience_id, strength))
        
        # Track activation history
        current_time = time.time()
        self._activation_history[experience.experience_id].append((current_time, strength))
        
        # Limit history size
        if len(self._activation_history[experience.experience_id]) > 100:
            self._activation_history[experience.experience_id] = \
                self._activation_history[experience.experience_id][-50:]
    
    def update_all_activations(self, all_experiences: Dict[str, Experience]):
        """
        Update activation levels for all experiences.
        
        This applies natural decay and spreading activation.
        
        Args:
            all_experiences: Dictionary of experience_id -> Experience
        """
        current_time = time.time()
        time_delta = current_time - self._last_update
        
        if time_delta < 0.1:  # Don't update too frequently
            return
        
        # Check if we should upgrade to GPU based on experience count
        self._check_and_upgrade_to_gpu(len(all_experiences))
        
        # Use GPU acceleration for large experience sets
        if self.use_gpu:
            self._gpu_update_activations(all_experiences, time_delta)
        else:
            self._cpu_update_activations(all_experiences, time_delta)
        
        self._last_update = current_time
    
    def _gpu_update_activations(self, all_experiences: Dict[str, Experience], time_delta: float):
        """GPU-accelerated activation updates."""
        try:
            # Rebuild tensors if experience set changed significantly
            if (self._gpu_activation_levels is None or 
                len(all_experiences) != self._gpu_activation_levels.shape[0]):
                self._rebuild_gpu_tensors(all_experiences)
                
            if self._gpu_activation_levels is None:
                return  # No experiences to process
            
            # Update GPU activation levels from current experience states
            for exp_id, experience in all_experiences.items():
                if exp_id in self._experience_id_to_index:
                    idx = self._experience_id_to_index[exp_id]
                    self._gpu_activation_levels[idx] = experience.activation_level
            
            # Apply decay (vectorized) with mixed precision computation
            decay_amount = torch.tensor(self.base_decay_rate * time_delta, dtype=self.compute_dtype, device=self.device)
            activations_compute = self._gpu_activation_levels.to(self.compute_dtype)
            activations_compute = torch.clamp(activations_compute - decay_amount, min=0.0)
            self._gpu_activation_levels = activations_compute.to(self.storage_dtype)
            
            # Process spreading activation (vectorized)
            self._gpu_process_spreading_activation()
            
            # Apply minimum activation threshold (vectorized)
            mask = (self._gpu_activation_levels > 0) & (self._gpu_activation_levels < self.min_activation)
            self._gpu_activation_levels[mask] = 0.0
            
            # Sync results back to experience objects
            self._sync_gpu_activations_to_experiences(all_experiences)
            
        except Exception as e:
            print(f"GPU activation update failed: {e}, falling back to CPU")
            self._cpu_update_activations(all_experiences, time_delta)
    
    def _cpu_update_activations(self, all_experiences: Dict[str, Experience], time_delta: float):
        """CPU fallback activation updates."""
        # Apply natural decay to all experiences
        self._apply_decay(all_experiences, time_delta)
        
        # Process spreading activation queue
        self._process_spreading_activation(all_experiences)
        
        # Cleanup very low activations
        self._cleanup_weak_activations(all_experiences)
    
    def _gpu_process_spreading_activation(self):
        """GPU-accelerated spreading activation."""
        if not self._spread_queue or self._gpu_similarity_matrix is None:
            return
            
        try:
            # Process spreading activation for each queued experience
            for exp_id, source_strength in self._spread_queue:
                if exp_id in self._experience_id_to_index:
                    source_idx = self._experience_id_to_index[exp_id]
                    
                    # Get similarity connections for this source experience
                    similarities = self._gpu_similarity_matrix[source_idx]
                    
                    # Apply similarity threshold (only spread to reasonably similar)
                    similarity_mask = similarities > 0.3
                    
                    # Calculate spread amounts (vectorized) with mixed precision
                    source_strength_tensor = torch.tensor(source_strength, dtype=self.compute_dtype, device=self.device)
                    spread_strength_tensor = torch.tensor(self.spread_strength, dtype=self.compute_dtype, device=self.device)
                    similarities_compute = similarities.to(self.compute_dtype)
                    spread_amounts = source_strength_tensor * similarities_compute * spread_strength_tensor
                    
                    # Apply minimum spread threshold
                    spread_mask = spread_amounts > 0.01
                    
                    # Combine masks
                    final_mask = similarity_mask & spread_mask
                    
                    # Apply spreading activation with mixed precision
                    activations_compute = self._gpu_activation_levels.to(self.compute_dtype)
                    activations_compute[final_mask] += spread_amounts[final_mask]
                    self._gpu_activation_levels = activations_compute.to(self.storage_dtype)
            
            # Clear the queue
            self._spread_queue.clear()
            
        except Exception as e:
            print(f"GPU spreading activation failed: {e}")
            # Fallback to CPU processing
            self._spread_queue.clear()  # Prevent infinite loop
    
    def get_activated_experiences(self, all_experiences: Dict[str, Experience], 
                                min_activation: float = 0.1) -> List[Experience]:
        """
        Get currently activated experiences (working memory).
        
        Args:
            all_experiences: All available experiences
            min_activation: Minimum activation level to include
            
        Returns:
            List of activated experiences, sorted by activation level (highest first)
        """
        activated = []
        for experience in all_experiences.values():
            if experience.activation_level >= min_activation:
                activated.append(experience)
        
        # Sort by activation level (highest first)
        activated.sort(key=lambda exp: exp.activation_level, reverse=True)
        return activated
    
    def get_working_memory_size(self, all_experiences: Dict[str, Experience], 
                               min_activation: float = 0.1) -> int:
        """Get the current working memory size (number of activated experiences)."""
        return len(self.get_activated_experiences(all_experiences, min_activation))
    
    def spread_activation_from_similar(self, source_experience: Experience,
                                     similar_experiences: List[Tuple[Experience, float]]):
        """
        Spread activation to similar experiences.
        
        Args:
            source_experience: Experience to spread activation from
            similar_experiences: List of (experience, similarity_score) tuples
        """
        source_activation = source_experience.activation_level
        
        if source_activation < self.min_activation:
            return  # Not enough activation to spread
        
        for target_experience, similarity in similar_experiences:
            # Activation spreads proportional to similarity and source activation
            spread_amount = (source_activation * similarity * self.spread_strength)
            
            if spread_amount > 0.01:  # Only spread significant amounts
                target_experience.activate(spread_amount)
    
    def boost_activation_by_prediction_error(self, experience: Experience):
        """
        Boost activation based on prediction error - surprising experiences get more attention.
        
        This uses adaptive boosting based on current system performance.
        Args:
            experience: Experience to potentially boost
        """
        # Adaptive boost calculation based on system learning state
        # Higher spread strength means we should boost more for pattern discovery
        adaptive_boost_factor = self.spread_strength * 2.0  # Scale with spread strength
        error_boost = experience.prediction_error * adaptive_boost_factor
        
        # Only boost if it's meaningful relative to current min activation threshold
        if error_boost > self.min_activation:
            experience.activate(error_boost)
            if error_boost > 0.1:  # Only log significant boosts
                print(f"Activation boost applied: {experience.experience_id[:8]} "
                      f"error={experience.prediction_error:.3f} boost={error_boost:.3f}")
    
    def get_activation_statistics(self, all_experiences: Dict[str, Experience]) -> Dict:
        """Get comprehensive activation statistics."""
        activations = [exp.activation_level for exp in all_experiences.values()]
        
        if not activations:
            return {
                'total_experiences': 0,
                'activated_count': 0,
                'avg_activation': 0.0,
                'max_activation': 0.0,
                'working_memory_size': 0
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
            }
        }
    
    def _apply_decay(self, all_experiences: Dict[str, Experience], time_delta: float):
        """Apply natural activation decay to all experiences."""
        decay_amount = self.base_decay_rate * time_delta
        
        for experience in all_experiences.values():
            experience.decay_activation(decay_amount)
    
    def _process_spreading_activation(self, all_experiences: Dict[str, Experience]):
        """Process the spreading activation queue."""
        if not self._spread_queue:
            return
        
        # Process each experience in the spread queue
        processed = []
        
        for exp_id, source_strength in self._spread_queue:
            source_exp = all_experiences.get(exp_id)
            if not source_exp:
                continue
            
            # Try to get cached similar experiences first
            cached_similar = self._cache_adapter.get_similar_experiences(exp_id)
            
            if cached_similar is not None:
                # Use cached similarities
                similar_experiences = cached_similar
            else:
                # Fall back to experience object similarities
                similar_experiences = source_exp.similar_experiences
                
                # Cache for future use with utility based on activation level
                utility_score = min(1.0, source_exp.activation_level * 2.0)
                self._cache_adapter.cache_similar_experiences(exp_id, similar_experiences, utility_score)
            
            # Spread activation to similar experiences
            for similar_id, similarity in similar_experiences.items():
                target_exp = all_experiences.get(similar_id)
                if target_exp and similarity > 0.3:  # Only spread to reasonably similar
                    spread_amount = source_strength * similarity * self.spread_strength
                    if spread_amount > 0.01:
                        target_exp.activate(spread_amount)
            
            processed.append((exp_id, source_strength))
        
        # Clear the queue
        self._spread_queue.clear()
    
    def _cleanup_weak_activations(self, all_experiences: Dict[str, Experience]):
        """Clean up experiences with very low activation levels."""
        for experience in all_experiences.values():
            if 0 < experience.activation_level < self.min_activation:
                experience.activation_level = 0.0  # Clear very weak activations
    
    def get_most_activated_contexts(self, all_experiences: Dict[str, Experience], 
                                  count: int = 5) -> List[Tuple[str, float, List[float]]]:
        """
        Get the context vectors of the most activated experiences.
        
        Args:
            all_experiences: All available experiences
            count: Number of top activated contexts to return
            
        Returns:
            List of (experience_id, activation_level, context_vector) tuples
        """
        activated = self.get_activated_experiences(all_experiences)
        
        results = []
        for exp in activated[:count]:
            results.append((
                exp.experience_id,
                exp.activation_level,
                exp.get_context_vector()
            ))
        
        return results
    
    def is_in_working_memory(self, experience: Experience, threshold: float = 0.1) -> bool:
        """Check if an experience is currently in working memory."""
        return experience.activation_level >= threshold
    
    def force_activate_experiences(self, experience_ids: List[str], 
                                 all_experiences: Dict[str, Experience],
                                 strength: float = 0.8):
        """
        Force activate a set of experiences (for bootstrapping or special situations).
        
        Args:
            experience_ids: List of experience IDs to activate
            all_experiences: All available experiences  
            strength: Activation strength to apply
        """
        activated_count = 0
        for exp_id in experience_ids:
            experience = all_experiences.get(exp_id)
            if experience:
                self.activate_experience(experience, strength)
                activated_count += 1
        
        if activated_count > 0:
            print(f"🔥 Force activated {activated_count} experiences")
    
    def clear_all_activations(self, all_experiences: Dict[str, Experience]):
        """Clear all activations (for testing or reset)."""
        for experience in all_experiences.values():
            experience.activation_level = 0.0
        
        self._spread_queue.clear()
        self._activation_history.clear()
        
        # Clear GPU tensors
        if self.use_gpu:
            self._gpu_activation_levels = None
            self._gpu_similarity_matrix = None
            self._experience_id_to_index.clear()
            self._index_to_experience_id.clear()
        
        print("🧹 All activations cleared (including GPU tensors)")
    
    def __str__(self) -> str:
        return f"ActivationDynamics(decay_rate={self.base_decay_rate}, spread_strength={self.spread_strength})"
    
    def __repr__(self) -> str:
        return self.__str__()