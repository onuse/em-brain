"""
GPU Pattern Analysis for Stream Processing

Detects recurring patterns in experience streams and pre-computes predictions
for common sequences. This is where real intelligence emerges - the brain
learns to recognize familiar situations and predict what comes next.

Examples:
- "When I see this sensory pattern, action X usually follows"
- "This sequence of experiences typically leads to this outcome"
- "In this context, these actions have high utility"

GPU-accelerated for real-time pattern recognition in massive experience streams.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import time
from collections import defaultdict, deque
from dataclasses import dataclass

# GPU acceleration with CUDA/MPS detection
try:
    import torch
    TORCH_AVAILABLE = True
    
    # Device selection hierarchy: CUDA > MPS > CPU
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    
    # Test GPU functionality
    GPU_FUNCTIONAL = False
    PREFERRED_DEVICE = 'cpu'
    
    if CUDA_AVAILABLE:
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('cuda')
            _ = test_tensor + 1
            GPU_FUNCTIONAL = True
            PREFERRED_DEVICE = 'cuda'
        except Exception:
            CUDA_AVAILABLE = False
    
    if not GPU_FUNCTIONAL and MPS_AVAILABLE:
        try:
            test_tensor = torch.tensor([1.0, 2.0, 3.0]).to('mps')
            _ = test_tensor + 1
            GPU_FUNCTIONAL = True
            PREFERRED_DEVICE = 'mps'
        except Exception:
            MPS_AVAILABLE = False
            
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
    MPS_AVAILABLE = False
    GPU_FUNCTIONAL = False
    PREFERRED_DEVICE = 'cpu'


@dataclass
class PatternSequence:
    """Represents a learned pattern sequence."""
    pattern_id: str
    context_vectors: List[np.ndarray]  # Sequence of sensory contexts
    action_vectors: List[np.ndarray]   # Sequence of actions taken
    outcome_vectors: List[np.ndarray]  # Sequence of outcomes
    frequency: int                     # How often this pattern occurs
    success_rate: float               # How successful this pattern is
    last_seen: float                  # Timestamp of last occurrence
    prediction_utility: float        # How useful for prediction


class GPUPatternAnalyzer:
    """
    GPU-accelerated pattern analysis for intelligent stream processing.
    
    Core principle: Real intelligence comes from recognizing patterns and
    predicting what happens next. This system learns common sequences
    and pre-computes predictions for familiar situations.
    """
    
    def __init__(self, use_gpu: bool = True, use_mixed_precision: bool = True):
        """
        Initialize GPU pattern analyzer.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            use_mixed_precision: Whether to use FP16 for memory efficiency
        """
        # GPU configuration - lazy initialization
        self.gpu_capable = use_gpu and GPU_FUNCTIONAL
        self.use_gpu = False  # Start with CPU, upgrade when pattern set is large enough
        self.device = 'cpu'  # Start with CPU
        self.use_mixed_precision = use_mixed_precision
        self.gpu_device = PREFERRED_DEVICE if self.gpu_capable else 'cpu'
        
        # Precision configuration - biological neural noise simulation
        self.compute_dtype = torch.float16 if self.use_mixed_precision else torch.float32
        self.storage_dtype = torch.float32  # Critical patterns stored in FP32
        
        # Pattern learning parameters
        self.min_pattern_length = 2      # Minimum sequence length to consider
        self.max_pattern_length = 5      # Maximum sequence length to track
        self.min_frequency = 3           # Minimum occurrences to consider a pattern
        self.similarity_threshold = 0.7  # How similar experiences must be to match pattern
        
        # Pattern storage
        self.learned_patterns: Dict[str, PatternSequence] = {}
        self.pattern_frequency: Dict[str, int] = defaultdict(int)
        self.experience_stream: deque = deque(maxlen=1000)  # Recent experience buffer
        
        # GPU tensors for fast pattern matching
        self._gpu_pattern_embeddings = None    # Pattern representations
        self._gpu_stream_buffer = None         # Recent experiences tensor
        self._pattern_id_to_index = {}         # Map pattern IDs to tensor indices
        self._stream_similarity_cache = {}     # Cache for stream similarity computations
        
        # Performance tracking
        self.patterns_discovered = 0
        self.pattern_predictions_made = 0
        self.pattern_prediction_accuracy = 0.0
        self.stream_processing_time = 0.0
        
        # Adaptive parameters (Strategy 5)
        self.pattern_learning_rate = 0.1
        self.pattern_decay_rate = 0.95      # How quickly unused patterns fade
        self.prediction_confidence_threshold = 0.6
        
        precision_info = f"FP16 compute, FP32 storage" if self.use_mixed_precision else "FP32"
        gpu_status = f"GPU capable: {self.gpu_capable} (lazy initialization enabled)"
        print(f"🔍 GPUPatternAnalyzer initialized - learning recurring sequences ({gpu_status}, {precision_info})")
    
    def _check_and_upgrade_to_gpu(self, num_patterns: int):
        """Check if we should upgrade to GPU based on number of patterns."""
        if not self.gpu_capable or self.use_gpu:
            return  # Already using GPU or not capable
        
        # Check with hardware adaptation system
        try:
            from ..utils.hardware_adaptation import should_use_gpu_for_pattern_analysis
            if should_use_gpu_for_pattern_analysis(num_patterns):
                self._upgrade_to_gpu()
        except ImportError:
            # Fallback to simple threshold
            if num_patterns >= 10:
                self._upgrade_to_gpu()
    
    def _upgrade_to_gpu(self):
        """Upgrade from CPU to GPU processing."""
        if not self.gpu_capable or self.use_gpu:
            return
        
        print(f"🚀 Upgrading pattern analysis to GPU ({self.gpu_device}) - pattern set large enough to benefit")
        
        self.use_gpu = True
        self.device = self.gpu_device
        
        # Rebuild existing GPU tensors if we have patterns
        if len(self.learned_patterns) > 0:
            self._rebuild_gpu_pattern_tensors()
    
    def add_experience_to_stream(self, experience_data: Dict[str, Any]):
        """
        Add new experience to the stream and analyze for patterns.
        
        Args:
            experience_data: Dictionary containing sensory_input, action_taken, outcome, etc.
        """
        # Extract vectors from experience
        context_vector = np.array(experience_data['sensory_input'])
        action_vector = np.array(experience_data.get('action_taken', []))
        outcome_vector = np.array(experience_data.get('outcome', []))
        timestamp = experience_data.get('timestamp', time.time())
        
        # Add to stream buffer with consistent key names
        stream_entry = {
            'context': context_vector,
            'action': action_vector,
            'outcome': outcome_vector,
            'timestamp': timestamp,
            'experience_id': experience_data.get('experience_id', f'exp_{len(self.experience_stream)}')
        }
        
        self.experience_stream.append(stream_entry)
        
        # Analyze for patterns once we have enough data
        if len(self.experience_stream) >= self.min_pattern_length:
            self._analyze_stream_for_patterns()
    
    def _analyze_stream_for_patterns(self):
        """Analyze recent experience stream for recurring patterns."""
        start_time = time.time()
        
        # Look for patterns of different lengths
        for pattern_length in range(self.min_pattern_length, 
                                   min(self.max_pattern_length + 1, len(self.experience_stream))):
            self._extract_patterns_of_length(pattern_length)
        
        # Update pattern statistics
        self._update_pattern_statistics()
        
        # Check if we should upgrade to GPU based on number of patterns
        self._check_and_upgrade_to_gpu(len(self.learned_patterns))
        
        # Rebuild GPU tensors if patterns changed significantly
        if len(self.learned_patterns) > 0 and self.use_gpu:
            self._rebuild_gpu_pattern_tensors()
        
        self.stream_processing_time = time.time() - start_time
    
    def _extract_patterns_of_length(self, pattern_length: int):
        """Extract patterns of specific length from experience stream."""
        stream_list = list(self.experience_stream)
        
        # Sliding window over recent experiences
        for i in range(len(stream_list) - pattern_length + 1):
            sequence = stream_list[i:i + pattern_length]
            
            # Create pattern signature
            pattern_signature = self._create_pattern_signature(sequence)
            
            # Check if this pattern already exists or is similar to existing ones
            similar_pattern_id = self._find_similar_pattern(pattern_signature, sequence)
            
            if similar_pattern_id:
                # Update existing pattern
                self._update_pattern(similar_pattern_id, sequence)
            else:
                # Create new pattern if we've seen this sequence enough times
                temp_pattern_id = f"temp_{pattern_signature}"
                self.pattern_frequency[temp_pattern_id] += 1
                
                if self.pattern_frequency[temp_pattern_id] >= self.min_frequency:
                    # Promote to learned pattern
                    self._create_new_pattern(pattern_signature, sequence)
    
    def _create_pattern_signature(self, sequence: List[Dict]) -> str:
        """Create a hash-like signature for a pattern sequence."""
        # Simple signature based on context similarity
        signature_parts = []
        for exp in sequence:
            context_hash = hash(tuple(np.round(exp['context'], 2)))  # Round for slight noise tolerance
            signature_parts.append(str(context_hash))
        
        return "_".join(signature_parts)
    
    def _find_similar_pattern(self, pattern_signature: str, sequence: List[Dict]) -> Optional[str]:
        """Find existing pattern similar to this sequence."""
        # First try exact signature match
        if pattern_signature in self.learned_patterns:
            return pattern_signature
        
        # Then try similarity-based matching
        for pattern_id, pattern in self.learned_patterns.items():
            if len(pattern.context_vectors) == len(sequence):
                # Check similarity of each step in sequence
                similarities = []
                for i, exp in enumerate(sequence):
                    context_sim = self._compute_vector_similarity(
                        exp['context'], pattern.context_vectors[i]
                    )
                    similarities.append(context_sim)
                
                avg_similarity = np.mean(similarities)
                if avg_similarity > self.similarity_threshold:
                    return pattern_id
        
        return None
    
    def _compute_vector_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        cosine_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        return (cosine_sim + 1.0) / 2.0  # Convert to 0-1 range
    
    def _create_new_pattern(self, pattern_signature: str, sequence: List[Dict]):
        """Create a new learned pattern from sequence."""
        context_vectors = [exp['context'] for exp in sequence]
        action_vectors = [exp['action'] for exp in sequence if len(exp['action']) > 0]
        outcome_vectors = [exp['outcome'] for exp in sequence if len(exp['outcome']) > 0]
        
        # Calculate initial success rate (placeholder - would be computed from actual outcomes)
        success_rate = 0.5  # Start neutral, will adapt based on usage
        
        pattern = PatternSequence(
            pattern_id=pattern_signature,
            context_vectors=context_vectors,
            action_vectors=action_vectors,
            outcome_vectors=outcome_vectors,
            frequency=self.pattern_frequency[f"temp_{pattern_signature}"],
            success_rate=success_rate,
            last_seen=time.time(),
            prediction_utility=0.5  # Will adapt based on prediction success
        )
        
        self.learned_patterns[pattern_signature] = pattern
        self.patterns_discovered += 1
        
        print(f"🔍 New pattern discovered: {pattern_signature[:20]}... (length={len(sequence)}, freq={pattern.frequency})")
    
    def _update_pattern(self, pattern_id: str, sequence: List[Dict]):
        """Update an existing pattern with new occurrence."""
        if pattern_id in self.learned_patterns:
            pattern = self.learned_patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_seen = time.time()
            
            # Adaptive update of pattern statistics
            # (In a full implementation, this would update based on actual prediction outcomes)
    
    def _update_pattern_statistics(self):
        """Update pattern statistics and apply decay to unused patterns."""
        current_time = time.time()
        patterns_to_remove = []
        
        for pattern_id, pattern in self.learned_patterns.items():
            # Apply time-based decay to unused patterns
            time_since_seen = current_time - pattern.last_seen
            if time_since_seen > 3600:  # 1 hour
                decay_factor = np.exp(-time_since_seen / 3600 * self.pattern_decay_rate)
                pattern.frequency = int(pattern.frequency * decay_factor)
                
                # Remove patterns that have decayed too much
                if pattern.frequency < self.min_frequency:
                    patterns_to_remove.append(pattern_id)
        
        # Clean up decayed patterns
        for pattern_id in patterns_to_remove:
            del self.learned_patterns[pattern_id]
    
    def _rebuild_gpu_pattern_tensors(self):
        """Rebuild GPU tensors when patterns change significantly."""
        if not self.use_gpu or len(self.learned_patterns) == 0:
            return
        
        try:
            # Build pattern embedding matrix
            pattern_embeddings = []
            pattern_ids = list(self.learned_patterns.keys())
            
            self._pattern_id_to_index = {pid: i for i, pid in enumerate(pattern_ids)}
            
            for pattern_id in pattern_ids:
                pattern = self.learned_patterns[pattern_id]
                # Create embedding from first context vector (could be more sophisticated)
                if len(pattern.context_vectors) > 0:
                    embedding = pattern.context_vectors[0]
                    pattern_embeddings.append(embedding)
            
            if pattern_embeddings:
                self._gpu_pattern_embeddings = torch.tensor(
                    np.array(pattern_embeddings), dtype=self.storage_dtype, device=self.device
                )
                
        except Exception as e:
            print(f"GPU pattern tensor rebuild failed: {e}, continuing with CPU")
            self.use_gpu = False
    
    def predict_next_experience(self, current_context: np.ndarray, 
                              recent_sequence: List[Dict]) -> Optional[Dict]:
        """
        Predict next experience based on learned patterns.
        
        Args:
            current_context: Current sensory context
            recent_sequence: Recent sequence of experiences
            
        Returns:
            Predicted next experience or None if no pattern matches
        """
        if len(self.learned_patterns) == 0:
            return None
        
        # Find matching patterns for current context and sequence
        matching_patterns = self._find_matching_patterns(current_context, recent_sequence)
        
        if not matching_patterns:
            return None
        
        # Select best pattern based on frequency and utility
        best_pattern = max(matching_patterns, 
                          key=lambda p: p.frequency * p.prediction_utility)
        
        # Generate prediction from pattern
        prediction = self._generate_prediction_from_pattern(best_pattern, current_context)
        
        if prediction:
            self.pattern_predictions_made += 1
            
        return prediction
    
    def _find_matching_patterns(self, current_context: np.ndarray, 
                               recent_sequence: List[Dict]) -> List[PatternSequence]:
        """Find patterns that match current context and sequence."""
        matching_patterns = []
        
        for pattern in self.learned_patterns.values():
            # Check if pattern length matches available sequence
            if len(pattern.context_vectors) > len(recent_sequence) + 1:
                continue
            
            # Check similarity of context sequence
            if len(recent_sequence) > 0:
                sequence_matches = True
                for i, exp in enumerate(recent_sequence[-(len(pattern.context_vectors)-1):]):
                    pattern_context = pattern.context_vectors[i]
                    similarity = self._compute_vector_similarity(exp['context'], pattern_context)
                    if similarity < self.similarity_threshold:
                        sequence_matches = False
                        break
                
                if not sequence_matches:
                    continue
            
            # Check current context similarity
            current_similarity = self._compute_vector_similarity(
                current_context, pattern.context_vectors[-1]
            )
            
            if current_similarity > self.similarity_threshold:
                matching_patterns.append(pattern)
        
        return matching_patterns
    
    def _generate_prediction_from_pattern(self, pattern: PatternSequence, 
                                        current_context: np.ndarray) -> Optional[Dict]:
        """Generate prediction based on learned pattern."""
        if len(pattern.action_vectors) == 0 or len(pattern.outcome_vectors) == 0:
            return None
        
        # Predict next action (average of actions in pattern)
        predicted_action = np.mean(pattern.action_vectors, axis=0)
        
        # Predict outcome (average of outcomes in pattern)
        predicted_outcome = np.mean(pattern.outcome_vectors, axis=0)
        
        # Calculate confidence based on pattern frequency and utility
        confidence = min(1.0, (pattern.frequency / 10.0) * pattern.prediction_utility)
        
        if confidence < self.prediction_confidence_threshold:
            return None
        
        return {
            'predicted_action': predicted_action.tolist(),
            'predicted_outcome': predicted_outcome.tolist(),
            'confidence': confidence,
            'pattern_id': pattern.pattern_id,
            'pattern_frequency': pattern.frequency,
            'prediction_method': 'pattern_analysis'
        }
    
    def record_prediction_outcome(self, pattern_id: str, prediction_success: float):
        """Record how well a pattern-based prediction worked."""
        if pattern_id in self.learned_patterns:
            pattern = self.learned_patterns[pattern_id]
            
            # Update prediction utility using running average
            old_utility = pattern.prediction_utility
            pattern.prediction_utility = (old_utility * 0.9 + prediction_success * 0.1)
            
            # Update overall accuracy
            if self.pattern_predictions_made > 0:
                self.pattern_prediction_accuracy = (
                    (self.pattern_prediction_accuracy * (self.pattern_predictions_made - 1) + 
                     prediction_success) / self.pattern_predictions_made
                )
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pattern analysis statistics."""
        if len(self.learned_patterns) == 0:
            return {
                'total_patterns': 0,
                'patterns_discovered': self.patterns_discovered,
                'predictions_made': self.pattern_predictions_made,
                'prediction_accuracy': 0.0,
                'avg_processing_time_ms': 0.0,
                'gpu_acceleration': self.use_gpu,
                'system_type': 'gpu_pattern_analysis'
            }
        
        # Analyze pattern characteristics
        pattern_lengths = [len(p.context_vectors) for p in self.learned_patterns.values()]
        pattern_frequencies = [p.frequency for p in self.learned_patterns.values()]
        pattern_utilities = [p.prediction_utility for p in self.learned_patterns.values()]
        
        return {
            'total_patterns': len(self.learned_patterns),
            'patterns_discovered': self.patterns_discovered,
            'predictions_made': self.pattern_predictions_made,
            'prediction_accuracy': self.pattern_prediction_accuracy,
            'avg_pattern_length': np.mean(pattern_lengths) if pattern_lengths else 0,
            'avg_pattern_frequency': np.mean(pattern_frequencies) if pattern_frequencies else 0,
            'avg_pattern_utility': np.mean(pattern_utilities) if pattern_utilities else 0,
            'most_frequent_pattern': max(self.learned_patterns.values(), 
                                       key=lambda p: p.frequency).pattern_id if self.learned_patterns else None,
            'avg_processing_time_ms': self.stream_processing_time * 1000,
            'gpu_acceleration': self.use_gpu,
            'mixed_precision': self.use_mixed_precision,
            'system_type': 'gpu_pattern_analysis'
        }
    
    def reset_patterns(self):
        """Reset all learned patterns."""
        self.learned_patterns.clear()
        self.pattern_frequency.clear()
        self.experience_stream.clear()
        
        # Reset GPU tensors
        if self.use_gpu:
            self._gpu_pattern_embeddings = None
            self._gpu_stream_buffer = None
            self._pattern_id_to_index.clear()
            self._stream_similarity_cache.clear()
        
        # Reset statistics
        self.patterns_discovered = 0
        self.pattern_predictions_made = 0
        self.pattern_prediction_accuracy = 0.0
        
        print("🔍 Pattern analysis reset - ready for fresh pattern discovery")