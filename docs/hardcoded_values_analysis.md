# Hardcoded Values Analysis for Brain Codebase

This analysis identifies all hardcoded numerical thresholds, magic numbers, and arbitrary constants that could potentially be made adaptive.

## 1. ACTIVATION SYSTEM

### activation/dynamics.py
- **Line 65**: `self.base_decay_rate = 0.02` - Base decay rate for activation levels
  - Controls: How fast experiences lose activation when not accessed
  - Candidate: YES - Could adapt based on memory effectiveness
  
- **Line 66**: `self.spread_strength = 0.1` - Strength of spreading activation
  - Controls: How much activation spreads to similar experiences
  - Candidate: YES - Could adapt based on learning success

- **Line 67**: `self.min_activation = 0.01` - Minimum activation threshold
  - Controls: When experiences are considered "inactive"
  - Candidate: YES - Could adapt based on memory effectiveness

- **Line 71**: `self.adaptation_rate = 0.1` - Rate of parameter adaptation
  - Controls: How fast the system adapts its parameters
  - Candidate: YES - Meta-learning parameter

- **Line 158**: `if len(recent_prediction_errors) < 5:` - Minimum data for adaptation
  - Controls: When adaptation can start
  - Candidate: MAYBE - Could be based on variance instead

- **Line 163**: `if len(self.recent_prediction_errors) > 50:` - History size limit
  - Controls: Memory consumption for error tracking
  - Candidate: NO - Infrastructure limit

- **Line 175**: `if error_trend > 0.1:` - Error trend threshold for decay adaptation
  - Controls: When to change decay rate
  - Candidate: YES - Could be relative to baseline

- **Line 185**: `if error_variance > 0.1:` - Error variance threshold for spread adaptation
  - Controls: When to change spread strength
  - Candidate: YES - Could be adaptive

- **Line 190**: `strength: float = 1.0` - Default activation strength
  - Controls: Default activation when bringing into working memory
  - Candidate: MAYBE - Could be context-dependent

- **Line 209**: `if len(self._activation_history[experience.experience_id]) > 100:` - History size
  - Controls: Memory consumption per experience
  - Candidate: NO - Infrastructure limit

- **Line 225**: `if time_delta < 0.1:` - Minimum update interval
  - Controls: Update frequency
  - Candidate: MAYBE - Could be performance-based

- **Line 229**: `if self.use_gpu and len(all_experiences) > 20:` - GPU threshold
  - Controls: When to use GPU vs CPU
  - Candidate: YES - Could be based on measured performance

- **Line 263**: `mask = (self._gpu_activation_levels > 0) & (self._gpu_activation_levels < self.min_activation)`
  - Controls: Cleanup threshold (uses min_activation)
  - Candidate: Already adaptive via min_activation

- **Line 299**: `similarity_mask = similarities > 0.3` - Similarity threshold for spreading
  - Controls: Which experiences receive spread activation
  - Candidate: YES - Could adapt based on network density

- **Line 308**: `spread_mask = spread_amounts > 0.01` - Minimum spread amount
  - Controls: Computational efficiency threshold
  - Candidate: MAYBE - Could be relative to total activation

- **Line 327**: `min_activation: float = 0.1` - Working memory threshold
  - Controls: What's considered "in working memory"
  - Candidate: YES - Could adapt based on capacity needs

- **Line 370**: `if spread_amount > 0.01:` - Minimum spread threshold
  - Controls: Computational efficiency
  - Candidate: Same as line 308

- **Line 389**: `if error_boost > 0.1:` - Logging threshold
  - Controls: When to log activation boosts
  - Candidate: NO - Logging preference

- **Line 407**: `working_memory_count = sum(1 for a in activations if a >= 0.1)`
  - Controls: Working memory size calculation
  - Candidate: Same as line 327

- **Lines 416-421**: Activation distribution thresholds (0.8, 0.5, 0.2, 0.1)
  - Controls: Categorization of activation levels
  - Candidate: YES - Could be based on distribution percentiles

- **Line 448**: `if target_exp and similarity > 0.3:` - Similarity threshold
  - Controls: Same as line 299
  - Candidate: YES

- **Line 450**: `if spread_amount > 0.01:` - Minimum spread
  - Controls: Same as line 308
  - Candidate: Same as line 308

- **Line 461**: `if 0 < experience.activation_level < self.min_activation:`
  - Controls: Cleanup threshold
  - Candidate: Already adaptive via min_activation

- **Line 488**: `threshold: float = 0.1` - Working memory check
  - Controls: Same as line 327
  - Candidate: Same as line 327

- **Line 494**: `strength: float = 0.8` - Force activation strength
  - Controls: Bootstrap activation strength
  - Candidate: MAYBE - Could be based on confidence

### activation/utility_based_activation.py
- **Line 81-82**: `self.initial_utility_learning_rate = 0.1`, `self.utility_learning_rate = 0.1`
  - Controls: How fast utility scores are learned
  - Candidate: YES - Already has meta-learning

- **Line 83**: `self.activation_persistence_factor = 0.9`
  - Controls: How long utility-based activation persists
  - Candidate: YES - Could adapt based on memory needs

- **Line 86**: `self.learning_rate_adaptation_rate = 0.1`
  - Controls: Meta-learning speed
  - Candidate: MAYBE - Meta-meta-learning?

- **Line 87-88**: `self.min_utility_learning_rate = 0.01`, `self.max_utility_learning_rate = 0.5`
  - Controls: Bounds on learning rate adaptation
  - Candidate: MAYBE - Could be based on stability

- **Line 241-247**: Utility combination weights (0.4, 0.2, 0.2, 0.1, 0.1)
  - Controls: How different utility sources are weighted
  - Candidate: YES - Could be learned from effectiveness

- **Line 260**: `utility_threshold = 0.1`
  - Controls: Minimum utility for activation
  - Candidate: YES - Could adapt based on resource constraints

- **Line 310**: `if utility_score > 0.1:` - Activation threshold
  - Controls: Same as line 260
  - Candidate: Same as line 260

- **Line 347-351**: Utility weights (0.4, 0.2, 0.2, 0.1, 0.1)
  - Controls: Same as lines 241-247
  - Candidate: Same as lines 241-247

- **Line 366-367**: Recent utility weighting (0.7 recent, 0.3 overall)
  - Controls: Recency bias in historical utility
  - Candidate: YES - Could adapt based on volatility

- **Line 385**: `if abs(activation_level - current_activation) < 0.2:`
  - Controls: Activation level similarity for success lookup
  - Candidate: MAYBE - Could be based on granularity

- **Line 405**: `return np.mean(connected_utilities) * 0.5` - Connection boost factor
  - Controls: How much connections boost utility
  - Candidate: YES - Could be learned

- **Line 424**: `if new_activation > 0.05:` - Decay cutoff
  - Controls: When to remove from active set
  - Candidate: YES - Could adapt based on capacity

- **Line 453**: `if len(self.prediction_utility_history[exp_id]) > 50:`
  - Controls: History size limit
  - Candidate: NO - Infrastructure limit

- **Line 465**: `if len(self.activation_success_tracking[rounded_activation]) > 20:`
  - Controls: Success tracking history size
  - Candidate: NO - Infrastructure limit

- **Line 484**: `if prediction_success > 0.6:` - Success threshold for connection learning
  - Controls: When to strengthen connections
  - Candidate: YES - Could adapt based on performance distribution

- **Line 517**: `if len(self.utility_learning_success_history) > 100:`
  - Controls: Meta-learning history size
  - Candidate: NO - Infrastructure limit

- **Line 521**: `if len(self.utility_learning_success_history) >= 10:`
  - Controls: When meta-learning can start
  - Candidate: MAYBE - Could be based on variance

- **Line 541-546**: Meta-learning thresholds (0.05, -0.05)
  - Controls: When to adjust learning rate
  - Candidate: YES - Could be based on noise level

- **Line 557-570**: `min_activation: float = 0.1` - Working memory thresholds
  - Controls: What's considered active
  - Candidate: Same as activation/dynamics.py line 327

- **Line 593**: `strong_connections += sum(1 for strength in connections.values() if strength > 0.5)`
  - Controls: What's considered a "strong" connection
  - Candidate: YES - Could be percentile-based

## 2. PREDICTION SYSTEM

### prediction/engine.py
- **Line 28**: `min_similar_experiences: int = 3`
  - Controls: Minimum experiences needed for consensus prediction
  - Candidate: YES - Could adapt based on confidence needs

- **Line 29**: `prediction_confidence_threshold: float = 0.3`
  - Controls: Minimum confidence to trust prediction
  - Candidate: YES - Could adapt based on risk tolerance

- **Line 30**: `success_error_threshold: float = 0.3`
  - Controls: What prediction error is considered "successful"
  - Candidate: YES - Could adapt based on task requirements

- **Line 31**: `bootstrap_randomness: float = 0.8`
  - Controls: Randomness when bootstrapping
  - Candidate: YES - Could decrease as learning progresses

- **Line 104**: `max_results=20, min_similarity=0.4`
  - Controls: Similarity search parameters
  - Candidate: YES - Could adapt based on experience density

- **Line 108**: `if self.pattern_analyzer and len(self.recent_experiences) >= 2:`
  - Controls: When pattern analysis can start
  - Candidate: MAYBE - Could be based on pattern complexity

- **Line 125**: `if pattern_prediction and pattern_prediction['confidence'] > self.prediction_confidence_threshold:`
  - Controls: Uses adaptive threshold
  - Candidate: Already adaptive

- **Line 172**: `if len(self.recent_experiences) > 50:`
  - Controls: Experience stream history size
  - Candidate: NO - Infrastructure limit

- **Line 210**: `success_weight = max(0.1, 1.0 - experience.prediction_error)`
  - Controls: Minimum success weight
  - Candidate: MAYBE - Could be based on distribution

- **Line 234**: `confidence = min(0.95, avg_similarity * (1.0 + weight_concentration * 0.1))`
  - Controls: Maximum confidence and concentration factor
  - Candidate: YES - Could be learned from calibration

- **Line 262**: `noise = np.random.normal(0, 0.3, action_dimensions)` - Exploration noise
  - Controls: Exploration amount
  - Candidate: YES - Could adapt based on learning progress

- **Line 265**: `confidence = 0.2` - Bootstrap confidence
  - Controls: Confidence when using similar experience
  - Candidate: YES - Could be based on similarity

- **Line 269**: `random_action = np.random.normal(0, 1.0, action_dimensions)` - Random action scale
  - Controls: Random action distribution
  - Candidate: YES - Could adapt to action space

- **Line 270**: `confidence = 0.1` - Pure random confidence
  - Controls: Confidence for pure random actions
  - Candidate: MAYBE - Could be slightly adaptive

- **Line 286**: `random_array = np.random.normal(0, 0.5, action_dimensions)` - Blend randomness
  - Controls: Randomness in blended actions
  - Candidate: YES - Could adapt based on confidence

- **Line 293**: `blended_confidence = confidence * 0.8` - Confidence reduction factor
  - Controls: How much blending reduces confidence
  - Candidate: MAYBE - Could be based on blend amount

- **Line 323**: `if len(self.prediction_accuracies) > 1000:`
  - Controls: Accuracy history limit
  - Candidate: NO - Infrastructure limit

## 3. SIMILARITY SYSTEM

### similarity/engine.py
- **Line 80**: `self._max_cache_size = 1000`
  - Controls: Similarity cache size
  - Candidate: NO - Infrastructure limit

- **Line 96**: `min_similarity: float = 0.3` - Default minimum similarity
  - Controls: Similarity threshold for results
  - Candidate: YES - Could adapt based on density

- **Line 125**: `if self.use_gpu and len(experience_vectors) > 50:`
  - Controls: GPU usage threshold
  - Candidate: YES - Could be based on measured performance

- **Line 154**: `min_similarity: float = 0.3` - Natural attention similarity threshold
  - Controls: Same as line 96
  - Candidate: Same as line 96

- **Line 227**: `max_distance = torch.sqrt(torch.tensor(len(target_vector) * 4.0, device=self.device))`
  - Controls: Distance normalization factor
  - Candidate: MAYBE - Could be based on actual distribution

- **Line 266**: `max_distance = np.sqrt(len(target_vector) * 4.0)`
  - Controls: Same as line 227
  - Candidate: Same as line 227

- **Line 278**: `attention_boost = max(natural_attention, base_similarity * 0.8)`
  - Controls: Hybrid mode attention boost factor
  - Candidate: YES - Could be learned

- **Line 283**: `utility_weight = experience.prediction_utility * 2.0`
  - Controls: Utility boost factor
  - Candidate: YES - Could be learned

- **Line 332**: `oldest_keys = list(self._cache.keys())[:100]`
  - Controls: Cache eviction batch size
  - Candidate: NO - Infrastructure choice

### similarity/learnable_similarity.py
- **Line 69**: `learning_rate: float = 0.01` - Initial learning rate
  - Controls: How fast similarity function learns
  - Candidate: YES - Already has meta-learning

- **Line 94**: `self.learning_rate_adaptation_rate = 0.1`
  - Controls: Meta-learning speed
  - Candidate: MAYBE - Meta-meta-learning?

- **Line 95-96**: `self.min_learning_rate = 0.001`, `self.max_learning_rate = 0.1`
  - Controls: Learning rate bounds
  - Candidate: MAYBE - Could be based on stability

- **Line 127**: `np.random.normal(0, 0.1, vector_dim)` - Initial weight randomization
  - Controls: Symmetry breaking amount
  - Candidate: NO - Initialization only

- **Line 132**: `self.interaction_matrix = np.eye(vector_dim) * 0.1`
  - Controls: Initial interaction strength
  - Candidate: NO - Initialization only

- **Line 202**: `max_distance = torch.sqrt(torch.tensor(2 * len(vec_a), dtype=torch.float32, device=self.device))`
  - Controls: Distance normalization
  - Candidate: MAYBE - Could be based on distribution

- **Line 236**: `max_distance = np.sqrt(2 * len(vec_a))`
  - Controls: Same as line 202
  - Candidate: Same as line 202

- **Line 262**: `self.similarity_predictions[round(similarity_score, 2)].append(prediction_success)`
  - Controls: Similarity score binning precision
  - Candidate: NO - Storage granularity

- **Line 274**: `if len(self.prediction_outcomes) > 1000:`
  - Controls: Outcome history limit
  - Candidate: NO - Infrastructure limit

- **Line 284**: `if len(self.prediction_outcomes) < 20:`
  - Controls: Minimum data for adaptation
  - Candidate: MAYBE - Could be based on variance

- **Line 293**: `if len(similarities) < 10:`
  - Controls: Minimum data for correlation
  - Candidate: MAYBE - Statistical requirement

- **Line 302**: `if correlation_before < 0.3:` - Poor correlation threshold
  - Controls: When to trigger adaptation
  - Candidate: YES - Could be based on noise level

- **Line 401**: `if success > 0.7 and similarity < 0.5:` - Good prediction from dissimilar
  - Controls: Adaptation trigger thresholds
  - Candidate: YES - Could be percentile-based

- **Line 407**: `elif success < 0.3 and similarity > 0.7:` - Bad prediction from similar
  - Controls: Adaptation trigger thresholds
  - Candidate: YES - Could be percentile-based

- **Line 412**: `self.feature_weights = np.maximum(0.1, self.feature_weights)`
  - Controls: Minimum feature weight
  - Candidate: MAYBE - Prevents zeros

- **Line 439**: `if outcome['success'] > 0.6:` - Success threshold for interaction learning
  - Controls: When to learn interactions
  - Candidate: YES - Could be adaptive

- **Line 454**: `learning_rate_tensor = torch.tensor(self.learning_rate * 0.1, dtype=self.compute_dtype, device=self.device)`
  - Controls: Interaction learning rate factor
  - Candidate: YES - Could be adaptive

- **Line 463**: `interaction_matrix_compute = torch.clamp(interaction_matrix_compute, min=-1.0, max=1.0)`
  - Controls: Interaction bounds
  - Candidate: MAYBE - Stability requirement

- **Line 478**: `if outcome['success'] > 0.6:` - Same as line 439
  - Controls: Same as line 439
  - Candidate: Same as line 439

- **Line 486**: `update = np.outer(feature_activation, feature_activation) * self.learning_rate * 0.1`
  - Controls: Same as line 454
  - Candidate: Same as line 454

- **Line 490**: `self.interaction_matrix = np.clip(self.interaction_matrix, -1.0, 1.0)`
  - Controls: Same as line 463
  - Candidate: Same as line 463

- **Line 512**: `if len(self.prediction_outcomes) < 10:`
  - Controls: Minimum data for meta-learning
  - Candidate: MAYBE - Statistical requirement

- **Line 529**: `if len(self.adaptation_success_history) > 50:`
  - Controls: Adaptation history limit
  - Candidate: NO - Infrastructure limit

- **Line 537-542**: Meta-learning thresholds (0.05, -0.05)
  - Controls: When to adjust learning rate
  - Candidate: YES - Could be based on noise level

### similarity/adaptive_attention.py
- **Line 27**: `adaptation_window: int = 50`
  - Controls: Learning sample tracking window
  - Candidate: MAYBE - Could be based on volatility

- **Line 35**: `self.attention_baseline = 0.5` - Initial baseline
  - Controls: Starting attention baseline
  - Candidate: NO - Neutral starting point

- **Line 44**: `self.adaptation_interval = 10.0` - Adaptation frequency
  - Controls: How often to adapt baseline
  - Candidate: YES - Could be based on experience rate

- **Line 76**: `attention_score = min(1.0, max(0.0, normalized_error * 0.8))` - Scaling factor
  - Controls: How error maps to attention
  - Candidate: YES - Could be learned

- **Line 79**: `if attention_score > 0.7:` - High attention threshold
  - Controls: What counts as "high attention"
  - Candidate: YES - Could be percentile-based

- **Line 81**: `elif attention_score < 0.3:` - Suppression threshold
  - Controls: What counts as "suppressed"
  - Candidate: YES - Could be percentile-based

- **Line 97**: `if len(self.accuracy_history) >= 10:`
  - Controls: Minimum data for velocity calculation
  - Candidate: MAYBE - Statistical requirement

- **Line 116**: `if len(self.learning_velocity_tracker) < 5:`
  - Controls: Minimum data for adaptation
  - Candidate: MAYBE - Statistical requirement

- **Line 127**: `adjustment_factor = 1.0 + (mean_velocity * 0.1)` - Velocity scaling
  - Controls: How velocity affects baseline
  - Candidate: YES - Could be learned

- **Line 132**: `adjustment_factor = 1.0 + (abs(mean_velocity) * 0.1)` - Same as above
  - Controls: Same as line 127
  - Candidate: Same as line 127

- **Line 141**: `if high_attention_rate > 0.2:` - Too much attention threshold
  - Controls: When to raise baseline
  - Candidate: YES - Could be based on capacity

- **Line 143**: `elif suppressed_rate > 0.6:` - Too much suppression threshold
  - Controls: When to lower baseline
  - Candidate: YES - Could be based on needs

- **Line 142**: `new_baseline = self.attention_baseline * 1.02` - Baseline adjustment rate
  - Controls: How fast baseline changes
  - Candidate: YES - Could be adaptive

- **Line 144**: `new_baseline = self.attention_baseline * 0.98` - Baseline adjustment rate
  - Controls: How fast baseline changes
  - Candidate: YES - Could be adaptive

- **Line 149**: `new_baseline = np.clip(new_baseline, 0.1, 2.0)` - Baseline bounds
  - Controls: Baseline limits
  - Candidate: MAYBE - Stability bounds

- **Line 155**: `if abs(baseline_change) > 0.01:` - Significant change threshold
  - Controls: Logging threshold
  - Candidate: NO - Logging preference

- **Line 278**: `attention_boost = max(natural_attention, base_similarity * 0.8)`
  - Controls: Hybrid retrieval boost factor
  - Candidate: YES - Could be learned

- **Line 283**: `utility_weight = experience.prediction_utility * 2.0`
  - Controls: Utility focus boost factor
  - Candidate: YES - Could be learned

- **Line 335**: `high_utility_count': sum(1 for u in utilities if u > 0.7)`
  - Controls: High utility threshold
  - Candidate: YES - Could be percentile-based

- **Line 339**: `clustered_memories': sum(1 for d in cluster_densities if d > 0.5)`
  - Controls: Clustered memory threshold
  - Candidate: YES - Could be based on distribution

- **Line 340**: `distinctive_memories': sum(1 for d in cluster_densities if d < 0.1)`
  - Controls: Distinctive memory threshold
  - Candidate: YES - Could be based on distribution

## 4. ADAPTIVE TRIGGER SYSTEM

### utils/adaptive_trigger.py
- **Line 28**: `min_experiences_between_adaptations: int = 5`
  - Controls: Adaptation frequency limit
  - Candidate: YES - Could be based on stability

- **Line 29**: `gradient_change_threshold: float = 0.3`
  - Controls: When gradient changes trigger adaptation
  - Candidate: YES - Already marked as adaptive

- **Line 30**: `plateau_detection_window: int = 20`
  - Controls: Window for plateau detection
  - Candidate: YES - Could be based on volatility

- **Line 31**: `surprise_threshold: float = 0.7`
  - Controls: What error level is "surprising"
  - Candidate: YES - Already marked as adaptive

- **Line 51**: `self.threshold_adaptation_rate = 0.1`
  - Controls: How fast thresholds adapt
  - Candidate: MAYBE - Meta-parameter

- **Line 55**: `self.prediction_errors = deque(maxlen=50)`
  - Controls: Error history size
  - Candidate: NO - Infrastructure limit

- **Line 60**: `self.error_gradients = deque(maxlen=10)`
  - Controls: Gradient history size
  - Candidate: NO - Infrastructure limit

- **Line 64**: `self.performance_windows = deque(maxlen=5)`
  - Controls: Performance window history
  - Candidate: NO - Infrastructure limit

- **Line 79**: `if len(self.prediction_errors) >= 5:`
  - Controls: Minimum data for gradient
  - Candidate: MAYBE - Statistical requirement

- **Line 85**: `if len(self.prediction_errors) >= 10:`
  - Controls: Minimum data for windows
  - Candidate: MAYBE - Statistical requirement

- **Line 104**: `if len(self.prediction_errors) < 10:`
  - Controls: Minimum data for triggers
  - Candidate: MAYBE - Statistical requirement

- **Line 138**: `if len(errors) < 3:`
  - Controls: Minimum data for gradient computation
  - Candidate: NO - Mathematical requirement

- **Line 158**: `if len(recent_prediction_errors) < 5:`
  - Controls: Minimum data for adaptation
  - Candidate: MAYBE - Statistical requirement

- **Line 167**: `if len(self.error_gradients) < 3:`
  - Controls: Minimum data for gradient trigger
  - Candidate: MAYBE - Statistical requirement

- **Line 191**: `if len(self.prediction_errors) < 15:`
  - Controls: Minimum data for performance trigger
  - Candidate: MAYBE - Statistical requirement

- **Line 204**: `if performance_degradation > 0.2:` - 20% degradation threshold
  - Controls: When poor performance triggers adaptation
  - Candidate: YES - Could be based on noise level

- **Line 220**: `if len(self.prediction_errors) < 10:`
  - Controls: Minimum data for surprise trigger
  - Candidate: MAYBE - Statistical requirement

- **Line 227**: `if surprise_rate > 0.6:` - 60% surprise rate threshold
  - Controls: When high surprise triggers adaptation
  - Candidate: YES - Could be adaptive

- **Line 243**: `if len(self.performance_windows) < 3:`
  - Controls: Minimum windows for plateau detection
  - Candidate: MAYBE - Statistical requirement

- **Line 253**: `if error_variance_across_windows < 0.01:` - Plateau variance threshold
  - Controls: What variance indicates plateau
  - Candidate: YES - Could be based on noise

- **Line 256**: `if current_error > 0.3:` - Minimum error for plateau trigger
  - Controls: Don't trigger on already-good performance
  - Candidate: YES - Could be adaptive

## Summary of Key Candidates for Adaptive Behavior:

### High Priority (Core behavior parameters):
1. Activation decay rates and thresholds
2. Spreading activation strength and thresholds
3. Working memory thresholds
4. Similarity thresholds for various operations
5. Prediction confidence thresholds
6. Learning rates (many already have meta-learning)
7. Utility combination weights
8. Attention baselines and thresholds
9. Adaptation trigger thresholds

### Medium Priority (Could improve adaptability):
1. GPU usage thresholds (based on performance)
2. Exploration noise parameters
3. Confidence scaling factors
4. Success/failure thresholds
5. History window sizes (some)
6. Percentile-based thresholds

### Low Priority (Infrastructure or safety):
1. Cache sizes and limits
2. History buffer sizes
3. Logging thresholds
4. Mathematical requirements
5. Safety bounds

The brain system already has significant adaptive capabilities through meta-learning in several components. The main opportunities for improvement are:
1. Making more thresholds relative to distributions rather than absolute
2. Adding meta-learning to parameters that don't have it yet
3. Creating cross-system parameter coordination
4. Making capacity-related parameters adapt to resource constraints