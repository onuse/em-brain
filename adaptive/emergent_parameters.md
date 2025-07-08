# Making Parameters Emergent

## Adaptive Similarity Weighting
```python
class AdaptiveSimilarity:
    """
    Learn which similarity measures actually predict success
    """
    def __init__(self):
        # Track success rates for different similarity components
        self.component_weights = {
            'context': 0.25,    # Start equal
            'action': 0.25,
            'sensory': 0.25,
            'neighborhood': 0.25
        }
        
        # Track prediction success for each component
        self.component_success_history = {
            'context': [],
            'action': [],
            'sensory': [],
            'neighborhood': []
        }
        
        self.learning_rate = 0.01
    
    def calculate_similarity(self, node1, node2, world_graph):
        """
        Calculate similarity using learned weights
        """
        similarities = {
            'context': euclidean_similarity(node1.mental_context, node2.mental_context),
            'action': euclidean_similarity(node1.action_taken, node2.action_taken),
            'sensory': euclidean_similarity(node1.actual_sensory, node2.actual_sensory),
            'neighborhood': neighborhood_similarity(node1, node2, world_graph)
        }
        
        # Weighted combination using learned weights
        total_similarity = sum(
            similarities[component] * self.component_weights[component]
            for component in similarities
        )
        
        return total_similarity, similarities
    
    def update_weights_from_prediction_success(self, used_similarities, prediction_accuracy):
        """
        Adjust weights based on how well predictions worked
        """
        for component, similarity_value in used_similarities.items():
            # Weight the update by how much this component contributed
            contribution = similarity_value * self.component_weights[component]
            update_strength = contribution * prediction_accuracy
            
            # Update the component weight toward better prediction
            if prediction_accuracy > 0.5:  # Good prediction
                self.component_weights[component] += self.learning_rate * update_strength
            else:  # Poor prediction
                self.component_weights[component] -= self.learning_rate * update_strength
        
        # Normalize weights to sum to 1
        total_weight = sum(self.component_weights.values())
        for component in self.component_weights:
            self.component_weights[component] /= total_weight
```

## Adaptive Decay Rate
```python
class AdaptiveDecay:
    """
    Adjust memory decay based on memory pressure and prediction quality
    """
    def __init__(self):
        self.base_decay_rate = 0.001
        self.current_decay_rate = 0.001
        self.graph_size_history = []
        self.prediction_accuracy_history = []
        
    def calculate_decay_rate(self, world_graph, recent_prediction_accuracy):
        """
        Adjust decay rate based on system performance
        """
        current_size = world_graph.node_count()
        
        # Memory pressure factor
        if current_size > 10000:  # Large graph
            memory_pressure = min(2.0, current_size / 10000.0)
        else:
            memory_pressure = 0.5  # Preserve memories when graph is small
        
        # Prediction quality factor
        avg_accuracy = sum(recent_prediction_accuracy) / len(recent_prediction_accuracy) if recent_prediction_accuracy else 0.5
        
        if avg_accuracy > 0.7:  # Good predictions - can afford to forget more
            quality_factor = 1.2
        elif avg_accuracy < 0.3:  # Poor predictions - preserve more memories
            quality_factor = 0.5
        else:
            quality_factor = 1.0
        
        # Combine factors
        self.current_decay_rate = self.base_decay_rate * memory_pressure * quality_factor
        
        # Clamp to reasonable bounds
        self.current_decay_rate = max(0.0001, min(self.current_decay_rate, 0.01))
        
        return self.current_decay_rate
```

## Adaptive Merge Threshold
```python
class AdaptiveMerging:
    """
    Learn optimal merge thresholds based on prediction success
    """
    def __init__(self):
        self.merge_threshold = 0.7  # Start with reasonable default
        self.threshold_history = []
        self.merge_success_history = []
        
    def should_merge_nodes(self, node1, node2, similarity_score):
        """
        Decide whether to merge based on learned threshold
        """
        return similarity_score > self.merge_threshold
    
    def update_threshold_from_merge_results(self, merged_pairs_and_outcomes):
        """
        Adjust threshold based on whether merges helped or hurt predictions
        """
        successful_merges = []
        failed_merges = []
        
        for merge_info in merged_pairs_and_outcomes:
            similarity_at_merge = merge_info['similarity']
            prediction_improvement = merge_info['prediction_improvement']
            
            if prediction_improvement > 0:
                successful_merges.append(similarity_at_merge)
            else:
                failed_merges.append(similarity_at_merge)
        
        if successful_merges and failed_merges:
            # Find the threshold that best separates good and bad merges
            successful_avg = sum(successful_merges) / len(successful_merges)
            failed_avg = sum(failed_merges) / len(failed_merges)
            
            # Move threshold toward the range that works
            if successful_avg > failed_avg:
                # Higher similarity merges work better
                self.merge_threshold += 0.01 * (successful_avg - self.merge_threshold)
            else:
                # Lower similarity merges work better
                self.merge_threshold -= 0.01 * (self.merge_threshold - failed_avg)
            
            # Clamp to reasonable bounds
            self.merge_threshold = max(0.3, min(self.merge_threshold, 0.95))
```

## Adaptive Traversal Depth
```python
class AdaptiveThinking:
    """
    Learn how much thinking depth is needed for different situations
    """
    def __init__(self):
        self.depth_success_history = {}  # depth -> success_rate
        self.context_depth_mapping = {}  # context_type -> optimal_depth
        
    def calculate_thinking_depth(self, mental_context, recent_prediction_accuracy):
        """
        Determine thinking depth based on context and past success
        """
        # Classify current context type (could be learned clustering)
        context_type = self.classify_context(mental_context)
        
        # Check if we've learned an optimal depth for this context type
        if context_type in self.context_depth_mapping:
            base_depth = self.context_depth_mapping[context_type]
        else:
            base_depth = 5  # Default
        
        # Increase depth if recent predictions have been poor
        if recent_prediction_accuracy < 0.4:
            uncertainty_bonus = 3
        elif recent_prediction_accuracy < 0.7:
            uncertainty_bonus = 1
        else:
            uncertainty_bonus = 0
        
        final_depth = base_depth + uncertainty_bonus
        return max(2, min(final_depth, 15))  # Reasonable bounds
    
    def update_depth_effectiveness(self, depth_used, context_type, prediction_success):
        """
        Learn which depths work best for which contexts
        """
        if context_type not in self.context_depth_mapping:
            self.context_depth_mapping[context_type] = depth_used
        else:
            # Simple moving average toward depths that work
            current_optimal = self.context_depth_mapping[context_type]
            if prediction_success > 0.6:
                # This depth worked well, move toward it
                self.context_depth_mapping[context_type] = (current_optimal * 0.9 + depth_used * 0.1)
            # If it didn't work well, stick with current optimal
    
    def classify_context(self, mental_context):
        """
        Simple context classification - could be made more sophisticated
        """
        # For now, just use first few dimensions as classification
        # In practice, this could use clustering or other learned representations
        context_signature = tuple(round(x, 1) for x in mental_context[:5])
        return hash(context_signature) % 100  # Simple bucketing
```

## Adaptive Consensus
```python
class AdaptiveConsensus:
    """
    Learn how many traversals are needed for reliable decisions
    """
    def __init__(self):
        self.traversal_count = 3  # Start with 3
        self.consensus_success_history = []
        
    def run_adaptive_consensus(self, mental_context, world_graph, max_depth):
        """
        Run variable number of traversals based on learned needs
        """
        predictions = []
        
        # Always run at least 2 traversals
        for i in range(max(2, self.traversal_count)):
            prediction = single_traversal(mental_context, world_graph, max_depth, 
                                        generate_random_seed())
            predictions.append(prediction)
        
        # Check for early consensus
        if len(predictions) >= 2 and predictions[0] == predictions[1]:
            consensus_strength = 'strong'
        elif len(predictions) >= 3:
            consensus_strength = self.evaluate_consensus_strength(predictions)
        else:
            consensus_strength = 'weak'
        
        return resolve_consensus(predictions), consensus_strength
    
    def update_traversal_needs(self, consensus_strength, prediction_success):
        """
        Adjust how many traversals we typically need
        """
        if consensus_strength == 'strong' and prediction_success > 0.7:
            # Strong consensus worked, maybe we can use fewer traversals
            self.traversal_count = max(2, self.traversal_count - 0.1)
        elif consensus_strength == 'weak' and prediction_success < 0.4:
            # Weak consensus failed, need more traversals
            self.traversal_count = min(7, self.traversal_count + 0.2)
        
        self.traversal_count = round(self.traversal_count)
```

## Integration Example
```python
class EmergentBrain:
    """
    Brain with adaptive parameters that learn from experience
    """
    def __init__(self):
        self.adaptive_similarity = AdaptiveSimilarity()
        self.adaptive_decay = AdaptiveDecay()
        self.adaptive_merging = AdaptiveMerging()
        self.adaptive_thinking = AdaptiveThinking()
        self.adaptive_consensus = AdaptiveConsensus()
        
        # Track prediction outcomes for learning
        self.recent_predictions = []
        self.recent_accuracies = []
    
    def generate_prediction_with_learning(self, mental_context, world_graph):
        """
        Generate prediction while learning optimal parameters
        """
        # Use adaptive thinking depth
        thinking_depth = self.adaptive_thinking.calculate_thinking_depth(
            mental_context, self.recent_accuracies[-10:]
        )
        
        # Use adaptive consensus
        prediction, consensus_strength = self.adaptive_consensus.run_adaptive_consensus(
            mental_context, world_graph, thinking_depth
        )
        
        # Store for later learning
        self.recent_predictions.append({
            'prediction': prediction,
            'consensus_strength': consensus_strength,
            'thinking_depth': thinking_depth,
            'context': mental_context.copy()
        })
        
        return prediction
    
    def learn_from_prediction_outcome(self, prediction_accuracy):
        """
        Update all adaptive systems based on prediction results
        """
        self.recent_accuracies.append(prediction_accuracy)
        
        # Update similarity weights
        if self.recent_predictions:
            last_pred_info = self.recent_predictions[-1]
            # Would need to track which similarities were used - simplified here
        
        # Update other adaptive systems
        self.adaptive_thinking.update_depth_effectiveness(
            last_pred_info['thinking_depth'],
            self.adaptive_thinking.classify_context(last_pred_info['context']),
            prediction_accuracy
        )
        
        self.adaptive_consensus.update_traversal_needs(
            last_pred_info['consensus_strength'],
            prediction_accuracy
        )
        
        # Keep only recent history
        if len(self.recent_predictions) > 100:
            self.recent_predictions = self.recent_predictions[-50:]
        if len(self.recent_accuracies) > 100:
            self.recent_accuracies = self.recent_accuracies[-50:]
```