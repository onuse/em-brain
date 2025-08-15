#!/usr/bin/env python3
"""
Memory Inspector - Analyze Brain's Internal Representations

This tool examines what the brain has "learned" and "remembered" from visual experiences.
It attempts to decode the brain's internal representations to show:

1. Pattern clusters - what visual patterns the brain groups together
2. Expectations - what the brain predicts should happen next
3. Learned features - what visual elements the brain has extracted
4. Memory associations - how different experiences are connected
5. Prototype reconstruction - attempting to visualize what the brain "remembers"

This is like doing a "memory scan" of the artificial brain to understand
its internal model of the world.
"""

import sys
import os
from pathlib import Path

# Add paths
brain_root = Path(__file__).parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

import cv2
import numpy as np
import time
import json
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any
import threading
import queue
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import brain components
from src.brain import MinimalBrain
from src.memory_gate import AdaptiveMemoryGate, MemoryConsolidator
from src.async_brain_maintenance import AsyncBrainMaintenance, BrainState
from src.emergent_memory_pressure import EmergentMemoryGate


class MemoryInspector:
    """Inspects and analyzes the brain's internal memory representations"""
    
    def __init__(self, brain: MinimalBrain, use_emergent_gate: bool = True):
        """Initialize memory inspector"""
        self.brain = brain
        self.memory_samples = []
        self.pattern_history = deque(maxlen=1000)
        self.prediction_history = deque(maxlen=100)
        self.similarity_matrix = None
        self.use_emergent_gate = use_emergent_gate
        
        if use_emergent_gate:
            # Use emergent constraint-based system
            self.memory_gate = EmergentMemoryGate(self)
            self.memory_consolidator = None  # Consolidation handled by emergent system
            print("🌊 Using emergent constraint-based memory system")
        else:
            # Use explicit threshold-based system
            self.memory_gate = AdaptiveMemoryGate(
                base_threshold=0.5,
                capacity_limit=5000,
                history_window=100
            )
            self.memory_consolidator = MemoryConsolidator(
                consolidation_interval=1000,
                similarity_threshold=0.8,
                min_cluster_size=5
            )
            print("⚙️ Using explicit threshold-based memory system")
        
        # Track frames for consolidation (legacy)
        self.frame_count = 0
        self.last_consolidation_frame = 0
        
        # Initialize async maintenance (optional)
        self.async_maintenance = None
        self._activity_level = 0.5  # Current activity level for quiet detection
        
    def capture_memory_snapshot(self, sensory_input: List[float], 
                               brain_output: Any, brain_info: Dict, 
                               attention_map: np.ndarray = None) -> Dict:
        """Capture a snapshot of current brain state with attention-gated memory formation"""
        self.frame_count += 1
        
        # Extract attention-weighted sensory input if attention map provided
        if attention_map is not None:
            # Use attention to weight and filter sensory input
            attended_input, attention_strength = self._extract_attended_patterns(
                sensory_input, attention_map
            )
            # Use attended input for memory decisions
            memory_input = attended_input
            attention_boost = attention_strength * 0.5  # Boost importance based on attention
        else:
            # Fallback to full input if no attention map
            memory_input = sensory_input
            attention_boost = 0.0
        
        # Calculate prediction error and novelty for memory gating
        prediction_error = brain_info.get('prediction_error', 0.5)
        if prediction_error == 0.5 and 'prediction_confidence' in brain_info:
            # Convert confidence to error if not provided
            prediction_error = 1.0 - brain_info['prediction_confidence']
        
        # Calculate novelty score using attention-weighted input
        novelty_score = self._calculate_novelty_score(memory_input)
        
        # Get processing time for constraint tracking
        processing_time_ms = brain_info.get('cycle_time_ms', None)
        
        # Check if we should store this memory
        if self.use_emergent_gate:
            # Emergent constraint-based decision with attention boost
            should_store, decision_info = self.memory_gate.should_store_memory(
                prediction_error=prediction_error,
                novelty_score=novelty_score,
                importance=attention_boost,  # Use attention as importance signal
                processing_time_ms=processing_time_ms
            )
        else:
            # Explicit threshold-based decision with attention boost
            should_store, decision_info = self.memory_gate.should_store_memory(
                prediction_error=prediction_error,
                novelty_score=novelty_score,
                importance=attention_boost,  # Use attention as importance signal
                sensory_input=np.array(memory_input),  # Use attended input
                brain_info=brain_info
            )
        
        # Always update pattern history for novelty calculation (use attended patterns)
        self.pattern_history.append(np.array(memory_input))
        
        if brain_output is not None:
            self.prediction_history.append(brain_output)
        
        # Only store memory if gate approves
        if should_store:
            snapshot = {
                'timestamp': time.time(),
                'sensory_input': np.array(memory_input),  # Store attended patterns
                'original_input': np.array(sensory_input),  # Keep original for reference
                'attention_strength': attention_boost if attention_map is not None else 0.0,
                'brain_output': brain_output,
                'brain_info': brain_info,
                'activation_pattern': self._extract_activation_pattern(memory_input),
                'memory_state': self._extract_memory_state(),
                'prediction_confidence': brain_info.get('confidence', 0.0) if brain_info else 0.0,
                'memory_decision': decision_info
            }
            
            self.memory_samples.append(snapshot)
            
            # Legacy consolidation for explicit system
            if not self.use_emergent_gate and self.memory_consolidator and \
               self.memory_consolidator.should_consolidate(self.frame_count):
                self._run_consolidation()
                self.memory_consolidator.last_consolidation = self.frame_count
            
            return snapshot
        
        # Return minimal info if not storing
        return {
            'timestamp': time.time(),
            'stored': False,
            'decision': decision_info
        }
    
    def _extract_activation_pattern(self, sensory_input: List[float]) -> Dict:
        """Extract activation pattern information"""
        input_array = np.array(sensory_input)
        
        # Find activation clusters
        threshold = 0.1
        active_indices = np.where(input_array > threshold)[0]
        
        # Calculate pattern statistics
        pattern_stats = {
            'active_neurons': len(active_indices),
            'total_neurons': len(input_array),
            'sparsity': len(active_indices) / len(input_array),
            'max_activation': np.max(input_array),
            'mean_activation': np.mean(input_array),
            'activation_variance': np.var(input_array),
            'active_indices': active_indices.tolist(),
            'activation_values': input_array[active_indices].tolist()
        }
        
        return pattern_stats
    
    def _extract_memory_state(self) -> Dict:
        """Extract current memory state from brain"""
        # This would ideally query the brain's internal memory systems
        # For now, we simulate based on pattern history
        
        if len(self.pattern_history) < 2:
            return {'memory_strength': 0.0, 'memory_type': 'none'}
        
        # Calculate memory strength based on pattern similarity
        current_pattern = self.pattern_history[-1]
        recent_patterns = list(self.pattern_history)[-10:]
        
        # Calculate similarity to recent patterns
        similarities = []
        for pattern in recent_patterns[:-1]:
            try:
                # Use cosine similarity instead of correlation to avoid division by zero
                dot_product = np.dot(current_pattern, pattern)
                norm_product = np.linalg.norm(current_pattern) * np.linalg.norm(pattern)
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append(similarity)
            except:
                continue
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        memory_strength = 1.0 - avg_similarity  # High novelty = strong memory formation
        
        return {
            'memory_strength': memory_strength,
            'memory_type': 'episodic' if memory_strength > 0.7 else 'working',
            'pattern_similarity': avg_similarity,
            'recent_pattern_count': len(recent_patterns)
        }
    
    def _calculate_novelty_score(self, sensory_input: List[float]) -> float:
        """Calculate novelty score for current input"""
        if len(self.pattern_history) < 5:
            return 0.6  # High novelty for early experiences
        
        current = np.array(sensory_input)
        recent_patterns = list(self.pattern_history)[-10:]  # Look at last 10 patterns for more sensitivity
        
        # Calculate average distance to recent patterns
        distances = []
        for pattern in recent_patterns:
            # Use cosine distance for consistency
            dot_product = np.dot(current, pattern)
            norms = np.linalg.norm(current) * np.linalg.norm(pattern)
            if norms > 0:
                similarity = dot_product / norms
                distance = 1.0 - similarity
                distances.append(distance)
        
        if distances:
            avg_distance = np.mean(distances)
            # More sensitive scaling for subtle changes (kitchens have small but meaningful changes)
            novelty = np.clip(avg_distance * 5.0, 0.0, 1.0)  # Increased sensitivity
        else:
            novelty = 0.5
        
        return novelty
    
    def _extract_attended_patterns(self, sensory_input: List[float], 
                                 attention_map: np.ndarray) -> Tuple[List[float], float]:
        """Extract attention-weighted patterns from sensory input"""
        try:
            sensory_array = np.array(sensory_input)
            h, w = attention_map.shape
            
            # Resize sensory input to match attention map dimensions
            input_spatial = self._sensory_to_spatial(sensory_array, w, h)
            
            # Apply attention weighting to sensory input
            attended_spatial = input_spatial * attention_map
            
            # Calculate attention strength
            attention_strength = np.mean(attention_map)
            
            # Only keep patterns above attention threshold
            attention_threshold = 0.3
            high_attention_mask = attention_map > attention_threshold
            
            if np.any(high_attention_mask):
                # Extract only attended regions
                attended_regions = attended_spatial[high_attention_mask]
                
                # Convert back to pattern vector
                # Pad or truncate to match original sensory input size
                if len(attended_regions) >= len(sensory_array):
                    attended_pattern = attended_regions[:len(sensory_array)]
                else:
                    # Pad with zeros if attended regions are smaller
                    attended_pattern = np.zeros(len(sensory_array))
                    attended_pattern[:len(attended_regions)] = attended_regions
                
                return attended_pattern.tolist(), attention_strength
            else:
                # No high attention areas, return low-weighted original
                return (sensory_array * 0.1).tolist(), attention_strength
                
        except Exception as e:
            print(f"Attention extraction error: {e}")
            # Fallback to original input
            return sensory_input, 0.0
    
    def _sensory_to_spatial(self, sensory_array: np.ndarray, width: int, height: int) -> np.ndarray:
        """Convert sensory vector to spatial representation matching attention map"""
        # Similar to _pattern_to_spatial but optimized for sensory input
        pattern_size = len(sensory_array)
        
        if pattern_size == width * height:
            # Perfect match - reshape directly
            return sensory_array.reshape(height, width)
        elif pattern_size < width * height:
            # Pad and reshape
            padded = np.zeros(width * height)
            padded[:pattern_size] = sensory_array
            return padded.reshape(height, width)
        else:
            # Downsample using interpolation
            grid_size = int(np.sqrt(pattern_size))
            temp_spatial = sensory_array[:grid_size*grid_size].reshape(grid_size, grid_size)
            return cv2.resize(temp_spatial, (width, height))
    
    def _run_consolidation(self):
        """Run memory consolidation in background"""
        print(f"\n🧹 Running memory consolidation at frame {self.frame_count}...")
        
        # Run consolidation
        self.memory_samples, consolidation_info = self.memory_consolidator.consolidate_memories(
            self.memory_samples, self.memory_gate
        )
        
        self.last_consolidation_frame = self.frame_count
        
        print(f"✅ Consolidation complete: {consolidation_info['start_count']} → {consolidation_info['end_count']} memories")
        print(f"   Removed: {consolidation_info['removed']}, Prototypes: {consolidation_info['prototypes_created']}")
    
    def analyze_pattern_clusters(self, n_clusters: int = 5) -> Dict:
        """Analyze how the brain clusters visual patterns"""
        if len(self.pattern_history) < n_clusters:
            return {'error': 'Not enough patterns for clustering'}
        
        # Convert pattern history to array
        patterns = np.array(list(self.pattern_history))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(patterns)
        
        # Analyze clusters
        cluster_analysis = {
            'n_clusters': n_clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'cluster_labels': cluster_labels,
            'cluster_sizes': [np.sum(cluster_labels == i) for i in range(n_clusters)],
            'cluster_statistics': []
        }
        
        # Calculate statistics for each cluster
        for i in range(n_clusters):
            cluster_patterns = patterns[cluster_labels == i]
            cluster_stats = {
                'cluster_id': i,
                'size': len(cluster_patterns),
                'mean_activation': np.mean(cluster_patterns),
                'activation_variance': np.var(cluster_patterns),
                'representative_pattern': kmeans.cluster_centers_[i],
                'pattern_diversity': np.mean(np.var(cluster_patterns, axis=0))
            }
            cluster_analysis['cluster_statistics'].append(cluster_stats)
        
        return cluster_analysis
    
    def extract_learned_features(self) -> Dict:
        """Extract what visual features the brain has learned"""
        if len(self.pattern_history) < 10:
            return {'error': 'Not enough patterns for feature extraction'}
        
        patterns = np.array(list(self.pattern_history))
        
        # Use PCA to find principal components (learned features)
        pca = PCA(n_components=min(8, patterns.shape[1]))
        pca_features = pca.fit_transform(patterns)
        
        # Analyze learned features
        feature_analysis = {
            'n_features': pca.n_components_,
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'feature_components': pca.components_,
            'feature_importance': pca.explained_variance_ratio_,
            'most_important_features': []
        }
        
        # Identify most important features
        for i, (component, importance) in enumerate(zip(pca.components_, pca.explained_variance_ratio_)):
            # Find which input dimensions contribute most to this feature
            top_indices = np.argsort(np.abs(component))[-3:]  # Top 3 contributors
            feature_info = {
                'feature_id': i,
                'importance': importance,
                'top_contributors': top_indices.tolist(),
                'contribution_weights': component[top_indices].tolist(),
                'feature_vector': component.tolist()
            }
            feature_analysis['most_important_features'].append(feature_info)
        
        return feature_analysis
    
    def analyze_expectations(self) -> Dict:
        """Analyze what the brain expects to happen next"""
        if len(self.prediction_history) < 5:
            return {'error': 'Not enough predictions for expectation analysis'}
        
        predictions = np.array([p for p in self.prediction_history if p is not None])
        if len(predictions) == 0:
            return {'error': 'No valid predictions found'}
        
        # Analyze prediction patterns
        expectation_analysis = {
            'prediction_count': len(predictions),
            'mean_prediction': np.mean(predictions, axis=0) if len(predictions.shape) > 1 else np.mean(predictions),
            'prediction_variance': np.var(predictions, axis=0) if len(predictions.shape) > 1 else np.var(predictions),
            'prediction_stability': self._calculate_prediction_stability(predictions),
            'expectation_strength': self._calculate_expectation_strength(predictions),
            'prediction_trends': self._analyze_prediction_trends(predictions)
        }
        
        return expectation_analysis
    
    def _calculate_prediction_stability(self, predictions: np.ndarray) -> float:
        """Calculate how stable the brain's predictions are"""
        if len(predictions) < 2:
            return 0.0
        
        # Calculate correlation between consecutive predictions
        correlations = []
        for i in range(len(predictions) - 1):
            try:
                if len(predictions.shape) > 1:
                    # Use cosine similarity for multi-dimensional predictions
                    dot_product = np.dot(predictions[i], predictions[i + 1])
                    norm_product = np.linalg.norm(predictions[i]) * np.linalg.norm(predictions[i + 1])
                    if norm_product > 0:
                        corr = dot_product / norm_product
                        correlations.append(corr)
                else:
                    # For single values, use difference-based stability
                    diff = abs(predictions[i] - predictions[i + 1])
                    stability = 1.0 / (1.0 + diff)
                    correlations.append(stability)
            except:
                continue
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_expectation_strength(self, predictions: np.ndarray) -> float:
        """Calculate how strong the brain's expectations are"""
        # Strong expectations = low variance in predictions
        if len(predictions.shape) > 1:
            variance = np.mean(np.var(predictions, axis=0))
        else:
            variance = np.var(predictions)
        
        # Convert variance to strength (inverse relationship)
        strength = 1.0 / (1.0 + variance)
        return strength
    
    def _analyze_prediction_trends(self, predictions: np.ndarray) -> Dict:
        """Analyze trends in predictions"""
        if len(predictions) < 3:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend over time
        if len(predictions.shape) > 1:
            # Multi-dimensional predictions
            trends = []
            for dim in range(predictions.shape[1]):
                trend = np.polyfit(range(len(predictions)), predictions[:, dim], 1)[0]
                trends.append(trend)
            
            return {
                'trend': 'multi_dimensional',
                'dimension_trends': trends,
                'overall_trend': np.mean(trends),
                'trend_strength': np.std(trends)
            }
        else:
            # Single-dimensional predictions
            trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
            return {
                'trend': 'increasing' if trend > 0.01 else 'decreasing' if trend < -0.01 else 'stable',
                'trend_slope': trend,
                'trend_strength': abs(trend)
            }
    
    def reconstruct_memory_prototypes(self, n_prototypes: int = 3) -> Dict:
        """Attempt to reconstruct what the brain 'remembers' about different experiences"""
        if len(self.memory_samples) < n_prototypes:
            return {'error': 'Not enough memory samples for reconstruction'}
        
        # Group memories by similarity
        patterns = np.array([sample['sensory_input'] for sample in self.memory_samples])
        
        # Cluster memories
        kmeans = KMeans(n_clusters=n_prototypes, random_state=42)
        memory_clusters = kmeans.fit_predict(patterns)
        
        prototypes = []
        for i in range(n_prototypes):
            cluster_memories = [self.memory_samples[j] for j in range(len(self.memory_samples)) 
                              if memory_clusters[j] == i]
            
            if cluster_memories:
                # Calculate prototype statistics
                cluster_patterns = [mem['sensory_input'] for mem in cluster_memories]
                prototype_pattern = np.mean(cluster_patterns, axis=0)
                
                # Find most representative memory
                similarities = []
                for pattern in cluster_patterns:
                    try:
                        # Use cosine similarity
                        dot_product = np.dot(prototype_pattern, pattern)
                        norm_product = np.linalg.norm(prototype_pattern) * np.linalg.norm(pattern)
                        if norm_product > 0:
                            similarity = dot_product / norm_product
                            similarities.append(similarity)
                        else:
                            similarities.append(0.0)
                    except:
                        similarities.append(0.0)
                
                best_match_idx = np.argmax(similarities) if similarities else 0
                representative_memory = cluster_memories[best_match_idx]
                
                prototype = {
                    'prototype_id': i,
                    'cluster_size': len(cluster_memories),
                    'prototype_pattern': prototype_pattern,
                    'representative_memory': representative_memory,
                    'memory_strength': np.mean([mem['memory_state']['memory_strength'] 
                                              for mem in cluster_memories if 'memory_state' in mem]),
                    'average_confidence': np.mean([mem['prediction_confidence'] 
                                                 for mem in cluster_memories]),
                    'pattern_variance': np.var(cluster_patterns, axis=0),
                    'formation_times': [mem['timestamp'] for mem in cluster_memories]
                }
                prototypes.append(prototype)
        
        return {
            'n_prototypes': n_prototypes,
            'prototypes': prototypes,
            'total_memories': len(self.memory_samples),
            'clustering_quality': self._calculate_clustering_quality(patterns, memory_clusters, kmeans.cluster_centers_)
        }
    
    def _calculate_clustering_quality(self, patterns: np.ndarray, 
                                    labels: np.ndarray, centers: np.ndarray) -> float:
        """Calculate quality of memory clustering"""
        if len(patterns) == 0:
            return 0.0
        
        # Calculate silhouette-like score
        total_score = 0.0
        for i, pattern in enumerate(patterns):
            cluster_id = labels[i]
            
            # Distance to own cluster center
            own_distance = np.linalg.norm(pattern - centers[cluster_id])
            
            # Distance to nearest other cluster center
            other_distances = [np.linalg.norm(pattern - center) 
                             for j, center in enumerate(centers) if j != cluster_id]
            nearest_other_distance = min(other_distances) if other_distances else own_distance
            
            # Calculate score (higher is better)
            if nearest_other_distance > own_distance:
                score = (nearest_other_distance - own_distance) / nearest_other_distance
            else:
                score = 0.0
            
            total_score += score
        
        return total_score / len(patterns)
    
    def generate_memory_report(self, output_file: str = None) -> Dict:
        """Generate comprehensive memory analysis report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"validation/memory_analysis/memory_report_{timestamp}.json"
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Perform all analyses
        report = {
            'timestamp': datetime.now().isoformat(),
            'brain_type': getattr(self.brain, 'brain_type', 'unknown'),
            'memory_samples_count': len(self.memory_samples),
            'pattern_history_length': len(self.pattern_history),
            'prediction_history_length': len(self.prediction_history),
            
            'pattern_clustering': self.analyze_pattern_clusters(),
            'learned_features': self.extract_learned_features(),
            'expectation_analysis': self.analyze_expectations(),
            'memory_prototypes': self.reconstruct_memory_prototypes(),
            
            'summary_statistics': {
                'avg_memory_strength': np.mean([s.get('memory_state', {}).get('memory_strength', 0) 
                                               for s in self.memory_samples]),
                'avg_prediction_confidence': np.mean([s['prediction_confidence'] 
                                                    for s in self.memory_samples]),
                'pattern_diversity': np.var([s['activation_pattern']['sparsity'] 
                                           for s in self.memory_samples]),
                'total_analysis_time': time.time()
            }
        }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Memory analysis report saved: {output_file}")
        
        # Generate human-readable summary
        summary_file = output_file.replace('.json', '_summary.md')
        self._generate_summary_report(report, summary_file)
        
        return report
    
    def _generate_summary_report(self, report: Dict, filename: str):
        """Generate human-readable summary report"""
        with open(filename, 'w') as f:
            f.write("# Brain Memory Analysis Report\n\n")
            f.write(f"**Generated**: {report['timestamp']}\n")
            f.write(f"**Brain Type**: {report['brain_type']}\n")
            f.write(f"**Memory Samples**: {report['memory_samples_count']}\n")
            f.write(f"**Pattern History**: {report['pattern_history_length']}\n\n")
            
            # Pattern clustering summary
            clustering = report.get('pattern_clustering', {})
            if 'error' not in clustering:
                f.write("## Pattern Clustering Analysis\n\n")
                f.write(f"The brain has organized its visual experiences into **{clustering['n_clusters']} distinct clusters**:\n\n")
                
                for i, cluster in enumerate(clustering['cluster_statistics']):
                    f.write(f"### Cluster {i + 1}\n")
                    f.write(f"- **Size**: {cluster['size']} experiences\n")
                    f.write(f"- **Average Activation**: {cluster['mean_activation']:.3f}\n")
                    f.write(f"- **Pattern Diversity**: {cluster['pattern_diversity']:.3f}\n\n")
            
            # Learned features summary
            features = report.get('learned_features', {})
            if 'error' not in features:
                f.write("## Learned Visual Features\n\n")
                f.write(f"The brain has extracted **{features['n_features']} primary visual features** from its experiences:\n\n")
                
                for i, feature in enumerate(features['most_important_features'][:3]):
                    f.write(f"### Feature {i + 1}\n")
                    f.write(f"- **Importance**: {feature['importance']:.3f}\n")
                    f.write(f"- **Top Contributors**: {feature['top_contributors']}\n")
                    f.write(f"- **Contribution Weights**: {[f'{w:.3f}' for w in feature['contribution_weights']]}\n\n")
            
            # Expectation analysis
            expectations = report.get('expectation_analysis', {})
            if 'error' not in expectations:
                f.write("## Prediction & Expectation Analysis\n\n")
                f.write(f"- **Prediction Stability**: {expectations['prediction_stability']:.3f}\n")
                f.write(f"- **Expectation Strength**: {expectations['expectation_strength']:.3f}\n")
                
                trends = expectations.get('prediction_trends', {})
                if trends.get('trend') != 'insufficient_data':
                    f.write(f"- **Prediction Trend**: {trends['trend']}\n")
                    if 'trend_slope' in trends:
                        f.write(f"- **Trend Slope**: {trends['trend_slope']:.6f}\n")
                f.write("\n")
            
            # Memory prototypes
            prototypes = report.get('memory_prototypes', {})
            if 'error' not in prototypes:
                f.write("## Memory Prototypes\n\n")
                f.write(f"The brain has formed **{prototypes['n_prototypes']} prototype memories**:\n\n")
                
                for prototype in prototypes['prototypes']:
                    f.write(f"### Prototype {prototype['prototype_id'] + 1}\n")
                    f.write(f"- **Cluster Size**: {prototype['cluster_size']} similar experiences\n")
                    f.write(f"- **Memory Strength**: {prototype['memory_strength']:.3f}\n")
                    f.write(f"- **Average Confidence**: {prototype['average_confidence']:.3f}\n")
                    f.write(f"- **Formation Times**: {len(prototype['formation_times'])} instances\n\n")
            
            # Summary statistics
            stats = report.get('summary_statistics', {})
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Average Memory Strength**: {stats.get('avg_memory_strength', 0):.3f}\n")
            f.write(f"- **Average Prediction Confidence**: {stats.get('avg_prediction_confidence', 0):.3f}\n")
            f.write(f"- **Pattern Diversity**: {stats.get('pattern_diversity', 0):.3f}\n")
            
            f.write(f"\n---\n*Analysis completed at {datetime.now().isoformat()}*\n")
        
        print(f"📝 Summary report saved: {filename}")
    
    def enable_async_maintenance(self, temporal_hierarchy=None):
        """Enable async background maintenance threads"""
        if self.async_maintenance is None:
            print("🔄 Starting async brain maintenance...")
            self.async_maintenance = AsyncBrainMaintenance(
                self.brain, 
                self,  # Memory inspector acts as memory system
                temporal_hierarchy
            )
            print("✅ Async maintenance threads running")
        
    def disable_async_maintenance(self):
        """Stop async maintenance threads"""
        if self.async_maintenance:
            print("⏹️ Stopping async maintenance...")
            self.async_maintenance.stop_all_threads()
            self.async_maintenance = None
            print("✅ Async maintenance stopped")
    
    def cleanup(self):
        """Clean up resources"""
        if self.use_emergent_gate and hasattr(self.memory_gate, 'stop'):
            self.memory_gate.stop()
        self.disable_async_maintenance()
    
    def update_activity_level(self, activity: float):
        """Update current activity level for quiet period detection"""
        self._activity_level = np.clip(activity, 0.0, 1.0)
        
        # Also update async maintenance if enabled
        if self.async_maintenance:
            # Record in load monitor
            self.async_maintenance.load_monitor.quiet_detector.record_activity(activity)
    
    def get_all_memories(self) -> List[Dict]:
        """Get all memories (for async consolidation)"""
        return self.memory_samples


def main():
    """Main function for memory inspection"""
    print("🧠 Memory Inspector - Analyzing Brain's Internal Representations")
    print("=" * 70)
    print("This tool will analyze what the brain has learned and remembered")
    print("from its visual experiences.")
    print()
    
    # For demonstration, create a brain and simulate some experiences
    brain = MinimalBrain(brain_type="sparse_goldilocks")
    inspector = MemoryInspector(brain)
    
    # Simulate some visual experiences
    print("🎯 Simulating visual experiences...")
    
    # Generate diverse visual patterns
    for i in range(100):
        # Create different types of patterns
        if i < 30:
            # Static patterns (like static scenes)
            pattern = np.random.normal(0.3, 0.1, brain.sensory_dim)
        elif i < 60:
            # Dynamic patterns (like movement)
            pattern = np.random.normal(0.7, 0.2, brain.sensory_dim)
        else:
            # Novel patterns (like new objects)
            pattern = np.random.normal(0.5, 0.3, brain.sensory_dim)
        
        # Ensure values are in valid range
        pattern = np.clip(pattern, 0, 1)
        
        # Process through brain
        brain_output, brain_info = brain.process_sensory_input(pattern.tolist())
        
        # Capture memory snapshot
        inspector.capture_memory_snapshot(pattern.tolist(), brain_output, brain_info)
        
        if (i + 1) % 25 == 0:
            print(f"  Processed {i + 1} experiences...")
    
    print("\n🔍 Analyzing brain's internal representations...")
    
    # Generate comprehensive report
    report = inspector.generate_memory_report()
    
    print("\n📊 Memory Analysis Complete!")
    print("=" * 40)
    
    # Print key insights
    if 'pattern_clustering' in report and 'error' not in report['pattern_clustering']:
        clusters = report['pattern_clustering']['n_clusters']
        print(f"🔗 Pattern Clusters: {clusters} distinct visual categories")
    
    if 'learned_features' in report and 'error' not in report['learned_features']:
        features = report['learned_features']['n_features']
        print(f"🎯 Learned Features: {features} primary visual features extracted")
    
    if 'memory_prototypes' in report and 'error' not in report['memory_prototypes']:
        prototypes = report['memory_prototypes']['n_prototypes']
        print(f"💾 Memory Prototypes: {prototypes} prototype memories formed")
    
    stats = report.get('summary_statistics', {})
    print(f"📈 Average Memory Strength: {stats.get('avg_memory_strength', 0):.3f}")
    print(f"🎯 Average Prediction Confidence: {stats.get('avg_prediction_confidence', 0):.3f}")
    
    print("\nCheck the generated reports for detailed analysis!")


if __name__ == "__main__":
    main()