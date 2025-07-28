#!/usr/bin/env python3
"""
Universal Memory Formation Interface - Modality-Agnostic Memory System

This system creates a unified memory formation interface that works with any
sensory modality while preserving the specialized nature of each signal type.

Core principles:
1. Attention-gated memory formation (only attended patterns get stored)
2. Sparse distributed representation for any signal type
3. Modality metadata preservation for cross-modal associations
4. Temporal correlation tracking across modalities
5. Efficient memory consolidation and retrieval
"""

import numpy as np
import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os

from ..attention.signal_attention import ModalityType, SignalShape, UniversalAttentionSystem


@dataclass
class MemoryPattern:
    """Universal memory pattern that works with any modality"""
    pattern_id: str
    modality: ModalityType
    signal_shape: SignalShape
    sparse_representation: np.ndarray
    attention_weight: float
    novelty_score: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cross-modal associations
    associated_patterns: List[str] = field(default_factory=list)
    temporal_context: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        return {
            'pattern_id': self.pattern_id,
            'modality': self.modality.value,
            'signal_shape': self.signal_shape.value,
            'sparse_representation': self.sparse_representation.tolist(),
            'attention_weight': float(self.attention_weight),
            'novelty_score': float(self.novelty_score),
            'timestamp': float(self.timestamp),
            'metadata': convert_numpy_types(self.metadata),
            'associated_patterns': self.associated_patterns,
            'temporal_context': self.temporal_context
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryPattern':
        """Create from dictionary"""
        return cls(
            pattern_id=data['pattern_id'],
            modality=ModalityType(data['modality']),
            signal_shape=SignalShape(data['signal_shape']),
            sparse_representation=np.array(data['sparse_representation']),
            attention_weight=data['attention_weight'],
            novelty_score=data['novelty_score'],
            timestamp=data['timestamp'],
            metadata=data.get('metadata', {}),
            associated_patterns=data.get('associated_patterns', []),
            temporal_context=data.get('temporal_context', [])
        )


class MemoryEncoder(ABC):
    """Abstract base class for modality-specific memory encoding"""
    
    @abstractmethod
    def encode_pattern(self, signal: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Encode attended signal into sparse representation"""
        pass
    
    @abstractmethod
    def extract_metadata(self, signal: np.ndarray, attention_map: np.ndarray) -> Dict[str, Any]:
        """Extract modality-specific metadata"""
        pass
    
    @abstractmethod
    def calculate_novelty(self, pattern: np.ndarray, existing_patterns: List[np.ndarray]) -> float:
        """Calculate novelty score against existing patterns"""
        pass


class VisualMemoryEncoder(MemoryEncoder):
    """Visual-specific memory encoding"""
    
    def __init__(self, target_sparsity: float = 0.02):
        self.target_sparsity = target_sparsity
    
    def encode_pattern(self, signal: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Encode visual signal into sparse representation"""
        # Apply attention mask to extract attended regions
        attended_signal = signal * attention_map
        
        # Flatten and normalize
        flat_signal = attended_signal.flatten()
        normalized = (flat_signal - np.mean(flat_signal)) / (np.std(flat_signal) + 1e-8)
        
        # Create sparse representation using top-k selection
        k = max(1, int(len(normalized) * self.target_sparsity))
        sparse_indices = np.argsort(np.abs(normalized))[-k:]
        
        sparse_pattern = np.zeros_like(normalized)
        sparse_pattern[sparse_indices] = normalized[sparse_indices]
        
        return sparse_pattern
    
    def extract_metadata(self, signal: np.ndarray, attention_map: np.ndarray) -> Dict[str, Any]:
        """Extract visual metadata"""
        attended_signal = signal * attention_map
        
        return {
            'signal_dimensions': signal.shape,
            'attention_coverage': np.mean(attention_map > 0.1),
            'brightness_mean': np.mean(attended_signal),
            'brightness_std': np.std(attended_signal),
            'edge_density': np.mean(np.abs(np.gradient(attended_signal))),
            'spatial_complexity': np.std(attended_signal)
        }
    
    def calculate_novelty(self, pattern: np.ndarray, existing_patterns: List[np.ndarray]) -> float:
        """Calculate visual novelty using cosine similarity"""
        if not existing_patterns:
            return 1.0
        
        similarities = []
        for existing in existing_patterns:
            # Cosine similarity
            dot_product = np.dot(pattern, existing)
            norms = np.linalg.norm(pattern) * np.linalg.norm(existing)
            similarity = dot_product / (norms + 1e-8)
            similarities.append(similarity)
        
        max_similarity = max(similarities)
        return 1.0 - max_similarity


class AudioMemoryEncoder(MemoryEncoder):
    """Audio-specific memory encoding"""
    
    def __init__(self, target_sparsity: float = 0.02):
        self.target_sparsity = target_sparsity
    
    def encode_pattern(self, signal: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Encode audio signal into sparse representation"""
        # Handle 1D and 2D audio signals
        if len(signal.shape) == 1:
            # 1D time series
            attended_signal = signal * attention_map
        else:
            # 2D spectrogram
            attended_signal = signal * attention_map
        
        # Flatten and normalize
        flat_signal = attended_signal.flatten()
        normalized = (flat_signal - np.mean(flat_signal)) / (np.std(flat_signal) + 1e-8)
        
        # Create sparse representation
        k = max(1, int(len(normalized) * self.target_sparsity))
        sparse_indices = np.argsort(np.abs(normalized))[-k:]
        
        sparse_pattern = np.zeros_like(normalized)
        sparse_pattern[sparse_indices] = normalized[sparse_indices]
        
        return sparse_pattern
    
    def extract_metadata(self, signal: np.ndarray, attention_map: np.ndarray) -> Dict[str, Any]:
        """Extract audio metadata"""
        attended_signal = signal * attention_map
        
        if len(signal.shape) == 1:
            # 1D time series
            return {
                'signal_dimensions': signal.shape,
                'attention_coverage': np.mean(attention_map > 0.1),
                'amplitude_mean': np.mean(np.abs(attended_signal)),
                'amplitude_std': np.std(attended_signal),
                'energy': np.mean(attended_signal**2),
                'zero_crossing_rate': np.mean(np.abs(np.diff(np.sign(attended_signal))))
            }
        else:
            # 2D spectrogram
            return {
                'signal_dimensions': signal.shape,
                'attention_coverage': np.mean(attention_map > 0.1),
                'spectral_centroid': np.mean(np.sum(attended_signal * np.arange(signal.shape[0])[:, None], axis=0)),
                'spectral_bandwidth': np.std(attended_signal),
                'spectral_rolloff': np.percentile(attended_signal, 85),
                'temporal_complexity': np.std(np.mean(attended_signal, axis=0))
            }
    
    def calculate_novelty(self, pattern: np.ndarray, existing_patterns: List[np.ndarray]) -> float:
        """Calculate audio novelty using correlation"""
        if not existing_patterns:
            return 1.0
        
        correlations = []
        for existing in existing_patterns:
            # Ensure same length for comparison
            if len(pattern) != len(existing):
                # Pad or truncate to match lengths
                min_len = min(len(pattern), len(existing))
                pattern_comp = pattern[:min_len]
                existing_comp = existing[:min_len]
            else:
                pattern_comp = pattern
                existing_comp = existing
            
            # Normalized cross-correlation
            if len(pattern_comp) > 1 and len(existing_comp) > 1:
                correlation = np.corrcoef(pattern_comp, existing_comp)[0, 1]
                if not np.isnan(correlation):
                    correlations.append(abs(correlation))
        
        if not correlations:
            return 1.0
        
        max_correlation = max(correlations)
        return 1.0 - max_correlation


class TactileMemoryEncoder(MemoryEncoder):
    """Tactile-specific memory encoding"""
    
    def __init__(self, target_sparsity: float = 0.02):
        self.target_sparsity = target_sparsity
    
    def encode_pattern(self, signal: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Encode tactile signal into sparse representation"""
        # Apply attention mask to extract attended pressure areas
        attended_signal = signal * attention_map
        
        # Flatten and normalize
        flat_signal = attended_signal.flatten()
        normalized = (flat_signal - np.mean(flat_signal)) / (np.std(flat_signal) + 1e-8)
        
        # Create sparse representation emphasizing pressure gradients
        k = max(1, int(len(normalized) * self.target_sparsity))
        sparse_indices = np.argsort(np.abs(normalized))[-k:]
        
        sparse_pattern = np.zeros_like(normalized)
        sparse_pattern[sparse_indices] = normalized[sparse_indices]
        
        return sparse_pattern
    
    def extract_metadata(self, signal: np.ndarray, attention_map: np.ndarray) -> Dict[str, Any]:
        """Extract tactile metadata"""
        attended_signal = signal * attention_map
        
        if len(signal.shape) == 1:
            # 1D pressure array
            return {
                'signal_dimensions': signal.shape,
                'attention_coverage': np.mean(attention_map > 0.1),
                'pressure_mean': np.mean(attended_signal),
                'pressure_std': np.std(attended_signal),
                'pressure_max': np.max(attended_signal),
                'contact_points': np.sum(attended_signal > 0.1),
                'gradient_magnitude': np.mean(np.abs(np.gradient(attended_signal)))
            }
        else:
            # 2D pressure map
            return {
                'signal_dimensions': signal.shape,
                'attention_coverage': np.mean(attention_map > 0.1),
                'pressure_mean': np.mean(attended_signal),
                'pressure_std': np.std(attended_signal),
                'pressure_max': np.max(attended_signal),
                'contact_area': np.sum(attended_signal > 0.1),
                'spatial_complexity': np.std(attended_signal),
                'edge_density': np.mean(np.abs(np.gradient(attended_signal)))
            }
    
    def calculate_novelty(self, pattern: np.ndarray, existing_patterns: List[np.ndarray]) -> float:
        """Calculate tactile novelty using pressure pattern similarity"""
        if not existing_patterns:
            return 1.0
        
        similarities = []
        for existing in existing_patterns:
            # Handle length differences
            if len(pattern) != len(existing):
                min_len = min(len(pattern), len(existing))
                pattern_comp = pattern[:min_len]
                existing_comp = existing[:min_len]
            else:
                pattern_comp = pattern
                existing_comp = existing
            
            # Use normalized dot product for tactile similarity
            dot_product = np.dot(pattern_comp, existing_comp)
            norms = np.linalg.norm(pattern_comp) * np.linalg.norm(existing_comp)
            similarity = dot_product / (norms + 1e-8)
            similarities.append(abs(similarity))
        
        max_similarity = max(similarities)
        return 1.0 - max_similarity


class MotorMemoryEncoder(MemoryEncoder):
    """Motor-specific memory encoding"""
    
    def __init__(self, target_sparsity: float = 0.02):
        self.target_sparsity = target_sparsity
    
    def encode_pattern(self, signal: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Encode motor signal into sparse representation"""
        # Apply attention mask to extract attended movement areas
        attended_signal = signal * attention_map
        
        # For motor signals, emphasize movement patterns
        if len(signal.shape) == 1:
            # 1D joint positions - encode velocity patterns
            velocity = np.diff(attended_signal, prepend=attended_signal[0])
            combined_signal = np.concatenate([attended_signal, velocity])
        else:
            # 2D motor map - flatten
            combined_signal = attended_signal.flatten()
        
        # Normalize
        normalized = (combined_signal - np.mean(combined_signal)) / (np.std(combined_signal) + 1e-8)
        
        # Create sparse representation
        k = max(1, int(len(normalized) * self.target_sparsity))
        sparse_indices = np.argsort(np.abs(normalized))[-k:]
        
        sparse_pattern = np.zeros_like(normalized)
        sparse_pattern[sparse_indices] = normalized[sparse_indices]
        
        return sparse_pattern
    
    def extract_metadata(self, signal: np.ndarray, attention_map: np.ndarray) -> Dict[str, Any]:
        """Extract motor metadata"""
        attended_signal = signal * attention_map
        
        if len(signal.shape) == 1:
            # 1D joint array
            velocity = np.diff(attended_signal, prepend=attended_signal[0])
            acceleration = np.diff(velocity, prepend=velocity[0])
            
            return {
                'signal_dimensions': signal.shape,
                'attention_coverage': np.mean(attention_map > 0.1),
                'position_mean': np.mean(attended_signal),
                'position_std': np.std(attended_signal),
                'velocity_mean': np.mean(np.abs(velocity)),
                'velocity_std': np.std(velocity),
                'acceleration_mean': np.mean(np.abs(acceleration)),
                'movement_complexity': np.std(velocity),
                'active_joints': np.sum(np.abs(velocity) > 0.01)
            }
        else:
            # 2D motor map
            velocities = np.diff(attended_signal, axis=1, prepend=attended_signal[:, 0:1])
            
            return {
                'signal_dimensions': signal.shape,
                'attention_coverage': np.mean(attention_map > 0.1),
                'position_mean': np.mean(attended_signal),
                'position_std': np.std(attended_signal),
                'velocity_mean': np.mean(np.abs(velocities)),
                'velocity_std': np.std(velocities),
                'coordination_index': np.corrcoef(attended_signal)[0, 1] if attended_signal.shape[0] > 1 else 0.0,
                'temporal_complexity': np.std(np.mean(velocities, axis=0))
            }
    
    def calculate_novelty(self, pattern: np.ndarray, existing_patterns: List[np.ndarray]) -> float:
        """Calculate motor novelty using movement pattern similarity"""
        if not existing_patterns:
            return 1.0
        
        similarities = []
        for existing in existing_patterns:
            # Handle length differences
            if len(pattern) != len(existing):
                min_len = min(len(pattern), len(existing))
                pattern_comp = pattern[:min_len]
                existing_comp = existing[:min_len]
            else:
                pattern_comp = pattern
                existing_comp = existing
            
            # Use dynamic time warping-like similarity for motor patterns
            # Simplified version using correlation
            if len(pattern_comp) > 1 and len(existing_comp) > 1:
                correlation = np.corrcoef(pattern_comp, existing_comp)[0, 1]
                if not np.isnan(correlation):
                    similarities.append(abs(correlation))
        
        if not similarities:
            return 1.0
        
        max_similarity = max(similarities)
        return 1.0 - max_similarity


class TemporalMemoryEncoder(MemoryEncoder):
    """Temporal-specific memory encoding"""
    
    def __init__(self, target_sparsity: float = 0.02):
        self.target_sparsity = target_sparsity
    
    def encode_pattern(self, signal: np.ndarray, attention_map: np.ndarray) -> np.ndarray:
        """Encode temporal signal into sparse representation"""
        # Apply attention mask to extract attended timing patterns
        attended_signal = signal * attention_map
        
        # For temporal signals, emphasize rhythmic patterns
        # Extract both signal and its rhythm features
        rhythm_features = self._extract_rhythm_features(attended_signal)
        combined_signal = np.concatenate([attended_signal, rhythm_features])
        
        # Normalize
        normalized = (combined_signal - np.mean(combined_signal)) / (np.std(combined_signal) + 1e-8)
        
        # Create sparse representation
        k = max(1, int(len(normalized) * self.target_sparsity))
        sparse_indices = np.argsort(np.abs(normalized))[-k:]
        
        sparse_pattern = np.zeros_like(normalized)
        sparse_pattern[sparse_indices] = normalized[sparse_indices]
        
        return sparse_pattern
    
    def _extract_rhythm_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract rhythm features from temporal signal"""
        if len(signal) < 4:
            return np.array([0.0])
        
        # Simple rhythm features
        intervals = np.diff(signal)
        interval_variance = np.var(intervals)
        
        # Autocorrelation peaks (simplified)
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find rhythm strength
        rhythm_strength = np.std(autocorr[:min(len(autocorr), len(signal)//2)])
        
        return np.array([interval_variance, rhythm_strength])
    
    def extract_metadata(self, signal: np.ndarray, attention_map: np.ndarray) -> Dict[str, Any]:
        """Extract temporal metadata"""
        attended_signal = signal * attention_map
        
        if len(signal.shape) == 1:
            intervals = np.diff(attended_signal)
            
            return {
                'signal_dimensions': signal.shape,
                'attention_coverage': np.mean(attention_map > 0.1),
                'signal_mean': np.mean(attended_signal),
                'signal_std': np.std(attended_signal),
                'interval_mean': np.mean(intervals),
                'interval_std': np.std(intervals),
                'rhythm_regularity': 1.0 / (np.std(intervals) + 1e-8),
                'temporal_complexity': np.std(attended_signal),
                'pattern_length': len(attended_signal)
            }
        else:
            raise ValueError("Temporal encoder only supports 1D signals")
    
    def calculate_novelty(self, pattern: np.ndarray, existing_patterns: List[np.ndarray]) -> float:
        """Calculate temporal novelty using rhythm pattern similarity"""
        if not existing_patterns:
            return 1.0
        
        similarities = []
        for existing in existing_patterns:
            # Handle length differences
            if len(pattern) != len(existing):
                min_len = min(len(pattern), len(existing))
                pattern_comp = pattern[:min_len]
                existing_comp = existing[:min_len]
            else:
                pattern_comp = pattern
                existing_comp = existing
            
            # Use phase-invariant similarity for temporal patterns
            if len(pattern_comp) > 1 and len(existing_comp) > 1:
                # Cross-correlation to find best alignment
                correlation = np.correlate(pattern_comp, existing_comp, mode='full')
                max_correlation = np.max(correlation) / (np.linalg.norm(pattern_comp) * np.linalg.norm(existing_comp) + 1e-8)
                similarities.append(abs(max_correlation))
        
        if not similarities:
            return 1.0
        
        max_similarity = max(similarities)
        return 1.0 - max_similarity


class UniversalMemorySystem:
    """Universal memory system that works with any signal modality"""
    
    def __init__(self, memory_capacity: int = 10000, consolidation_threshold: float = 0.7):
        self.memory_capacity = memory_capacity
        self.consolidation_threshold = consolidation_threshold
        
        # Initialize GPU device detection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.use_gpu = True
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            self.use_gpu = True
        else:
            self.device = torch.device('cpu')
            self.use_gpu = False
        
        # Memory storage
        self.memory_patterns: Dict[str, MemoryPattern] = {}
        self.modality_indices: Dict[ModalityType, List[str]] = defaultdict(list)
        self.temporal_sequence: List[str] = []
        
        # Modality-specific encoders
        self.encoders = {
            ModalityType.VISUAL: VisualMemoryEncoder(),
            ModalityType.AUDIO: AudioMemoryEncoder(),
            ModalityType.TACTILE: TactileMemoryEncoder(),
            ModalityType.MOTOR: MotorMemoryEncoder(),
            ModalityType.TEMPORAL: TemporalMemoryEncoder(),
        }
        
        # Attention system for cross-modal associations
        self.attention_system = UniversalAttentionSystem()
        
        # Statistics
        self.formation_stats = {
            'total_patterns': 0,
            'attended_patterns': 0,
            'consolidated_patterns': 0,
            'cross_modal_associations': 0
        }
    
    def form_memory(self, signal: np.ndarray, modality: ModalityType, 
                   attention_map: np.ndarray, brain_output: Optional[np.ndarray] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Optional[MemoryPattern]:
        """
        Form memory from attended signal
        
        Args:
            signal: Input signal of any dimensionality
            modality: Signal modality type
            attention_map: Attention map from universal attention system
            brain_output: Optional brain response for association
            metadata: Optional additional metadata
            
        Returns:
            Created memory pattern or None if attention too low
        """
        try:
            # Check if attention is sufficient for memory formation
            attention_strength = np.mean(attention_map)
            if attention_strength < 0.3:  # Threshold for memory formation
                return None
            
            # Get modality-specific encoder
            if modality not in self.encoders:
                print(f"No encoder for modality {modality}")
                return None
            
            encoder = self.encoders[modality]
            
            # Encode pattern
            sparse_pattern = encoder.encode_pattern(signal, attention_map)
            
            # Extract metadata
            pattern_metadata = encoder.extract_metadata(signal, attention_map)
            if metadata:
                pattern_metadata.update(metadata)
            
            # Calculate novelty
            existing_patterns = [self.memory_patterns[pid].sparse_representation 
                               for pid in self.modality_indices[modality]]
            novelty_score = encoder.calculate_novelty(sparse_pattern, existing_patterns)
            
            # Create memory pattern
            pattern_id = f"{modality.value}_{time.time():.6f}"
            signal_shape = self._get_signal_shape(signal)
            
            memory_pattern = MemoryPattern(
                pattern_id=pattern_id,
                modality=modality,
                signal_shape=signal_shape,
                sparse_representation=sparse_pattern,
                attention_weight=attention_strength,
                novelty_score=novelty_score,
                timestamp=time.time(),
                metadata=pattern_metadata
            )
            
            # Add temporal context
            if self.temporal_sequence:
                # Link to recent patterns
                recent_patterns = self.temporal_sequence[-5:]  # Last 5 patterns
                memory_pattern.temporal_context = recent_patterns
            
            # Cross-modal associations
            if brain_output is not None:
                self._create_cross_modal_associations(memory_pattern, brain_output)
            
            # Store memory
            self.memory_patterns[pattern_id] = memory_pattern
            self.modality_indices[modality].append(pattern_id)
            self.temporal_sequence.append(pattern_id)
            
            # Update statistics
            self.formation_stats['total_patterns'] += 1
            self.formation_stats['attended_patterns'] += 1
            
            # Memory consolidation
            if len(self.memory_patterns) > self.memory_capacity:
                self._consolidate_memories()
            
            return memory_pattern
            
        except Exception as e:
            print(f"Memory formation error: {e}")
            return None
    
    def _get_signal_shape(self, signal: np.ndarray) -> SignalShape:
        """Determine signal shape type"""
        if signal.ndim == 0:
            return SignalShape.SCALAR
        elif signal.ndim == 1:
            return SignalShape.VECTOR
        elif signal.ndim == 2:
            return SignalShape.MATRIX
        else:
            return SignalShape.TENSOR
    
    def _create_cross_modal_associations(self, memory_pattern: MemoryPattern, 
                                       brain_output: np.ndarray):
        """Create associations with other modalities based on brain response"""
        # Find patterns with similar brain responses
        current_time = time.time()
        time_window = 2.0  # 2 second window
        
        # Get candidate patterns for association
        candidate_patterns = []
        for pattern_id, pattern in self.memory_patterns.items():
            # Only consider recent patterns from different modalities
            if (pattern.modality != memory_pattern.modality and 
                current_time - pattern.timestamp < time_window):
                candidate_patterns.append((pattern_id, pattern))
        
        if not candidate_patterns:
            return
        
        # Calculate similarities using GPU if available
        if self.use_gpu and len(candidate_patterns) > 5:
            similar_patterns = self._calculate_similarities_gpu(
                memory_pattern, candidate_patterns, brain_output
            )
        else:
            similar_patterns = self._calculate_similarities_cpu(
                memory_pattern, candidate_patterns, brain_output
            )
        
        if similar_patterns:
            memory_pattern.associated_patterns = similar_patterns
            self.formation_stats['cross_modal_associations'] += 1
    
    def _calculate_similarities_gpu(self, memory_pattern: MemoryPattern, 
                                  candidate_patterns: List[Tuple[str, MemoryPattern]], 
                                  brain_output: np.ndarray) -> List[str]:
        """GPU-accelerated similarity calculation"""
        try:
            # Convert target pattern to tensor
            target_tensor = torch.from_numpy(memory_pattern.sparse_representation.astype(np.float32)).to(self.device)
            
            # Stack candidate patterns into a single tensor
            candidate_tensors = []
            pattern_ids = []
            
            for pattern_id, pattern in candidate_patterns:
                candidate_tensors.append(torch.from_numpy(pattern.sparse_representation.astype(np.float32)))
                pattern_ids.append(pattern_id)
            
            if not candidate_tensors:
                return []
                
            # Stack and move to GPU
            candidates = torch.stack(candidate_tensors).to(self.device)
            
            # Calculate cosine similarities in batch
            target_norm = F.normalize(target_tensor.unsqueeze(0), p=2, dim=1)
            candidate_norms = F.normalize(candidates, p=2, dim=1)
            
            similarities = torch.mm(target_norm, candidate_norms.t()).squeeze()
            
            # Find patterns with similarity > 0.7
            similar_mask = similarities > 0.7
            similar_indices = torch.nonzero(similar_mask, as_tuple=True)[0]
            
            # Convert back to pattern IDs
            similar_patterns = [pattern_ids[i] for i in similar_indices.cpu().numpy()]
            
            return similar_patterns
            
        except Exception as e:
            # Fallback to CPU calculation
            return self._calculate_similarities_cpu(memory_pattern, candidate_patterns, brain_output)
    
    def _calculate_similarities_cpu(self, memory_pattern: MemoryPattern, 
                                  candidate_patterns: List[Tuple[str, MemoryPattern]], 
                                  brain_output: np.ndarray) -> List[str]:
        """CPU fallback for similarity calculation"""
        similar_patterns = []
        
        for pattern_id, pattern in candidate_patterns:
            # Calculate cosine similarity between sparse representations
            dot_product = np.dot(memory_pattern.sparse_representation, pattern.sparse_representation)
            norm_product = (np.linalg.norm(memory_pattern.sparse_representation) * 
                          np.linalg.norm(pattern.sparse_representation))
            
            similarity = dot_product / (norm_product + 1e-8)
            
            if similarity > 0.7:
                similar_patterns.append(pattern_id)
        
        return similar_patterns
    
    def _consolidate_memories(self):
        """Consolidate memories when capacity is reached"""
        # Remove oldest, least attended patterns
        patterns_by_score = sorted(
            self.memory_patterns.items(),
            key=lambda x: x[1].attention_weight * x[1].novelty_score
        )
        
        # Remove bottom 10%
        remove_count = max(1, len(patterns_by_score) // 10)
        for i in range(remove_count):
            pattern_id, pattern = patterns_by_score[i]
            
            # Remove from all indices
            del self.memory_patterns[pattern_id]
            if pattern_id in self.modality_indices[pattern.modality]:
                self.modality_indices[pattern.modality].remove(pattern_id)
            if pattern_id in self.temporal_sequence:
                self.temporal_sequence.remove(pattern_id)
        
        self.formation_stats['consolidated_patterns'] += remove_count
    
    def retrieve_patterns(self, modality: Optional[ModalityType] = None,
                         time_window: Optional[float] = None,
                         min_attention: float = 0.0) -> List[MemoryPattern]:
        """Retrieve memory patterns with optional filtering"""
        patterns = []
        current_time = time.time()
        
        for pattern_id, pattern in self.memory_patterns.items():
            # Filter by modality
            if modality and pattern.modality != modality:
                continue
            
            # Filter by time window
            if time_window and current_time - pattern.timestamp > time_window:
                continue
            
            # Filter by attention
            if pattern.attention_weight < min_attention:
                continue
            
            patterns.append(pattern)
        
        # Sort by recency and attention
        patterns.sort(key=lambda p: p.timestamp * p.attention_weight, reverse=True)
        return patterns
    
    def get_cross_modal_associations(self, pattern_id: str) -> List[MemoryPattern]:
        """Get cross-modal associations for a pattern"""
        if pattern_id not in self.memory_patterns:
            return []
        
        pattern = self.memory_patterns[pattern_id]
        associated_patterns = []
        
        for assoc_id in pattern.associated_patterns:
            if assoc_id in self.memory_patterns:
                associated_patterns.append(self.memory_patterns[assoc_id])
        
        return associated_patterns
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics"""
        stats = self.formation_stats.copy()
        stats.update({
            'current_patterns': len(self.memory_patterns),
            'visual_patterns': len(self.modality_indices[ModalityType.VISUAL]),
            'audio_patterns': len(self.modality_indices[ModalityType.AUDIO]),
            'avg_attention': np.mean([p.attention_weight for p in self.memory_patterns.values()]) if self.memory_patterns else 0,
            'avg_novelty': np.mean([p.novelty_score for p in self.memory_patterns.values()]) if self.memory_patterns else 0
        })
        return stats
    
    def add_encoder(self, modality: ModalityType, encoder: MemoryEncoder):
        """Add a new modality-specific encoder"""
        self.encoders[modality] = encoder
    
    def save_memories(self, filepath: str):
        """Save memories to file"""
        try:
            data = {
                'patterns': [p.to_dict() for p in self.memory_patterns.values()],
                'statistics': self.get_statistics(),
                'timestamp': time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Saved {len(self.memory_patterns)} memories to {filepath}")
            
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    def load_memories(self, filepath: str):
        """Load memories from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Clear existing memories
            self.memory_patterns.clear()
            self.modality_indices.clear()
            self.temporal_sequence.clear()
            
            # Load patterns
            for pattern_data in data['patterns']:
                pattern = MemoryPattern.from_dict(pattern_data)
                self.memory_patterns[pattern.pattern_id] = pattern
                self.modality_indices[pattern.modality].append(pattern.pattern_id)
                self.temporal_sequence.append(pattern.pattern_id)
            
            # Sort temporal sequence by timestamp
            self.temporal_sequence.sort(key=lambda pid: self.memory_patterns[pid].timestamp)
            
            print(f"Loaded {len(self.memory_patterns)} memories from {filepath}")
            
        except Exception as e:
            print(f"Error loading memories: {e}")
    
    def cleanup(self):
        """Clean up memory system"""
        self.memory_patterns.clear()
        self.modality_indices.clear()
        self.temporal_sequence.clear()
        
        # Reset statistics
        self.formation_stats = {
            'total_patterns': 0,
            'attended_patterns': 0,
            'consolidated_patterns': 0,
            'cross_modal_associations': 0
        }