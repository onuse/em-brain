#!/usr/bin/env python3
"""
Pattern-Based Attention System

Implements attention through pattern salience rather than spatial gradients.
This is more biologically plausible and coordinate-free.

Key principles:
1. Attention emerges from pattern surprise/novelty/importance
2. No spatial coordinates or gradients needed
3. Patterns compete for limited attention resources
4. Cross-modal patterns can bind through synchrony
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from collections import deque
import time

from .field_types import FieldDynamicsFamily


@dataclass
class PatternSignature:
    """Represents a memorable pattern configuration."""
    pattern_id: str
    pattern_tensor: torch.Tensor
    modality: str  # visual, audio, motor, etc.
    frequency: float  # How often seen
    last_seen: float  # Timestamp
    importance: float  # Learned importance
    associations: Set[str]  # Associated pattern IDs


@dataclass 
class AttentionFocus:
    """Represents current attention focus."""
    primary_pattern: Optional[str] = None
    supporting_patterns: List[str] = None
    focus_strength: float = 0.0
    duration: float = 0.0
    
    def __post_init__(self):
        if self.supporting_patterns is None:
            self.supporting_patterns = []


class PatternBasedAttention:
    """
    Attention system based on pattern salience and surprise.
    
    This replaces gradient-based attention with pattern-based attention,
    making the system more organic and coordinate-free.
    """
    
    def __init__(self,
                 field_shape: Tuple[int, ...],
                 attention_capacity: int = 5,
                 device: torch.device = torch.device('cpu'),
                 quiet_mode: bool = False):
        """
        Initialize pattern-based attention.
        
        Args:
            field_shape: Shape of the unified field
            attention_capacity: Max patterns that can be attended simultaneously
            device: Computation device
            quiet_mode: Suppress debug output
        """
        self.field_shape = field_shape
        self.attention_capacity = attention_capacity
        self.device = device
        self.quiet_mode = quiet_mode
        
        # Pattern memory
        self.known_patterns: Dict[str, PatternSignature] = {}
        self.pattern_history = deque(maxlen=100)
        
        # Attention state
        self.current_focus = AttentionFocus()
        self.attention_buffer: List[Tuple[str, float]] = []  # (pattern_id, salience)
        
        # Salience factors
        self.novelty_weight = 0.4
        self.surprise_weight = 0.3
        self.importance_weight = 0.3
        
        # Pattern statistics for surprise detection
        self.pattern_statistics = {
            'mean_energy': 0.5,
            'mean_variance': 0.1,
            'mean_frequency': 0.5
        }
        
        # Cross-modal binding through synchrony
        self.synchrony_threshold = 0.8
        self.binding_window = 0.1  # 100ms
        
        if not quiet_mode:
            print(f"👁️  Pattern-Based Attention initialized")
            print(f"   Attention capacity: {attention_capacity} patterns")
            print(f"   No coordinates or gradients!")
    
    def process_field_patterns(self, 
                             field: torch.Tensor,
                             sensory_patterns: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Process field patterns to determine attention focus.
        
        Args:
            field: Current unified field state
            sensory_patterns: Optional modality-specific patterns
            
        Returns:
            Attention state including focus and salience
        """
        # Calculate salience for each pattern
        pattern_saliences = []
        
        # Process sensory patterns first (higher priority)
        if sensory_patterns:
            for modality, pattern_tensor in sensory_patterns.items():
                if pattern_tensor is not None and len(pattern_tensor) > 0:
                    # Extract patterns from sensory input
                    sensory_subpatterns = self._extract_sensory_patterns(pattern_tensor, modality)
                    
                    for subpattern_data in sensory_subpatterns:
                        subpattern = subpattern_data['tensor']
                        
                        # Calculate pattern salience with modality boost
                        salience = self._calculate_pattern_salience(subpattern, modality)
                        salience *= 1.5  # Boost sensory patterns
                        
                        # Check if this is a known pattern
                        pattern_id = self._find_or_create_pattern(subpattern, modality)
                        
                        pattern_saliences.append((pattern_id, salience))
        
        # Extract patterns from field (lower priority)
        field_patterns = self._extract_field_patterns(field)
        
        for pattern_data in field_patterns:
            pattern_tensor = pattern_data['tensor']
            modality = pattern_data.get('modality', 'field')
            
            # Calculate pattern salience
            salience = self._calculate_pattern_salience(pattern_tensor, modality)
            
            # Check if this is a known pattern
            pattern_id = self._find_or_create_pattern(pattern_tensor, modality)
            
            pattern_saliences.append((pattern_id, salience))
        
        # Sort by salience and select top patterns
        pattern_saliences.sort(key=lambda x: x[1], reverse=True)
        self.attention_buffer = pattern_saliences[:self.attention_capacity]
        
        # Update attention focus
        self._update_attention_focus()
        
        # Detect cross-modal bindings
        bindings = self._detect_pattern_bindings(sensory_patterns)
        
        return {
            'focus': self.current_focus,
            'attended_patterns': [p[0] for p in self.attention_buffer],
            'salience_map': self._create_salience_map(field),
            'bindings': bindings,
            'pattern_count': len(self.known_patterns)
        }
    
    def _extract_sensory_patterns(self, sensory_input: torch.Tensor, modality: str) -> List[Dict[str, Any]]:
        """Extract patterns specifically from sensory input."""
        patterns = []
        
        # Ensure tensor format
        if not isinstance(sensory_input, torch.Tensor):
            sensory_input = torch.tensor(sensory_input, device=self.device)
        
        input_size = len(sensory_input)
        
        # 1. Whole pattern (if meaningful)
        if self._is_meaningful_pattern(sensory_input):
            patterns.append({
                'tensor': sensory_input.clone(),
                'type': 'whole',
                'modality': modality,
                'size': input_size
            })
        
        # 2. Sliding windows for local patterns
        window_sizes = [4, 8] if input_size >= 8 else [2, 4]
        for window_size in window_sizes:
            if window_size <= input_size:
                stride = max(1, window_size // 2)
                for i in range(0, input_size - window_size + 1, stride):
                    pattern = sensory_input[i:i + window_size]
                    if self._is_meaningful_pattern(pattern):
                        patterns.append({
                            'tensor': pattern,
                            'type': 'local',
                            'modality': modality,
                            'size': window_size
                        })
        
        # 3. Statistical signature
        if input_size >= 4:
            stat_pattern = torch.tensor([
                torch.mean(sensory_input).item(),
                torch.std(sensory_input).item() if input_size > 1 else 0.0,
                torch.max(sensory_input).item(),
                torch.min(sensory_input).item()
            ], device=self.device)
            
            patterns.append({
                'tensor': stat_pattern,
                'type': 'statistical',
                'modality': modality,
                'size': 4
            })
        
        return patterns
    
    def _extract_field_patterns(self, field: torch.Tensor) -> List[Dict[str, Any]]:
        """Extract meaningful patterns from the field."""
        patterns = []
        
        # Flatten field for pattern analysis
        flat_field = field.flatten()
        field_size = len(flat_field)
        
        # 1. Extract local patterns (sliding windows)
        window_sizes = [8, 16, 32]
        for window_size in window_sizes:
            if window_size <= field_size:
                for i in range(0, field_size - window_size + 1, window_size // 2):
                    pattern = flat_field[i:i + window_size]
                    if self._is_meaningful_pattern(pattern):
                        patterns.append({
                            'tensor': pattern,
                            'type': 'local',
                            'size': window_size
                        })
        
        # 2. Extract frequency patterns (via FFT)
        if field_size >= 16:
            try:
                # Real FFT for pattern extraction
                freq_pattern = torch.fft.rfft(flat_field)
                freq_magnitudes = torch.abs(freq_pattern)
                
                # Find dominant frequencies
                top_k = min(5, len(freq_magnitudes))
                top_freqs, _ = torch.topk(freq_magnitudes, top_k)
                
                patterns.append({
                    'tensor': top_freqs,
                    'type': 'frequency',
                    'size': top_k
                })
            except:
                pass
        
        # 3. Extract statistical patterns
        stat_pattern = torch.tensor([
            torch.mean(field).item(),
            torch.std(field).item(),
            torch.min(field).item(),
            torch.max(field).item(),
            self._calculate_entropy(field)
        ], device=self.device)
        
        patterns.append({
            'tensor': stat_pattern,
            'type': 'statistical',
            'size': 5
        })
        
        return patterns
    
    def _is_meaningful_pattern(self, pattern: torch.Tensor) -> bool:
        """Check if a pattern is meaningful (not noise)."""
        # Pattern is meaningful if it has structure
        if len(pattern) < 2:
            return False
        
        # Check variance (not flat)
        if torch.var(pattern).item() < 0.001:
            return False
        
        # Check range (not near zero)
        if torch.max(torch.abs(pattern)).item() < 0.01:
            return False
        
        return True
    
    def _calculate_pattern_salience(self, pattern: torch.Tensor, modality: str) -> float:
        """
        Calculate salience of a pattern based on novelty, surprise, and importance.
        """
        # Novelty: How new/unique is this pattern?
        novelty = self._calculate_novelty(pattern)
        
        # Surprise: How unexpected is this pattern?
        surprise = self._calculate_surprise(pattern)
        
        # Importance: Learned or innate importance
        importance = self._calculate_importance(pattern, modality)
        
        # Weighted combination
        salience = (self.novelty_weight * novelty +
                   self.surprise_weight * surprise +
                   self.importance_weight * importance)
        
        return salience
    
    def _calculate_novelty(self, pattern: torch.Tensor) -> float:
        """Calculate how novel/unique a pattern is."""
        if len(self.pattern_history) == 0:
            return 1.0
        
        # Compare to recent patterns
        min_similarity = 1.0
        for past_pattern_id in self.pattern_history:
            if past_pattern_id in self.known_patterns:
                past_pattern = self.known_patterns[past_pattern_id].pattern_tensor
                if len(past_pattern) == len(pattern):
                    similarity = torch.nn.functional.cosine_similarity(
                        pattern.unsqueeze(0),
                        past_pattern.unsqueeze(0)
                    ).item()
                    min_similarity = min(min_similarity, similarity)
        
        return 1.0 - min_similarity
    
    def _calculate_surprise(self, pattern: torch.Tensor) -> float:
        """Calculate how surprising a pattern is given expectations."""
        # Surprise based on deviation from expected statistics
        pattern_energy = torch.mean(torch.abs(pattern)).item()
        pattern_variance = torch.var(pattern).item()
        
        # How much does this deviate from expected?
        energy_surprise = abs(pattern_energy - self.pattern_statistics['mean_energy'])
        variance_surprise = abs(pattern_variance - self.pattern_statistics['mean_variance'])
        
        # Update running statistics (learning)
        alpha = 0.01
        self.pattern_statistics['mean_energy'] = (
            (1 - alpha) * self.pattern_statistics['mean_energy'] + 
            alpha * pattern_energy
        )
        self.pattern_statistics['mean_variance'] = (
            (1 - alpha) * self.pattern_statistics['mean_variance'] + 
            alpha * pattern_variance
        )
        
        # Normalize surprise
        surprise = np.tanh(energy_surprise + variance_surprise)
        
        return surprise
    
    def _calculate_importance(self, pattern: torch.Tensor, modality: str) -> float:
        """Calculate learned or innate importance of a pattern."""
        # Base importance by modality
        modality_importance = {
            'visual': 0.7,
            'audio': 0.6,
            'motor': 0.8,
            'field': 0.5
        }.get(modality, 0.5)
        
        # Boost importance for certain pattern characteristics
        # High energy patterns (potential threats/opportunities)
        if torch.mean(torch.abs(pattern)).item() > 0.8:
            modality_importance *= 1.5
        
        # Rhythmic patterns (biological relevance)
        if self._is_rhythmic(pattern):
            modality_importance *= 1.3
        
        return min(1.0, modality_importance)
    
    def _is_rhythmic(self, pattern: torch.Tensor) -> bool:
        """Check if pattern has rhythmic structure."""
        if len(pattern) < 4:
            return False
        
        # Simple rhythm detection via autocorrelation
        pattern_np = pattern.cpu().numpy()
        autocorr = np.correlate(pattern_np, pattern_np, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Look for peaks in autocorrelation (rhythm)
        if len(autocorr) > 2:
            peaks = np.where(np.diff(np.sign(np.diff(autocorr))) < 0)[0]
            return len(peaks) > 1
        
        return False
    
    def _calculate_entropy(self, field: torch.Tensor) -> float:
        """Calculate field entropy as a measure of disorder/information."""
        # Normalize field to probabilities
        field_positive = field - torch.min(field) + 1e-8
        field_probs = field_positive / torch.sum(field_positive)
        
        # Calculate entropy
        entropy = -torch.sum(field_probs * torch.log2(field_probs + 1e-8))
        
        return entropy.item()
    
    def _find_or_create_pattern(self, pattern: torch.Tensor, modality: str) -> str:
        """Find existing pattern or create new one."""
        # Check if pattern matches known patterns
        for pattern_id, known_pattern in self.known_patterns.items():
            if (len(known_pattern.pattern_tensor) == len(pattern) and
                known_pattern.modality == modality):
                similarity = torch.nn.functional.cosine_similarity(
                    pattern.unsqueeze(0),
                    known_pattern.pattern_tensor.unsqueeze(0)
                ).item()
                
                if similarity > 0.9:  # Close enough to be same pattern
                    # Update pattern statistics
                    known_pattern.frequency += 1
                    known_pattern.last_seen = time.time()
                    self.pattern_history.append(pattern_id)
                    return pattern_id
        
        # Create new pattern
        pattern_id = f"pat_{len(self.known_patterns)}_{modality[:3]}"
        new_pattern = PatternSignature(
            pattern_id=pattern_id,
            pattern_tensor=pattern.clone().detach(),
            modality=modality,
            frequency=1.0,
            last_seen=time.time(),
            importance=0.5,
            associations=set()
        )
        
        self.known_patterns[pattern_id] = new_pattern
        self.pattern_history.append(pattern_id)
        
        if not self.quiet_mode:
            print(f"👁️  New pattern discovered: {pattern_id}")
        
        return pattern_id
    
    def _update_attention_focus(self):
        """Update current attention focus based on attention buffer."""
        if not self.attention_buffer:
            self.current_focus = AttentionFocus()
            return
        
        # Primary focus is highest salience
        primary_id, primary_salience = self.attention_buffer[0]
        
        # Check if focus is shifting
        if primary_id != self.current_focus.primary_pattern:
            if not self.quiet_mode and self.current_focus.primary_pattern:
                print(f"👁️  Attention shift: {self.current_focus.primary_pattern} → {primary_id}")
            
            self.current_focus = AttentionFocus(
                primary_pattern=primary_id,
                supporting_patterns=[p[0] for p in self.attention_buffer[1:]],
                focus_strength=primary_salience,
                duration=0.0
            )
        else:
            # Maintain focus
            self.current_focus.duration += 0.05  # Assume 50ms cycles
            self.current_focus.focus_strength = primary_salience
            self.current_focus.supporting_patterns = [p[0] for p in self.attention_buffer[1:]]
    
    def _detect_pattern_bindings(self, 
                                sensory_patterns: Optional[Dict[str, torch.Tensor]]) -> List[Set[str]]:
        """Detect cross-modal pattern bindings through synchrony."""
        if not sensory_patterns or len(sensory_patterns) < 2:
            return []
        
        bindings = []
        modalities = list(sensory_patterns.keys())
        
        # Check for synchronous patterns across modalities
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                pattern1 = sensory_patterns[mod1]
                pattern2 = sensory_patterns[mod2]
                
                # Check synchrony (simplified - correlation)
                if len(pattern1) == len(pattern2):
                    correlation = torch.nn.functional.cosine_similarity(
                        pattern1.unsqueeze(0),
                        pattern2.unsqueeze(0)
                    ).item()
                    
                    if correlation > self.synchrony_threshold:
                        # Patterns are synchronous - bind them
                        binding = {f"{mod1}_pattern", f"{mod2}_pattern"}
                        bindings.append(binding)
                        
                        if not self.quiet_mode:
                            print(f"🔗 Cross-modal binding: {mod1} ↔ {mod2}")
        
        return bindings
    
    def _create_salience_map(self, field: torch.Tensor) -> torch.Tensor:
        """Create a salience map showing attention distribution."""
        salience_map = torch.zeros_like(field)
        
        # For each attended pattern, increase salience in relevant regions
        for pattern_id, salience in self.attention_buffer:
            if pattern_id in self.known_patterns:
                # Simplified: add salience uniformly
                # In practice, would localize to pattern regions
                salience_map += salience * 0.2
        
        # Normalize
        if torch.max(salience_map) > 0:
            salience_map = salience_map / torch.max(salience_map)
        
        return salience_map
    
    def modulate_field_with_attention(self, 
                                    field: torch.Tensor,
                                    attention_state: Dict[str, Any]) -> torch.Tensor:
        """Modulate field activity based on attention state."""
        if not attention_state.get('focus'):
            return field
        
        focus = attention_state['focus']
        if not focus.primary_pattern:
            return field
        
        # Boost activity in attended regions
        modulated_field = field.clone()
        
        # Simple modulation: scale by focus strength
        attention_boost = 1.0 + focus.focus_strength * 0.5
        modulated_field = modulated_field * attention_boost
        
        # Suppress unattended regions (optional)
        if focus.focus_strength > 0.7:
            # Strong focus - suppress non-attended
            suppression = 0.8
            modulated_field = modulated_field * suppression
        
        return modulated_field
    
    def get_attention_metrics(self) -> Dict[str, Any]:
        """Get metrics about attention system performance."""
        return {
            'known_patterns': len(self.known_patterns),
            'current_focus': self.current_focus.primary_pattern,
            'focus_strength': self.current_focus.focus_strength,
            'focus_duration': self.current_focus.duration,
            'attended_count': len(self.attention_buffer),
            'pattern_statistics': self.pattern_statistics,
            'capacity_usage': len(self.attention_buffer) / self.attention_capacity
        }