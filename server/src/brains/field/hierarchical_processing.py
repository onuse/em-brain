#!/usr/bin/env python3
"""
Efficient Hierarchical Processing

Performance-optimized hierarchical processing that provides multi-level intelligence
without significant computational overhead. Uses temporal processing, attention guidance,
and multi-resolution fields for efficient hierarchy management.

Key Performance Strategies:
1. Temporal Processing: Different levels run at different frequencies
2. Attention-Guided: Only process hierarchies where attention is focused
3. Multi-Resolution: Parallel field resolutions with cross-coupling
4. Lazy Evaluation: Cache and reuse results when possible
5. Bounded Complexity: Hard limits on computational overhead
"""

import torch
import time
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    from .field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience, 
        FieldNativeAction
    )
    from .adaptive_field_impl import FieldImplementation
    from .attention_guided import AttentionGuidedProcessing
except ImportError:
    from field_types import (
        FieldDynamicsFamily, FieldDimension, UnifiedFieldExperience,
        FieldNativeAction
    )
    from adaptive_field_impl import FieldImplementation
    from attention_guided import AttentionGuidedProcessing


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical processing."""
    enable_temporal_hierarchy: bool = True      # Different frequencies per level
    enable_attention_guidance: bool = True     # Focus on attention regions only
    enable_multi_resolution: bool = True       # Multiple field resolutions
    
    # Temporal frequencies (cycles between processing)
    coarse_frequency: int = 5                  # Process every 5 cycles
    relational_frequency: int = 2             # Process every 2 cycles
    detailed_frequency: int = 1               # Process every cycle
    
    # Resolution ratios
    coarse_resolution_ratio: float = 0.25     # 25% of full resolution
    relational_resolution_ratio: float = 0.5  # 50% of full resolution
    
    # Performance bounds
    max_hierarchy_time_ms: float = 100.0      # Hard limit on hierarchy time
    attention_threshold: float = 0.1          # Minimum attention for processing
    cache_lifetime_cycles: int = 10           # How long to cache results


@dataclass
class HierarchicalLevel:
    """Represents one level in the hierarchy."""
    name: str
    resolution: int
    frequency: int                # Process every N cycles
    last_processed: int = 0      # Last cycle when processed
    cached_patterns: Dict = None
    processing_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.cached_patterns is None:
            self.cached_patterns = {}


class EfficientHierarchicalProcessing:
    """
    Efficient hierarchical processing that maintains real-time performance.
    
    Provides sophisticated multi-level intelligence through optimized algorithms
    that respect computational constraints.
    """
    
    def __init__(self, field_impl: FieldImplementation,
                 attention_processor: Optional[AttentionGuidedProcessing] = None,
                 config: Optional[HierarchicalConfig] = None,
                 base_resolution: int = 100,
                 quiet_mode: bool = False):
        
        self.field_impl = field_impl
        self.attention_processor = attention_processor
        self.config = config or HierarchicalConfig()
        self.base_resolution = base_resolution
        self.quiet_mode = quiet_mode
        
        # Initialize hierarchical levels
        self.levels = self._initialize_hierarchical_levels()
        
        # Processing state
        self.current_cycle = 0
        self.total_hierarchy_time_ms = 0.0
        self.hierarchy_enabled = True
        
        # Multi-resolution fields (if enabled)
        self.multi_resolution_fields = {}
        if self.config.enable_multi_resolution:
            self._initialize_multi_resolution_fields()
        
        # Performance tracking
        self.performance_metrics = {
            'total_cycles': 0,
            'hierarchy_overhead_ms': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'levels_processed': defaultdict(int)
        }
        
        if not quiet_mode:
            print(f"🏗️ Efficient Hierarchical Processing initialized")
            print(f"   Levels: {len(self.levels)} (coarse → detailed)")
            print(f"   Multi-resolution: {self.config.enable_multi_resolution}")
            print(f"   Attention-guided: {self.config.enable_attention_guidance}")
    
    def _initialize_hierarchical_levels(self) -> List[HierarchicalLevel]:
        """Initialize the hierarchical processing levels."""
        coarse_res = int(self.base_resolution * self.config.coarse_resolution_ratio)
        relational_res = int(self.base_resolution * self.config.relational_resolution_ratio)
        
        return [
            HierarchicalLevel(
                name="coarse",
                resolution=coarse_res,
                frequency=self.config.coarse_frequency
            ),
            HierarchicalLevel(
                name="relational", 
                resolution=relational_res,
                frequency=self.config.relational_frequency
            ),
            HierarchicalLevel(
                name="detailed",
                resolution=self.base_resolution,
                frequency=self.config.detailed_frequency
            )
        ]
    
    def _initialize_multi_resolution_fields(self):
        """Initialize multiple resolution fields for parallel processing."""
        # Create smaller resolution field implementations for coarse processing
        # This would require creating simplified field implementations
        # For now, we'll use the main field but sample at different resolutions
        self.multi_resolution_fields = {
            'coarse': {'resolution': self.levels[0].resolution, 'sampling_rate': 4},
            'relational': {'resolution': self.levels[1].resolution, 'sampling_rate': 2},
            'detailed': {'resolution': self.levels[2].resolution, 'sampling_rate': 1}
        }
    
    def process_hierarchical_intelligence(self, dt: float = 0.1,
                                        attention_regions: Optional[List] = None) -> Dict[str, Any]:
        """
        Main hierarchical processing with performance optimization.
        
        Returns hierarchy results while respecting time constraints.
        """
        if not self.hierarchy_enabled:
            return {'hierarchy_enabled': False}
        
        start_time = time.time()
        self.current_cycle += 1
        hierarchy_results = {}
        
        # Get attention regions if available
        active_attention_regions = []
        if self.config.enable_attention_guidance and self.attention_processor:
            attention_stats = self.attention_processor.get_attention_statistics()
            if 'attention_regions' in attention_stats:
                active_attention_regions = attention_stats['attention_regions']
        
        # Process each level based on temporal frequency
        for level in self.levels:
            if self._should_process_level(level):
                level_start = time.time()
                
                # Check time budget
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.config.max_hierarchy_time_ms * 0.8:
                    # Running out of time budget - skip remaining levels
                    break
                
                level_results = self._process_hierarchical_level(
                    level, active_attention_regions
                )
                hierarchy_results[level.name] = level_results
                
                level.last_processed = self.current_cycle
                level.processing_time_ms = (time.time() - level_start) * 1000
                self.performance_metrics['levels_processed'][level.name] += 1
        
        # Update performance metrics
        total_time_ms = (time.time() - start_time) * 1000
        self.total_hierarchy_time_ms += total_time_ms
        self.performance_metrics['total_cycles'] += 1
        self.performance_metrics['hierarchy_overhead_ms'] += total_time_ms
        
        # Auto-disable if consistently too slow
        if total_time_ms > self.config.max_hierarchy_time_ms:
            self._handle_performance_budget_exceeded(total_time_ms)
        
        hierarchy_results['processing_time_ms'] = total_time_ms
        hierarchy_results['hierarchy_enabled'] = self.hierarchy_enabled
        
        return hierarchy_results
    
    def _should_process_level(self, level: HierarchicalLevel) -> bool:
        """Determine if a level should be processed this cycle."""
        cycles_since_last = self.current_cycle - level.last_processed
        return cycles_since_last >= level.frequency
    
    def _process_hierarchical_level(self, level: HierarchicalLevel,
                                  attention_regions: List) -> Dict[str, Any]:
        """Process a single hierarchical level efficiently."""
        
        # Check cache first (lazy evaluation)
        cache_key = self._generate_cache_key(level, attention_regions)
        if self._can_use_cached_result(level, cache_key):
            self.performance_metrics['cache_hits'] += 1
            return level.cached_patterns.get(cache_key, {})
        
        self.performance_metrics['cache_misses'] += 1
        
        # Performance safety: timeout for level processing
        level_start = time.time()
        max_level_time = self.config.max_hierarchy_time_ms / 1000.0 / len(self.levels)
        
        # Get field gradients for this level
        gradients = self.field_impl.compute_field_gradients()
        
        # Check if we have time to process
        if (time.time() - level_start) > max_level_time * 0.5:
            # Already used half the budget just getting gradients - skip processing
            return {}
        
        # Process based on level and attention guidance with timeout protection
        try:
            if level.name == "coarse":
                patterns = self._process_coarse_level(gradients, attention_regions, level)
            elif level.name == "relational":
                patterns = self._process_relational_level(gradients, attention_regions, level)
            else:  # detailed
                patterns = self._process_detailed_level(gradients, attention_regions, level)
            
            # Check if processing took too long
            processing_time = time.time() - level_start
            if processing_time > max_level_time:
                # This level is too slow - increase its frequency (process less often)
                level.frequency = min(level.frequency * 2, 10)
                if not self.quiet_mode:
                    print(f"⚠️ {level.name} level too slow ({processing_time*1000:.1f}ms), reducing frequency to {level.frequency}")
            
        except Exception as e:
            # Graceful degradation - return empty patterns if processing fails
            if not self.quiet_mode:
                print(f"⚠️ {level.name} level processing failed: {e}")
            patterns = {}
        
        # Cache results
        level.cached_patterns[cache_key] = patterns
        self._cleanup_cache(level)
        
        return patterns
    
    def _process_coarse_level(self, gradients: Dict, attention_regions: List, 
                            level: HierarchicalLevel) -> Dict[str, Any]:
        """Process coarse-level patterns efficiently (optimized)."""
        patterns = {}
        
        # Limit processing to first few gradients for performance
        limited_gradients = dict(list(gradients.items())[:3])  # Only process first 3
        sampling_rate = self.multi_resolution_fields['coarse']['sampling_rate']
        
        for grad_name, grad_tensor in limited_gradients.items():
            if grad_tensor.numel() == 0:
                continue
            
            # Ultra-efficient coarse sampling - much sparser sampling
            if grad_tensor.dim() >= 3:
                # More aggressive sampling for coarse level
                coarse_sampling = sampling_rate * 2  # Even coarser
                sampled = grad_tensor[::coarse_sampling, ::coarse_sampling, ::coarse_sampling]
            else:
                sampled = grad_tensor[::sampling_rate * 2]
            
            # Very simple pattern detection - just check if above threshold
            if sampled.numel() > 0 and sampled.numel() < 1000:  # Safety check for size
                mean_activation = torch.mean(torch.abs(sampled)).item()
                
                # Simple threshold without expensive std calculation
                if mean_activation > 0.1:  # Fixed threshold instead of computed
                    patterns[f"coarse_{grad_name}"] = {
                        'strength': mean_activation,
                        'level': 'coarse'
                    }
        
        return patterns
    
    def _process_relational_level(self, gradients: Dict, attention_regions: List,
                                level: HierarchicalLevel) -> Dict[str, Any]:
        """Process relational-level patterns efficiently."""
        patterns = {}
        
        sampling_rate = self.multi_resolution_fields['relational']['sampling_rate']
        
        # Focus on relationships between gradient patterns
        grad_names = list(gradients.keys())
        
        for i, grad1_name in enumerate(grad_names[:3]):  # Limit to first 3 for performance
            for j, grad2_name in enumerate(grad_names[i+1:4], i+1):  # Limit combinations
                grad1 = gradients[grad1_name]
                grad2 = gradients[grad2_name]
                
                if grad1.numel() == 0 or grad2.numel() == 0:
                    continue
                
                # Sample at relational resolution
                if grad1.dim() >= 3:
                    sampled1 = grad1[::sampling_rate, ::sampling_rate, ::sampling_rate]
                    sampled2 = grad2[::sampling_rate, ::sampling_rate, ::sampling_rate]
                else:
                    sampled1 = grad1[::sampling_rate]
                    sampled2 = grad2[::sampling_rate]
                
                # Simple relationship measure - activation correlation
                min_size = min(sampled1.numel(), sampled2.numel())
                if min_size > 1:
                    s1_flat = sampled1.flatten()[:min_size]
                    s2_flat = sampled2.flatten()[:min_size]
                    
                    # Use simple correlation approximation (faster than torch.corrcoef)
                    mean1, mean2 = torch.mean(s1_flat), torch.mean(s2_flat)
                    correlation = torch.mean((s1_flat - mean1) * (s2_flat - mean2))
                    
                    if abs(correlation.item()) > 0.1:  # Significant relationship
                        patterns[f"relation_{grad1_name}_{grad2_name}"] = {
                            'correlation': correlation.item(),
                            'level': 'relational'
                        }
        
        return patterns
    
    def _process_detailed_level(self, gradients: Dict, attention_regions: List,
                              level: HierarchicalLevel) -> Dict[str, Any]:
        """Process detailed-level patterns efficiently."""
        patterns = {}
        
        # Focus on attention regions if available
        focus_regions = attention_regions if self.config.enable_attention_guidance else []
        
        # Process only where attention is focused (if attention guidance enabled)
        if focus_regions:
            for region in focus_regions[:3]:  # Limit to 3 regions for performance
                region_type = region.get('type', 'unknown')
                region_intensity = region.get('intensity', 0.0)
                
                if region_intensity > self.config.attention_threshold:
                    patterns[f"detailed_{region_type}"] = {
                        'intensity': region_intensity,
                        'age': region.get('age', 0.0),
                        'level': 'detailed'
                    }
        else:
            # No attention regions - use simple detailed analysis
            for grad_name, grad_tensor in list(gradients.items())[:2]:  # Limit for performance
                if grad_tensor.numel() > 0:
                    max_activation = torch.max(torch.abs(grad_tensor)).item()
                    if max_activation > 0.1:
                        patterns[f"detailed_{grad_name}"] = {
                            'max_activation': max_activation,
                            'level': 'detailed'
                        }
        
        return patterns
    
    def _generate_cache_key(self, level: HierarchicalLevel, attention_regions: List) -> str:
        """Generate cache key for level results."""
        # Simple key based on level name and attention region count
        attention_count = len(attention_regions) if attention_regions else 0
        return f"{level.name}_{attention_count}_{self.current_cycle // level.frequency}"
    
    def _can_use_cached_result(self, level: HierarchicalLevel, cache_key: str) -> bool:
        """Check if cached result can be used."""
        if cache_key not in level.cached_patterns:
            return False
        
        # Simple cache validation - results valid for cache_lifetime_cycles
        cache_cycle = int(cache_key.split('_')[-1])
        current_cache_cycle = self.current_cycle // level.frequency
        
        return (current_cache_cycle - cache_cycle) < (self.config.cache_lifetime_cycles // level.frequency)
    
    def _cleanup_cache(self, level: HierarchicalLevel):
        """Clean up old cache entries."""
        if len(level.cached_patterns) > 10:  # Keep cache small
            # Remove oldest entries
            keys_to_remove = list(level.cached_patterns.keys())[:-5]  # Keep last 5
            for key in keys_to_remove:
                del level.cached_patterns[key]
    
    def _handle_performance_budget_exceeded(self, time_ms: float):
        """Handle cases where hierarchy processing is too slow."""
        if not self.quiet_mode:
            print(f"⚠️ Hierarchy processing exceeded budget: {time_ms:.1f}ms > {self.config.max_hierarchy_time_ms}ms")
        
        # Adaptive response - reduce frequency of slower levels
        for level in self.levels:
            if level.processing_time_ms > self.config.max_hierarchy_time_ms / len(self.levels):
                level.frequency = min(level.frequency * 2, 10)  # Reduce frequency, cap at 10
                if not self.quiet_mode:
                    print(f"   Reduced {level.name} frequency to every {level.frequency} cycles")
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy processing statistics."""
        avg_overhead = 0.0
        if self.performance_metrics['total_cycles'] > 0:
            avg_overhead = (self.performance_metrics['hierarchy_overhead_ms'] / 
                          self.performance_metrics['total_cycles'])
        
        return {
            'hierarchy_enabled': self.hierarchy_enabled,
            'total_cycles': self.performance_metrics['total_cycles'],
            'average_overhead_ms': avg_overhead,
            'cache_hit_rate': (self.performance_metrics['cache_hits'] / 
                             max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])),
            'levels_processed': dict(self.performance_metrics['levels_processed']),
            'level_frequencies': {level.name: level.frequency for level in self.levels},
            'level_processing_times': {level.name: level.processing_time_ms for level in self.levels}
        }
    
    def enable_hierarchy(self, enable: bool = True):
        """Enable or disable hierarchical processing."""
        self.hierarchy_enabled = enable
    
    def adjust_performance_budget(self, max_time_ms: float):
        """Adjust the performance budget for hierarchy processing."""
        self.config.max_hierarchy_time_ms = max_time_ms