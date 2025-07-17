#!/usr/bin/env python3
"""
Unified Cortical Column Storage Architecture

Key insight: Instead of multiple separate storage systems, use a single
cortical column architecture that serves all brain systems.

Current problem:
- sensory_stream.storage (separate)
- emergent_hierarchy.storage (separate)  
- emergent_competition.storage (separate)
- column_storage (separate)

Each system searches independently → massive duplication and inefficiency

Solution: Unified cortical architecture where all patterns flow through
a single column-organized storage system that serves all brain functions.

This matches biology: cortical columns serve multiple functions simultaneously.
"""

import time
import torch
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    from .sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage
    from .cortical_columns import CorticalColumnStorage
except ImportError:
    from sparse_representations import SparsePattern, SparsePatternEncoder, SparsePatternStorage
    from cortical_columns import CorticalColumnStorage


class UnifiedCorticalStorage:
    """
    Single cortical column storage system that serves all brain functions.
    
    Instead of separate storage systems, all patterns flow through unified
    cortical columns that serve:
    - Stream storage (sensory, motor, temporal)
    - Temporal hierarchy queries
    - Competitive dynamics
    - Cross-stream co-activation
    
    This eliminates duplication and enables cross-system pattern sharing.
    """
    
    def __init__(self, pattern_dim: int, max_patterns: int = 1_000_000, 
                 max_columns: int = 1000, quiet_mode: bool = False):
        self.pattern_dim = pattern_dim
        self.max_patterns = max_patterns
        self.quiet_mode = quiet_mode
        
        # Single unified storage system
        self.base_storage = SparsePatternStorage(
            pattern_dim=pattern_dim,
            max_patterns=max_patterns,
            sparsity=0.02,
            quiet_mode=quiet_mode
        )
        
        # Cortical column organization
        self.column_storage = CorticalColumnStorage(
            self.base_storage,
            max_columns=max_columns,
            quiet_mode=quiet_mode
        )
        
        # Pattern encoders for different input types
        self.encoders = {
            'sensory': SparsePatternEncoder(pattern_dim, sparsity=0.02, quiet_mode=True),
            'motor': SparsePatternEncoder(pattern_dim, sparsity=0.02, quiet_mode=True),
            'temporal': SparsePatternEncoder(pattern_dim, sparsity=0.02, quiet_mode=True),
            'unified': SparsePatternEncoder(pattern_dim, sparsity=0.02, quiet_mode=True)
        }
        
        # Stream-specific pattern tracking
        self.stream_patterns = {
            'sensory': set(),
            'motor': set(),
            'temporal': set(),
            'hierarchy': set(),
            'competition': set()
        }
        
        # Cross-stream pattern relationships
        self.pattern_relationships = defaultdict(set)  # pattern_id -> {related_pattern_ids}
        
        # Performance tracking
        self.unified_stores = 0
        self.unified_retrievals = 0
        self.cross_stream_hits = 0
        
        if not quiet_mode:
            print(f"🧠 Unified Cortical Storage initialized")
            print(f"   Pattern dimension: {pattern_dim}")
            print(f"   Max patterns: {max_patterns:,}")
            print(f"   Max columns: {max_columns}")
            print(f"   🎯 Single storage system serves all brain functions")
    
    def store_pattern(self, dense_pattern: torch.Tensor, 
                     stream_type: str, 
                     pattern_id: str = None,
                     timestamp: float = None) -> str:
        """
        Store pattern in unified cortical system.
        
        All patterns flow through the same cortical columns regardless of
        which brain system is storing them.
        """
        if timestamp is None:
            timestamp = time.time()
        
        if pattern_id is None:
            pattern_id = f"{stream_type}_{int(timestamp * 1000000)}"
        
        # Encode as sparse pattern
        encoder = self.encoders.get(stream_type, self.encoders['unified'])
        sparse_pattern = encoder.encode_top_k(dense_pattern, pattern_id)
        
        # Store through unified cortical columns
        stored_pattern_id = self.column_storage.store_pattern_with_columns(sparse_pattern)
        
        # Track which stream this pattern belongs to
        self.stream_patterns[stream_type].add(stored_pattern_id)
        
        # Update cross-stream relationships
        self._update_cross_stream_relationships(stored_pattern_id, stream_type)
        
        self.unified_stores += 1
        return stored_pattern_id
    
    def find_similar_patterns(self, query_pattern: torch.Tensor,
                            stream_type: str = 'unified',
                            k: int = 10,
                            min_similarity: float = 0.1,
                            cross_stream: bool = True) -> List[Tuple[SparsePattern, float]]:
        """
        Find similar patterns using unified cortical columns.
        
        Key optimization: Single search serves all brain systems.
        Cross-stream patterns can be discovered automatically.
        """
        start_time = time.time()
        
        # Encode query as sparse pattern
        encoder = self.encoders.get(stream_type, self.encoders['unified'])
        query_sparse = encoder.encode_top_k(
            query_pattern, 
            f"query_{stream_type}_{int(time.time() * 1000000)}"
        )
        
        # Use cortical columns for O(1) clustering
        clustered_patterns = self.column_storage.get_clustered_patterns(
            query_sparse, max_columns=5
        )
        
        # Flatten clusters into ranked patterns
        similar_patterns = []
        for cluster in clustered_patterns:
            similar_patterns.extend(cluster)
        
        # Sort by similarity and apply filters
        similar_patterns.sort(key=lambda x: x[1], reverse=True)
        
        # Apply stream filtering if not cross-stream
        if not cross_stream and stream_type != 'unified':
            filtered_patterns = []
            target_stream_patterns = self.stream_patterns.get(stream_type, set())
            
            for pattern, similarity in similar_patterns:
                if pattern.pattern_id in target_stream_patterns:
                    filtered_patterns.append((pattern, similarity))
            
            similar_patterns = filtered_patterns
        
        # Apply similarity threshold and k limit
        filtered_patterns = [
            (pattern, similarity) for pattern, similarity in similar_patterns
            if similarity >= min_similarity
        ][:k]
        
        # Track cross-stream hits
        if cross_stream:
            for pattern, similarity in filtered_patterns:
                if pattern.pattern_id not in self.stream_patterns.get(stream_type, set()):
                    self.cross_stream_hits += 1
        
        self.unified_retrievals += 1
        return filtered_patterns
    
    def get_stream_patterns(self, stream_type: str) -> Set[str]:
        """Get patterns belonging to specific stream."""
        return self.stream_patterns.get(stream_type, set())
    
    def get_cross_stream_relationships(self, pattern_id: str) -> Set[str]:
        """Get patterns related to this pattern across streams."""
        return self.pattern_relationships.get(pattern_id, set())
    
    def _update_cross_stream_relationships(self, pattern_id: str, stream_type: str):
        """Update cross-stream pattern relationships."""
        # Find similar patterns in other streams
        pattern = self.base_storage.pattern_index.get(pattern_id)
        if pattern is None:
            return
        
        # Look for similar patterns in other streams
        for other_stream, other_patterns in self.stream_patterns.items():
            if other_stream == stream_type:
                continue
            
            # Sample some patterns from other stream for relationship discovery
            sample_patterns = list(other_patterns)[-10:]  # Recent patterns
            
            for other_pattern_id in sample_patterns:
                other_pattern = self.base_storage.pattern_index.get(other_pattern_id)
                if other_pattern is None:
                    continue
                
                # Calculate similarity
                similarity = pattern.jaccard_similarity(other_pattern)
                
                # Create relationship if sufficiently similar
                if similarity > 0.3:  # Threshold for cross-stream relationship
                    self.pattern_relationships[pattern_id].add(other_pattern_id)
                    self.pattern_relationships[other_pattern_id].add(pattern_id)
    
    def get_unified_stats(self) -> Dict[str, Any]:
        """Get statistics about unified cortical storage."""
        base_stats = self.base_storage.get_pattern_stats()
        column_stats = self.column_storage.get_column_stats()
        
        # Stream distribution
        stream_distribution = {
            stream: len(patterns) 
            for stream, patterns in self.stream_patterns.items()
        }
        
        # Cross-stream relationships
        total_relationships = sum(len(relations) for relations in self.pattern_relationships.values())
        
        return {
            'base_storage': base_stats,
            'cortical_columns': column_stats,
            'stream_distribution': stream_distribution,
            'total_patterns': base_stats['pattern_count'],
            'total_columns': column_stats['num_columns'],
            'cross_stream_relationships': total_relationships,
            'unified_stores': self.unified_stores,
            'unified_retrievals': self.unified_retrievals,
            'cross_stream_hits': self.cross_stream_hits,
            'cross_stream_hit_rate': self.cross_stream_hits / max(1, self.unified_retrievals),
            'storage_efficiency': 'unified' if base_stats['pattern_count'] > 0 else 'empty'
        }


def demonstrate_unified_cortical_performance():
    """Demonstrate performance gains from unified cortical storage."""
    print("🧠 UNIFIED CORTICAL STORAGE DEMONSTRATION")
    print("=" * 60)
    
    # Create unified storage
    unified_storage = UnifiedCorticalStorage(
        pattern_dim=100,
        max_patterns=100000,
        max_columns=100,
        quiet_mode=True
    )
    
    print("Testing unified storage performance...")
    
    # Store patterns from different streams
    patterns_per_stream = 50
    streams = ['sensory', 'motor', 'temporal']
    
    store_start = time.time()
    stored_patterns = []
    
    for stream in streams:
        for i in range(patterns_per_stream):
            # Generate stream-specific pattern
            if stream == 'sensory':
                pattern = torch.randn(100) * 0.5 + torch.tensor([1.0] * 100)
            elif stream == 'motor':
                pattern = torch.randn(100) * 0.3 + torch.tensor([0.5] * 100)
            else:  # temporal
                pattern = torch.randn(100) * 0.2 + torch.tensor([0.2] * 100)
            
            pattern_id = unified_storage.store_pattern(pattern, stream)
            stored_patterns.append((pattern, stream, pattern_id))
    
    store_time = time.time() - store_start
    
    # Test retrieval performance
    print(f"Storage time: {store_time:.3f}s for {len(stored_patterns)} patterns")
    
    # Test unified retrieval
    retrieval_start = time.time()
    query_pattern = stored_patterns[0][0]  # Use first pattern as query
    
    # Single search serves all streams
    similar_patterns = unified_storage.find_similar_patterns(
        query_pattern, 
        stream_type='unified',
        k=10,
        cross_stream=True
    )
    
    retrieval_time = time.time() - retrieval_start
    
    print(f"Unified retrieval time: {retrieval_time*1000:.2f}ms")
    print(f"Similar patterns found: {len(similar_patterns)}")
    
    # Test cross-stream pattern discovery
    cross_stream_patterns = [
        (pattern, similarity) for pattern, similarity in similar_patterns
        if pattern.pattern_id not in unified_storage.get_stream_patterns('sensory')
    ]
    
    print(f"Cross-stream patterns discovered: {len(cross_stream_patterns)}")
    
    # Show statistics
    stats = unified_storage.get_unified_stats()
    print(f"\nUnified Storage Statistics:")
    print(f"   Total patterns: {stats['total_patterns']}")
    print(f"   Total columns: {stats['total_columns']}")
    print(f"   Stream distribution: {stats['stream_distribution']}")
    print(f"   Cross-stream relationships: {stats['cross_stream_relationships']}")
    print(f"   Cross-stream hit rate: {stats['cross_stream_hit_rate']:.1%}")
    
    print(f"\n✅ Unified cortical storage eliminates duplicate searches")
    print(f"   Single search serves all brain systems")
    print(f"   Cross-stream pattern discovery automatic")
    print(f"   🧠 Biology-inspired unified architecture")


if __name__ == "__main__":
    demonstrate_unified_cortical_performance()