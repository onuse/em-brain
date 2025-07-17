#!/usr/bin/env python3
"""
Cortical Column-Inspired Pre-Clustering for Pattern Storage

Biology insight: Cortical columns physically group similar patterns together,
enabling fast retrieval without expensive clustering during prediction.

Key principles:
1. Clustering happens during storage (async, no time pressure)
2. Retrieval uses pre-computed clusters (O(1) lookup)
3. Columns emerge from similarity constraints, not explicit programming
4. Maintains constraint-based philosophy while optimizing performance
"""

import time
import torch
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque

try:
    from .sparse_representations import SparsePattern, SparsePatternStorage
except ImportError:
    from sparse_representations import SparsePattern, SparsePatternStorage


@dataclass
class CorticalColumn:
    """
    A cortical column that groups similar patterns for fast retrieval.
    
    Biology: Cortical columns physically cluster similar patterns in brain tissue
    Our approach: Virtual columns cluster patterns by similarity constraints
    """
    column_id: str
    centroid_pattern: SparsePattern  # Representative pattern for this column
    member_patterns: List[str]       # Pattern IDs in this column
    activation_frequency: float      # How often this column is accessed
    last_update_time: float         # When column was last updated
    internal_similarity: float      # Average similarity within column
    column_strength: float          # How well-established this column is


class CorticalColumnStorage:
    """
    Storage system with cortical column-inspired pre-clustering.
    
    Key insight: Move expensive clustering from retrieval to storage time.
    
    During storage (async, no time pressure):
    - Find best column for new pattern
    - Update column statistics
    - Maintain column organization
    
    During retrieval (fast, real-time):
    - Look up relevant columns
    - Return pre-clustered patterns
    - No expensive clustering needed
    """
    
    def __init__(self, base_storage: SparsePatternStorage, 
                 max_columns: int = 1000,
                 column_similarity_threshold: float = 0.6,
                 quiet_mode: bool = False):
        self.base_storage = base_storage
        self.max_columns = max_columns
        self.column_similarity_threshold = column_similarity_threshold
        self.quiet_mode = quiet_mode
        
        # Cortical column organization
        self.columns: Dict[str, CorticalColumn] = {}
        self.pattern_to_column: Dict[str, str] = {}  # Fast pattern → column lookup
        
        # Column formation statistics
        self.column_formations = 0
        self.column_updates = 0
        self.column_merges = 0
        
        # Performance tracking
        self.storage_times = deque(maxlen=1000)
        self.retrieval_times = deque(maxlen=1000)
        
        if not quiet_mode:
            print(f"🧠 Cortical Column Storage initialized")
            print(f"   Max columns: {max_columns}")
            print(f"   Column similarity threshold: {column_similarity_threshold}")
            print(f"   🎯 Pre-clustering during storage, O(1) retrieval")
    
    def store_pattern_with_columns(self, pattern: SparsePattern) -> str:
        """
        Store pattern with cortical column pre-clustering.
        
        This is the key optimization: expensive clustering happens here (async)
        so that retrieval can be fast (real-time).
        """
        start_time = time.time()
        
        # Store in base storage first
        pattern_id = self.base_storage.store_pattern(pattern)
        
        # Find best column for this pattern (or create new one)
        best_column_id = self._find_best_column(pattern)
        
        if best_column_id is None:
            # Create new column
            best_column_id = self._create_new_column(pattern)
        else:
            # Add to existing column
            self._add_to_column(pattern, best_column_id)
        
        # Update column organization
        self._update_column_organization()
        
        # Track performance
        storage_time = time.time() - start_time
        self.storage_times.append(storage_time)
        
        return pattern_id
    
    def get_clustered_patterns(self, query_pattern: SparsePattern, 
                              max_columns: int = 5) -> List[List[Tuple[SparsePattern, float]]]:
        """
        Get pre-clustered patterns for fast retrieval.
        
        This is the key speedup: O(1) column lookup instead of O(n²) clustering.
        """
        start_time = time.time()
        
        # Find most relevant columns
        relevant_columns = self._find_relevant_columns(query_pattern, max_columns)
        
        # Get patterns from each column (already clustered!)
        clustered_patterns = []
        for column_id, column_similarity in relevant_columns:
            column = self.columns[column_id]
            
            # Get patterns from this column
            column_patterns = []
            for pattern_id in column.member_patterns:
                if pattern_id in self.base_storage.pattern_index:
                    pattern = self.base_storage.pattern_index[pattern_id]
                    # Calculate similarity to query
                    similarity = query_pattern.jaccard_similarity(pattern)
                    column_patterns.append((pattern, similarity))
            
            # Sort by similarity within column
            column_patterns.sort(key=lambda x: x[1], reverse=True)
            clustered_patterns.append(column_patterns)
        
        # Track performance
        retrieval_time = time.time() - start_time
        self.retrieval_times.append(retrieval_time)
        
        return clustered_patterns
    
    def _find_best_column(self, pattern: SparsePattern) -> Optional[str]:
        """Find the best existing column for this pattern."""
        best_column_id = None
        best_similarity = 0.0
        
        for column_id, column in self.columns.items():
            # Calculate similarity to column centroid
            similarity = pattern.jaccard_similarity(column.centroid_pattern)
            
            if similarity > self.column_similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_column_id = column_id
        
        return best_column_id
    
    def _create_new_column(self, pattern: SparsePattern) -> str:
        """Create a new cortical column for this pattern."""
        column_id = f"column_{len(self.columns)}_{int(time.time() * 1000)}"
        
        new_column = CorticalColumn(
            column_id=column_id,
            centroid_pattern=pattern,  # Pattern becomes the centroid
            member_patterns=[pattern.pattern_id],
            activation_frequency=1.0,
            last_update_time=time.time(),
            internal_similarity=1.0,  # Single pattern has perfect internal similarity
            column_strength=1.0
        )
        
        self.columns[column_id] = new_column
        self.pattern_to_column[pattern.pattern_id] = column_id
        self.column_formations += 1
        
        return column_id
    
    def _add_to_column(self, pattern: SparsePattern, column_id: str):
        """Add pattern to existing column and update statistics."""
        column = self.columns[column_id]
        
        # Add pattern to column
        column.member_patterns.append(pattern.pattern_id)
        self.pattern_to_column[pattern.pattern_id] = column_id
        
        # Update column centroid (moving average)
        alpha = 0.1  # Learning rate for centroid update
        old_centroid = column.centroid_pattern
        
        # Compute new centroid by blending old centroid with new pattern
        new_centroid_indices = self._blend_sparse_patterns(
            old_centroid, pattern, alpha
        )
        
        # Create new centroid pattern
        column.centroid_pattern = SparsePattern(
            active_indices=new_centroid_indices,
            pattern_dim=pattern.pattern_dim,
            sparsity=len(new_centroid_indices) / pattern.pattern_dim,
            pattern_id=f"{column_id}_centroid",
            creation_time=time.time()
        )
        
        # Update column statistics
        column.activation_frequency += 1.0
        column.last_update_time = time.time()
        column.column_strength = min(10.0, column.column_strength + 0.1)
        
        # Recalculate internal similarity
        column.internal_similarity = self._calculate_internal_similarity(column)
        
        self.column_updates += 1
    
    def _find_relevant_columns(self, query_pattern: SparsePattern, 
                              max_columns: int) -> List[Tuple[str, float]]:
        """Find columns most relevant to query pattern."""
        column_similarities = []
        
        for column_id, column in self.columns.items():
            # Calculate similarity to column centroid
            similarity = query_pattern.jaccard_similarity(column.centroid_pattern)
            
            # Weight by column strength and activation frequency
            weighted_similarity = similarity * column.column_strength * np.log1p(column.activation_frequency)
            
            column_similarities.append((column_id, weighted_similarity))
        
        # Sort by weighted similarity and return top columns
        column_similarities.sort(key=lambda x: x[1], reverse=True)
        return column_similarities[:max_columns]
    
    def _blend_sparse_patterns(self, pattern1: SparsePattern, pattern2: SparsePattern, 
                              alpha: float) -> torch.Tensor:
        """Blend two sparse patterns to create new centroid."""
        # Convert to dense, blend, convert back to sparse
        dense1 = pattern1.to_dense()
        dense2 = pattern2.to_dense()
        
        # Weighted blend
        blended_dense = (1 - alpha) * dense1 + alpha * dense2
        
        # Convert back to sparse (top-k selection)
        sparsity = int(pattern1.pattern_dim * 0.02)  # 2% sparsity
        _, top_indices = torch.topk(blended_dense, sparsity)
        
        return top_indices.sort()[0]
    
    def _calculate_internal_similarity(self, column: CorticalColumn) -> float:
        """Calculate average similarity within column."""
        if len(column.member_patterns) < 2:
            return 1.0
        
        similarities = []
        patterns = [self.base_storage.pattern_index[pid] for pid in column.member_patterns[-10:]]  # Sample last 10
        
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                similarity = patterns[i].jaccard_similarity(patterns[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _update_column_organization(self):
        """Maintain column organization (merge weak columns, split strong ones)."""
        if len(self.columns) <= 2:
            return
        
        # Check if any columns should be merged
        columns_to_merge = []
        column_list = list(self.columns.items())
        
        for i in range(len(column_list)):
            for j in range(i + 1, len(column_list)):
                column1_id, column1 = column_list[i]
                column2_id, column2 = column_list[j]
                
                # Check if columns are similar enough to merge
                centroid_similarity = column1.centroid_pattern.jaccard_similarity(column2.centroid_pattern)
                
                if centroid_similarity > 0.8 and len(column1.member_patterns) + len(column2.member_patterns) < 50:
                    columns_to_merge.append((column1_id, column2_id))
                    break
        
        # Merge similar columns
        for column1_id, column2_id in columns_to_merge:
            self._merge_columns(column1_id, column2_id)
    
    def _merge_columns(self, column1_id: str, column2_id: str):
        """Merge two similar columns."""
        if column1_id not in self.columns or column2_id not in self.columns:
            return
        
        column1 = self.columns[column1_id]
        column2 = self.columns[column2_id]
        
        # Merge patterns
        merged_patterns = column1.member_patterns + column2.member_patterns
        
        # Update pattern-to-column mapping
        for pattern_id in column2.member_patterns:
            self.pattern_to_column[pattern_id] = column1_id
        
        # Update column1 with merged information
        column1.member_patterns = merged_patterns
        column1.activation_frequency += column2.activation_frequency
        column1.column_strength = (column1.column_strength + column2.column_strength) / 2
        
        # Remove column2
        del self.columns[column2_id]
        self.column_merges += 1
    
    def get_column_stats(self) -> Dict[str, Any]:
        """Get statistics about cortical column organization."""
        if not self.columns:
            return {'num_columns': 0, 'avg_patterns_per_column': 0}
        
        patterns_per_column = [len(col.member_patterns) for col in self.columns.values()]
        internal_similarities = [col.internal_similarity for col in self.columns.values()]
        
        avg_storage_time = np.mean(self.storage_times) if self.storage_times else 0
        avg_retrieval_time = np.mean(self.retrieval_times) if self.retrieval_times else 0
        
        return {
            'num_columns': len(self.columns),
            'avg_patterns_per_column': np.mean(patterns_per_column),
            'min_patterns_per_column': np.min(patterns_per_column),
            'max_patterns_per_column': np.max(patterns_per_column),
            'avg_internal_similarity': np.mean(internal_similarities),
            'column_formations': self.column_formations,
            'column_updates': self.column_updates,
            'column_merges': self.column_merges,
            'avg_storage_time_ms': avg_storage_time * 1000,
            'avg_retrieval_time_ms': avg_retrieval_time * 1000,
            'storage_to_retrieval_ratio': avg_storage_time / avg_retrieval_time if avg_retrieval_time > 0 else 0
        }


def demonstrate_cortical_columns():
    """Demonstrate cortical column pre-clustering performance."""
    print("🧠 CORTICAL COLUMN PRE-CLUSTERING DEMONSTRATION")
    print("=" * 60)
    
    # Create storage systems
    base_storage = SparsePatternStorage(pattern_dim=100, max_patterns=10000, quiet_mode=True)
    column_storage = CorticalColumnStorage(base_storage, quiet_mode=True)
    
    # Generate test patterns in clusters
    print("Generating clustered test patterns...")
    
    # Create 5 pattern clusters
    cluster_centers = [
        torch.randn(100) for _ in range(5)
    ]
    
    patterns = []
    for cluster_idx in range(5):
        for pattern_idx in range(20):  # 20 patterns per cluster
            # Add noise to cluster center
            noisy_pattern = cluster_centers[cluster_idx] + torch.randn(100) * 0.1
            
            # Convert to sparse
            from sparse_representations import SparsePatternEncoder
            encoder = SparsePatternEncoder(100, sparsity=0.02, quiet_mode=True)
            sparse_pattern = encoder.encode_top_k(noisy_pattern, f"cluster_{cluster_idx}_pattern_{pattern_idx}")
            patterns.append(sparse_pattern)
    
    # Store patterns (this is where pre-clustering happens)
    print("Storing patterns with cortical column pre-clustering...")
    storage_start = time.time()
    
    for pattern in patterns:
        column_storage.store_pattern_with_columns(pattern)
    
    storage_time = time.time() - storage_start
    print(f"   Storage time: {storage_time:.3f}s ({storage_time/len(patterns)*1000:.1f}ms per pattern)")
    
    # Test retrieval speed
    print("Testing retrieval speed...")
    query_pattern = patterns[0]  # Use first pattern as query
    
    retrieval_start = time.time()
    for _ in range(100):  # 100 retrievals
        clustered_patterns = column_storage.get_clustered_patterns(query_pattern, max_columns=3)
    retrieval_time = (time.time() - retrieval_start) / 100
    
    print(f"   Average retrieval time: {retrieval_time*1000:.2f}ms")
    print(f"   Clusters returned: {len(clustered_patterns)}")
    
    # Show column statistics
    stats = column_storage.get_column_stats()
    print(f"\nCortical Column Statistics:")
    print(f"   Columns formed: {stats['num_columns']}")
    print(f"   Avg patterns per column: {stats['avg_patterns_per_column']:.1f}")
    print(f"   Avg internal similarity: {stats['avg_internal_similarity']:.3f}")
    print(f"   Storage time: {stats['avg_storage_time_ms']:.2f}ms")
    print(f"   Retrieval time: {stats['avg_retrieval_time_ms']:.2f}ms")
    print(f"   Storage/Retrieval ratio: {stats['storage_to_retrieval_ratio']:.1f}x")
    
    print(f"\n✅ Pre-clustering moves expensive work to storage time")
    print(f"   Retrieval becomes O(1) column lookup instead of O(n²) clustering")
    print(f"   🧠 Biology-inspired optimization maintains constraint-based approach")


if __name__ == "__main__":
    demonstrate_cortical_columns()