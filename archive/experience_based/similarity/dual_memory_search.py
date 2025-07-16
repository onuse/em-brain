"""
Dual Memory Search

Extends similarity search to work across both working memory (recent, unconsolidated)
and long-term memory (consolidated experiences). This enables immediate use of recent
experiences in reasoning without waiting for consolidation.

Biological inspiration:
- Working memory has higher activation/salience
- Both memory systems contribute to pattern matching
- Recent experiences can override older patterns temporarily
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import time

from ..experience.working_memory import WorkingMemoryBuffer, WorkingMemoryItem
from ..experience.storage import ExperienceStorage, Experience
from .engine import SimilarityEngine


class DualMemorySearch:
    """
    Similarity search across both working memory and long-term memory.
    
    Features:
    - Unified search interface for both memory systems
    - Working memory gets activation boost (recency bias)
    - Can blend or prioritize results based on context
    - Maintains separate statistics for each memory type
    """
    
    def __init__(self,
                 similarity_engine: SimilarityEngine,
                 working_memory: WorkingMemoryBuffer,
                 experience_storage: ExperienceStorage,
                 working_memory_weight: float = 1.5):
        """
        Initialize dual memory search.
        
        Args:
            similarity_engine: The similarity computation engine
            working_memory: Working memory buffer
            experience_storage: Long-term memory storage
            working_memory_weight: Boost factor for working memory items
        """
        self.similarity_engine = similarity_engine
        self.working_memory = working_memory
        self.experience_storage = experience_storage
        self.working_memory_weight = working_memory_weight
        
        # Statistics
        self.total_searches = 0
        self.working_memory_hits = 0
        self.long_term_hits = 0
        self.blended_results = 0
        
        print(f"🔍 DualMemorySearch initialized")
        print(f"   Searches both working memory and long-term storage")
        print(f"   Working memory weight: {working_memory_weight}x")
    
    def search(self,
               query_vector: np.ndarray,
               k: int = 10,
               similarity_threshold: float = 0.7,
               prefer_working_memory: bool = False) -> List[Tuple[Any, float]]:
        """
        Search for similar experiences across both memory systems.
        
        Args:
            query_vector: The vector to search for
            k: Number of results to return
            similarity_threshold: Minimum similarity score
            prefer_working_memory: Whether to prioritize working memory results
            
        Returns:
            List of (experience, similarity_score) tuples, sorted by similarity
        """
        self.total_searches += 1
        search_start = time.time()
        
        # Search working memory
        working_memory_results = self._search_working_memory(
            query_vector, 
            k * 2,  # Get more candidates
            similarity_threshold
        )
        
        # Search long-term memory
        long_term_results = self._search_long_term_memory(
            query_vector,
            k * 2,  # Get more candidates
            similarity_threshold
        )
        
        # Combine and rank results
        combined_results = self._combine_results(
            working_memory_results,
            long_term_results,
            k,
            prefer_working_memory
        )
        
        # Track statistics
        for result, _ in combined_results:
            if isinstance(result, WorkingMemoryItem):
                self.working_memory_hits += 1
            else:
                self.long_term_hits += 1
        
        if working_memory_results and long_term_results:
            self.blended_results += 1
        
        return combined_results
    
    def _search_working_memory(self,
                              query_vector: np.ndarray,
                              k: int,
                              threshold: float) -> List[Tuple[WorkingMemoryItem, float]]:
        """Search working memory for similar experiences."""
        results = []
        
        # Get active working memory items with their weights
        wm_items = self.working_memory.get_experiences_for_matching()
        
        for item, activation_weight in wm_items:
            # Compute similarity using the main engine
            context_vector = item.get_context_vector()
            
            # Ensure vectors have compatible dimensions
            if len(context_vector) == len(query_vector):
                similarity = self.similarity_engine.compute_similarity(
                    query_vector, 
                    context_vector
                )
                
                # Apply working memory boost and activation weight
                boosted_similarity = similarity * self.working_memory_weight * activation_weight
                
                if boosted_similarity >= threshold:
                    results.append((item, boosted_similarity))
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _search_long_term_memory(self,
                                query_vector: np.ndarray,
                                k: int,
                                threshold: float) -> List[Tuple[Experience, float]]:
        """Search long-term memory for similar experiences."""
        results = []
        
        # Get all experiences from storage
        experiences = list(self.experience_storage.get_all_experiences())
        
        if not experiences:
            return results
        
        # Simple similarity computation for each experience
        for exp in experiences[:k*2]:  # Limit for performance
            try:
                # Get context vector for comparison
                context_vector = exp.get_context_vector()
                
                # Compute similarity
                if len(context_vector) == len(query_vector):
                    similarity = self.similarity_engine.compute_similarity(
                        query_vector.tolist(),
                        context_vector.tolist()
                    )
                    
                    if similarity >= threshold:
                        results.append((exp, similarity))
            except Exception:
                continue  # Skip invalid experiences
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _combine_results(self,
                        working_memory_results: List[Tuple[Any, float]],
                        long_term_results: List[Tuple[Any, float]],
                        k: int,
                        prefer_working_memory: bool) -> List[Tuple[Any, float]]:
        """
        Combine and rank results from both memory systems.
        
        Handles deduplication and ranking strategies.
        """
        # If preferring working memory, boost their scores further
        if prefer_working_memory:
            working_memory_results = [
                (item, score * 1.2) for item, score in working_memory_results
            ]
        
        # Combine all results
        all_results = working_memory_results + long_term_results
        
        # Sort by score
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k (could implement deduplication here if needed)
        return all_results[:k]
    
    def get_memory_distribution(self, results: List[Tuple[Any, float]]) -> Dict[str, int]:
        """
        Analyze which memory system contributed results.
        
        Useful for understanding memory usage patterns.
        """
        distribution = {
            'working_memory': 0,
            'long_term_memory': 0
        }
        
        for result, _ in results:
            if isinstance(result, WorkingMemoryItem):
                distribution['working_memory'] += 1
            else:
                distribution['long_term_memory'] += 1
        
        return distribution
    
    def adapt_search_strategy(self, cognitive_mode: str):
        """
        Adapt search strategy based on cognitive state.
        
        Args:
            cognitive_mode: 'autopilot', 'focused', or 'deep_think'
        """
        if cognitive_mode == 'autopilot':
            # Rely more on long-term patterns
            self.working_memory_weight = 1.2
        elif cognitive_mode == 'focused':
            # Balanced approach
            self.working_memory_weight = 1.5
        elif cognitive_mode == 'deep_think':
            # Emphasize recent context
            self.working_memory_weight = 2.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dual memory search statistics."""
        total_hits = self.working_memory_hits + self.long_term_hits
        
        if total_hits > 0:
            wm_percentage = (self.working_memory_hits / total_hits) * 100
            lt_percentage = (self.long_term_hits / total_hits) * 100
        else:
            wm_percentage = 0.0
            lt_percentage = 0.0
        
        return {
            'total_searches': self.total_searches,
            'working_memory_hits': self.working_memory_hits,
            'long_term_hits': self.long_term_hits,
            'blended_results': self.blended_results,
            'working_memory_percentage': wm_percentage,
            'long_term_percentage': lt_percentage,
            'working_memory_weight': self.working_memory_weight
        }
    
    def print_search_report(self):
        """Print dual memory search statistics."""
        stats = self.get_statistics()
        
        print(f"\n🔍 DUAL MEMORY SEARCH REPORT")
        print(f"=" * 40)
        print(f"📊 Total searches: {stats['total_searches']:,}")
        print(f"🧠 Working memory hits: {stats['working_memory_hits']:,} ({stats['working_memory_percentage']:.1f}%)")
        print(f"💾 Long-term hits: {stats['long_term_hits']:,} ({stats['long_term_percentage']:.1f}%)")
        print(f"🔀 Blended results: {stats['blended_results']:,}")
        print(f"⚖️  Working memory weight: {stats['working_memory_weight']}x")
    
    def __str__(self) -> str:
        stats = self.get_statistics()
        return f"DualMemorySearch(WM: {stats['working_memory_percentage']:.0f}%, LT: {stats['long_term_percentage']:.0f}%)"