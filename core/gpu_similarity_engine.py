#!/usr/bin/env python3
"""
GPU-accelerated similarity engine for massively parallel similarity calculations.
Replaces the serial CPU loop with vectorized GPU operations.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
from collections import defaultdict

from .accelerated_similarity import MPS_FUNCTIONAL, TORCH_AVAILABLE


class GPUSimilarityEngine:
    """
    GPU-accelerated similarity engine using PyTorch MPS.
    
    Converts serial similarity calculations into parallel GPU operations
    for massive speedup on large node graphs.
    """
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and MPS_FUNCTIONAL
        self.device = 'mps' if self.use_gpu else 'cpu'
        
        # Performance tracking
        self.total_queries = 0
        self.total_gpu_time = 0.0
        self.total_cpu_time = 0.0
        self.batch_sizes = []
        
        # Caching for repeated queries
        self.context_cache = {}
        self.cache_hits = 0
        
        if self.use_gpu:
            print(f"🚀 GPU Similarity Engine initialized (device: {self.device})")
        else:
            print("⚠️  GPU Similarity Engine falling back to CPU")
    
    def find_most_similar_batch(self, target_context: List[float], 
                               candidate_contexts: List[List[float]],
                               candidate_strengths: List[float] = None) -> Tuple[int, float]:
        """
        Find the most similar context from a batch of candidates using GPU vectorization.
        
        Args:
            target_context: The context to match against
            candidate_contexts: List of contexts to compare with
            candidate_strengths: Optional strengths for weighting similarities
            
        Returns:
            Tuple of (best_index, best_similarity)
        """
        if not candidate_contexts:
            return -1, 0.0
        
        self.total_queries += 1
        batch_size = len(candidate_contexts)
        self.batch_sizes.append(batch_size)
        
        # Create cache key for this query
        cache_key = self._create_cache_key(target_context, candidate_contexts)
        if cache_key in self.context_cache:
            self.cache_hits += 1
            return self.context_cache[cache_key]
        
        if self.use_gpu:
            result = self._gpu_similarity_search(target_context, candidate_contexts, candidate_strengths)
        else:
            result = self._cpu_similarity_search(target_context, candidate_contexts, candidate_strengths)
        
        # Cache the result
        self.context_cache[cache_key] = result
        
        # Limit cache size
        if len(self.context_cache) > 1000:
            # Remove oldest half of cache
            keys_to_remove = list(self.context_cache.keys())[:500]
            for key in keys_to_remove:
                del self.context_cache[key]
        
        return result
    
    def _gpu_similarity_search(self, target_context: List[float], 
                              candidate_contexts: List[List[float]],
                              candidate_strengths: List[float] = None) -> Tuple[int, float]:
        """GPU-accelerated similarity search."""
        start_time = time.perf_counter()
        
        try:
            # Convert to tensors
            target_tensor = torch.tensor(target_context, dtype=torch.float32, device=self.device)
            
            # Stack all candidate contexts into a single tensor
            candidates_tensor = torch.stack([
                torch.tensor(context, dtype=torch.float32, device=self.device) 
                for context in candidate_contexts
            ])
            
            # Calculate similarities using vectorized operations
            similarities = self._calculate_euclidean_similarity_batch(target_tensor, candidates_tensor)
            
            # Apply strength weighting if provided
            if candidate_strengths:
                strengths_tensor = torch.tensor(candidate_strengths, dtype=torch.float32, device=self.device)
                # Weight similarity by node strength (stronger memories are more relevant)
                similarities = similarities * (1.0 + strengths_tensor * 0.1)
            
            # Find best match
            best_idx = torch.argmax(similarities).item()
            best_similarity = similarities[best_idx].item()
            
            end_time = time.perf_counter()
            self.total_gpu_time += (end_time - start_time)
            
            return best_idx, best_similarity
            
        except Exception as e:
            print(f"⚠️  GPU similarity search failed: {e}, falling back to CPU")
            return self._cpu_similarity_search(target_context, candidate_contexts, candidate_strengths)
    
    def _cpu_similarity_search(self, target_context: List[float], 
                              candidate_contexts: List[List[float]],
                              candidate_strengths: List[float] = None) -> Tuple[int, float]:
        """CPU fallback similarity search."""
        start_time = time.perf_counter()
        
        best_idx = -1
        best_similarity = -1.0
        
        for i, candidate_context in enumerate(candidate_contexts):
            similarity = self._calculate_euclidean_similarity(target_context, candidate_context)
            
            # Apply strength weighting if provided
            if candidate_strengths:
                similarity = similarity * (1.0 + candidate_strengths[i] * 0.1)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
        
        end_time = time.perf_counter()
        self.total_cpu_time += (end_time - start_time)
        
        return best_idx, best_similarity
    
    def _calculate_euclidean_similarity_batch(self, target: torch.Tensor, 
                                            candidates: torch.Tensor) -> torch.Tensor:
        """Calculate Euclidean similarity for a batch of candidates (GPU optimized)."""
        # Calculate squared differences
        diff_squared = (target.unsqueeze(0) - candidates) ** 2
        
        # Sum along the feature dimension to get squared distances
        distances_squared = torch.sum(diff_squared, dim=1)
        
        # Take square root to get Euclidean distances
        distances = torch.sqrt(distances_squared)
        
        # Convert to similarity (0 to 1)
        # Assuming values roughly -2 to +2, max distance is sqrt(len * 16)
        max_possible_distance = torch.sqrt(torch.tensor(len(target) * 16.0, device=self.device))
        
        # Convert distance to similarity
        similarities = torch.clamp(1.0 - (distances / max_possible_distance), 0.0, 1.0)
        
        return similarities
    
    def _calculate_euclidean_similarity(self, context1: List[float], 
                                      context2: List[float]) -> float:
        """CPU version of Euclidean similarity calculation."""
        if len(context1) != len(context2):
            return 0.0
        
        if not context1 or not context2:
            return 0.0
        
        # Euclidean distance
        distance = sum((a - b) ** 2 for a, b in zip(context1, context2)) ** 0.5
        max_possible_distance = (len(context1) * 16.0) ** 0.5  # Assuming values roughly -2 to +2
        
        # Convert distance to similarity (0 to 1)
        if max_possible_distance == 0:
            return 1.0 if distance == 0 else 0.0
        
        similarity = max(0.0, 1.0 - (distance / max_possible_distance))
        return similarity
    
    def _create_cache_key(self, target: List[float], 
                         candidates: List[List[float]]) -> str:
        """Create a cache key for the similarity query."""
        # Create a hash-based key (simplified)
        target_hash = hash(tuple(target))
        candidates_hash = hash(tuple(tuple(c) for c in candidates[:5]))  # Use first 5 for performance
        return f"{target_hash}_{candidates_hash}_{len(candidates)}"
    
    def batch_similarities(self, target_context: List[float], 
                          candidate_contexts: List[List[float]]) -> List[float]:
        """
        Calculate similarities for all candidates and return the full list.
        Useful for debugging and analysis.
        """
        if not candidate_contexts:
            return []
        
        if self.use_gpu:
            return self._gpu_batch_similarities(target_context, candidate_contexts)
        else:
            return self._cpu_batch_similarities(target_context, candidate_contexts)
    
    def _gpu_batch_similarities(self, target_context: List[float], 
                               candidate_contexts: List[List[float]]) -> List[float]:
        """GPU version of batch similarities."""
        try:
            target_tensor = torch.tensor(target_context, dtype=torch.float32, device=self.device)
            candidates_tensor = torch.stack([
                torch.tensor(context, dtype=torch.float32, device=self.device) 
                for context in candidate_contexts
            ])
            
            similarities = self._calculate_euclidean_similarity_batch(target_tensor, candidates_tensor)
            return similarities.cpu().tolist()
            
        except Exception as e:
            print(f"⚠️  GPU batch similarities failed: {e}, falling back to CPU")
            return self._cpu_batch_similarities(target_context, candidate_contexts)
    
    def _cpu_batch_similarities(self, target_context: List[float], 
                               candidate_contexts: List[List[float]]) -> List[float]:
        """CPU version of batch similarities."""
        return [
            self._calculate_euclidean_similarity(target_context, candidate_context)
            for candidate_context in candidate_contexts
        ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_time = self.total_gpu_time + self.total_cpu_time
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        
        return {
            'total_queries': self.total_queries,
            'total_time': total_time,
            'avg_query_time': total_time / self.total_queries if self.total_queries > 0 else 0,
            'gpu_time': self.total_gpu_time,
            'cpu_time': self.total_cpu_time,
            'gpu_percentage': (self.total_gpu_time / total_time * 100) if total_time > 0 else 0,
            'avg_batch_size': avg_batch_size,
            'cache_hits': self.cache_hits,
            'cache_hit_rate': (self.cache_hits / self.total_queries * 100) if self.total_queries > 0 else 0,
            'device': self.device,
            'gpu_available': self.use_gpu
        }
    
    def clear_cache(self):
        """Clear the similarity cache."""
        self.context_cache.clear()
        self.cache_hits = 0
    
    def warmup(self, context_size: int = 8, batch_size: int = 100):
        """Warm up the GPU with a test calculation."""
        if not self.use_gpu:
            return
        
        print(f"🔥 Warming up GPU with {batch_size} contexts...")
        
        # Create dummy data
        target = [0.1 * i for i in range(context_size)]
        candidates = [
            [0.1 * i + 0.01 * j for i in range(context_size)]
            for j in range(batch_size)
        ]
        
        # Run warmup
        start_time = time.perf_counter()
        self.find_most_similar_batch(target, candidates)
        end_time = time.perf_counter()
        
        print(f"🔥 GPU warmup completed in {(end_time - start_time)*1000:.2f}ms")


# Global instance
_gpu_similarity_engine = None

def get_gpu_similarity_engine() -> GPUSimilarityEngine:
    """Get the global GPU similarity engine instance."""
    global _gpu_similarity_engine
    if _gpu_similarity_engine is None:
        _gpu_similarity_engine = GPUSimilarityEngine()
        _gpu_similarity_engine.warmup()
    return _gpu_similarity_engine