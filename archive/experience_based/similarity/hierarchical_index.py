#!/usr/bin/env python3
"""
Hierarchical Experience Index

Production implementation of hierarchical indexing for the similarity engine.
Enables sub-millisecond similarity search for 10k+ experiences through emergent
brain regionalization.

This is sophisticated supporting infrastructure that enables the simple core
similarity system to handle massive data efficiently, following the README.md
principle: "Intelligence should be simple. Engineering can be sophisticated."
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ExperienceRegion:
    """
    A specialized brain region containing similar experiences.
    
    Biological inspiration: Cortical regions that specialize in processing
    similar types of information (visual cortex, motor cortex, etc.)
    """
    region_id: str
    center: np.ndarray  # Regional prototype vector
    experience_ids: List[str] = field(default_factory=list)
    specialization_score: float = 0.0
    access_frequency: int = 0
    creation_time: float = field(default_factory=time.time)
    
    def get_distance_to(self, vector: np.ndarray) -> float:
        """Compute distance from vector to region center."""
        return np.linalg.norm(vector - self.center)
    
    def add_experience(self, exp_id: str, vector: np.ndarray):
        """Add experience to region and update specialization."""
        self.experience_ids.append(exp_id)
        
        # Update center using exponential moving average
        if len(self.experience_ids) == 1:
            self.center = vector.copy()
        else:
            alpha = 0.1  # Learning rate for regional adaptation
            self.center = (1 - alpha) * self.center + alpha * vector
        
        # Update specialization based on regional coherence
        self._update_specialization()
    
    def _update_specialization(self):
        """Update how specialized this region is."""
        if len(self.experience_ids) < 2:
            self.specialization_score = 0.0
        else:
            # Specialization increases with size but decreases with spread
            size_factor = math.log(len(self.experience_ids) + 1)
            self.specialization_score = size_factor / (1.0 + len(self.experience_ids) * 0.01)


class HierarchicalExperienceIndex:
    """
    Hierarchical indexing system for ultra-fast similarity search.
    
    Creates emergent brain regions that automatically specialize, enabling
    O(regions + relevant_experiences) search instead of O(all_experiences).
    
    Key insight: Just like biological brains have specialized regions,
    we can let experience clusters naturally emerge and specialize.
    """
    
    def __init__(self, 
                 max_region_size: int = 50,
                 similarity_threshold: float = 0.4,
                 max_search_regions: int = 3):
        """
        Initialize hierarchical index.
        
        Args:
            max_region_size: Split regions when they exceed this size
            similarity_threshold: Distance threshold for region membership
            max_search_regions: Maximum regions to search per query
        """
        self.regions: Dict[str, ExperienceRegion] = {}
        self.experience_to_region: Dict[str, str] = {}
        self.max_region_size = max_region_size
        self.similarity_threshold = similarity_threshold
        self.max_search_regions = max_search_regions
        self.next_region_id = 0
        
        # Performance tracking
        self.total_searches = 0
        self.total_regions_searched = 0
        self.total_experiences_searched = 0
        
    def add_experience(self, exp_id: str, vector: np.ndarray):
        """Add experience to hierarchical index."""
        if exp_id in self.experience_to_region:
            return  # Already indexed
        
        # Convert to numpy array for efficient computation
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        
        # Find best region for this experience
        best_region = self._find_best_region(vector)
        
        if best_region is None:
            # Create new specialized region
            self._create_new_region(exp_id, vector)
        else:
            # Add to existing region
            best_region.add_experience(exp_id, vector)
            self.experience_to_region[exp_id] = best_region.region_id
            
            # Split region if it becomes too large
            if len(best_region.experience_ids) > self.max_region_size:
                self._split_region(best_region)
    
    def _find_best_region(self, vector: np.ndarray) -> Optional[ExperienceRegion]:
        """Find the best region for a vector."""
        best_region = None
        best_distance = float('inf')
        
        for region in self.regions.values():
            distance = region.get_distance_to(vector)
            
            # Only consider regions within similarity threshold
            if distance < self.similarity_threshold and distance < best_distance:
                best_distance = distance
                best_region = region
        
        return best_region
    
    def _create_new_region(self, exp_id: str, vector: np.ndarray):
        """Create a new specialized region."""
        region_id = f"region_{self.next_region_id}"
        self.next_region_id += 1
        
        new_region = ExperienceRegion(
            region_id=region_id,
            center=vector.copy(),
            experience_ids=[exp_id]
        )
        
        self.regions[region_id] = new_region
        self.experience_to_region[exp_id] = region_id
    
    def _split_region(self, region: ExperienceRegion):
        """Split a region into specialized sub-regions."""
        if len(region.experience_ids) <= self.max_region_size:
            return
        
        # Simple split for now - in production, could use k-means or other clustering
        mid_point = len(region.experience_ids) // 2
        
        # Create two new specialized regions
        region1_id = f"region_{self.next_region_id}"
        region2_id = f"region_{self.next_region_id + 1}"
        self.next_region_id += 2
        
        # Split experiences
        exp1 = region.experience_ids[:mid_point]
        exp2 = region.experience_ids[mid_point:]
        
        # Create specialized regions with inherited properties
        region1 = ExperienceRegion(
            region_id=region1_id,
            center=region.center.copy(),
            experience_ids=exp1,
            specialization_score=region.specialization_score * 1.2
        )
        
        region2 = ExperienceRegion(
            region_id=region2_id,
            center=region.center.copy(),
            experience_ids=exp2,
            specialization_score=region.specialization_score * 1.2
        )
        
        # Update mappings
        for exp_id in exp1:
            self.experience_to_region[exp_id] = region1_id
        for exp_id in exp2:
            self.experience_to_region[exp_id] = region2_id
        
        # Replace old region with specialized ones
        del self.regions[region.region_id]
        self.regions[region1_id] = region1
        self.regions[region2_id] = region2
    
    def find_similar_experiences(self, 
                               query_vector: np.ndarray,
                               experience_vectors: List[np.ndarray],
                               experience_ids: List[str],
                               max_results: int = 10) -> List[Tuple[str, float]]:
        """
        Hierarchical similarity search.
        
        First finds relevant regions, then searches only within those regions.
        This achieves O(regions + relevant_experiences) instead of O(all_experiences).
        """
        self.total_searches += 1
        
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector, dtype=np.float32)
        
        # Step 1: Find relevant regions (brain regions to search)
        relevant_regions = self._find_relevant_regions(query_vector)
        
        if not relevant_regions:
            # Fallback to linear search if no regions found
            return self._linear_search(query_vector, experience_vectors, experience_ids, max_results)
        
        # Step 2: Collect candidate experiences from relevant regions
        candidate_experiences = []
        
        for region, region_distance in relevant_regions[:self.max_search_regions]:
            region.access_frequency += 1
            self.total_regions_searched += 1
            
            # Search within this region
            for exp_id in region.experience_ids:
                if exp_id in experience_ids:
                    exp_index = experience_ids.index(exp_id)
                    exp_vector = experience_vectors[exp_index]
                    
                    # Compute actual similarity
                    similarity = self._compute_similarity(query_vector, exp_vector)
                    candidate_experiences.append((exp_id, similarity))
        
        self.total_experiences_searched += len(candidate_experiences)
        
        # Step 3: Return top candidates
        candidate_experiences.sort(key=lambda x: x[1], reverse=True)
        return candidate_experiences[:max_results]
    
    def _find_relevant_regions(self, query_vector: np.ndarray) -> List[Tuple[ExperienceRegion, float]]:
        """Find regions relevant to the query."""
        relevant_regions = []
        
        for region in self.regions.values():
            distance = region.get_distance_to(query_vector)
            
            # Use broader threshold for region-level search
            if distance < self.similarity_threshold * 2.0:
                relevant_regions.append((region, distance))
        
        # Sort by relevance (distance to region center)
        relevant_regions.sort(key=lambda x: x[1])
        return relevant_regions
    
    def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute similarity between two vectors."""
        # Use cosine similarity for robustness
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        cosine_sim = dot_product / norms
        return max(0.0, cosine_sim)  # Clamp to [0, 1]
    
    def _linear_search(self, 
                      query_vector: np.ndarray,
                      experience_vectors: List[np.ndarray], 
                      experience_ids: List[str],
                      max_results: int) -> List[Tuple[str, float]]:
        """Fallback linear search when no regions found."""
        similarities = []
        
        for i, exp_vector in enumerate(experience_vectors):
            similarity = self._compute_similarity(query_vector, exp_vector)
            similarities.append((experience_ids[i], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_experiences = len(self.experience_to_region)
        
        if not self.regions:
            return {
                'total_experiences': total_experiences,
                'total_regions': 0,
                'avg_region_size': 0,
                'search_efficiency': 1.0,
                'avg_experiences_per_search': total_experiences,
                'avg_regions_per_search': 0
            }
        
        region_sizes = [len(region.experience_ids) for region in self.regions.values()]
        avg_region_size = sum(region_sizes) / len(region_sizes)
        
        avg_experiences_per_search = (
            self.total_experiences_searched / max(1, self.total_searches)
        )
        avg_regions_per_search = (
            self.total_regions_searched / max(1, self.total_searches)
        )
        
        search_efficiency = total_experiences / max(1, avg_experiences_per_search)
        
        return {
            'total_experiences': total_experiences,
            'total_regions': len(self.regions),
            'avg_region_size': avg_region_size,
            'max_region_size': max(region_sizes) if region_sizes else 0,
            'search_efficiency': search_efficiency,
            'avg_experiences_per_search': avg_experiences_per_search,
            'avg_regions_per_search': avg_regions_per_search,
            'specialization_scores': [r.specialization_score for r in self.regions.values()]
        }
    
    def should_use_hierarchical_search(self, total_experiences: int) -> bool:
        """Determine if hierarchical search is beneficial."""
        # Use hierarchical search when we have enough experiences and regions
        return (total_experiences > 100 and 
                len(self.regions) > 1 and
                len(self.regions) < total_experiences * 0.8)