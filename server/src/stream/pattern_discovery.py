"""
Pattern Discovery System

Discovers structure in raw information streams through prediction patterns.
This is where "experiences" emerge from continuous data flow.

Key insight: Experience boundaries appear where prediction patterns change.
Action/outcome relationships emerge from temporal prediction success.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import time

# GPU acceleration imports
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available()
    if GPU_AVAILABLE:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    else:
        DEVICE = torch.device('cpu')
    print(f"🚀 Pattern Discovery GPU: {DEVICE}")
except ImportError:
    torch = None
    GPU_AVAILABLE = False
    DEVICE = None
    print("⚠️  PyTorch not available - using CPU-only pattern discovery")


class PatternDiscovery:
    """
    Discovers emergent structure in pure information streams.
    
    No hardcoded concepts - structure emerges from:
    1. Prediction boundaries (where predictability changes)
    2. Causal patterns (what predicts what)
    3. Temporal regularities (repeating sequences)
    """
    
    def __init__(self, prediction_window: int = 5, emergence_threshold: float = 0.7):
        """
        Initialize pattern discovery system.
        
        Args:
            prediction_window: Window size for prediction analysis
            emergence_threshold: Threshold for pattern emergence
        """
        self.prediction_window = prediction_window
        self.emergence_threshold = emergence_threshold
        
        # Discovered patterns (emerge over time)
        self.prediction_boundaries = []  # Where predictability changes
        self.causal_patterns = defaultdict(list)  # What predicts what
        self.temporal_motifs = []  # Repeating sequences
        
        # Pattern emergence tracking
        self.boundary_candidates = defaultdict(float)
        self.motif_candidates = defaultdict(int)
        
        # Learning state
        self.patterns_discovered = 0
        self.discovery_start_time = time.time()
        
        print("PatternDiscovery initialized - structure will emerge from prediction")
    
    def analyze_stream_segment(self, vectors: List[np.ndarray]) -> Dict[str, Any]:
        """
        Analyze a stream segment for emergent patterns.
        
        This is the core of Strategy 1 - finding structure without imposing it.
        
        Args:
            vectors: Sequence of raw vectors from stream
            
        Returns:
            Analysis results including discovered patterns
        """
        if len(vectors) < self.prediction_window:
            return {'status': 'insufficient_data', 'vectors_analyzed': len(vectors)}
        
        analysis = {
            'vectors_analyzed': len(vectors),
            'boundaries_found': [],
            'causal_links': [],
            'motifs_detected': []
        }
        
        # 1. Find prediction boundaries (experience boundaries emerge here)
        boundaries = self._find_prediction_boundaries(vectors)
        analysis['boundaries_found'] = boundaries
        
        # 2. Discover causal patterns (action->outcome emerges here)
        causal = self._discover_causal_patterns(vectors)
        analysis['causal_links'] = causal
        
        # 3. Detect temporal motifs (behavioral patterns emerge here)
        motifs = self._detect_temporal_motifs(vectors)
        analysis['motifs_detected'] = motifs
        
        # Update discovered patterns
        self._update_pattern_emergence(boundaries, causal, motifs)
        
        return analysis
    
    def _find_prediction_boundaries(self, vectors: List[np.ndarray]) -> List[int]:
        """
        Find where prediction patterns change significantly.
        
        These boundaries naturally segment the stream into "experiences"
        without us defining what an experience is.
        """
        if len(vectors) < self.prediction_window:
            return []
            
        # GPU vectorized version
        if GPU_AVAILABLE and torch is not None and len(vectors) > 20:
            return self._find_boundaries_gpu(vectors)
        
        # CPU fallback for small sequences
        return self._find_boundaries_cpu(vectors)
    
    def _find_boundaries_gpu(self, vectors: List[np.ndarray]) -> List[int]:
        """GPU-accelerated boundary detection using vectorized operations."""
        # Convert to GPU tensor
        vector_tensor = torch.tensor(np.array(vectors), dtype=torch.float32, device=DEVICE)
        n_vectors, dim = vector_tensor.shape
        
        boundaries = []
        window_size = self.prediction_window - 1
        
        # Vectorized sliding window predictions
        if n_vectors <= window_size:
            return []
            
        # Create sliding windows using unfold
        windows = vector_tensor.unfold(0, window_size, 1)  # Shape: (n_windows, dim, window_size)
        targets = vector_tensor[window_size:n_vectors]  # Shape: (n_windows, dim)
        
        # Vectorized predictions (mean of each window)
        predictions = windows.mean(dim=2)  # Shape: (n_windows, dim)
        
        # Vectorized prediction errors
        errors = torch.norm(targets - predictions, dim=1)  # Shape: (n_windows,)
        
        # Find error changes
        if len(errors) > 1:
            error_changes = torch.abs(errors[1:] - errors[:-1])
            boundary_mask = error_changes > 0.5
            boundary_indices = torch.where(boundary_mask)[0] + window_size + 1
            
            # Convert back to CPU and update candidates
            for idx in boundary_indices.cpu().numpy():
                boundaries.append(int(idx))
                self.boundary_candidates[int(idx)] += float(error_changes[idx - window_size - 1])
        
        return boundaries
    
    def _find_boundaries_cpu(self, vectors: List[np.ndarray]) -> List[int]:
        """CPU fallback for boundary detection."""
        boundaries = []
        
        # Slide window through stream
        for i in range(len(vectors) - self.prediction_window):
            # Try to predict next vector from previous ones
            context = vectors[i:i+self.prediction_window-1]
            target = vectors[i+self.prediction_window-1]
            
            # Simple prediction: average of context
            prediction = np.mean(context, axis=0)
            
            # Prediction error
            error = np.linalg.norm(target - prediction)
            
            # Check if this is a boundary (high prediction error)
            if i > 0:
                # Compare to previous prediction error
                prev_context = vectors[i-1:i+self.prediction_window-2]
                prev_target = vectors[i+self.prediction_window-2]
                prev_prediction = np.mean(prev_context, axis=0)
                prev_error = np.linalg.norm(prev_target - prev_prediction)
                
                # Significant change in predictability suggests boundary
                error_change = abs(error - prev_error)
                if error_change > 0.5:  # Threshold for significant change
                    boundaries.append(i + self.prediction_window - 1)
                    self.boundary_candidates[i] += error_change
        
        return boundaries
    
    def _discover_causal_patterns(self, vectors: List[np.ndarray]) -> List[Dict]:
        """
        Discover what vectors tend to predict other vectors.
        
        This is how action->outcome relationships emerge without
        us defining what actions or outcomes are.
        """
        if len(vectors) < 3:
            return []
            
        # GPU vectorized version for larger sequences
        if GPU_AVAILABLE and torch is not None and len(vectors) > 50:
            return self._discover_causal_gpu(vectors)
        
        # CPU fallback
        return self._discover_causal_cpu(vectors)
    
    def _discover_causal_gpu(self, vectors: List[np.ndarray]) -> List[Dict]:
        """GPU-accelerated causal pattern discovery."""
        vector_tensor = torch.tensor(np.array(vectors), dtype=torch.float32, device=DEVICE)
        n_vectors = len(vector_tensor)
        
        if n_vectors < 3:
            return []
        
        causal_links = []
        
        # Get antecedent-consequent pairs
        antecedents = vector_tensor[:-2]  # All but last 2
        consequents = vector_tensor[1:-1]  # Offset by 1
        
        # Vectorized similarity computation
        # Compute pairwise distances between all antecedents
        antecedent_dists = torch.cdist(antecedents, antecedents, p=2)
        antecedent_norms = torch.norm(antecedents, dim=1, keepdim=True)
        antecedent_similarities = 1.0 - antecedent_dists / (antecedent_norms + 0.001)
        
        # Find similar antecedents (similarity > 0.8)
        similar_mask = antecedent_similarities > 0.8
        
        # For each antecedent, compute outcome similarities
        for i in range(len(antecedents)):
            similar_indices = torch.where(similar_mask[i])[0]
            similar_indices = similar_indices[similar_indices != i]  # Exclude self
            
            if len(similar_indices) > 0:
                # Get consequents for similar antecedents
                similar_consequents = consequents[similar_indices]
                current_consequent = consequents[i:i+1]
                
                # Vectorized outcome similarity computation
                outcome_dists = torch.cdist(current_consequent, similar_consequents, p=2)
                consequent_norm = torch.norm(current_consequent, dim=1, keepdim=True)
                outcome_similarities = 1.0 - outcome_dists / (consequent_norm + 0.001)
                
                pattern_strength = torch.sum(outcome_similarities).item()
                
                if pattern_strength > 1.0:
                    causal_links.append({
                        'antecedent_idx': i,
                        'consequent_idx': i + 1,
                        'strength': pattern_strength,
                        'pattern': 'predictive'
                    })
        
        return causal_links
    
    def _discover_causal_cpu(self, vectors: List[np.ndarray]) -> List[Dict]:
        """CPU fallback for causal pattern discovery."""
        causal_links = []
        
        # Look for consistent prediction patterns
        for i in range(len(vectors) - 2):
            antecedent = vectors[i]
            consequent = vectors[i + 1]
            
            # Check if this pattern repeats
            pattern_strength = 0.0
            for j in range(len(vectors) - 2):
                if j != i:
                    candidate = vectors[j]
                    similarity = 1.0 - np.linalg.norm(antecedent - candidate) / (np.linalg.norm(antecedent) + 0.001)
                    
                    if similarity > 0.8:  # Similar antecedent
                        next_vec = vectors[j + 1]
                        outcome_similarity = 1.0 - np.linalg.norm(consequent - next_vec) / (np.linalg.norm(consequent) + 0.001)
                        pattern_strength += outcome_similarity
            
            if pattern_strength > 1.0:  # Found repeating causal pattern
                causal_links.append({
                    'antecedent_idx': i,
                    'consequent_idx': i + 1,
                    'strength': pattern_strength,
                    'pattern': 'predictive'
                })
        
        return causal_links
    
    def _detect_temporal_motifs(self, vectors: List[np.ndarray]) -> List[Dict]:
        """
        Detect repeating sequences (behavioral motifs).
        
        These emerge as natural "chunks" of behavior without
        us defining what behaviors are.
        """
        if len(vectors) < 3:
            return []
            
        # GPU vectorized version for larger sequences  
        if GPU_AVAILABLE and torch is not None and len(vectors) > 30:
            return self._detect_motifs_gpu(vectors)
        
        # CPU fallback
        return self._detect_motifs_cpu(vectors)
    
    def _detect_motifs_gpu(self, vectors: List[np.ndarray]) -> List[Dict]:
        """GPU-accelerated motif detection using vectorized operations."""
        vector_tensor = torch.tensor(np.array(vectors), dtype=torch.float32, device=DEVICE)
        motifs = []
        min_motif_length = 3
        max_motif_length = min(10, len(vectors) // 2)
        discovered_regions = set()
        
        # Process each motif length
        for length in range(min_motif_length, max_motif_length):
            if len(vector_tensor) < length:
                continue
                
            # Create all possible subsequences of this length
            n_subsequences = len(vector_tensor) - length + 1
            if n_subsequences <= 1:
                continue
            
            # Use unfold to create sliding windows
            subsequences = vector_tensor.unfold(0, length, 1)  # Shape: (n_subs, dim, length)
            subsequences = subsequences.transpose(1, 2)  # Shape: (n_subs, length, dim)
            
            # Compute pairwise distances between all subsequences
            # Flatten each subsequence for distance computation
            flat_subs = subsequences.reshape(n_subsequences, -1)  # Shape: (n_subs, length*dim)
            
            # Compute pairwise distances
            distances = torch.cdist(flat_subs, flat_subs, p=2) / length  # Normalize by length
            
            # Find similar motifs (distance < 0.3)
            similarity_mask = distances < 0.3
            
            # Process each potential motif
            for start in range(n_subsequences):
                # Skip if this region is already covered
                if any(start >= covered[0] and start < covered[1] for covered in discovered_regions):
                    continue
                    
                # Find all positions similar to this motif
                similar_positions = torch.where(similarity_mask[start])[0]
                similar_positions = similar_positions[similar_positions != start]  # Exclude self
                
                if len(similar_positions) >= 2:  # Motif repeats
                    # Create pattern signature
                    candidate_motif = vector_tensor[start:start + length]
                    pattern_signature = tuple(torch.round(torch.mean(candidate_motif, dim=0), decimals=2).cpu().numpy())
                    motif_key = f"pattern_{length}_{hash(pattern_signature) % 10000}"
                    
                    # Only add if this exact pattern hasn't been seen before
                    if motif_key not in self.motif_candidates:
                        discovered_regions.add((start, start + length))
                        positions = [start] + similar_positions.cpu().tolist()
                        occurrences = len(positions) - 1  # Exclude original position from count
                        
                        self.motif_candidates[motif_key] = occurrences
                        
                        motifs.append({
                            'start_idx': start,
                            'length': length,
                            'occurrences': occurrences,
                            'positions': positions,
                            'motif_id': motif_key,
                            'pattern_signature': pattern_signature
                        })
                    else:
                        # Update existing motif with new occurrences
                        self.motif_candidates[motif_key] += len(similar_positions)
        
        return motifs
    
    def _detect_motifs_cpu(self, vectors: List[np.ndarray]) -> List[Dict]:
        """CPU fallback for motif detection."""
        motifs = []
        min_motif_length = 3
        max_motif_length = 10
        discovered_regions = set()  # Track covered regions to avoid overlaps
        
        # Look for repeating sequences
        for length in range(min_motif_length, min(max_motif_length, len(vectors) // 2)):
            for start in range(len(vectors) - length):
                # Skip if this region is already covered by a discovered motif
                region_key = (start, start + length)
                if any(start >= covered[0] and start < covered[1] for covered in discovered_regions):
                    continue
                    
                candidate_motif = vectors[start:start + length]
                
                # Count occurrences of this motif
                occurrences = 0
                positions = [start]  # Include original position
                
                for i in range(len(vectors) - length):
                    if i != start:
                        window = vectors[i:i + length]
                        
                        # Check similarity of sequences
                        distances = [np.linalg.norm(candidate_motif[j] - window[j]) 
                                   for j in range(length)]
                        avg_distance = np.mean(distances)
                        
                        if avg_distance < 0.3:  # Similar sequence
                            occurrences += 1
                            positions.append(i)
                
                if occurrences >= 2:  # Motif repeats
                    # Create unique motif signature based on pattern, not position
                    pattern_signature = tuple(np.round(np.mean(candidate_motif, axis=0), 2))
                    motif_key = f"pattern_{length}_{hash(pattern_signature) % 10000}"
                    
                    # Only add if this exact pattern hasn't been seen before
                    if motif_key not in self.motif_candidates:
                        discovered_regions.add(region_key)
                        self.motif_candidates[motif_key] = occurrences
                        
                        motifs.append({
                            'start_idx': start,
                            'length': length,
                            'occurrences': occurrences,
                            'positions': positions,
                            'motif_id': motif_key,
                            'pattern_signature': pattern_signature
                        })
                    else:
                        # Update existing motif with new occurrences
                        self.motif_candidates[motif_key] += occurrences
        
        return motifs
    
    def _update_pattern_emergence(self, boundaries: List[int], 
                                 causal: List[Dict], 
                                 motifs: List[Dict]):
        """Update emerging patterns based on new discoveries."""
        
        # Make boundaries more permissive - any boundary with significant change becomes strong
        for boundary in boundaries:
            # Lower threshold for boundary emergence (0.3 instead of 0.7)
            self.boundary_candidates[boundary] += 1.0  # Accumulate evidence faster
            if self.boundary_candidates[boundary] > 0.3:  # More permissive threshold
                if boundary not in self.prediction_boundaries:
                    self.prediction_boundaries.append(boundary)
                    self.patterns_discovered += 1
                    print(f"Experience boundary emerged at position {boundary}")
        
        # Strong causal patterns become recognized relationships  
        for link in causal:
            if link['strength'] > 0.5:  # Lower threshold for causal patterns
                key = (link['antecedent_idx'], link['consequent_idx'])
                self.causal_patterns[key].append(link)
        
        # Frequent motifs become recognized behaviors
        for motif in motifs:
            if self.motif_candidates[motif['motif_id']] > 3:
                # Check if this motif pattern is already recognized
                motif_exists = any(existing['motif_id'] == motif['motif_id'] 
                                 for existing in self.temporal_motifs)
                
                if not motif_exists:
                    self.temporal_motifs.append(motif)
                    self.patterns_discovered += 1
                    print(f"Behavioral motif emerged: length {motif['length']}, pattern {motif['motif_id']}, occurs {motif['occurrences']} times")
    
    def get_emergent_structure(self) -> Dict[str, Any]:
        """
        Get the structure that has emerged from the raw stream.
        
        This is the payoff - structure discovered rather than imposed.
        """
        # Sort boundaries
        sorted_boundaries = sorted(self.prediction_boundaries)
        
        # Extract emergent "experiences" based on boundaries
        emergent_experiences = []
        
        # Method 1: Create experiences from boundary segments
        if len(sorted_boundaries) > 1:
            for i in range(len(sorted_boundaries) - 1):
                emergent_experiences.append({
                    'start': sorted_boundaries[i],
                    'end': sorted_boundaries[i + 1],
                    'type': 'boundary_segment',
                    'discovered_at': time.time()
                })
        
        # Method 2: Create experiences from significant motifs (fallback and supplement)
        # Use the strongest motifs as experience candidates
        strong_motifs = [m for m in self.temporal_motifs if m['occurrences'] >= 5]
        for motif in strong_motifs[:10]:  # Limit to top 10 to avoid overflow
            emergent_experiences.append({
                'start': motif['start_idx'],
                'end': motif['start_idx'] + motif['length'],
                'type': 'motif_segment', 
                'motif_length': motif['length'],
                'motif_occurrences': motif['occurrences'],
                'discovered_at': time.time()
            })
        
        # Method 3: If still no experiences, create from any boundaries found  
        if not emergent_experiences and len(sorted_boundaries) >= 1:
            # Create experiences around each boundary
            for boundary in sorted_boundaries:
                start = max(0, boundary - 3)
                end = boundary + 3
                emergent_experiences.append({
                    'start': start,
                    'end': end,
                    'type': 'boundary_window',
                    'boundary_position': boundary,
                    'discovered_at': time.time()
                })
        
        return {
            'patterns_discovered': self.patterns_discovered,
            'discovery_duration': time.time() - self.discovery_start_time,
            'emergent_experiences': emergent_experiences,
            'prediction_boundaries': sorted_boundaries,
            'causal_relationships': dict(self.causal_patterns),
            'behavioral_motifs': self.temporal_motifs,
            'emergence_statistics': {
                'boundary_candidates': len(self.boundary_candidates),
                'strong_boundaries': len(self.prediction_boundaries),
                'motif_candidates': len(self.motif_candidates),
                'recognized_motifs': len(self.temporal_motifs),
                'causal_patterns': len(self.causal_patterns)
            }
        }
    
    def reset_discovery(self):
        """Reset pattern discovery to start fresh."""
        self.prediction_boundaries.clear()
        self.causal_patterns.clear()
        self.temporal_motifs.clear()
        self.boundary_candidates.clear()
        self.motif_candidates.clear()
        self.patterns_discovered = 0
        self.discovery_start_time = time.time()
        print("Pattern discovery reset - awaiting emergence")