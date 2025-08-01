#!/usr/bin/env python3
"""
Example: GPU-Optimized Pattern System

Shows how to reduce 700 transfers to 1 transfer per cycle
"""

import torch
import time

class CurrentPatternSystem:
    """Current implementation - lots of GPU->CPU transfers"""
    def __init__(self, device='cuda'):
        self.device = device
        self.patterns = []
        
    def match_pattern(self, field_state):
        best_similarity = 0.0
        best_idx = -1
        
        # BAD: Loop with .item() calls
        for i, pattern in enumerate(self.patterns):
            similarity = torch.cosine_similarity(
                field_state.flatten(), 
                pattern.flatten(), 
                dim=0
            ).item()  # GPU->CPU transfer!
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = i
                
        return best_idx, best_similarity
        
    def add_pattern(self, pattern):
        self.patterns.append(pattern.clone())


class GPUOptimizedPatternSystem:
    """GPU-optimized - single batched operation"""
    def __init__(self, max_patterns=10000, pattern_dim=512, device='cuda'):
        self.device = device
        self.pattern_bank = torch.zeros(max_patterns, pattern_dim, device=device)
        self.pattern_count = 0
        
    def match_pattern(self, field_state):
        if self.pattern_count == 0:
            return -1, 0.0
            
        # GOOD: Single batched operation
        field_flat = field_state.flatten()[:512]  # Ensure consistent size
        
        # Normalize for cosine similarity
        field_norm = field_flat / (field_flat.norm() + 1e-8)
        patterns_norm = self.pattern_bank[:self.pattern_count] / (
            self.pattern_bank[:self.pattern_count].norm(dim=1, keepdim=True) + 1e-8
        )
        
        # Compute ALL similarities at once
        similarities = torch.matmul(field_norm.unsqueeze(0), patterns_norm.T).squeeze()
        
        # Get best match (still on GPU)
        best_similarity, best_idx = similarities.max(dim=0)
        
        # Only transfer final result if needed
        return best_idx.item(), best_similarity.item()
        
    def match_pattern_gpu_only(self, field_state):
        """Even better - keep results on GPU"""
        if self.pattern_count == 0:
            return None, None
            
        field_flat = field_state.flatten()[:512]
        field_norm = field_flat / (field_flat.norm() + 1e-8)
        patterns_norm = self.pattern_bank[:self.pattern_count] / (
            self.pattern_bank[:self.pattern_count].norm(dim=1, keepdim=True) + 1e-8
        )
        
        similarities = torch.matmul(field_norm.unsqueeze(0), patterns_norm.T).squeeze()
        best_similarity, best_idx = similarities.max(dim=0)
        
        # Return GPU tensors - no transfer!
        return best_idx, best_similarity
        
    def add_pattern(self, pattern):
        if self.pattern_count < len(self.pattern_bank):
            self.pattern_bank[self.pattern_count] = pattern.flatten()[:512]
            self.pattern_count += 1


def benchmark():
    """Compare performance of both approaches"""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Running on device: {device}")
    
    # Create test data
    field_state = torch.randn(32, 32, 32, 64, device=device)
    n_patterns = 1000
    
    # Current approach
    current_system = CurrentPatternSystem(device)
    for i in range(n_patterns):
        pattern = torch.randn_like(field_state)
        current_system.add_pattern(pattern)
    
    start = time.time()
    for _ in range(10):
        idx, sim = current_system.match_pattern(field_state)
    current_time = time.time() - start
    
    # GPU-optimized approach
    pattern_dim = 512
    gpu_system = GPUOptimizedPatternSystem(
        max_patterns=10000, 
        pattern_dim=pattern_dim, 
        device=device
    )
    
    for i in range(n_patterns):
        pattern = torch.randn_like(field_state)
        gpu_system.add_pattern(pattern)
    
    start = time.time()
    for _ in range(10):
        idx, sim = gpu_system.match_pattern(field_state)
    gpu_time = time.time() - start
    
    # GPU-only (no transfers)
    start = time.time()
    for _ in range(10):
        idx, sim = gpu_system.match_pattern_gpu_only(field_state)
    gpu_only_time = time.time() - start
    
    print(f"\nResults for {n_patterns} patterns:")
    print(f"Current approach: {current_time*1000:.1f}ms ({n_patterns*10} GPU->CPU transfers)")
    print(f"GPU-optimized: {gpu_time*1000:.1f}ms (20 GPU->CPU transfers)")
    print(f"GPU-only: {gpu_only_time*1000:.1f}ms (0 GPU->CPU transfers)")
    print(f"\nSpeedup: {current_time/gpu_time:.1f}x")
    print(f"Speedup (GPU-only): {current_time/gpu_only_time:.1f}x")


if __name__ == "__main__":
    benchmark()