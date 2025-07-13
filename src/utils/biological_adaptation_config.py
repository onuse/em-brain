"""
Biologically Plausible Adaptation Configuration

Based on neuroscience research on real brain adaptation timescales:
- Synaptic plasticity: minutes to hours (hundreds of experiences)
- Network reorganization: days (thousands of experiences) 
- Parameter adaptation: gradual (1-2% changes)
- Stability: adaptation should be rare, not constant

This config makes Phase 2 adaptations much slower and more brain-like.
"""

class BiologicalAdaptationConfig:
    """Biologically inspired adaptation parameters."""
    
    # Adaptation frequency (much slower than every 5 experiences)
    MIN_EXPERIENCES_BETWEEN_ADAPTATIONS = 100  # vs 5 (20x slower)
    
    # Gradual adaptation rates (vs aggressive 10% changes)
    SYNAPTIC_ADAPTATION_RATE = 0.01  # 1% vs 10% (10x more gradual)
    NETWORK_ADAPTATION_RATE = 0.005  # 0.5% for network-level changes
    BOUND_ADAPTATION_RATE = 0.002    # 0.2% for parameter bounds (vs 2%)
    
    # Threshold adaptation (less sensitivity)
    THRESHOLD_ADAPTATION_RATE = 0.02  # vs 0.1 (5x less sensitive)
    
    # Performance monitoring (less hair-trigger)
    PERFORMANCE_DEGRADATION_THRESHOLD = 50.0  # vs 20% (allow more variation)
    PERFORMANCE_WINDOW_SIZE = 30  # vs 10 (longer smoothing window)
    
    # Memory pressure adaptation (slower response)
    MEMORY_PRESSURE_ADAPTATION_RATE = 0.01  # vs 0.05 (5x slower)
    
    # Plateau detection (longer patience)
    PLATEAU_DETECTION_WINDOW = 200  # vs 20 (10x longer patience)
    GRADIENT_CHANGE_THRESHOLD = 0.5  # vs 0.3 (less sensitive)
    
    # Meta-learning rates (much more conservative)
    LEARNING_RATE_ADAPTATION_RATE = 0.02  # vs 0.1 (5x slower)
    UTILITY_LEARNING_ADAPTATION_RATE = 0.01  # vs 0.1 (10x slower)


def apply_biological_adaptation_config():
    """
    Apply biologically plausible adaptation rates to Phase 2 systems.
    
    Call this to make Phase 2 adaptations brain-like instead of aggressive.
    """
    config = BiologicalAdaptationConfig()
    
    print("🧠 Applying biologically plausible adaptation timescales...")
    print(f"   Adaptation frequency: every {config.MIN_EXPERIENCES_BETWEEN_ADAPTATIONS} experiences (vs 5)")
    print(f"   Adaptation rates: {config.SYNAPTIC_ADAPTATION_RATE:.1%} (vs 10%)")
    print(f"   Performance threshold: {config.PERFORMANCE_DEGRADATION_THRESHOLD}% (vs 20%)")
    
    return config


def get_biological_brain_config():
    """Get MinimalBrain init parameters with biological adaptation rates."""
    
    return {
        'enable_phase2_adaptations': True,
        'biological_adaptation': True,  # Signal to use these configs
        'adaptation_config': apply_biological_adaptation_config()
    }


# Research references for these timescales:
RESEARCH_NOTES = """
Biological Adaptation Timescales (Research References):

1. Synaptic Plasticity:
   - LTP/LTD: minutes to hours (Malenka & Bear, 2004)
   - Spike-timing dependent plasticity: 100s of pairings (Bi & Poo, 1998)
   
2. Network Reorganization:
   - Cortical map plasticity: days to weeks (Buonomano & Merzenich, 1998)
   - Hippocampal place cell remapping: hours (Leutgeb et al., 2005)
   
3. Learning Rate Adaptation:
   - Dopamine learning rate: hundreds of trials (Behrens et al., 2007)
   - Meta-learning: thousands of experiences (Wang et al., 2016)
   
4. Working Memory:
   - Prefrontal stability: minutes to hours (Goldman-Rakic, 1995)
   - Attention parameters: stable within tasks (Posner & Petersen, 1990)
   
Phase 2 current config (every 5 experiences, 10% changes) is ~100x too fast.
"""