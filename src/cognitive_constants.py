"""
Cognitive DNA: Irreducible Constants of Our Artificial Mind

These values define the fundamental "species" characteristics of our cognitive 
architecture. Unlike adaptive parameters that emerge from experience, these 
represent the hardcoded constraints that define what kind of intelligence 
this system IS.

Think of this as the cognitive equivalent of biological DNA - the unchanging
foundation that enables all emergent behaviors while constraining the system
to a specific "cognitive niche."

Research Foundation:
- Based on neuroscience research about fundamental cognitive constraints
- Informed by robotics requirements for real-time control
- Optimized for continuous learning in dynamic environments
- Validated through empirical testing of robot behavior

Last Updated: January 2025 (following Evolution Phase 2 + Hardware Adaptation Integration)
"""

from typing import Dict, Any
import time

# Hardware adaptation integration
try:
    from .utils.hardware_adaptation import get_adaptive_cognitive_limits
    HARDWARE_ADAPTATION_AVAILABLE = True
except ImportError:
    HARDWARE_ADAPTATION_AVAILABLE = False
    def get_adaptive_cognitive_limits():
        return {}
    print("⚠️  Hardware adaptation not available, using static constants")

# =============================================================================
# CORE IDENTITY: What Kind of Mind This Is
# =============================================================================

COGNITIVE_SPECIES = {
    "name": "Robotic Continuous Learner",
    "niche": "Real-time robotic intelligence with adaptive behavior",
    "architecture": "4-system emergent (experience + similarity + activation + prediction)",
    "learning_paradigm": "Continuous experiential learning",
    "specialization": "Embodied intelligence in dynamic environments"
}

# =============================================================================
# TEMPORAL DNA: Time Constants That Define Our "Speed of Thought"
# =============================================================================

class TemporalConstants:
    """Time-based constants that define the system's temporal niche."""
    
    # Control Loop Timing (Real-time robotics constraint)
    MAX_CONTROL_CYCLE_TIME = 0.1  # 100ms - upper bound for real-time control
    TARGET_CONTROL_CYCLE_TIME = 0.05  # 50ms - target for responsive behavior
    MIN_CONTROL_CYCLE_TIME = 0.01  # 10ms - lower bound for system stability
    
    # Memory Consolidation Timing (Biological inspiration: sleep cycles)
    EXPERIENCE_BATCH_TIME = 300.0  # 5 minutes - natural batch processing window
    MEMORY_CONSOLIDATION_INTERVAL = 3600.0  # 1 hour - major memory reorganization
    LONG_TERM_ADAPTATION_WINDOW = 86400.0  # 24 hours - major parameter adjustments
    
    # Learning Response Times (Based on prediction error feedback loops)
    IMMEDIATE_ADAPTATION_WINDOW = 1.0  # 1 second - immediate error correction
    SHORT_TERM_LEARNING_WINDOW = 60.0  # 1 minute - pattern recognition stabilization
    MEDIUM_TERM_LEARNING_WINDOW = 600.0  # 10 minutes - behavioral pattern formation

# =============================================================================
# COGNITIVE CAPACITY DNA: Information Processing Limits
# =============================================================================

class CognitiveCapacityConstants:
    """Core capacity constraints that define the system's information processing niche.
    
    Note: Many of these are now hardware-adaptive rather than fixed constants.
    The static values serve as fallbacks when hardware adaptation is unavailable.
    """
    
    # Working Memory Architecture (Hardware-adaptive, these are fallback defaults)
    DEFAULT_WORKING_MEMORY_TARGET = 15  # Experiences actively maintained (fallback)
    MIN_WORKING_MEMORY_SIZE = 5  # Absolute minimum for coherent behavior
    MAX_WORKING_MEMORY_SIZE = 100  # Absolute maximum (hardware adaptation can scale up)
    
    # Experience Integration Limits (Hardware-adaptive)
    MAX_SIMILARITY_SEARCH_SIZE = 1000  # Fallback maximum (hardware adaptation overrides)
    SIMILARITY_COMPUTATION_BATCH_SIZE = 50  # Fallback GPU threshold (hardware-adaptive)
    EXPERIENCE_INTEGRATION_DEPTH = 100  # How many experiences influence one decision
    
    @classmethod
    def get_working_memory_limit(cls) -> int:
        """Get current hardware-adaptive working memory limit."""
        if HARDWARE_ADAPTATION_AVAILABLE:
            limits = get_adaptive_cognitive_limits()
            return limits.get('working_memory_limit', cls.DEFAULT_WORKING_MEMORY_TARGET)
        return cls.DEFAULT_WORKING_MEMORY_TARGET
    
    @classmethod
    def get_similarity_search_limit(cls) -> int:
        """Get current hardware-adaptive similarity search limit."""
        if HARDWARE_ADAPTATION_AVAILABLE:
            limits = get_adaptive_cognitive_limits()
            return limits.get('similarity_search_limit', cls.MAX_SIMILARITY_SEARCH_SIZE)
        return cls.MAX_SIMILARITY_SEARCH_SIZE
    
    @classmethod
    def get_batch_processing_threshold(cls) -> int:
        """Get current hardware-adaptive batch processing threshold."""
        if HARDWARE_ADAPTATION_AVAILABLE:
            limits = get_adaptive_cognitive_limits()
            return limits.get('batch_processing_threshold', cls.SIMILARITY_COMPUTATION_BATCH_SIZE)
        return cls.SIMILARITY_COMPUTATION_BATCH_SIZE
    
    # Pattern Recognition Granularity (Balances generalization vs specificity)
    MIN_PATTERN_LENGTH = 2  # Shortest meaningful pattern
    MAX_PATTERN_LENGTH = 10  # Longest trackable sequence
    PATTERN_RECOGNITION_WINDOW = 50  # Experiences analyzed for patterns

# =============================================================================
# PREDICTION ERROR DNA: The Fundamental Drive
# =============================================================================

class PredictionErrorConstants:
    """The core drive that defines what the system optimizes for."""
    
    # Optimal Prediction Error Range (The system's "sweet spot" for learning)
    # Research basis: Zone of proximal development (Vygotsky), optimal challenge theory
    OPTIMAL_PREDICTION_ERROR_TARGET = 0.3  # Sweet spot for learnable complexity
    MIN_VIABLE_PREDICTION_ERROR = 0.1  # Below this = stagnation (too easy)
    MAX_VIABLE_PREDICTION_ERROR = 0.8  # Above this = chaos (too hard)
    
    # Error Tolerance for Stability (Prevents oscillation around target)
    PREDICTION_ERROR_TOLERANCE = 0.05  # ±0.05 around target is acceptable
    ERROR_GRADIENT_SENSITIVITY = 0.1  # How much change triggers adaptation
    
    # Bootstrap Values (Stable starting points for virgin system)
    DEFAULT_PREDICTION_ERROR = 0.5  # Moderate challenge for cold start
    DEFAULT_PREDICTION_UTILITY = 0.5  # Neutral utility for new experiences
    DEFAULT_CONFIDENCE = 0.1  # Low confidence for untested predictions

# =============================================================================
# SENSORY-MOTOR DNA: Interface Constraints
# =============================================================================

class SensoryMotorConstants:
    """Constants that define the system's interface with the physical world."""
    
    # Vector Dimensions (Communication protocol constraints)
    SENSORY_VECTOR_SIZE = 16  # Standard input dimension
    ACTION_VECTOR_SIZE = 4   # Standard output dimension
    MAX_VECTOR_SIZE = 1024   # Protocol safety limit
    
    # Sensory Resolution (Balances precision vs processing speed)
    FLOAT_PRECISION = 32  # IEEE 754 float32 for cross-platform compatibility
    SENSORY_NOISE_TOLERANCE = 0.01  # Minimum detectable difference
    ACTION_QUANTIZATION = 0.01  # Minimum meaningful action difference
    
    # Real-world Interface Timing (Robot hardware constraints)
    SENSOR_REFRESH_RATE = 20.0  # Hz - maximum sensor reading frequency
    MOTOR_COMMAND_LATENCY = 0.02  # 20ms - typical motor response time
    COMMUNICATION_TIMEOUT = 1.0  # 1 second - network communication limit

# =============================================================================
# STABILITY DNA: System Robustness Constants
# =============================================================================

class StabilityConstants:
    """Constants that ensure system robustness and prevent pathological states."""
    
    # Numerical Stability (Prevents mathematical edge cases)
    MIN_SIMILARITY_VALUE = 0.001  # Prevents division by zero
    MAX_ACTIVATION_VALUE = 10.0   # Prevents exponential explosion
    MIN_ACTIVATION_VALUE = 0.001  # Prevents complete deactivation
    
    # Learning Stability (Prevents runaway adaptation)
    MAX_LEARNING_RATE = 0.5      # Upper bound on adaptation speed
    MIN_LEARNING_RATE = 0.0001   # Lower bound to maintain learning ability
    ADAPTATION_MOMENTUM = 0.9    # Smooths parameter changes
    
    # Memory Stability (Prevents memory pathologies)
    MAX_EXPERIENCES_PER_SECOND = 100  # Prevents memory overflow
    MIN_SIMILARITY_FOR_STORAGE = 0.0  # Always store (no filtering)
    EXPERIENCE_REDUNDANCY_THRESHOLD = 0.99  # Near-identical experience filtering

# =============================================================================
# EVOLUTION PHASE 2 DNA: Biologically-Inspired Efficiency Mechanisms
# =============================================================================

class PerformancePressureConstants:
    """Constants for natural performance pressure emergence (Evolution system)."""
    
    # Performance Monitoring (Biological: metabolic efficiency pressure)
    PERFORMANCE_DEGRADATION_THRESHOLD = 0.2  # 20% slowdown triggers concern
    CONSOLIDATION_PRESSURE_THRESHOLD = 0.1   # Triggers natural "sleep" cycles
    TENSOR_FRAGMENTATION_THRESHOLD = 0.05    # GPU memory efficiency pressure
    MAX_CONSECUTIVE_BAD_CYCLES = 10          # Before rollback recommendation
    
    # Natural Adaptation Rates (Biological: gradual neural plasticity)
    PERFORMANCE_ADAPTATION_RATE = 0.02       # Conservative system evolution
    CONSOLIDATION_PRESSURE_ACCUMULATION = 0.001  # Tiny pressure per cycle
    FRAGMENTATION_ACCUMULATION = 0.0001      # Builds slowly over time

class CognitiveEnergyConstants:
    """Energy budget constraints for biological realism (Evolution 3: Sparse Activation).
    
    Energy budgets are now hardware-adaptive, scaling with computational capacity.
    """
    
    # Energy Budget (Hardware-adaptive, these are fallback defaults)
    BASE_COGNITIVE_ENERGY_BUDGET = 20        # Fallback: ~20 active experiences
    MIN_COGNITIVE_ENERGY_BUDGET = 5          # Absolute survival minimum
    MAX_COGNITIVE_ENERGY_BUDGET = 100        # Hardware adaptation can scale up to 100
    
    @classmethod
    def get_cognitive_energy_budget(cls) -> int:
        """Get current hardware-adaptive cognitive energy budget."""
        if HARDWARE_ADAPTATION_AVAILABLE:
            limits = get_adaptive_cognitive_limits()
            return limits.get('cognitive_energy_budget', cls.BASE_COGNITIVE_ENERGY_BUDGET)
        return cls.BASE_COGNITIVE_ENERGY_BUDGET
    
    # Energy Adaptation (Performance pressure affects available energy)
    PRESSURE_ENERGY_FACTOR = 0.3             # High pressure reduces available energy
    FRAGMENTATION_ENERGY_FACTOR = 0.2        # Fragmentation costs energy
    ENERGY_RECOVERY_RATE = 0.01              # Gradual energy recovery

class SmartStorageConstants:
    """Intelligent memory management constants (Evolution 1: Smart Storage)."""
    
    # Storage Intelligence (Biological: memory consolidation during sleep)
    NOVELTY_WEIGHT_IN_STORAGE = 0.6          # How much novelty affects storage decisions
    UTILITY_WEIGHT_IN_STORAGE = 0.4          # How much utility affects storage decisions
    BASE_STORAGE_THRESHOLD = 0.3             # Store top 70% by default
    MAX_STORAGE_SELECTIVITY = 0.8            # Most selective: store only top 20%
    
    # Memory Pressure Management (Biological: forgetting low-utility memories)
    EXPERIENCE_SOFT_LIMIT = 2000             # Start pressure management
    MEMORY_PRESSURE_BUILD_RATE = 1000.0      # Experiences where pressure builds
    UTILITY_CLEANUP_PERCENTAGE = 0.25        # Max 25% removal per cleanup cycle
    CLEANUP_TARGET_PERCENTAGE = 0.1          # Remove 10% of excess per cycle

class AttentionMechanismConstants:
    """Natural attention and focus constants (Evolution 2: Attention Systems)."""
    
    # Attention Routing (Biological: selective attention mechanisms)
    ATTENTION_ROUTING_THRESHOLD = 20          # Experiences where attention helps
    ATTENTION_ADAPTATION_WINDOW = 50          # Recent samples for baseline adaptation
    ATTENTION_BASELINE_START = 0.5            # Neutral starting attention baseline
    
    # Attention Distribution (Biological: attention focus vs breadth)
    HIGH_ATTENTION_RATE_THRESHOLD = 0.2      # 20% high attention is optimal
    SUPPRESSION_RATE_THRESHOLD = 0.6         # 60% suppression is too much
    ATTENTION_BASELINE_BOUNDS = (0.1, 2.0)   # Attention can vary 10x
    BASELINE_ADAPTATION_RATE = 0.02          # Conservative attention evolution

# =============================================================================
# EMERGENCE BOUNDARIES: What CAN vs CANNOT Adapt
# =============================================================================

class EmergenceBoundaries:
    """Defines what aspects of the system can adapt vs what must remain constant."""
    
    # IMMUTABLE: Core architecture (these define the species)
    IMMUTABLE_CONSTANTS = {
        'system_count': 4,  # Experience, Similarity, Activation, Prediction
        'core_drive': 'prediction_error_minimization',
        'vector_sizes': (16, 4),  # Sensory input, action output
        'precision': 'float32',
        'real_time_constraint': True
    }
    
    # BOUNDED_ADAPTIVE: Can adapt within limits (species characteristics)
    BOUNDED_ADAPTIVE = {
        'optimal_prediction_error': (0.1, 0.8),
        'working_memory_size': (5, 30),
        'learning_rates': (0.0001, 0.5),
        'similarity_thresholds': (0.0, 1.0),
        'activation_decay_rates': (0.001, 0.1),
        # Evolution Phase 2 Lite additions
        'cognitive_energy_budget': (5, 50),
        'storage_selectivity': (0.0, 0.8),
        'attention_baseline': (0.1, 2.0),
        'consolidation_pressure': (0.0, 0.3),
        'performance_adaptation_rate': (0.001, 0.1)
    }
    
    # FULLY_EMERGENT: No constraints (behavioral flexibility)
    FULLY_EMERGENT = {
        'similarity_weights',
        'activation_patterns', 
        'prediction_strategies',
        'behavioral_sequences',
        'motor_skills',
        'spatial_knowledge',
        'temporal_patterns'
    }

# =============================================================================
# COGNITIVE PROFILE: System Capabilities and Limitations
# =============================================================================

def get_cognitive_profile() -> Dict[str, Any]:
    """
    Get complete cognitive profile of this artificial mind.
    
    Returns detailed specification of capabilities, limitations, and design choices
    that define this particular cognitive architecture.
    """
    return {
        "cognitive_species": COGNITIVE_SPECIES,
        "creation_timestamp": time.time(),
        "version": "5.0_evolution_phase2",
        
        "temporal_niche": {
            "response_time_range": (TemporalConstants.MIN_CONTROL_CYCLE_TIME, 
                                  TemporalConstants.MAX_CONTROL_CYCLE_TIME),
            "learning_timescales": {
                "immediate": TemporalConstants.IMMEDIATE_ADAPTATION_WINDOW,
                "short_term": TemporalConstants.SHORT_TERM_LEARNING_WINDOW,
                "medium_term": TemporalConstants.MEDIUM_TERM_LEARNING_WINDOW,
                "long_term": TemporalConstants.LONG_TERM_ADAPTATION_WINDOW
            }
        },
        
        "cognitive_capacity": {
            "working_memory_range": (CognitiveCapacityConstants.MIN_WORKING_MEMORY_SIZE,
                                   CognitiveCapacityConstants.MAX_WORKING_MEMORY_SIZE),
            "pattern_complexity": (CognitiveCapacityConstants.MIN_PATTERN_LENGTH,
                                 CognitiveCapacityConstants.MAX_PATTERN_LENGTH),
            "experience_integration_depth": CognitiveCapacityConstants.EXPERIENCE_INTEGRATION_DEPTH
        },
        
        "drive_system": {
            "primary_drive": "adaptive_prediction_error_minimization",
            "optimal_challenge_level": PredictionErrorConstants.OPTIMAL_PREDICTION_ERROR_TARGET,
            "challenge_tolerance": (PredictionErrorConstants.MIN_VIABLE_PREDICTION_ERROR,
                                  PredictionErrorConstants.MAX_VIABLE_PREDICTION_ERROR)
        },
        
        "interface_constraints": {
            "sensory_dimensions": SensoryMotorConstants.SENSORY_VECTOR_SIZE,
            "action_dimensions": SensoryMotorConstants.ACTION_VECTOR_SIZE,
            "temporal_resolution": 1.0 / SensoryMotorConstants.SENSOR_REFRESH_RATE,
            "motor_latency": SensoryMotorConstants.MOTOR_COMMAND_LATENCY
        },
        
        "adaptation_boundaries": {
            "immutable": list(EmergenceBoundaries.IMMUTABLE_CONSTANTS.keys()),
            "bounded_adaptive": list(EmergenceBoundaries.BOUNDED_ADAPTIVE.keys()),
            "fully_emergent": list(EmergenceBoundaries.FULLY_EMERGENT)
        },
        
        "evolution_systems": {
            "performance_pressure": {
                "consolidation_threshold": PerformancePressureConstants.CONSOLIDATION_PRESSURE_THRESHOLD,
                "adaptation_rate": PerformancePressureConstants.PERFORMANCE_ADAPTATION_RATE,
                "max_bad_cycles": PerformancePressureConstants.MAX_CONSECUTIVE_BAD_CYCLES
            },
            "cognitive_energy": {
                "base_budget": CognitiveEnergyConstants.BASE_COGNITIVE_ENERGY_BUDGET,
                "energy_range": (CognitiveEnergyConstants.MIN_COGNITIVE_ENERGY_BUDGET,
                                CognitiveEnergyConstants.MAX_COGNITIVE_ENERGY_BUDGET),
                "recovery_rate": CognitiveEnergyConstants.ENERGY_RECOVERY_RATE
            },
            "smart_storage": {
                "novelty_weight": SmartStorageConstants.NOVELTY_WEIGHT_IN_STORAGE,
                "utility_weight": SmartStorageConstants.UTILITY_WEIGHT_IN_STORAGE,
                "storage_threshold": SmartStorageConstants.BASE_STORAGE_THRESHOLD,
                "memory_pressure_limit": SmartStorageConstants.EXPERIENCE_SOFT_LIMIT
            },
            "attention_mechanisms": {
                "routing_threshold": AttentionMechanismConstants.ATTENTION_ROUTING_THRESHOLD,
                "adaptation_window": AttentionMechanismConstants.ATTENTION_ADAPTATION_WINDOW,
                "baseline_bounds": AttentionMechanismConstants.ATTENTION_BASELINE_BOUNDS
            }
        }
    }

# =============================================================================
# DESIGN PHILOSOPHY DOCUMENTATION
# =============================================================================

DESIGN_PHILOSOPHY = """
COGNITIVE DNA DESIGN PHILOSOPHY

1. BIOLOGICAL INSPIRATION
   - Every biological species has genetic constraints that define its cognitive niche
   - BUT biological brains adapt their processing to their physical capabilities
   - Humans: ~100ms neural transmission, 7±2 working memory, ~300ms decision time
   - Our system: ~50ms target cycles, hardware-adaptive working memory, ~30ms decisions
   - Like biology: fundamental algorithms are fixed, but capacity scales with "brain size"

2. HARDWARE ADAPTATION REALISM
   - Core algorithms are fixed (the cognitive "DNA")
   - But capacity limits adapt to actual hardware capabilities
   - Same brain architecture runs optimally on Raspberry Pi or GPU server
   - Hardware discovery and dynamic adaptation replace hardcoded limits
   - True constants: protocol contracts (vector sizes), numerical stability bounds
   - Adaptive constants: memory limits, processing thresholds, energy budgets

3. EMERGENCE VS ENGINEERING BALANCE
   - Constants define the BOUNDARY CONDITIONS for emergence
   - Within these boundaries, unlimited behavioral complexity can emerge
   - Without boundaries, systems become unstable or computationally intractable

4. SCIENTIFIC REPRODUCIBILITY
   - Documenting these constants enables replication and comparison
   - Different cognitive niches would have different constant values
   - This enables scientific study of cognitive architecture variants

5. EVOLUTIONARY PERSPECTIVE
   - These constants could themselves evolve over longer timescales
   - Different environments might select for different cognitive constants
   - This framework enables studying cognitive evolution and adaptation

The key insight: Even maximally emergent systems need some irreducible foundation.
Making this foundation explicit is scientific honesty, not engineering compromise.
"""

# =============================================================================
# VALIDATION AND TESTING
# =============================================================================

def validate_cognitive_constants():
    """Validate that all cognitive constants are within reasonable ranges."""
    
    # Temporal constraints must be physically realizable
    assert TemporalConstants.MIN_CONTROL_CYCLE_TIME > 0.001  # 1ms minimum
    assert TemporalConstants.MAX_CONTROL_CYCLE_TIME < 1.0    # 1s maximum
    
    # Memory constraints must be computationally feasible
    assert CognitiveCapacityConstants.MIN_WORKING_MEMORY_SIZE >= 1
    assert CognitiveCapacityConstants.MAX_WORKING_MEMORY_SIZE <= 100
    
    # Prediction error must form valid optimization target
    assert 0.0 < PredictionErrorConstants.OPTIMAL_PREDICTION_ERROR_TARGET < 1.0
    assert PredictionErrorConstants.MIN_VIABLE_PREDICTION_ERROR < PredictionErrorConstants.MAX_VIABLE_PREDICTION_ERROR
    
    # Interface constants must match protocol specifications
    assert SensoryMotorConstants.SENSORY_VECTOR_SIZE == 16
    assert SensoryMotorConstants.ACTION_VECTOR_SIZE == 4
    
    # Evolution Phase 2 constants validation
    assert 0.0 <= PerformancePressureConstants.CONSOLIDATION_PRESSURE_THRESHOLD <= 1.0
    assert PerformancePressureConstants.PERFORMANCE_ADAPTATION_RATE > 0.0
    assert PerformancePressureConstants.MAX_CONSECUTIVE_BAD_CYCLES >= 1
    
    assert CognitiveEnergyConstants.MIN_COGNITIVE_ENERGY_BUDGET > 0
    assert CognitiveEnergyConstants.MAX_COGNITIVE_ENERGY_BUDGET > CognitiveEnergyConstants.MIN_COGNITIVE_ENERGY_BUDGET
    assert CognitiveEnergyConstants.BASE_COGNITIVE_ENERGY_BUDGET >= CognitiveEnergyConstants.MIN_COGNITIVE_ENERGY_BUDGET
    
    assert 0.0 <= SmartStorageConstants.NOVELTY_WEIGHT_IN_STORAGE <= 1.0
    assert 0.0 <= SmartStorageConstants.UTILITY_WEIGHT_IN_STORAGE <= 1.0
    assert abs(SmartStorageConstants.NOVELTY_WEIGHT_IN_STORAGE + SmartStorageConstants.UTILITY_WEIGHT_IN_STORAGE - 1.0) < 0.01
    
    assert AttentionMechanismConstants.ATTENTION_ROUTING_THRESHOLD > 0
    assert AttentionMechanismConstants.ATTENTION_BASELINE_BOUNDS[1] > AttentionMechanismConstants.ATTENTION_BASELINE_BOUNDS[0]
    
    print("✅ All cognitive constants validated successfully (including Evolution Phase 2)")

if __name__ == "__main__":
    validate_cognitive_constants()
    profile = get_cognitive_profile()
    print(f"🧬 Cognitive DNA loaded: {profile['cognitive_species']['name']}")
    print(f"🎯 Primary drive: {profile['drive_system']['primary_drive']}")
    print(f"⚡ Response time: {profile['temporal_niche']['response_time_range']} seconds")
    print(f"🧠 Working memory: {profile['cognitive_capacity']['working_memory_range']} experiences (static range)")
    print(f"🔬 Evolution systems: {len(profile['evolution_systems'])} biological optimization mechanisms")
    print(f"📊 Version: {profile['version']}")
    
    # Show current hardware-adaptive limits if available
    if HARDWARE_ADAPTATION_AVAILABLE:
        adaptive_limits = get_adaptive_cognitive_limits()
        print(f"🔧 Hardware-adaptive limits:")
        print(f"   Working memory: {adaptive_limits.get('working_memory_limit', 'N/A')}")
        print(f"   Similarity search: {adaptive_limits.get('similarity_search_limit', 'N/A')}")
        print(f"   Cognitive energy: {adaptive_limits.get('cognitive_energy_budget', 'N/A')}")
    else:
        print(f"⚙️  Using static limits (hardware adaptation not available)")