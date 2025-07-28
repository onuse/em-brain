"""
Cognitive Configuration Integration

This module integrates the cognitive constants into the dynamic brain architecture,
replacing scattered hardcoded values with centralized, scientifically-grounded parameters.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os

# Import cognitive constants
from .cognitive_constants import (
    TemporalConstants,
    CognitiveCapacityConstants,
    PredictionErrorConstants,
    StabilityConstants,
    PerformancePressureConstants,
    CognitiveEnergyConstants,
    SmartStorageConstants,
    AttentionMechanismConstants
)


@dataclass
class BrainConfig:
    """Configuration for DynamicUnifiedFieldBrain with cognitive constants."""
    
    # Field dynamics parameters
    field_evolution_rate: float = field(
        default_factory=lambda: PerformancePressureConstants.PERFORMANCE_ADAPTATION_RATE * 5
    )  # 0.1 = moderate field evolution
    
    field_decay_rate: float = field(
        default_factory=lambda: 1.0 - StabilityConstants.MIN_ACTIVATION_VALUE
    )  # 0.999 = very slow decay
    
    field_diffusion_rate: float = field(
        default_factory=lambda: PerformancePressureConstants.PERFORMANCE_ADAPTATION_RATE * 2.5
    )  # 0.05 = moderate diffusion
    
    # Activation parameters
    activation_threshold: float = field(
        default_factory=lambda: StabilityConstants.MIN_ACTIVATION_VALUE
    )  # 0.001 = minimum activity
    
    activation_max: float = field(
        default_factory=lambda: StabilityConstants.MAX_ACTIVATION_VALUE
    )  # 10.0 = maximum activity
    
    
    # Prediction and confidence
    default_prediction_confidence: float = field(
        default_factory=lambda: PredictionErrorConstants.DEFAULT_CONFIDENCE * 5
    )  # 0.5 = neutral starting confidence
    
    optimal_prediction_error: float = field(
        default_factory=lambda: PredictionErrorConstants.OPTIMAL_PREDICTION_ERROR_TARGET
    )  # 0.3 = sweet spot for learning
    
    prediction_error_tolerance: float = field(
        default_factory=lambda: PredictionErrorConstants.PREDICTION_ERROR_TOLERANCE
    )  # 0.05 = acceptable variance
    
    # Cognitive autopilot thresholds
    autopilot_confidence_threshold: float = field(
        default_factory=lambda: 0.90  # High confidence for autopilot
    )
    
    focused_confidence_threshold: float = field(
        default_factory=lambda: 0.70  # Moderate confidence for focused mode
    )
    
    # Spontaneous dynamics
    spontaneous_rate: float = field(
        default_factory=lambda: StabilityConstants.MIN_ACTIVATION_VALUE
    )  # 0.001 = minimal spontaneous activity
    
    resting_potential: float = field(
        default_factory=lambda: StabilityConstants.MIN_ACTIVATION_VALUE * 10
    )  # 0.01 = baseline activity
    
    # Attention and novelty
    attention_threshold: float = field(
        default_factory=lambda: PredictionErrorConstants.DEFAULT_CONFIDENCE
    )  # 0.1 = minimum for attention
    
    attention_decay_rate: float = field(
        default_factory=lambda: 1.0 - PerformancePressureConstants.PERFORMANCE_ADAPTATION_RATE * 2.5
    )  # 0.95 = slow attention decay
    
    novelty_boost: float = field(
        default_factory=lambda: PredictionErrorConstants.OPTIMAL_PREDICTION_ERROR_TARGET
    )  # 0.3 = extra attention for novel patterns
    
    # Stability parameters
    topology_stability_threshold: float = field(
        default_factory=lambda: PerformancePressureConstants.PERFORMANCE_ADAPTATION_RATE
    )  # 0.02 = topology change threshold
    
    field_energy_dissipation_rate: float = field(
        default_factory=lambda: StabilityConstants.ADAPTATION_MOMENTUM
    )  # 0.9 = energy dissipation
    
    # Constraint discovery
    constraint_discovery_rate: float = field(
        default_factory=lambda: PredictionErrorConstants.ERROR_GRADIENT_SENSITIVITY * 1.5
    )  # 0.15 = moderate discovery rate
    
    constraint_enforcement_strength: float = field(
        default_factory=lambda: PredictionErrorConstants.OPTIMAL_PREDICTION_ERROR_TARGET
    )  # 0.3 = moderate enforcement
    
    # Working memory (hardware-adaptive)
    working_memory_limit: int = field(
        default_factory=lambda: CognitiveCapacityConstants.get_working_memory_limit()
    )
    
    # Energy budget (hardware-adaptive)
    cognitive_energy_budget: int = field(
        default_factory=lambda: CognitiveEnergyConstants.get_cognitive_energy_budget()
    )
    
    # Pattern recognition
    min_pattern_length: int = field(
        default_factory=lambda: CognitiveCapacityConstants.MIN_PATTERN_LENGTH
    )
    
    max_pattern_length: int = field(
        default_factory=lambda: CognitiveCapacityConstants.MAX_PATTERN_LENGTH
    )
    
    # Pattern-based systems (coordinate-free mainline)
    pattern_attention: bool = True  # Use pattern-based attention


@dataclass 
class BlendedRealityConfig:
    """Configuration for blended reality with cognitive constants."""
    
    # Imprint strength parameters
    base_imprint_strength: float = field(
        default_factory=lambda: StabilityConstants.MAX_LEARNING_RATE
    )  # 0.5 = baseline imprint
    
    min_imprint_strength: float = field(
        default_factory=lambda: PredictionErrorConstants.DEFAULT_CONFIDENCE
    )  # 0.1 = minimum imprint
    
    # Confidence smoothing
    confidence_smoothing_rate: float = field(
        default_factory=lambda: PredictionErrorConstants.ERROR_GRADIENT_SENSITIVITY
    )  # 0.1 = moderate smoothing
    
    # Dream mode parameters
    dream_threshold_cycles: int = 100  # Cycles before dream mode
    dream_transition_rate: float = field(
        default_factory=lambda: PerformancePressureConstants.PERFORMANCE_ADAPTATION_RATE
    )  # 0.02 = gradual transition
    
    # Spontaneous dynamics blending
    max_spontaneous_weight: float = field(
        default_factory=lambda: 1.0 - AttentionMechanismConstants.HIGH_ATTENTION_RATE_THRESHOLD
    )  # 0.8 = maximum fantasy


@dataclass
class SensorProcessingConfig:
    """Configuration for confidence-based sensor processing."""
    
    # Sensor check probabilities by mode
    autopilot_sensor_probability: float = field(
        default_factory=lambda: AttentionMechanismConstants.HIGH_ATTENTION_RATE_THRESHOLD
    )  # 0.2 = check 20% in autopilot
    
    focused_sensor_probability: float = field(
        default_factory=lambda: PredictionErrorConstants.DEFAULT_PREDICTION_ERROR
    )  # 0.5 = check 50% in focused
    
    deep_think_sensor_probability: float = field(
        default_factory=lambda: StabilityConstants.ADAPTATION_MOMENTUM
    )  # 0.9 = check 90% in deep think
    
    # Confidence thresholds match brain config
    autopilot_threshold: float = 0.9
    focused_threshold: float = 0.7


class CognitiveConfigManager:
    """
    Manages cognitive configuration with hardware adaptation and environment variables.
    """
    
    def __init__(self):
        self.brain_config = BrainConfig()
        self.blended_reality_config = BlendedRealityConfig()
        self.sensor_config = SensorProcessingConfig()
        
        # Override from environment if available
        self._load_environment_overrides()
    
    def _load_environment_overrides(self):
        """Load configuration overrides from environment variables."""
        # Example: BRAIN_FIELD_EVOLUTION_RATE=0.2
        for config_obj in [self.brain_config, self.blended_reality_config, self.sensor_config]:
            config_name = config_obj.__class__.__name__.replace('Config', '').upper()
            
            for field_name in config_obj.__dataclass_fields__:
                env_var = f"BRAIN_{config_name}_{field_name.upper()}"
                if env_var in os.environ:
                    try:
                        value = os.environ[env_var]
                        # Convert to appropriate type
                        field_type = config_obj.__dataclass_fields__[field_name].type
                        if field_type == float:
                            setattr(config_obj, field_name, float(value))
                        elif field_type == int:
                            setattr(config_obj, field_name, int(value))
                        elif field_type == bool:
                            setattr(config_obj, field_name, value.lower() in ('true', '1', 'yes'))
                        print(f"🔧 Override {field_name}: {value} (from {env_var})")
                    except Exception as e:
                        print(f"⚠️  Failed to parse {env_var}: {e}")
    
    def get_brain_config_dict(self) -> Dict[str, Any]:
        """Get brain configuration as dictionary for compatibility."""
        return {
            k: getattr(self.brain_config, k) 
            for k in self.brain_config.__dataclass_fields__
        }
    
    def get_temporal_config(self) -> Dict[str, float]:
        """Get temporal configuration from cognitive constants."""
        return {
            'control_cycle_target': TemporalConstants.TARGET_CONTROL_CYCLE_TIME,
            'control_cycle_max': TemporalConstants.MAX_CONTROL_CYCLE_TIME,
            'immediate_adaptation': TemporalConstants.IMMEDIATE_ADAPTATION_WINDOW,
            'short_term_learning': TemporalConstants.SHORT_TERM_LEARNING_WINDOW,
            'memory_consolidation': TemporalConstants.MEMORY_CONSOLIDATION_INTERVAL
        }
    
    def validate_config(self) -> bool:
        """Validate configuration against cognitive boundaries."""
        # Check prediction error bounds
        if not (PredictionErrorConstants.MIN_VIABLE_PREDICTION_ERROR <= 
                self.brain_config.optimal_prediction_error <= 
                PredictionErrorConstants.MAX_VIABLE_PREDICTION_ERROR):
            print("⚠️  Optimal prediction error out of viable bounds")
            return False
        
        # Check activation bounds
        if not (StabilityConstants.MIN_ACTIVATION_VALUE <= 
                self.brain_config.activation_threshold <= 
                StabilityConstants.MAX_ACTIVATION_VALUE):
            print("⚠️  Activation threshold out of bounds")
            return False
        
        # Check confidence thresholds
        if not (0 < self.brain_config.focused_confidence_threshold < 
                self.brain_config.autopilot_confidence_threshold <= 1.0):
            print("⚠️  Invalid confidence thresholds")
            return False
        
        return True


# Global instance
_cognitive_config: Optional[CognitiveConfigManager] = None


def get_cognitive_config(quiet: bool = False) -> CognitiveConfigManager:
    """Get the global cognitive configuration instance."""
    global _cognitive_config
    if _cognitive_config is None:
        _cognitive_config = CognitiveConfigManager()
        if _cognitive_config.validate_config() and not quiet:
            print("✅ Cognitive configuration validated")
    return _cognitive_config


def reset_cognitive_config():
    """Reset the global configuration (mainly for testing)."""
    global _cognitive_config
    _cognitive_config = None