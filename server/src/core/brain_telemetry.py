"""
Brain Telemetry Interface

Provides clean, read-only access to brain internals for monitoring and testing.
This is completely separate from the cognitive pipeline and does not interfere
with normal brain operation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time


@dataclass
class BrainTelemetry:
    """Complete telemetry snapshot from a brain"""
    # Core metrics
    brain_cycles: int
    field_energy: float
    prediction_confidence: float
    cognitive_mode: str
    
    # Memory and learning
    working_memory_size: int
    active_constraints: int
    memory_regions: int
    experiences_stored: int
    
    # Field dynamics
    field_dimensions: int
    tensor_dimensions: int
    max_activation: float
    field_stability: float
    
    # Performance
    last_cycle_time_ms: float
    average_cycle_time_ms: float
    
    # Advanced features
    blended_reality_ratio: Dict[str, float]  # reality vs fantasy weights
    attention_focus: Optional[List[float]]   # Current attention coordinates
    phase_state: str                         # stable, chaotic, etc.
    
    # Prediction system
    prediction_error: Optional[float]
    prediction_history: List[float]          # Recent confidence values
    improvement_rate: float
    
    # Timestamp
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class IBrainTelemetryProvider(ABC):
    """Interface for objects that can provide brain telemetry"""
    
    @abstractmethod
    def get_telemetry(self) -> BrainTelemetry:
        """Get current telemetry snapshot"""
        pass
    
    @abstractmethod
    def get_telemetry_history(self, num_samples: int = 10) -> List[BrainTelemetry]:
        """Get recent telemetry history"""
        pass


class BrainTelemetryAdapter:
    """
    Adapter that extracts telemetry from DynamicUnifiedFieldBrain.
    
    This keeps telemetry extraction logic separate from the brain itself,
    maintaining clean separation of concerns.
    """
    
    def __init__(self, brain):
        """
        Initialize adapter with reference to brain.
        
        Args:
            brain: DynamicUnifiedFieldBrain instance or DynamicBrainWrapper
        """
        # Handle wrapped brains
        if hasattr(brain, 'brain'):
            # This is a wrapper, get the actual brain
            self.brain = brain.brain
        else:
            # Direct brain reference
            self.brain = brain
        self.telemetry_history = []
        self.max_history = 100
    
    def get_telemetry(self) -> BrainTelemetry:
        """Extract current telemetry from brain"""
        # Get brain state
        brain_state = self.brain.get_brain_state() if hasattr(self.brain, 'get_brain_state') else {}
        
        # Get field statistics
        field_stats = self.brain.get_field_statistics() if hasattr(self.brain, 'get_field_statistics') else {}
        
        # Get blended reality state
        blend_state = {'reality_weight': 1.0, 'fantasy_weight': 0.0}
        if hasattr(self.brain, 'blended_reality') and self.brain.blended_reality:
            blend_state = self.brain.blended_reality.get_blend_state()
        
        # Get prediction history
        pred_history = []
        if hasattr(self.brain, '_prediction_confidence_history'):
            pred_history = list(self.brain._prediction_confidence_history)[-10:]
        
        # Calculate improvement rate
        improvement_rate = 0.0
        if len(pred_history) >= 10:
            early = sum(pred_history[:5]) / 5
            late = sum(pred_history[-5:]) / 5
            improvement_rate = late - early
        
        # Get phase state
        phase_state = "unknown"
        if hasattr(self.brain, 'enhanced_dynamics') and self.brain.enhanced_dynamics:
            phase_state = self.brain.enhanced_dynamics.current_phase
        
        # Create telemetry snapshot
        telemetry = BrainTelemetry(
            # Core metrics
            brain_cycles=getattr(self.brain, 'brain_cycles', 0),
            field_energy=field_stats.get('field_energy', 0.0),
            prediction_confidence=getattr(self.brain, '_current_prediction_confidence', 0.5),
            cognitive_mode=brain_state.get('cognitive_mode', 'unknown'),
            
            # Memory and learning
            working_memory_size=len(getattr(self.brain, 'working_memory', [])),
            active_constraints=brain_state.get('active_constraints', 0),
            memory_regions=len(getattr(self.brain, 'memory_regions', [])),
            experiences_stored=len(getattr(self.brain, 'experiences', [])),
            
            # Field dynamics
            field_dimensions=getattr(self.brain, 'total_dimensions', 0),
            tensor_dimensions=len(getattr(self.brain, 'tensor_shape', [])),
            max_activation=field_stats.get('max_activation', 0.0),
            field_stability=field_stats.get('stability_index', 0.0),
            
            # Performance
            last_cycle_time_ms=brain_state.get('cycle_time_ms', 0.0),
            average_cycle_time_ms=0.0,  # Would need to track this
            
            # Advanced features
            blended_reality_ratio={
                'reality': blend_state.get('reality_weight', 1.0),
                'fantasy': blend_state.get('fantasy_weight', 0.0)
            },
            attention_focus=None,  # Could extract from attention system
            phase_state=phase_state,
            
            # Prediction system
            prediction_error=getattr(self.brain, '_last_prediction_error', None),
            prediction_history=pred_history,
            improvement_rate=improvement_rate
        )
        
        # Store in history
        self.telemetry_history.append(telemetry)
        if len(self.telemetry_history) > self.max_history:
            self.telemetry_history.pop(0)
        
        return telemetry
    
    def get_telemetry_history(self, num_samples: int = 10) -> List[BrainTelemetry]:
        """Get recent telemetry history"""
        return self.telemetry_history[-num_samples:]
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get a summary of telemetry for quick monitoring"""
        telemetry = self.get_telemetry()
        
        return {
            'cycles': telemetry.brain_cycles,
            'energy': round(telemetry.field_energy, 6),
            'confidence': round(telemetry.prediction_confidence, 3),
            'mode': telemetry.cognitive_mode,
            'phase': telemetry.phase_state,
            'memory_regions': telemetry.memory_regions,
            'constraints': telemetry.active_constraints,
            'blend': f"{telemetry.blended_reality_ratio['reality']:.0%}R/{telemetry.blended_reality_ratio['fantasy']:.0%}F"
        }


class SessionTelemetryWrapper:
    """
    Wraps a BrainSession to provide telemetry access.
    """
    
    def __init__(self, session):
        """
        Initialize wrapper.
        
        Args:
            session: BrainSession instance
        """
        self.session = session
        self.telemetry_adapter = None
        
        # Create telemetry adapter if brain supports it
        if hasattr(session, 'brain'):
            self.telemetry_adapter = BrainTelemetryAdapter(session.brain)
    
    def get_telemetry(self) -> Optional[BrainTelemetry]:
        """Get telemetry from wrapped session"""
        if self.telemetry_adapter:
            return self.telemetry_adapter.get_telemetry()
        return None
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get telemetry summary"""
        if self.telemetry_adapter:
            return self.telemetry_adapter.get_telemetry_summary()
        return {'error': 'No telemetry available'}