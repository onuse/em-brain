#!/usr/bin/env python3
"""
Field Brain TCP Adapter

Adapter that makes the GenericFieldBrain compatible with the existing
TCP server infrastructure. This allows the field brain to work with
the established communication protocol without changing the server.
"""

import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from .generic_brain import GenericFieldBrain, StreamCapabilities
from .field_logger import FieldBrainLogger


@dataclass
class FieldBrainConfig:
    """Configuration for field brain TCP adapter."""
    spatial_resolution: int = 20
    temporal_window: float = 10.0
    field_evolution_rate: float = 0.1
    constraint_discovery_rate: float = 0.15
    sensory_dimensions: int = 16
    motor_dimensions: int = 4
    quiet_mode: bool = False
    
    # Performance mode settings
    performance_mode: str = "balanced"
    enable_enhanced_dynamics: bool = True
    enable_attention_guidance: bool = True
    enable_hierarchical_processing: bool = True
    enable_attention_super_resolution: bool = False
    attention_base_resolution: int = 50
    attention_focus_resolution: int = 100
    hierarchical_max_time_ms: float = 40.0
    target_cycle_time_ms: float = 150.0


class FieldBrainTCPAdapter:
    """
    Adapter that makes GenericFieldBrain compatible with MinimalTCPServer.
    
    This provides the same interface as MinimalBrain but uses the generic
    field brain internally, enabling field-native intelligence over TCP.
    """
    
    def __init__(self, config: FieldBrainConfig = None):
        """Initialize the field brain TCP adapter."""
        self.config = config or FieldBrainConfig()
        
        # Initialize field-specific logger
        self.logger = FieldBrainLogger(quiet_mode=self.config.quiet_mode)
        
        # Create generic field brain
        self.field_brain = GenericFieldBrain(
            spatial_resolution=self.config.spatial_resolution,
            temporal_window=self.config.temporal_window,
            field_evolution_rate=self.config.field_evolution_rate,
            constraint_discovery_rate=self.config.constraint_discovery_rate,
            enable_enhanced_dynamics=self.config.enable_enhanced_dynamics,
            enable_attention_guidance=self.config.enable_attention_guidance,
            enable_hierarchical_processing=self.config.enable_hierarchical_processing,
            hierarchical_max_time_ms=self.config.hierarchical_max_time_ms,
            quiet_mode=self.config.quiet_mode
        )
        
        # Negotiate stream capabilities automatically
        self._negotiate_capabilities()
        
        # Tracking for MinimalBrain compatibility
        self.total_cycles = 0
        self.total_experiences = 0
        self.total_predictions = 0
        self.brain_start_time = time.time()
        
        # Interface properties
        self.sensory_dim = self.config.sensory_dimensions
        self.motor_dim = self.config.motor_dimensions
        
        # Log field brain initialization
        config_dict = {
            'spatial_resolution': self.config.spatial_resolution,
            'temporal_window': self.config.temporal_window,
            'field_evolution_rate': self.config.field_evolution_rate,
            'constraint_discovery_rate': self.config.constraint_discovery_rate
        }
        self.logger.log_initialization(config_dict)
        
        if not self.config.quiet_mode:
            print(f"🧠 FieldBrainTCPAdapter ready: {self.sensory_dim}D → {self.motor_dim}D")
    
    def _negotiate_capabilities(self):
        """Set up stream capabilities for the field brain."""
        capabilities = StreamCapabilities(
            input_dimensions=self.config.sensory_dimensions,
            output_dimensions=self.config.motor_dimensions,
            input_labels=[f"sensor_{i}" for i in range(self.config.sensory_dimensions)],
            output_labels=[f"action_{i}" for i in range(self.config.motor_dimensions)],
            input_ranges=[(-1.0, 1.0)] * self.config.sensory_dimensions,
            output_ranges=[(-1.0, 1.0)] * self.config.motor_dimensions,
            update_frequency_hz=20.0,  # Typical robot update rate
            latency_ms=50.0            # Expected network latency
        )
        
        success = self.field_brain.negotiate_stream_capabilities(capabilities)
        if not success:
            raise RuntimeError("Failed to negotiate stream capabilities with field brain")
    
    def process_sensory_input(self, sensory_input: List[float], 
                            action_dimensions: int = 4) -> Tuple[List[float], Dict[str, Any]]:
        """
        Process sensory input through the field brain and return action commands.
        
        This is the main interface that the TCP server expects.
        """
        # Validate input
        if len(sensory_input) != self.sensory_dim:
            # Pad or truncate to expected dimensions
            if len(sensory_input) < self.sensory_dim:
                sensory_input = sensory_input + [0.0] * (self.sensory_dim - len(sensory_input))
            else:
                sensory_input = sensory_input[:self.sensory_dim]
        
        # Validate action dimensions
        if action_dimensions != self.motor_dim:
            if not self.config.quiet_mode:
                print(f"⚠️ Action dimension mismatch: requested {action_dimensions}, using {self.motor_dim}")
        
        # Process through field brain
        cycle_start_time = time.time()
        try:
            output_stream, brain_state = self.field_brain.process_input_stream(sensory_input)
            
            # Log learning progress metrics
            if brain_state:
                # Log field energy for learning tracking
                field_energy = brain_state.get('field_total_energy', 0.0)
                if field_energy > 0:
                    self.logger.log_field_energy(field_energy)
                
                # Log field coordinates for exploration tracking
                field_coordinates = brain_state.get('field_coordinates')
                if field_coordinates is not None:
                    self.logger.log_field_coordinates(field_coordinates)
            
            # Calculate cycle timing
            cycle_time_ms = (time.time() - cycle_start_time) * 1000.0
            
            # Ensure output has correct dimensions
            if len(output_stream) != self.motor_dim:
                if len(output_stream) < self.motor_dim:
                    output_stream = output_stream + [0.0] * (self.motor_dim - len(output_stream))
                else:
                    output_stream = output_stream[:self.motor_dim]
            
            # Update tracking
            self.total_cycles += 1
            self.total_experiences += 1
            self.total_predictions += 1
            
            # Log field-specific metrics
            sparse_regions = brain_state.get('active_regions', 0)
            field_update_time = brain_state.get('field_update_time_ms', 0.0)
            self.logger.log_field_cycle(cycle_time_ms, sparse_regions, field_update_time)
            
            # Check for field evolution/consolidation events
            if brain_state.get('field_evolved', False):
                self.logger.log_field_evolution()
            
            if brain_state.get('field_consolidated', False):
                compression_ratio = brain_state.get('compression_ratio', 0.0)
                memory_usage = brain_state.get('memory_usage_mb', 0.0)
                self.logger.log_field_consolidation(compression_ratio, memory_usage)
            
            # Periodic performance reporting
            self.logger.maybe_report_performance()
            
            # Create compatible brain state response
            brain_state_response = {
                'prediction_method': 'field_dynamics',
                'prediction_confidence': min(1.0, brain_state.get('field_total_energy', 0.0) / 1000.0),
                'field_energy': brain_state.get('field_total_energy', 0.0),
                'field_activation': brain_state.get('field_mean_activation', 0.0),
                'topology_regions': brain_state.get('topology_regions', 0),
                'brain_cycles': brain_state.get('brain_cycles', 0),
                'processing_mode': 'field_native',
                'dynamics_families': brain_state.get('family_activities', {}),
                'stream_status': brain_state.get('stream_capabilities', {})
            }
            
            return output_stream, brain_state_response
            
        except Exception as e:
            # Fallback response on error
            cycle_time_ms = (time.time() - cycle_start_time) * 1000.0
            self.logger.log_error(f"Field processing failed: {str(e)}", f"Cycle {self.total_cycles}")
            
            # Still log the cycle time for performance tracking
            self.logger.log_field_cycle(cycle_time_ms, 0, 0.0)
            
            # Return neutral action
            neutral_action = [0.0] * self.motor_dim
            error_state = {
                'prediction_method': 'field_error_fallback',
                'prediction_confidence': 0.0,
                'error': str(e),
                'field_energy': 0.0,
                'processing_mode': 'error_recovery'
            }
            
            return neutral_action, error_state
    
    def store_experience(self, sensory_input: List[float], action_taken: List[float], 
                        outcome: List[float], predicted_action: List[float]) -> str:
        """
        Store experience for learning (compatibility method).
        
        The field brain handles learning intrinsically through field evolution,
        so this is mainly for TCP server compatibility.
        """
        # The field brain learns intrinsically through field imprinting
        # We don't need separate experience storage
        experience_id = f"field_exp_{self.total_experiences}_{int(time.time())}"
        
        if not self.config.quiet_mode and self.total_experiences < 3:
            print(f"💾 Field brain experience: {experience_id} (intrinsic learning)")
        
        return experience_id
    
    def get_brain_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics for TCP server monitoring."""
        
        # Get field brain stats
        field_stats = self.field_brain.get_field_memory_stats()
        brain_state = self.field_brain._get_brain_state()
        
        # Calculate uptime and performance metrics
        uptime = time.time() - self.brain_start_time
        cycles_per_minute = self.total_cycles / max(1, uptime / 60)
        
        return {
            'brain_summary': {
                'total_cycles': self.total_cycles,
                'total_experiences': self.total_experiences, 
                'total_predictions': self.total_predictions,
                'uptime_seconds': uptime,
                'cycles_per_minute': cycles_per_minute,
                'brain_type': 'field_native',
                'processing_mode': 'unified_field_dynamics'
            },
            'field_brain': {
                'field_dimensions': self.field_brain.total_dimensions,
                'spatial_resolution': self.field_brain.spatial_resolution,
                'field_energy': field_stats.get('field_energy', 0.0),
                'max_activation': field_stats.get('max_activation', 0.0),
                'mean_activation': field_stats.get('mean_activation', 0.0),
                'memory_size_mb': field_stats.get('memory_size_mb', 0.0),
                'topology_regions': field_stats.get('topology_regions', 0),
                'nonzero_elements': field_stats.get('nonzero_elements', 0),
                'sparsity': field_stats.get('sparsity', 1.0),
                'field_evolution_cycles': field_stats.get('field_evolution_cycles', 0)
            },
            'dynamics_families': brain_state.get('family_activities', {}),
            'stream_interface': {
                'input_dimensions': self.sensory_dim,
                'output_dimensions': self.motor_dim,
                'capabilities_negotiated': brain_state.get('stream_capabilities', {}).get('negotiated', False),
                'adaptive_mappings': True
            },
            'performance': {
                'last_cycle_time_ms': brain_state.get('last_cycle_time_ms', 0.0),
                'field_processing': 'real_time',
                'memory_efficiency': 'intrinsic_field_persistence',
                'learning_type': 'continuous_field_evolution'
            }
        }
    
    def get_brain_statistics(self) -> Dict[str, Any]:
        """Get brain statistics (alias for compatibility with MinimalBrain interface)."""
        return self.get_brain_stats()
    
    def save_field_state(self, filepath: str) -> bool:
        """Save the field brain state (delegate to field brain)."""
        return self.field_brain.save_field_state(filepath, compress=True)
    
    def load_field_state(self, filepath: str) -> bool:
        """Load the field brain state (delegate to field brain)."""
        success = self.field_brain.load_field_state(filepath)
        if success:
            # Re-negotiate capabilities after loading
            self._negotiate_capabilities()
        return success
    
    def consolidate_field(self, strength: float = 0.1) -> int:
        """Perform field consolidation (delegate to field brain)."""
        if hasattr(self.field_brain, 'consolidate_field'):
            return self.field_brain.consolidate_field(strength)
        return 0
    
    def get_field_capabilities(self) -> Dict[str, Any]:
        """Get field brain capabilities for introspection."""
        return {
            'brain_type': 'unified_field_native',
            'field_dimensions': self.field_brain.total_dimensions,
            'spatial_resolution': self.field_brain.spatial_resolution,
            'temporal_window': self.field_brain.temporal_window,
            'stream_interface': {
                'input_dimensions': self.sensory_dim,
                'output_dimensions': self.motor_dim,
                'adaptive_mapping': True,
                'capability_negotiation': True
            },
            'memory_system': 'intrinsic_field_topology',
            'learning_system': 'continuous_field_evolution',
            'persistence': 'field_state_compression',
            'biological_optimizations': True,
            'tcp_compatibility': True
        }
    
    def __str__(self) -> str:
        return f"FieldBrainTCPAdapter({self.sensory_dim}D→{self.motor_dim}D, {self.total_cycles} cycles)"
    
    def __repr__(self) -> str:
        return self.__str__()


def create_field_brain_tcp_adapter(sensory_dimensions: int = 16,
                                   motor_dimensions: int = 4,
                                   spatial_resolution: int = 20,
                                   quiet_mode: bool = False) -> FieldBrainTCPAdapter:
    """
    Create a field brain TCP adapter with specified configuration.
    
    Args:
        sensory_dimensions: Number of input sensor dimensions
        motor_dimensions: Number of output motor dimensions  
        spatial_resolution: Field brain spatial resolution
        quiet_mode: Suppress initialization output
        
    Returns:
        Configured field brain TCP adapter
    """
    config = FieldBrainConfig(
        sensory_dimensions=sensory_dimensions,
        motor_dimensions=motor_dimensions,
        spatial_resolution=spatial_resolution,
        quiet_mode=quiet_mode
    )
    
    return FieldBrainTCPAdapter(config)


if __name__ == "__main__":
    # Test the adapter
    print("🧪 Testing FieldBrainTCPAdapter...")
    
    # Create adapter
    adapter = create_field_brain_tcp_adapter(
        sensory_dimensions=16,
        motor_dimensions=4,
        spatial_resolution=8,
        quiet_mode=False
    )
    
    print(f"\n🧠 Testing TCP compatibility:")
    print(f"   Adapter: {adapter}")
    print(f"   Interface: {adapter.sensory_dim}D sensors → {adapter.motor_dim}D actions")
    
    # Test processing
    print(f"\n🌊 Testing sensory processing:")
    for i in range(3):
        # Create test sensory input
        sensory_input = [0.5 + 0.2 * (i % 3) + j * 0.1 for j in range(16)]
        
        # Process through adapter
        actions, brain_state = adapter.process_sensory_input(sensory_input)
        
        print(f"   Cycle {i+1}:")
        print(f"      Input:  {[f'{x:.2f}' for x in sensory_input[:4]]}...")
        print(f"      Output: {[f'{x:.2f}' for x in actions]}")
        print(f"      Method: {brain_state['prediction_method']}")
        print(f"      Confidence: {brain_state['prediction_confidence']:.3f}")
    
    # Test statistics
    print(f"\n📊 Testing brain statistics:")
    stats = adapter.get_brain_stats()
    print(f"   Total cycles: {stats['brain_summary']['total_cycles']}")
    print(f"   Field energy: {stats['field_brain']['field_energy']:.3f}")
    print(f"   Processing mode: {stats['brain_summary']['processing_mode']}")
    print(f"   Stream interface: {stats['stream_interface']['input_dimensions']}D→{stats['stream_interface']['output_dimensions']}D")
    
    print(f"\n✅ FieldBrainTCPAdapter test complete!")
    print(f"   Ready for TCP server integration")
    print(f"   Compatible with existing communication protocol")
    print(f"   Field-native intelligence over TCP sockets")