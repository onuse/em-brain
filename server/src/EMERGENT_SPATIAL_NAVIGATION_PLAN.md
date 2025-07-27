# Emergent Spatial Navigation Implementation Plan

## Vision

Transform the brain's current coordinate-based navigation into a truly emergent system where spatial concepts arise from field dynamics, not from programmed coordinates.

## Current State Analysis

### Existing Embryonic Systems

1. **Enhanced Dynamics (enhanced_dynamics.py)**
   - ✓ Attractor/repulsor dynamics
   - ✓ Phase transitions based on field energy
   - ✓ Energy redistribution mechanisms
   - Missing: Attractors as navigational "places"

2. **Spontaneous Dynamics (spontaneous_dynamics.py)**
   - ✓ Traveling waves in field
   - ✓ Coherent internal activity patterns
   - ✓ Homeostatic field maintenance
   - Missing: Wave patterns encoding spatial relationships

3. **Topology Regions (in core_brain.py)**
   - ✓ Tracks high-activation stable regions
   - ✓ Memory formation at specific locations
   - ✗ But uses absolute coordinates!
   - Missing: Relationships between regions

4. **Field Evolution**
   - ✓ Natural field decay and diffusion
   - ✓ Gradient-based dynamics
   - ✗ But gradients are in coordinate space!
   - Missing: Field evolution as movement intention

5. **Sensorimotor Coupling (robot_interface.py)**
   - ✓ Correlates sensory and motor signals
   - ✗ But still maps to fixed coordinates
   - Missing: Movement emerging from field dynamics

### Critical Gap: Field-Motor Translation

Current approach:
```
Sensory → Coordinates → Field Position → Spatial Gradients → Motors
```

Desired approach:
```
Sensory → Field Impression → Field Evolution → Motor Emergence
```

## Implementation Plan

### Phase 1: Remove Coordinate Dependencies

1. **Modify Field Imprinting**
   - Stop mapping sensors to spatial coordinates
   - Let full sensory pattern create field impressions
   - Use pattern similarity for "nearness" not coordinate distance

2. **Replace Gradient Following**
   - Current: Calculate gradients in spatial dimensions
   - New: Let field evolution naturally create motor tendencies
   - Movement emerges from field seeking lower energy states

### Phase 2: Implement Field-Based "Places"

1. **Enhance Topology Regions**
   ```python
   class EmergentPlace:
       field_signature: torch.Tensor  # Full field state, not coordinates
       stability_score: float         # How stable this configuration is
       visit_count: int              # Strengthens with revisits
       connections: Dict[str, float]  # Relationships to other places
   ```

2. **Place Recognition**
   - Use field similarity (cosine distance) not coordinate distance
   - Places emerge from stable attractors in enhanced dynamics
   - Recognition strengthens place stability

3. **Place Relationships**
   - Learn transitions between places through experience
   - No fixed topology - relationships are learned
   - Distance = effort/transitions, not meters

### Phase 3: Emergent Navigation

1. **Field Tension Navigation**
   ```python
   def compute_navigational_tension(self, current_field, desired_place):
       # Field naturally wants to evolve toward remembered states
       tension = desired_place.field_signature - current_field
       # This tension drives field evolution
       return tension
   ```

2. **Motor Emergence from Field Evolution**
   ```python
   def field_evolution_to_motor(self, field_delta):
       # Different field dynamics couple to different motor aspects
       
       # Oscillatory patterns → forward/backward rhythm
       oscillatory_change = extract_oscillatory_component(field_delta)
       forward_drive = oscillatory_to_forward_mapping(oscillatory_change)
       
       # Flow patterns → turning tendencies  
       flow_change = extract_flow_component(field_delta)
       turning_drive = flow_to_rotation_mapping(flow_change)
       
       # Energy redistribution → speed modulation
       energy_change = extract_energy_component(field_delta)
       speed_modulation = energy_to_speed_mapping(energy_change)
       
       return combine_motor_drives(forward_drive, turning_drive, speed_modulation)
   ```

### Phase 4: Integration Points

1. **With Attention System**
   - Attention creates temporary attractors
   - Salient features become navigational landmarks
   - Attention modulates place stability

2. **With Enhanced Dynamics**
   - Phase transitions mark place boundaries
   - Attractors ARE places
   - Energy landscape defines navigability

3. **With Spontaneous Dynamics**
   - Traveling waves explore possible transitions
   - Spontaneous activity maintains place memories
   - Wave patterns encode spatial relationships

## Key Implementation Files to Modify

1. **dynamic_unified_brain_full.py**
   - Remove coordinate mapping in `_robot_sensors_to_field_experience`
   - Replace gradient-based navigation in `_unified_field_to_robot_action`
   - Enhance `topology_regions` to store field signatures

2. **Create: emergent_spatial_dynamics.py**
   - Implement field-based place detection
   - Create motor emergence from field evolution
   - Build topological navigation layer

3. **Enhance: enhanced_dynamics.py**
   - Make attractors explicitly spatial
   - Add place persistence mechanisms
   - Implement field tension dynamics

4. **Modify: robot_interface.py**
   - Remove coordinate calculations
   - Implement pattern-based sensory mapping
   - Create emergent motor coupling

## Success Metrics

1. **Navigation without coordinates**
   - Robot can return to previously visited "places"
   - Places defined by field states, not positions

2. **Emergent spatial relationships**
   - Brain learns which places are "near" through experience
   - Spatial topology emerges from navigation patterns

3. **Natural movement**
   - Movement arises from field evolution
   - No explicit path planning needed

4. **Generalization**
   - Can navigate in new environments
   - Uses learned spatial concepts, not memorized coordinates

## Risks and Mitigations

1. **Risk**: Complete loss of navigation ability
   - **Mitigation**: Implement incrementally, maintain coordinate fallback

2. **Risk**: Computational overhead
   - **Mitigation**: Use existing field dynamics, minimize new computations

3. **Risk**: Learning too slow
   - **Mitigation**: Bootstrap with enhanced dynamics attractors

## Next Steps

1. Create proof-of-concept in emergent_spatial_dynamics.py
2. Test with simple navigation scenarios
3. Gradually reduce coordinate dependencies
4. Validate emergent behaviors
5. Full integration once proven

This plan transforms navigation from programmed behavior to emergent phenomenon, aligning with the biological inspiration of the project.