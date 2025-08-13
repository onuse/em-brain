#!/usr/bin/env python3
"""
Field Brain Behavioral Test Implementation

Tests the PureFieldBrain's emergent behaviors through the brain-brainstem interface,
focusing on field dynamics and self-modification capabilities.

This test verifies that intelligence emerges from field tensions rather than
programmed behaviors.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import queue

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent / "server"))

# Import behavioral test framework
from behavioral_test_strategy import (
    BehavioralTestSuite,
    BehavioralAnalyzer,
    SafetyVerifier,
    LearningProgressionTracker,
    BehaviorCategory,
    TestScenario
)

# Import brainstem components
from brainstem.integrated_brainstem import IntegratedBrainstem, BrainstemConfig
from brainstem.brain_client import BrainServerConfig

# Import telemetry for monitoring field dynamics
try:
    from src.core.telemetry_client import TelemetryClient
except ImportError:
    TelemetryClient = None


# ==================== FIELD-SPECIFIC METRICS ====================

@dataclass
class FieldDynamicsMetric:
    """Metrics specific to field brain dynamics"""
    name: str
    description: str
    expected_behavior: str
    
    def evaluate_field_state(self, field_telemetry: Dict) -> Dict[str, Any]:
        """Evaluate field dynamics from telemetry"""
        
        if not field_telemetry:
            return {'error': 'No field telemetry available'}
            
        evaluation = {
            'metric': self.name,
            'timestamp': time.time()
        }
        
        # Extract field-specific measurements
        if self.name == "field_coherence":
            evaluation['value'] = self._measure_coherence(field_telemetry)
        elif self.name == "gradient_flow":
            evaluation['value'] = self._measure_gradient_flow(field_telemetry)
        elif self.name == "self_modification_rate":
            evaluation['value'] = self._measure_self_modification(field_telemetry)
        elif self.name == "regional_specialization":
            evaluation['value'] = self._measure_specialization(field_telemetry)
        elif self.name == "prediction_tension":
            evaluation['value'] = self._measure_prediction_tension(field_telemetry)
            
        return evaluation
    
    def _measure_coherence(self, telemetry: Dict) -> float:
        """Measure field coherence (should be stable but not rigid)"""
        field_variance = telemetry.get('field_variance', 1.0)
        # Optimal coherence: not too rigid (var > 0.1) but not chaotic (var < 2.0)
        if 0.1 < field_variance < 2.0:
            return 1.0 - abs(field_variance - 0.5) / 1.5
        return 0.0
    
    def _measure_gradient_flow(self, telemetry: Dict) -> float:
        """Measure gradient flow (indicates active processing)"""
        gradient_magnitude = telemetry.get('gradient_magnitude', 0.0)
        # Good flow: moderate gradients (not flat, not explosive)
        if 0.01 < gradient_magnitude < 1.0:
            return min(1.0, gradient_magnitude * 2)
        return 0.0
    
    def _measure_self_modification(self, telemetry: Dict) -> float:
        """Measure self-modification through evolution channels"""
        evolution_activity = telemetry.get('evolution_channel_activity', 0.0)
        # Healthy self-modification: active but controlled
        return min(1.0, evolution_activity / 0.5)
    
    def _measure_specialization(self, telemetry: Dict) -> float:
        """Measure regional specialization in field"""
        regional_variance = telemetry.get('regional_variance', 0.0)
        # Specialization emerges as regions differentiate
        return min(1.0, regional_variance / 0.3)
    
    def _measure_prediction_tension(self, telemetry: Dict) -> float:
        """Measure prediction error creating tension for learning"""
        prediction_error = telemetry.get('prediction_error', 1.0)
        # Moderate error drives learning
        if 0.1 < prediction_error < 0.6:
            return 1.0 - abs(prediction_error - 0.3) / 0.3
        return 0.2  # Some baseline tension


# ==================== FIELD BRAIN TEST SCENARIOS ====================

class FieldEmergenceScenario(TestScenario):
    """Test emergence of behavior from field dynamics"""
    
    def __init__(self):
        super().__init__(
            name="Field Emergence Test",
            description="Verifies behaviors emerge from field tensions, not programming",
            duration_cycles=200,
            required_behaviors=[
                BehaviorCategory.HOMEOSTASIS,  # Field should self-stabilize
                BehaviorCategory.EXPLORATORY,  # Curiosity from field tensions
                BehaviorCategory.PREDICTIVE    # Prediction through field dynamics
            ],
            prohibited_behaviors=[
                BehaviorCategory.CATATONIC,    # Field shouldn't freeze
                BehaviorCategory.PATHOLOGICAL  # No self-destructive patterns
            ],
            metrics=[],  # Will use field-specific metrics
            environmental_conditions={
                'stimulus_complexity': 'gradual',
                'field_perturbation': 'minimal'
            }
        )
        
        # Add field-specific metrics
        self.field_metrics = [
            FieldDynamicsMetric(
                "field_coherence",
                "Stability of field patterns",
                "Stable but flexible patterns"
            ),
            FieldDynamicsMetric(
                "gradient_flow",
                "Information flow through gradients",
                "Active gradient dynamics"
            ),
            FieldDynamicsMetric(
                "self_modification_rate",
                "Evolution channel activity",
                "Controlled self-modification"
            )
        ]
    
    def generate_sensory_sequence(self, cycle: int) -> List[float]:
        """Generate gradually complexifying sensory patterns"""
        
        sensors = [0.0] * 16
        
        # Start with simple periodic pattern
        if cycle < 50:
            sensors[0] = 0.5 + 0.3 * np.sin(cycle * 0.1)
            sensors[1:4] = [0.5, 0.5, 0.5]
            
        # Add spatial structure
        elif cycle < 100:
            phase = cycle * 0.1
            sensors[0] = 0.5 + 0.3 * np.sin(phase)
            sensors[1] = 0.5 + 0.2 * np.sin(phase + np.pi/3)
            sensors[2] = 0.5 + 0.2 * np.sin(phase + 2*np.pi/3)
            sensors[3] = 0.5 + 0.2 * np.sin(phase + np.pi)
            
        # Introduce novelty and complexity
        else:
            # Complex but learnable pattern
            for i in range(4):
                sensors[i] = 0.5 + 0.3 * np.sin(cycle * 0.1 + i * np.pi/4)
            # Occasional surprises
            if cycle % 20 == 0:
                sensors[np.random.randint(4, 16)] = np.random.random()
                
        return sensors


class SelfModificationScenario(TestScenario):
    """Test the brain's ability to self-modify through experience"""
    
    def __init__(self):
        super().__init__(
            name="Self-Modification Test",
            description="Verifies brain modifies itself through evolution channels",
            duration_cycles=300,
            required_behaviors=[
                BehaviorCategory.ADAPTIVE,
                BehaviorCategory.HABITUAL
            ],
            prohibited_behaviors=[
                BehaviorCategory.OSCILLATORY
            ],
            metrics=[],
            environmental_conditions={
                'task_switching': True,
                'reward_structure': 'implicit'
            }
        )
        
        self.field_metrics = [
            FieldDynamicsMetric(
                "self_modification_rate",
                "Evolution parameter changes",
                "Active self-modification"
            ),
            FieldDynamicsMetric(
                "regional_specialization",
                "Development of specialized regions",
                "Increasing specialization"
            )
        ]
        
        self.current_task = 0
        self.task_switch_cycles = [75, 150, 225]
    
    def generate_sensory_sequence(self, cycle: int) -> List[float]:
        """Generate task-switching scenarios requiring adaptation"""
        
        # Switch tasks at predetermined cycles
        if cycle in self.task_switch_cycles:
            self.current_task = (self.current_task + 1) % 3
            print(f"      Task switch to: {['navigation', 'tracking', 'avoidance'][self.current_task]}")
        
        sensors = [0.0] * 16
        
        if self.current_task == 0:  # Navigation task
            # Line following pattern
            deviation = np.sin(cycle * 0.05)
            sensors[1] = 0.3 if deviation < -0.5 else 0.7
            sensors[2] = 0.7 if abs(deviation) < 0.5 else 0.3
            sensors[3] = 0.3 if deviation > 0.5 else 0.7
            sensors[0] = 0.5  # Clear path
            
        elif self.current_task == 1:  # Tracking task
            # Moving target to track
            target_pos = 0.5 + 0.4 * np.sin(cycle * 0.03)
            sensors[0] = target_pos  # Distance varies with target
            sensors[1:4] = [target_pos - 0.1, target_pos, target_pos + 0.1]
            
        else:  # Avoidance task
            # Obstacles appearing periodically
            if (cycle % 30) < 15:
                sensors[0] = 0.2  # Close obstacle
                sensors[1] = 0.8  # Obstacle on right
            else:
                sensors[0] = 0.8  # Clear ahead
                sensors[3] = 0.8  # Obstacle on left
                
        return sensors


class IntrinsicMotivationScenario(TestScenario):
    """Test intrinsic curiosity and exploration without rewards"""
    
    def __init__(self):
        super().__init__(
            name="Intrinsic Motivation Test",
            description="Verifies exploration driven by internal field tensions",
            duration_cycles=250,
            required_behaviors=[
                BehaviorCategory.EXPLORATORY,
                BehaviorCategory.CREATIVE
            ],
            prohibited_behaviors=[
                BehaviorCategory.CATATONIC
            ],
            metrics=[],
            environmental_conditions={
                'external_rewards': 'none',
                'environment': 'open_ended'
            }
        )
        
        self.field_metrics = [
            FieldDynamicsMetric(
                "prediction_tension",
                "Prediction error driving exploration",
                "Maintained prediction tension"
            ),
            FieldDynamicsMetric(
                "gradient_flow",
                "Active information seeking",
                "Sustained gradient activity"
            )
        ]
        
        self.discovered_patterns = set()
        self.environment_state = np.random.RandomState(42)
    
    def generate_sensory_sequence(self, cycle: int) -> List[float]:
        """Generate open environment with discoverable patterns"""
        
        sensors = [0.0] * 16
        
        # Base environmental noise
        sensors = self.environment_state.random(16) * 0.2
        
        # Hidden patterns to discover (no explicit reward)
        # Pattern 1: Periodic structure in space
        x = (cycle % 100) / 100.0
        if 0.3 < x < 0.7:
            sensors[0] += 0.5
            self.discovered_patterns.add("spatial_pattern_1")
            
        # Pattern 2: Temporal rhythm
        if cycle % 17 == 0:
            sensors[1:4] = [0.8, 0.8, 0.8]
            self.discovered_patterns.add("temporal_pattern_1")
            
        # Pattern 3: Correlation structure
        if sensors[5] > 0.15:  # Contingent on random value
            sensors[6] = sensors[5] * 2
            sensors[7] = sensors[5] * 3
            self.discovered_patterns.add("correlation_pattern_1")
            
        # Pattern 4: Emerges from robot's own behavior
        # (would need motor history in real implementation)
        
        return sensors.tolist()


# ==================== FIELD BRAIN ANALYZER ====================

class FieldBrainAnalyzer(BehavioralAnalyzer):
    """Extended analyzer for field brain specifics"""
    
    def __init__(self):
        super().__init__()
        self.field_history: List[Dict] = []
        self.emergence_markers: List[Dict] = []
        
    def analyze_field_dynamics(self, telemetry: Dict) -> Dict[str, Any]:
        """Analyze field-specific dynamics"""
        
        analysis = {
            'timestamp': time.time(),
            'field_health': self._assess_field_health(telemetry),
            'emergence_indicators': self._detect_emergence(telemetry),
            'self_organization': self._measure_self_organization(telemetry)
        }
        
        self.field_history.append(analysis)
        
        # Check for emergence milestones
        for indicator, value in analysis['emergence_indicators'].items():
            if value > 0.7:  # Threshold for emergence
                self.emergence_markers.append({
                    'indicator': indicator,
                    'value': value,
                    'cycle': len(self.field_history)
                })
                print(f"      🌟 Emergence detected: {indicator} = {value:.2f}")
                
        return analysis
    
    def _assess_field_health(self, telemetry: Dict) -> str:
        """Assess overall field health"""
        
        # Check key field indicators
        variance = telemetry.get('field_variance', 0)
        gradients = telemetry.get('gradient_magnitude', 0)
        evolution = telemetry.get('evolution_activity', 0)
        
        if variance < 0.01:
            return "frozen"
        elif variance > 10:
            return "chaotic"
        elif gradients < 0.001:
            return "flat"
        elif evolution > 2.0:
            return "unstable_evolution"
        elif 0.1 < variance < 2.0 and 0.01 < gradients < 1.0:
            return "healthy"
        else:
            return "suboptimal"
    
    def _detect_emergence(self, telemetry: Dict) -> Dict[str, float]:
        """Detect emergence signatures in field dynamics"""
        
        indicators = {}
        
        # Spontaneous pattern formation
        if 'regional_variance' in telemetry:
            indicators['pattern_formation'] = min(1.0, telemetry['regional_variance'] / 0.5)
        
        # Self-organized criticality
        if 'gradient_cascade_events' in telemetry:
            # Power law distribution indicates criticality
            indicators['criticality'] = telemetry.get('criticality_score', 0.0)
        
        # Hierarchical organization
        if 'cross_scale_correlation' in telemetry:
            indicators['hierarchy'] = telemetry['cross_scale_correlation']
        
        # Predictive coherence
        if 'prediction_accuracy' in telemetry:
            indicators['predictive_coherence'] = telemetry['prediction_accuracy']
            
        return indicators
    
    def _measure_self_organization(self, telemetry: Dict) -> float:
        """Measure degree of self-organization"""
        
        # Combine multiple indicators
        organization = 0.0
        weights = 0.0
        
        if 'entropy_reduction' in telemetry:
            organization += telemetry['entropy_reduction'] * 2.0
            weights += 2.0
            
        if 'pattern_stability' in telemetry:
            organization += telemetry['pattern_stability'] * 1.5
            weights += 1.5
            
        if 'information_integration' in telemetry:
            organization += telemetry['information_integration'] * 1.0
            weights += 1.0
            
        return organization / weights if weights > 0 else 0.0
    
    def generate_field_report(self) -> Dict[str, Any]:
        """Generate field-specific analysis report"""
        
        if not self.field_history:
            return {'error': 'No field data collected'}
            
        # Analyze field health over time
        health_states = [h['field_health'] for h in self.field_history]
        health_distribution = {}
        for state in health_states:
            health_distribution[state] = health_distribution.get(state, 0) + 1
            
        # Count emergence events
        emergence_summary = {}
        for marker in self.emergence_markers:
            indicator = marker['indicator']
            emergence_summary[indicator] = emergence_summary.get(indicator, 0) + 1
        
        # Calculate self-organization trend
        org_values = [h['self_organization'] for h in self.field_history[-50:]]
        org_trend = "increasing" if len(org_values) > 1 and org_values[-1] > org_values[0] else "stable"
        
        return {
            'total_cycles': len(self.field_history),
            'health_distribution': health_distribution,
            'emergence_events': emergence_summary,
            'self_organization_trend': org_trend,
            'emergence_markers': self.emergence_markers[:10],  # Top 10
            'recommendation': self._field_recommendation(health_distribution, emergence_summary)
        }
    
    def _field_recommendation(self, health: Dict, emergence: Dict) -> str:
        """Generate field-specific recommendation"""
        
        healthy_ratio = health.get('healthy', 0) / sum(health.values()) if health else 0
        
        if healthy_ratio < 0.5:
            return "⚠️ Field dynamics unhealthy - adjust parameters"
        elif len(emergence) < 2:
            return "🔄 Limited emergence - extend testing or adjust field size"
        elif len(emergence) >= 5:
            return "✅ Rich emergent dynamics observed"
        else:
            return "✓ Field showing healthy emergence"


# ==================== INTEGRATED FIELD TEST ====================

class FieldBrainBehavioralTest:
    """Complete behavioral test for field brain system"""
    
    def __init__(self, use_real_brain: bool = True):
        self.use_real_brain = use_real_brain
        
        # Initialize components
        self.brainstem_config = BrainstemConfig(
            brain_server_config=BrainServerConfig(
                host="localhost",
                port=9999
            ),
            use_mock_brain=not use_real_brain,
            enable_local_reflexes=True,
            safety_override=True,
            update_rate_hz=20.0
        )
        
        self.brainstem = IntegratedBrainstem(self.brainstem_config)
        self.analyzer = FieldBrainAnalyzer()
        self.safety_verifier = SafetyVerifier()
        self.learning_tracker = LearningProgressionTracker()
        
        # Telemetry for field monitoring
        self.telemetry_client = TelemetryClient() if TelemetryClient else None
        
        # Test scenarios
        self.scenarios = [
            FieldEmergenceScenario(),
            SelfModificationScenario(),
            IntrinsicMotivationScenario()
        ]
        
    def run_test(self, total_cycles: int = 750) -> Dict[str, Any]:
        """Run complete field brain behavioral test"""
        
        print("\n" + "="*60)
        print("🧠 FIELD BRAIN BEHAVIORAL TEST")
        print("="*60)
        print(f"Testing PureFieldBrain emergent behaviors")
        print(f"Mode: {'Real Brain Server' if self.use_real_brain else 'Mock Brain'}")
        
        # Connect to brain
        if self.use_real_brain:
            connected = self.brainstem.connect()
            if not connected:
                print("❌ Failed to connect to brain server")
                print("   Start server with: python3 server/brain.py")
                return {'error': 'Connection failed'}
                
            # Connect telemetry
            if self.telemetry_client:
                if self.telemetry_client.connect():
                    print("✅ Telemetry connected for field monitoring")
                else:
                    print("⚠️  Telemetry unavailable - limited field analysis")
        
        results = {
            'test_type': 'field_brain_behavioral',
            'start_time': time.time(),
            'scenarios': {}
        }
        
        cycle_count = 0
        
        # Run scenarios
        for scenario in self.scenarios:
            print(f"\n📋 Scenario: {scenario.name}")
            print(f"   {scenario.description}")
            
            scenario_results = self._run_field_scenario(
                scenario, cycle_count
            )
            results['scenarios'][scenario.name] = scenario_results
            cycle_count += scenario.duration_cycles
            
            # Check for critical issues
            if scenario_results.get('critical_failure'):
                print("   🚨 Critical failure detected")
                break
        
        # Generate reports
        results['behavioral_analysis'] = self.analyzer.generate_report()
        results['field_analysis'] = self.analyzer.generate_field_report()
        results['safety_report'] = self.safety_verifier.generate_safety_report()
        results['learning_analysis'] = self.learning_tracker.analyze_progression()
        
        # Print summary
        self._print_field_summary(results)
        
        return results
    
    def _run_field_scenario(
        self,
        scenario: Any,
        start_cycle: int
    ) -> Dict[str, Any]:
        """Run a field-specific scenario"""
        
        results = {
            'name': scenario.name,
            'field_metrics': {},
            'behavioral_observations': [],
            'critical_failure': False
        }
        
        for cycle in range(scenario.duration_cycles):
            global_cycle = start_cycle + cycle
            
            # Generate sensory input
            sensors = scenario.generate_sensory_sequence(cycle)
            
            # Process through brainstem
            motor_response = self.brainstem.process_cycle(sensors)
            
            # Analyze behavior
            behavior = self.analyzer.analyze_motor_pattern(
                motor_response.get('motors', [0, 0, 0, 0])
            )
            results['behavioral_observations'].append(behavior['behavior'].value)
            
            # Get field telemetry if available
            if self.telemetry_client and global_cycle % 10 == 0:
                telemetry = self._get_field_telemetry()
                if telemetry:
                    field_analysis = self.analyzer.analyze_field_dynamics(telemetry)
                    
                    # Evaluate field metrics
                    if hasattr(scenario, 'field_metrics'):
                        for metric in scenario.field_metrics:
                            evaluation = metric.evaluate_field_state(telemetry)
                            results['field_metrics'][metric.name] = evaluation
            
            # Safety checks
            safe = self.safety_verifier.verify_collision_avoidance(
                sensors[0], motor_response.get('motors', []), global_cycle
            )
            
            if not safe and cycle < 10:
                results['critical_failure'] = True
            
            # Progress update
            if cycle % 50 == 0:
                print(f"      Cycle {cycle}/{scenario.duration_cycles}")
                if hasattr(scenario, 'discovered_patterns'):
                    print(f"      Patterns discovered: {len(scenario.discovered_patterns)}")
        
        return results
    
    def _get_field_telemetry(self) -> Optional[Dict]:
        """Get field telemetry from brain server"""
        
        if not self.telemetry_client:
            return None
            
        try:
            # Get session for our brainstem
            sessions = self.telemetry_client.get_all_sessions()
            if sessions:
                session_id = sessions[0]  # Use first available
                telemetry = self.telemetry_client.get_session_telemetry(session_id)
                
                if telemetry:
                    # Extract field-specific data
                    return {
                        'field_variance': telemetry.field_stats.get('variance', 0),
                        'gradient_magnitude': telemetry.gradient_stats.get('magnitude', 0),
                        'prediction_error': telemetry.prediction_error,
                        'evolution_activity': telemetry.evolution_stats.get('activity', 0),
                        'regional_variance': telemetry.field_stats.get('regional_variance', 0),
                        'confidence': telemetry.confidence
                    }
        except Exception as e:
            pass  # Telemetry not critical
            
        return None
    
    def _print_field_summary(self, results: Dict[str, Any]):
        """Print field-specific test summary"""
        
        print("\n" + "="*60)
        print("📊 FIELD BRAIN TEST SUMMARY")
        print("="*60)
        
        # Field analysis
        field = results.get('field_analysis', {})
        print("\n🌊 Field Dynamics:")
        for state, count in field.get('health_distribution', {}).items():
            print(f"   {state}: {count} cycles")
        
        print("\n✨ Emergence Events:")
        for indicator, count in field.get('emergence_events', {}).items():
            print(f"   {indicator}: {count} occurrences")
        
        # Behavioral analysis
        behavioral = results.get('behavioral_analysis', {})
        print("\n🧠 Observed Behaviors:")
        achievements = behavioral.get('achievements', [])
        for achievement in achievements[:5]:
            print(f"   ✅ {achievement}")
            
        concerns = behavioral.get('concerns', [])
        for concern in concerns[:3]:
            print(f"   ⚠️ {concern}")
        
        # Safety
        safety = results.get('safety_report', {})
        print(f"\n🛡️ Safety Performance:")
        print(f"   Score: {safety.get('safety_score', 0):.1%}")
        print(f"   {safety.get('recommendation', '')}")
        
        # Final assessment
        print("\n🎯 Field Brain Assessment:")
        print(f"   {field.get('recommendation', 'No field data available')}")
        print(f"   {behavioral.get('recommendation', '')}")
        
        # Check for true emergence
        emergence_count = len(field.get('emergence_events', {}))
        if emergence_count >= 3:
            print("\n   ✨ TRUE EMERGENCE ACHIEVED - Behaviors arising from field dynamics!")
        elif emergence_count > 0:
            print("\n   🌱 Emergence beginning - Continue development")
        else:
            print("\n   ⏳ No clear emergence yet - May need more cycles or parameter tuning")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Field Brain Behavioral Test")
    parser.add_argument('--mock', action='store_true', help='Use mock brain instead of real server')
    parser.add_argument('--cycles', type=int, default=750, help='Total test cycles')
    args = parser.parse_args()
    
    # Run test
    test = FieldBrainBehavioralTest(use_real_brain=not args.mock)
    results = test.run_test(total_cycles=args.cycles)
    
    # Save results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"field_brain_test_{timestamp}.json"
    
    # Convert non-serializable objects
    def make_serializable(obj):
        if hasattr(obj, 'value'):
            return obj.value
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    with open(filename, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    
    print(f"\n📁 Results saved to: {filename}")