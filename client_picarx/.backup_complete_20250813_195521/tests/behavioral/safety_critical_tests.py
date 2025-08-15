#!/usr/bin/env python3
"""
Safety-Critical Behavioral Tests

Ensures the brainstem safety reflexes function correctly even when:
- Brain server is unavailable
- Brain produces dangerous commands
- Sensors indicate critical conditions
- System is under stress

These tests are MANDATORY before deployment on physical hardware.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from brainstem.integrated_brainstem import IntegratedBrainstem, BrainstemConfig
from brainstem.brain_client import BrainServerConfig


# ==================== SAFETY TEST CATEGORIES ====================

class SafetyTestCategory(Enum):
    """Categories of safety-critical tests"""
    COLLISION_PREVENTION = "collision_prevention"
    CLIFF_AVOIDANCE = "cliff_avoidance"
    THERMAL_PROTECTION = "thermal_protection"
    POWER_MANAGEMENT = "power_management"
    EMERGENCY_STOP = "emergency_stop"
    SENSOR_FAILURE = "sensor_failure"
    BRAIN_FAILURE = "brain_failure"
    CONTRADICTORY_COMMANDS = "contradictory_commands"


@dataclass
class SafetyTestCase:
    """Individual safety test case"""
    name: str
    category: SafetyTestCategory
    description: str
    sensor_pattern: List[float]
    expected_response: str
    max_response_time_ms: int
    is_critical: bool  # If True, failure means DO NOT DEPLOY
    
    def evaluate_response(
        self,
        motor_response: Dict[str, Any],
        response_time_ms: float
    ) -> Tuple[bool, str]:
        """Evaluate if response meets safety requirements"""
        
        passed = True
        details = []
        
        # Check response time
        if response_time_ms > self.max_response_time_ms:
            passed = False
            details.append(f"Response too slow: {response_time_ms:.1f}ms > {self.max_response_time_ms}ms")
        
        # Check motor response based on test type
        motors = motor_response.get('motors', [0, 0, 0, 0])
        
        if self.category == SafetyTestCategory.COLLISION_PREVENTION:
            # Should stop or reverse
            if motors[0] > 0.1:  # Still moving forward
                passed = False
                details.append(f"Failed to stop: forward motor = {motors[0]:.2f}")
                
        elif self.category == SafetyTestCategory.CLIFF_AVOIDANCE:
            # Should stop and possibly back up
            if motors[0] > 0 or motors[1] > 0:
                passed = False
                details.append(f"Moving toward cliff: motors = {motors[:2]}")
                
        elif self.category == SafetyTestCategory.EMERGENCY_STOP:
            # All motors should be zero
            total_motion = sum(abs(m) for m in motors)
            if total_motion > 0.01:
                passed = False
                details.append(f"Failed to stop: total motion = {total_motion:.3f}")
                
        elif self.category == SafetyTestCategory.THERMAL_PROTECTION:
            # Should reduce activity
            total_motion = sum(abs(m) for m in motors)
            if total_motion > 0.3:
                passed = False
                details.append(f"Excessive motion during thermal event: {total_motion:.2f}")
        
        return passed, " | ".join(details) if details else "OK"


# ==================== SAFETY TEST SCENARIOS ====================

class CollisionPreventionTests:
    """Test collision prevention reflexes"""
    
    @staticmethod
    def get_test_cases() -> List[SafetyTestCase]:
        return [
            SafetyTestCase(
                name="Immediate obstacle",
                category=SafetyTestCategory.COLLISION_PREVENTION,
                description="Object detected 10cm ahead",
                sensor_pattern=[0.1] + [0.5]*15,  # Ultrasonic shows 10cm
                expected_response="Stop or reverse",
                max_response_time_ms=100,
                is_critical=True
            ),
            SafetyTestCase(
                name="Approaching wall",
                category=SafetyTestCategory.COLLISION_PREVENTION,
                description="Wall approaching at 20cm",
                sensor_pattern=[0.2] + [0.5]*15,
                expected_response="Slow down significantly",
                max_response_time_ms=200,
                is_critical=True
            ),
            SafetyTestCase(
                name="Side obstacle",
                category=SafetyTestCategory.COLLISION_PREVENTION,
                description="Obstacle on right side",
                sensor_pattern=[0.5, 0.9, 0.5, 0.3] + [0.5]*12,
                expected_response="Turn left",
                max_response_time_ms=300,
                is_critical=False
            )
        ]


class CliffAvoidanceTests:
    """Test cliff detection reflexes"""
    
    @staticmethod
    def get_test_cases() -> List[SafetyTestCase]:
        return [
            SafetyTestCase(
                name="Cliff ahead",
                category=SafetyTestCategory.CLIFF_AVOIDANCE,
                description="Cliff detected in front",
                sensor_pattern=[2.0] + [0.5]*10 + [1.0] + [0.5]*4,  # Cliff sensor active
                expected_response="Immediate stop and reverse",
                max_response_time_ms=50,
                is_critical=True
            ),
            SafetyTestCase(
                name="Edge detection",
                category=SafetyTestCategory.CLIFF_AVOIDANCE,
                description="Approaching table edge",
                sensor_pattern=[0.8] + [0.5]*10 + [0.7] + [0.5]*4,  # Partial cliff
                expected_response="Stop and turn",
                max_response_time_ms=100,
                is_critical=True
            )
        ]


class EmergencyStopTests:
    """Test emergency stop functionality"""
    
    @staticmethod
    def get_test_cases() -> List[SafetyTestCase]:
        return [
            SafetyTestCase(
                name="Emergency stop signal",
                category=SafetyTestCategory.EMERGENCY_STOP,
                description="Emergency stop triggered",
                sensor_pattern=[0.5]*15 + [1.0],  # Last channel is emergency
                expected_response="All motors stop",
                max_response_time_ms=20,
                is_critical=True
            ),
            SafetyTestCase(
                name="Multiple critical sensors",
                category=SafetyTestCategory.EMERGENCY_STOP,
                description="Multiple safety violations",
                sensor_pattern=[0.05, 0.9, 0.9, 0.9] + [0.5]*7 + [1.0] + [0.5]*4,
                expected_response="Complete stop",
                max_response_time_ms=30,
                is_critical=True
            )
        ]


class BrainFailureTests:
    """Test behavior when brain fails"""
    
    @staticmethod
    def get_test_cases() -> List[SafetyTestCase]:
        # These tests will be run with disconnected brain
        return [
            SafetyTestCase(
                name="Brain offline navigation",
                category=SafetyTestCategory.BRAIN_FAILURE,
                description="Navigate with brain disconnected",
                sensor_pattern=[0.3, 0.5, 0.7, 0.5] + [0.5]*12,
                expected_response="Basic obstacle avoidance",
                max_response_time_ms=500,
                is_critical=False
            ),
            SafetyTestCase(
                name="Brain offline emergency",
                category=SafetyTestCategory.BRAIN_FAILURE,
                description="Emergency during brain failure",
                sensor_pattern=[0.1] + [0.5]*15,
                expected_response="Local reflex activation",
                max_response_time_ms=100,
                is_critical=True
            )
        ]


class StressTests:
    """Test under adverse conditions"""
    
    @staticmethod
    def get_test_cases() -> List[SafetyTestCase]:
        return [
            SafetyTestCase(
                name="Sensor noise",
                category=SafetyTestCategory.SENSOR_FAILURE,
                description="Noisy sensor data",
                sensor_pattern=[np.random.random() for _ in range(16)],
                expected_response="Cautious movement",
                max_response_time_ms=200,
                is_critical=False
            ),
            SafetyTestCase(
                name="Contradictory signals",
                category=SafetyTestCategory.CONTRADICTORY_COMMANDS,
                description="Conflicting sensor inputs",
                sensor_pattern=[0.1, 0.1, 0.9, 0.1] + [0.5]*12,  # Close but path clear?
                expected_response="Prioritize safety",
                max_response_time_ms=150,
                is_critical=True
            ),
            SafetyTestCase(
                name="Thermal overload",
                category=SafetyTestCategory.THERMAL_PROTECTION,
                description="High temperature detected",
                sensor_pattern=[0.5]*12 + [75.0] + [0.5]*3,  # CPU temp = 75°C
                expected_response="Reduce activity",
                max_response_time_ms=1000,
                is_critical=False
            )
        ]


# ==================== SAFETY TEST RUNNER ====================

class SafetyCriticalTestRunner:
    """Runs comprehensive safety tests"""
    
    def __init__(self, brainstem: Optional[IntegratedBrainstem] = None):
        """Initialize test runner
        
        Args:
            brainstem: Existing brainstem instance, or None to create one
        """
        
        if brainstem:
            self.brainstem = brainstem
        else:
            # Create brainstem with mock brain for isolated testing
            config = BrainstemConfig(
                brain_server_config=BrainServerConfig(
                    host="localhost",
                    port=9999
                ),
                use_mock_brain=True,  # Use mock for safety tests
                enable_local_reflexes=True,
                safety_override=True,
                update_rate_hz=50.0  # Higher rate for safety testing
            )
            self.brainstem = IntegratedBrainstem(config)
        
        # Collect all test cases
        self.test_cases = (
            CollisionPreventionTests.get_test_cases() +
            CliffAvoidanceTests.get_test_cases() +
            EmergencyStopTests.get_test_cases() +
            BrainFailureTests.get_test_cases() +
            StressTests.get_test_cases()
        )
        
        self.results = {
            'passed': [],
            'failed': [],
            'critical_failures': [],
            'response_times': []
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all safety tests"""
        
        print("\n" + "="*60)
        print("🛡️ SAFETY-CRITICAL TEST SUITE")
        print("="*60)
        print(f"Running {len(self.test_cases)} safety tests")
        print("CRITICAL tests must pass for deployment\n")
        
        for i, test_case in enumerate(self.test_cases, 1):
            result = self._run_single_test(test_case, i)
            
            if not result['passed']:
                self.results['failed'].append(result)
                if test_case.is_critical:
                    self.results['critical_failures'].append(result)
                    print(f"   🚨 CRITICAL FAILURE: {test_case.name}")
            else:
                self.results['passed'].append(result)
            
            self.results['response_times'].append(result['response_time_ms'])
            
            # Brief pause between tests
            time.sleep(0.1)
        
        # Generate summary
        summary = self._generate_summary()
        self._print_summary(summary)
        
        return summary
    
    def _run_single_test(
        self,
        test_case: SafetyTestCase,
        test_num: int
    ) -> Dict[str, Any]:
        """Run a single safety test"""
        
        print(f"[{test_num}/{len(self.test_cases)}] {test_case.name}")
        print(f"   Category: {test_case.category.value}")
        print(f"   Critical: {'YES' if test_case.is_critical else 'No'}")
        
        # Special handling for brain failure tests
        if test_case.category == SafetyTestCategory.BRAIN_FAILURE:
            # Ensure brain is disconnected
            if self.brainstem.brain_client.is_connected():
                self.brainstem.brain_client.disconnect()
                time.sleep(0.1)
        
        # Measure response time
        start_time = time.perf_counter()
        
        # Process sensor input through brainstem
        motor_response = self.brainstem.process_cycle(test_case.sensor_pattern)
        
        response_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Evaluate response
        passed, details = test_case.evaluate_response(motor_response, response_time_ms)
        
        # Check if local reflex was activated
        reflex_activated = self.brainstem.reflex_active
        
        result = {
            'test_name': test_case.name,
            'category': test_case.category.value,
            'is_critical': test_case.is_critical,
            'passed': passed,
            'response_time_ms': response_time_ms,
            'reflex_activated': reflex_activated,
            'motor_response': motor_response.get('motors', []),
            'details': details
        }
        
        # Print result
        if passed:
            print(f"   ✅ PASSED ({response_time_ms:.1f}ms) - {details}")
        else:
            print(f"   ❌ FAILED ({response_time_ms:.1f}ms) - {details}")
        
        if reflex_activated:
            print(f"   🔄 Local reflex activated")
        
        return result
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate test summary"""
        
        total_tests = len(self.test_cases)
        passed_count = len(self.results['passed'])
        failed_count = len(self.results['failed'])
        critical_failures = len(self.results['critical_failures'])
        
        # Calculate statistics
        avg_response_time = np.mean(self.results['response_times'])
        max_response_time = np.max(self.results['response_times'])
        
        # Group failures by category
        failures_by_category = {}
        for failure in self.results['failed']:
            category = failure['category']
            failures_by_category[category] = failures_by_category.get(category, 0) + 1
        
        # Determine deployment readiness
        deployment_ready = critical_failures == 0
        
        # Generate recommendations
        recommendations = []
        if critical_failures > 0:
            recommendations.append("🚨 DO NOT DEPLOY - Critical safety failures detected")
        
        if avg_response_time > 200:
            recommendations.append("⚠️ Response times too slow for real-time safety")
        
        if failed_count > total_tests * 0.2:
            recommendations.append("⚠️ High failure rate - review safety logic")
        
        if not recommendations:
            recommendations.append("✅ System meets safety requirements")
        
        return {
            'total_tests': total_tests,
            'passed': passed_count,
            'failed': failed_count,
            'critical_failures': critical_failures,
            'pass_rate': passed_count / total_tests,
            'avg_response_time_ms': avg_response_time,
            'max_response_time_ms': max_response_time,
            'failures_by_category': failures_by_category,
            'deployment_ready': deployment_ready,
            'recommendations': recommendations,
            'detailed_results': self.results
        }
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print test summary"""
        
        print("\n" + "="*60)
        print("📊 SAFETY TEST SUMMARY")
        print("="*60)
        
        print(f"\n📈 Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        print(f"   Failed: {summary['failed']}")
        print(f"   Critical Failures: {summary['critical_failures']}")
        
        print(f"\n⏱️ Performance:")
        print(f"   Average Response: {summary['avg_response_time_ms']:.1f}ms")
        print(f"   Maximum Response: {summary['max_response_time_ms']:.1f}ms")
        
        if summary['failures_by_category']:
            print(f"\n❌ Failures by Category:")
            for category, count in summary['failures_by_category'].items():
                print(f"   {category}: {count}")
        
        print(f"\n🎯 Recommendations:")
        for rec in summary['recommendations']:
            print(f"   {rec}")
        
        print(f"\n{'='*60}")
        if summary['deployment_ready']:
            print("✅ DEPLOYMENT READY - All critical safety tests passed")
        else:
            print("🚨 NOT READY FOR DEPLOYMENT - Critical safety issues detected")
        print("="*60)


# ==================== CONTINUOUS SAFETY MONITOR ====================

class ContinuousSafetyMonitor:
    """Monitors safety during regular operation"""
    
    def __init__(self, brainstem: IntegratedBrainstem):
        self.brainstem = brainstem
        self.violation_history = []
        self.reflex_history = []
        self.monitoring = False
        
    def start_monitoring(self):
        """Start continuous safety monitoring"""
        self.monitoring = True
        print("🔍 Safety monitoring active")
    
    def check_cycle(
        self,
        sensors: List[float],
        motor_response: Dict[str, Any],
        cycle: int
    ) -> Dict[str, Any]:
        """Check safety for a single cycle"""
        
        if not self.monitoring:
            return {}
        
        safety_status = {
            'cycle': cycle,
            'violations': [],
            'warnings': [],
            'reflex_active': self.brainstem.reflex_active
        }
        
        motors = motor_response.get('motors', [0, 0, 0, 0])
        
        # Check collision risk
        if sensors[0] < 0.15 and motors[0] > 0.2:
            safety_status['violations'].append("Moving toward obstacle")
        
        # Check cliff risk
        if sensors[11] > 0.5 and (motors[0] > 0 or motors[1] > 0):
            safety_status['violations'].append("Moving toward cliff")
        
        # Check thermal
        if len(sensors) > 12 and sensors[12] > 70:
            if sum(abs(m) for m in motors) > 0.5:
                safety_status['warnings'].append("High activity during thermal event")
        
        # Track violations
        if safety_status['violations']:
            self.violation_history.append(safety_status)
        
        if safety_status['reflex_active']:
            self.reflex_history.append(cycle)
        
        return safety_status
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get accumulated safety metrics"""
        
        return {
            'total_violations': len(self.violation_history),
            'recent_violations': len([v for v in self.violation_history if v['cycle'] > max(0, cycle - 100)]),
            'reflex_activations': len(self.reflex_history),
            'monitoring_active': self.monitoring
        }


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Safety-Critical Tests")
    parser.add_argument('--real-brain', action='store_true', 
                       help='Test with real brain server (default: mock)')
    parser.add_argument('--quick', action='store_true',
                       help='Run only critical tests')
    args = parser.parse_args()
    
    # Configure brainstem
    if args.real_brain:
        print("Testing with REAL brain server...")
        config = BrainstemConfig(
            brain_server_config=BrainServerConfig(
                host="localhost",
                port=9999
            ),
            use_mock_brain=False,
            enable_local_reflexes=True,
            safety_override=True,
            update_rate_hz=50.0
        )
    else:
        print("Testing with MOCK brain (safety reflexes only)...")
        config = BrainstemConfig(
            brain_server_config=BrainServerConfig(
                host="localhost",
                port=9999
            ),
            use_mock_brain=True,
            enable_local_reflexes=True,
            safety_override=True,
            update_rate_hz=50.0
        )
    
    brainstem = IntegratedBrainstem(config)
    
    # Run safety tests
    runner = SafetyCriticalTestRunner(brainstem)
    
    if args.quick:
        # Filter to critical tests only
        runner.test_cases = [tc for tc in runner.test_cases if tc.is_critical]
        print(f"Running {len(runner.test_cases)} CRITICAL tests only")
    
    results = runner.run_all_tests()
    
    # Save results
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"safety_test_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        # Remove non-serializable items
        clean_results = {k: v for k, v in results.items() if k != 'detailed_results'}
        json.dump(clean_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {filename}")
    
    # Exit with appropriate code
    sys.exit(0 if results['deployment_ready'] else 1)