#!/usr/bin/env python3
"""
Comprehensive Brain Test Suite
==============================
Tests the INFRASTRUCTURE thoroughly while letting intelligence EMERGE freely.

Philosophy:
- Test that the plumbing works, not what the brain learns
- Validate safety boundaries without constraining behavior
- Catch implementation bugs before they affect the robot
- Monitor for pathological states that could damage hardware
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

from src.brains.field.pure_field_brain import create_pure_field_brain
from src.brains.field.optimized_field_ops import OptimizedFieldOps


@dataclass
class TestResult:
    """Result of a single test"""
    name: str
    passed: bool
    message: str
    severity: str  # 'critical', 'warning', 'info'
    metrics: Dict[str, Any] = None


class BrainTestSuite:
    """
    Comprehensive test suite for PureFieldBrain.
    
    Tests are organized by criticality:
    1. CRITICAL: Must pass or robot could be damaged
    2. IMPORTANT: Should pass for proper operation
    3. DIAGNOSTIC: Helpful for debugging and optimization
    """
    
    def __init__(self, device: str = None, verbose: bool = True):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.results: List[TestResult] = []
        
    def run_all_tests(self, brain_size: str = 'tiny') -> Dict[str, Any]:
        """Run complete test suite"""
        print("="*60)
        print("🧠 PureFieldBrain Comprehensive Test Suite")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Brain size: {brain_size}")
        print()
        
        # Critical Safety Tests
        self._run_critical_tests(brain_size)
        
        # Field Dynamics Tests
        self._run_field_dynamics_tests(brain_size)
        
        # Performance Tests
        self._run_performance_tests(brain_size)
        
        # Emergence Monitoring
        self._run_emergence_tests(brain_size)
        
        # Summarize results
        return self._summarize_results()
    
    def _run_critical_tests(self, brain_size: str):
        """Tests that MUST pass for safe robot operation"""
        print("\n🚨 CRITICAL SAFETY TESTS")
        print("-"*40)
        
        # Test 1: Motor output bounds
        result = self._test_motor_bounds(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test 2: NaN/Inf propagation
        result = self._test_nan_handling(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test 3: Memory stability
        result = self._test_memory_stability(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test 4: Gradient explosion
        result = self._test_gradient_stability(brain_size)
        self.results.append(result)
        self._print_result(result)
    
    def _test_motor_bounds(self, brain_size: str) -> TestResult:
        """Ensure motor outputs NEVER exceed safe bounds"""
        brain = create_pure_field_brain(size=brain_size, device=self.device)
        
        # Test with extreme inputs
        test_cases = [
            torch.zeros(10, device=self.device),  # Zero input
            torch.ones(10, device=self.device) * 1000,  # Huge input
            torch.randn(10, device=self.device) * 100,  # Large random
            torch.ones(10, device=self.device) * float('inf'),  # Infinity
        ]
        
        all_safe = True
        max_output = 0
        
        for i, test_input in enumerate(test_cases):
            output = brain(test_input)
            
            # Check bounds [-1, 1]
            if output.abs().max().item() > 1.0:
                all_safe = False
                
            max_output = max(max_output, output.abs().max().item())
        
        return TestResult(
            name="Motor Output Bounds",
            passed=all_safe,
            message=f"Max output: {max_output:.3f} (must be ≤1.0)",
            severity="critical",
            metrics={"max_motor_output": max_output}
        )
    
    def _test_nan_handling(self, brain_size: str) -> TestResult:
        """Ensure NaN/Inf inputs don't propagate"""
        brain = create_pure_field_brain(size=brain_size, device=self.device)
        
        # Create input with NaN and Inf
        bad_input = torch.randn(10, device=self.device)
        bad_input[2] = float('nan')
        bad_input[5] = float('inf')
        bad_input[7] = float('-inf')
        
        # Process
        output = brain(bad_input)
        
        # Check output is clean
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        passed = not (has_nan or has_inf)
        
        return TestResult(
            name="NaN/Inf Handling",
            passed=passed,
            message="Output clean" if passed else "NaN/Inf propagated!",
            severity="critical"
        )
    
    def _test_memory_stability(self, brain_size: str) -> TestResult:
        """Test for memory leaks over many cycles"""
        brain = create_pure_field_brain(size=brain_size, device=self.device)
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            start_memory = torch.cuda.memory_allocated()
        
        # Run many cycles
        for _ in range(100):
            input_data = torch.randn(10, device=self.device)
            _ = brain(input_data)
        
        if self.device == 'cuda':
            end_memory = torch.cuda.memory_allocated()
            memory_growth = (end_memory - start_memory) / 1024**2  # MB
            
            # Allow up to 150MB growth (initial GPU allocation and caching)
            # This is mostly one-time allocation, not a leak
            passed = memory_growth < 150.0
            message = f"Memory growth: {memory_growth:.2f}MB"
        else:
            passed = True
            message = "CPU memory (not measured)"
            memory_growth = 0
        
        return TestResult(
            name="Memory Stability",
            passed=passed,
            message=message,
            severity="critical",
            metrics={"memory_growth_mb": memory_growth}
        )
    
    def _test_gradient_stability(self, brain_size: str) -> TestResult:
        """Ensure gradients don't explode"""
        brain = create_pure_field_brain(size=brain_size, device=self.device, aggressive=True)
        
        max_gradient = 0
        field_explosion = False
        
        # Run with aggressive learning
        for i in range(50):
            input_data = torch.randn(10, device=self.device) * 2
            output = brain(input_data)
            
            # Check field magnitude
            field_norm = torch.norm(brain.field).item()
            if field_norm > 1000 or torch.isnan(brain.field).any():
                field_explosion = True
                break
                
            max_gradient = max(max_gradient, field_norm)
        
        passed = not field_explosion
        
        return TestResult(
            name="Gradient Stability",
            passed=passed,
            message=f"Max field norm: {max_gradient:.1f}",
            severity="critical",
            metrics={"max_field_norm": max_gradient}
        )
    
    def _run_field_dynamics_tests(self, brain_size: str):
        """Test field dynamics behavior"""
        print("\n🔬 FIELD DYNAMICS TESTS")
        print("-"*40)
        
        # Test evolution convergence
        result = self._test_field_evolution(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test cross-scale coherence
        result = self._test_cross_scale_flow(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test diffusion behavior
        result = self._test_diffusion_behavior(brain_size)
        self.results.append(result)
        self._print_result(result)
    
    def _test_field_evolution(self, brain_size: str) -> TestResult:
        """Test that field evolves over time"""
        brain = create_pure_field_brain(size=brain_size, device=self.device)
        
        # Capture initial field state
        initial_field = brain.field.clone()
        
        # Run several cycles
        for _ in range(10):
            input_data = torch.randn(10, device=self.device)
            _ = brain(input_data)
        
        # Check field changed
        field_change = (brain.field - initial_field).abs().mean().item()
        
        # Field should change but not explode
        passed = 0.001 < field_change < 10.0
        
        return TestResult(
            name="Field Evolution",
            passed=passed,
            message=f"Mean field change: {field_change:.4f}",
            severity="important",
            metrics={"field_change": field_change}
        )
    
    def _test_cross_scale_flow(self, brain_size: str) -> TestResult:
        """Test information flow between scales (if multi-level)"""
        # Use 'large' to ensure multiple levels
        if brain_size in ['tiny', 'small', 'medium']:
            return TestResult(
                name="Cross-Scale Flow",
                passed=True,
                message="Single level (N/A)",
                severity="diagnostic"
            )
        
        brain = create_pure_field_brain(size='large', device=self.device)
        
        # Run some cycles to establish patterns
        for _ in range(20):
            input_data = torch.randn(10, device=self.device)
            _ = brain(input_data)
        
        # Check cross-scale coherence metric
        coherence = brain.emergence_metrics.get('cross_scale_coherence', 0)
        
        # Should have some coherence but not perfect (that would indicate no dynamics)
        passed = 0.1 < coherence < 0.9
        
        return TestResult(
            name="Cross-Scale Flow",
            passed=passed,
            message=f"Coherence: {coherence:.3f}",
            severity="diagnostic",
            metrics={"coherence": coherence}
        )
    
    def _test_diffusion_behavior(self, brain_size: str) -> TestResult:
        """Test that diffusion smooths appropriately"""
        ops = OptimizedFieldOps(self.device)
        
        # Create sharp field
        field = torch.zeros(1, 32, 16, 16, 16, device=self.device)
        field[0, :, 8, 8, 8] = 1.0  # Point source
        
        # Apply diffusion
        diffused = ops.optimized_diffusion(field, diffusion_rate=0.5)
        
        # Check smoothing occurred
        max_before = field.max().item()
        max_after = diffused.max().item()
        spread = (diffused > 0.01).sum().item()
        
        # Should spread and reduce peak
        passed = max_after < max_before and spread > 100
        
        return TestResult(
            name="Diffusion Behavior",
            passed=passed,
            message=f"Peak: {max_before:.2f}→{max_after:.2f}, Spread: {spread}",
            severity="diagnostic",
            metrics={"peak_reduction": max_before - max_after, "spread": spread}
        )
    
    def _run_performance_tests(self, brain_size: str):
        """Test performance characteristics"""
        print("\n⚡ PERFORMANCE TESTS")
        print("-"*40)
        
        # Test cycle time
        result = self._test_cycle_time(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test optimization speedup
        result = self._test_optimization_speedup(brain_size)
        self.results.append(result)
        self._print_result(result)
    
    def _test_cycle_time(self, brain_size: str) -> TestResult:
        """Measure brain cycle time"""
        brain = create_pure_field_brain(size=brain_size, device=self.device)
        
        # Warmup
        for _ in range(10):
            _ = brain(torch.randn(10, device=self.device))
        
        # Time cycles
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        cycles = 100
        
        for _ in range(cycles):
            _ = brain(torch.randn(10, device=self.device))
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        ms_per_cycle = (elapsed / cycles) * 1000
        
        # Target: <10ms on dev hardware (will be <1ms on production)
        passed = ms_per_cycle < 10.0 if self.device == 'cuda' else True
        
        return TestResult(
            name="Cycle Time",
            passed=passed,
            message=f"{ms_per_cycle:.2f}ms/cycle ({1000/ms_per_cycle:.1f}Hz)",
            severity="important",
            metrics={"ms_per_cycle": ms_per_cycle, "hz": 1000/ms_per_cycle}
        )
    
    def _test_optimization_speedup(self, brain_size: str) -> TestResult:
        """Test that optimizations provide speedup"""
        if self.device != 'cuda':
            return TestResult(
                name="Optimization Speedup",
                passed=True,
                message="GPU required",
                severity="diagnostic"
            )
        
        # Create test field
        field = torch.randn(1, 64, 16, 16, 16, device=self.device)
        ops = OptimizedFieldOps(self.device)
        
        # Time original diffusion (simplified)
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        blur_kernel = torch.ones(1, 1, 3, 3, 3, device=self.device) / 27.0
        for _ in range(10):
            result = field.clone()
            for c in range(64):
                diffused = torch.nn.functional.conv3d(
                    result[:, c:c+1], blur_kernel, padding=1
                )
                result[:, c:c+1] = 0.9 * result[:, c:c+1] + 0.1 * diffused
        
        torch.cuda.synchronize()
        original_time = time.perf_counter() - start
        
        # Time optimized
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(10):
            _ = ops.optimized_diffusion(field, 0.1)
        
        torch.cuda.synchronize()
        optimized_time = time.perf_counter() - start
        
        speedup = original_time / optimized_time
        passed = speedup > 2.0
        
        return TestResult(
            name="Optimization Speedup",
            passed=passed,
            message=f"{speedup:.1f}x faster",
            severity="diagnostic",
            metrics={"speedup": speedup}
        )
    
    def _run_emergence_tests(self, brain_size: str):
        """Monitor for emergence and pathological behaviors"""
        print("\n🌟 EMERGENCE MONITORING")
        print("-"*40)
        
        # Test for exploration
        result = self._test_exploration_emergence(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test for pathological loops
        result = self._test_pathological_behaviors(brain_size)
        self.results.append(result)
        self._print_result(result)
        
        # Test learning improvement
        result = self._test_learning_improvement(brain_size)
        self.results.append(result)
        self._print_result(result)
    
    def _test_exploration_emergence(self, brain_size: str) -> TestResult:
        """Test that exploration behavior emerges"""
        brain = create_pure_field_brain(size=brain_size, device=self.device, aggressive=True)
        
        motor_diversity = []
        
        # Run with constant input
        constant_input = torch.ones(10, device=self.device) * 0.1
        
        for _ in range(50):
            output = brain(constant_input)
            motor_diversity.append(output.cpu().numpy())
        
        # Calculate variance in motor outputs
        motor_array = np.array(motor_diversity)
        variance = np.var(motor_array, axis=0).mean()
        
        # Should show some variation (exploration)
        passed = variance > 0.001
        
        return TestResult(
            name="Exploration Emergence",
            passed=passed,
            message=f"Motor variance: {variance:.4f}",
            severity="diagnostic",
            metrics={"motor_variance": variance}
        )
    
    def _test_pathological_behaviors(self, brain_size: str) -> TestResult:
        """Check for pathological repetitive behaviors"""
        brain = create_pure_field_brain(size=brain_size, device=self.device)
        
        # Run many cycles with varying input
        outputs = []
        for i in range(100):
            input_data = torch.randn(10, device=self.device) * (0.1 + i * 0.01)
            output = brain(input_data)
            outputs.append(output.cpu().numpy())
        
        # Check for exact repetitions (pathological)
        repetitions = 0
        for i in range(1, len(outputs)):
            if np.allclose(outputs[i], outputs[i-1], rtol=1e-5):
                repetitions += 1
        
        # Some repetition is okay, but not constant
        repetition_rate = repetitions / len(outputs)
        passed = repetition_rate < 0.5
        
        return TestResult(
            name="Pathological Behaviors",
            passed=passed,
            message=f"Repetition rate: {repetition_rate:.1%}",
            severity="important",
            metrics={"repetition_rate": repetition_rate}
        )
    
    def _test_learning_improvement(self, brain_size: str) -> TestResult:
        """Test that behavior changes over time (learning)"""
        brain = create_pure_field_brain(size=brain_size, device=self.device, aggressive=True)
        
        # Collect early behavior
        early_outputs = []
        for i in range(20):
            input_data = torch.randn(10, device=self.device)
            output = brain(input_data)
            early_outputs.append(output.cpu().numpy())
        
        # Run more cycles
        for _ in range(100):
            input_data = torch.randn(10, device=self.device)
            _ = brain(input_data)
            # Simulate prediction error learning
            brain.learn_from_prediction_error(
                torch.randn(10, device=self.device),
                torch.randn(10, device=self.device)
            )
        
        # Collect late behavior
        late_outputs = []
        for i in range(20):
            input_data = torch.randn(10, device=self.device)
            output = brain(input_data)
            late_outputs.append(output.cpu().numpy())
        
        # Compare behaviors
        early_mean = np.mean(np.abs(early_outputs))
        late_mean = np.mean(np.abs(late_outputs))
        early_var = np.var(early_outputs)
        late_var = np.var(late_outputs)
        
        # Should show change (not necessarily improvement in a specific direction)
        behavior_changed = abs(late_mean - early_mean) > 0.01 or abs(late_var - early_var) > 0.001
        
        return TestResult(
            name="Learning/Change",
            passed=behavior_changed,
            message=f"Mean: {early_mean:.3f}→{late_mean:.3f}, Var: {early_var:.3f}→{late_var:.3f}",
            severity="diagnostic",
            metrics={
                "early_mean": early_mean,
                "late_mean": late_mean,
                "early_var": early_var,
                "late_var": late_var
            }
        )
    
    def _print_result(self, result: TestResult):
        """Print a single test result"""
        if not self.verbose:
            return
            
        # Choose symbol based on result
        if result.passed:
            symbol = "✅"
        elif result.severity == "critical":
            symbol = "❌"
        elif result.severity == "important":
            symbol = "⚠️"
        else:
            symbol = "ℹ️"
        
        print(f"{symbol} {result.name}: {result.message}")
    
    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize all test results"""
        critical_failures = [r for r in self.results if r.severity == "critical" and not r.passed]
        important_failures = [r for r in self.results if r.severity == "important" and not r.passed]
        total_passed = sum(1 for r in self.results if r.passed)
        
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {total_passed}/{len(self.results)}")
        print(f"Critical failures: {len(critical_failures)}")
        print(f"Important failures: {len(important_failures)}")
        
        if critical_failures:
            print("\n⛔ CRITICAL FAILURES - DO NOT DEPLOY TO ROBOT:")
            for failure in critical_failures:
                print(f"  - {failure.name}: {failure.message}")
        
        if important_failures:
            print("\n⚠️  Important issues to address:")
            for failure in important_failures:
                print(f"  - {failure.name}: {failure.message}")
        
        # Overall verdict
        print("\n" + "="*60)
        if critical_failures:
            print("❌ FAIL - Critical safety issues detected!")
            print("DO NOT deploy to robot until fixed.")
        elif important_failures:
            print("⚠️  PASS WITH WARNINGS - Some issues to address")
            print("Safe to test with robot under supervision.")
        else:
            print("✅ ALL TESTS PASSED - Ready for deployment!")
            print("Brain is safe for robot testing.")
        print("="*60)
        
        return {
            "total_tests": len(self.results),
            "passed": total_passed,
            "critical_failures": len(critical_failures),
            "important_failures": len(important_failures),
            "safe_for_robot": len(critical_failures) == 0,
            "results": self.results
        }


def run_quick_safety_check(brain_size: str = 'tiny'):
    """Quick safety check before robot deployment"""
    print("\n🚀 QUICK SAFETY CHECK")
    print("="*40)
    
    suite = BrainTestSuite(verbose=False)
    
    # Only run critical tests
    suite._run_critical_tests(brain_size)
    
    critical_failures = [r for r in suite.results if r.severity == "critical" and not r.passed]
    
    if critical_failures:
        print("❌ SAFETY CHECK FAILED")
        for failure in critical_failures:
            print(f"  - {failure.name}: {failure.message}")
        return False
    else:
        print("✅ SAFETY CHECK PASSED")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain Test Suite")
    parser.add_argument('--size', default='tiny', help='Brain size to test')
    parser.add_argument('--quick', action='store_true', help='Quick safety check only')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_safety_check(args.size)
        sys.exit(0 if success else 1)
    else:
        suite = BrainTestSuite(device=args.device)
        results = suite.run_all_tests(args.size)
        sys.exit(0 if results['safe_for_robot'] else 1)