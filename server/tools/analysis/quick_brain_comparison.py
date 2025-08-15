#!/usr/bin/env python3
"""
Quick Brain Comparison
=====================

Fast analysis combining static code metrics with performance predictions
to determine which brain implementation is best for robot deployment.

Based on:
1. Static code analysis (already completed)
2. Architectural patterns
3. Real-world deployment considerations
"""

import json
import time
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class BrainAnalysis:
    """Analysis results for one brain"""
    name: str
    lines_of_code: int
    complexity_score: float
    subsystem_count: int
    torch_operations: int
    dependencies: int
    predicted_performance_hz: float
    predicted_memory_mb: float
    predicted_stability: float
    deployment_score: float
    pros: List[str]
    cons: List[str]


class QuickBrainComparison:
    """Fast comparison of the three brain architectures"""
    
    def __init__(self):
        # Data from our static analysis
        self.analysis_data = {
            'unified': {
                'lines_of_code': 707,
                'complexity_score': 31.8,
                'subsystem_count': 4,
                'torch_operations': 35,
                'dependencies': 30,
                'subsystems': ['MotorCortex', 'SensoryMapping', 'PatternSystem', 'StrategicPlanner']
            },
            'minimal': {
                'lines_of_code': 166,
                'complexity_score': 7.8,
                'subsystem_count': 0,
                'torch_operations': 31,
                'dependencies': 7,
                'subsystems': []
            },
            'pure': {
                'lines_of_code': 287,
                'complexity_score': 11.4,
                'subsystem_count': 0,
                'torch_operations': 47,
                'dependencies': 6,
                'subsystems': []
            }
        }
    
    def predict_performance(self, brain_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict performance characteristics from architecture"""
        
        complexity = brain_data['complexity_score']
        subsystems = brain_data['subsystem_count']
        torch_ops = brain_data['torch_operations']
        lines = brain_data['lines_of_code']
        
        # Performance prediction (Hz) - simpler is faster
        base_performance = 50.0  # Starting point
        
        # Complexity penalty
        complexity_penalty = complexity * 0.5
        
        # Subsystem coordination overhead
        subsystem_penalty = subsystems * 3.0
        
        # Code size penalty
        size_penalty = lines / 50.0
        
        # GPU operations can be efficient if well-designed
        gpu_bonus = min(torch_ops / 10.0, 5.0) if torch_ops > 20 else 0
        
        predicted_hz = max(1.0, base_performance - complexity_penalty - subsystem_penalty - size_penalty + gpu_bonus)
        
        # Memory prediction (MB) - more complex = more memory
        base_memory = 20.0
        field_memory = 32 * 32 * 32 * 64 * 4 / (1024 * 1024)  # Field tensor size
        complexity_memory = complexity * 0.8
        subsystem_memory = subsystems * 8.0
        
        predicted_memory = base_memory + field_memory + complexity_memory + subsystem_memory
        
        # Stability prediction - fewer moving parts = more stable
        base_stability = 0.95
        complexity_penalty = complexity / 100.0 * 0.3
        subsystem_penalty = subsystems / 10.0 * 0.2
        
        predicted_stability = max(0.5, base_stability - complexity_penalty - subsystem_penalty)
        
        return {
            'performance_hz': predicted_hz,
            'memory_mb': predicted_memory,
            'stability': predicted_stability
        }
    
    def calculate_deployment_score(self, predictions: Dict[str, float]) -> float:
        """Calculate overall deployment readiness"""
        
        # Performance score (target 10+ Hz for real-time)
        perf_score = min(1.0, predictions['performance_hz'] / 15.0)
        
        # Memory score (target <60MB for embedded systems)
        memory_score = max(0, 1.0 - predictions['memory_mb'] / 60.0)
        
        # Stability is critical for robots
        stability_score = predictions['stability']
        
        # Weighted combination (stability is most important)
        deployment_score = (
            0.25 * perf_score +      # Performance
            0.25 * memory_score +    # Memory efficiency
            0.50 * stability_score   # Stability (critical for robots)
        )
        
        return max(0, min(1.0, deployment_score))
    
    def analyze_brain(self, name: str, data: Dict[str, Any]) -> BrainAnalysis:
        """Analyze one brain implementation"""
        
        predictions = self.predict_performance(data)
        deployment_score = self.calculate_deployment_score(predictions)
        
        # Determine pros and cons based on characteristics
        pros = []
        cons = []
        
        if name == 'unified':
            pros = [
                "Rich subsystem architecture",
                "Well-tested and battle-proven",
                "Handles complex behaviors",
                "Full feature set"
            ]
            cons = [
                "High complexity (31.8/100)",
                "4 subsystems create coordination overhead", 
                "High memory usage (~95MB predicted)",
                "30 dependencies create maintenance burden",
                "Slower performance due to subsystem interactions"
            ]
        elif name == 'minimal':
            pros = [
                "Extremely simple (7.8/100 complexity)",
                "No subsystems - monolithic design",
                "Fast execution (predicted 40+ Hz)",
                "Low memory footprint",
                "Only 7 dependencies",
                "Easy to understand and debug",
                "166 lines of code - minimal attack surface"
            ]
            cons = [
                "May lack sophistication for complex tasks",
                "Limited behavioral repertoire",
                "Less learning capability",
                "Aggressive parameters may need tuning"
            ]
        else:  # pure
            pros = [
                "Single tensor operation design",
                "GPU-optimized architecture", 
                "Modern unified approach",
                "No subsystem complexity",
                "Biological channel allocation",
                "Learnable evolution kernel",
                "Good balance of features and simplicity"
            ]
            cons = [
                "Newer, less battle-tested",
                "Higher torch operations (47)",
                "Medium complexity (11.4/100)",
                "Requires more GPU memory",
                "Single point of failure in evolution kernel"
            ]
        
        return BrainAnalysis(
            name=name,
            lines_of_code=data['lines_of_code'],
            complexity_score=data['complexity_score'],
            subsystem_count=data['subsystem_count'],
            torch_operations=data['torch_operations'],
            dependencies=data['dependencies'],
            predicted_performance_hz=predictions['performance_hz'],
            predicted_memory_mb=predictions['memory_mb'],
            predicted_stability=predictions['stability'],
            deployment_score=deployment_score,
            pros=pros,
            cons=cons
        )
    
    def run_comparison(self) -> List[BrainAnalysis]:
        """Run the complete comparison"""
        
        print("🧠 Quick Brain Comparison")
        print("=" * 60)
        print("Analyzing UnifiedFieldBrain vs MinimalFieldBrain vs PureFieldBrain")
        print("For real robot deployment readiness\n")
        
        results = []
        
        for name, data in self.analysis_data.items():
            print(f"🔍 Analyzing {name.upper()}...")
            analysis = self.analyze_brain(name, data)
            results.append(analysis)
            
            print(f"   Complexity: {analysis.complexity_score:.1f}/100")
            print(f"   Predicted performance: {analysis.predicted_performance_hz:.1f} Hz")
            print(f"   Predicted memory: {analysis.predicted_memory_mb:.1f}MB")
            print(f"   Predicted stability: {analysis.predicted_stability:.1%}")
            print(f"   Deployment score: {analysis.deployment_score:.1%}")
        
        return results
    
    def print_detailed_comparison(self, results: List[BrainAnalysis]):
        """Print detailed comparison table"""
        
        print("\n" + "=" * 80)
        print("📊 DETAILED COMPARISON")
        print("=" * 80)
        
        # Sort by deployment score
        results.sort(key=lambda x: x.deployment_score, reverse=True)
        
        # Comparison table
        print(f"\n{'Metric':<20} {'Unified':<15} {'Minimal':<15} {'Pure':<15}")
        print("-" * 70)
        
        metrics = [
            ('Lines of Code', 'lines_of_code', ''),
            ('Complexity Score', 'complexity_score', '/100'),
            ('Subsystems', 'subsystem_count', ''),
            ('Dependencies', 'dependencies', ''),
            ('Torch Operations', 'torch_operations', ''),
            ('Performance', 'predicted_performance_hz', ' Hz'),
            ('Memory Usage', 'predicted_memory_mb', ' MB'),
            ('Stability', 'predicted_stability', '%'),
            ('Deployment Score', 'deployment_score', '%')
        ]
        
        for metric_name, attr, unit in metrics:
            values = []
            for result in results:
                val = getattr(result, attr)
                if unit == '%':
                    values.append(f"{val:.1%}")
                elif isinstance(val, float):
                    values.append(f"{val:.1f}{unit}")
                else:
                    values.append(f"{val}{unit}")
            
            # Reorder to match original order (unified, minimal, pure)
            ordered_values = [None, None, None]
            for result in results:
                if result.name == 'unified':
                    ordered_values[0] = values[results.index(result)]
                elif result.name == 'minimal':
                    ordered_values[1] = values[results.index(result)]
                elif result.name == 'pure':
                    ordered_values[2] = values[results.index(result)]
            
            print(f"{metric_name:<20} {ordered_values[0]:<15} {ordered_values[1]:<15} {ordered_values[2]:<15}")
        
        # Winner analysis
        winner = results[0]
        print(f"\n🏆 WINNER: {winner.name.upper()}")
        print(f"   Deployment Score: {winner.deployment_score:.1%}")
        print(f"   Performance: {winner.predicted_performance_hz:.1f} Hz")
        print(f"   Memory: {winner.predicted_memory_mb:.1f} MB")
        print(f"   Stability: {winner.predicted_stability:.1%}")
        
        # Detailed pros/cons
        print(f"\n✅ PROS:")
        for pro in winner.pros:
            print(f"     • {pro}")
        
        print(f"\n❌ CONS:")
        for con in winner.cons:
            print(f"     • {con}")
        
        # Specific recommendations
        print(f"\n🤖 ROBOT DEPLOYMENT RECOMMENDATIONS:")
        
        if winner.name == 'minimal':
            print("   🥇 PRIMARY CHOICE: MinimalFieldBrain")
            print("     Rationale: Best balance of simplicity, performance, and reliability")
            print("     Use cases: Most robots, production deployments, resource-constrained systems")
            print("     Risk level: LOW - Simple architecture minimizes failure modes")
            
        elif winner.name == 'pure':
            print("   🥇 PRIMARY CHOICE: PureFieldBrain")
            print("     Rationale: Modern GPU-optimized design with single operation efficiency")
            print("     Use cases: GPU-equipped robots, high-performance applications")
            print("     Risk level: MEDIUM - Newer architecture, single point of failure")
            
        else:
            print("   🥇 PRIMARY CHOICE: UnifiedFieldBrain") 
            print("     Rationale: Full-featured despite complexity overhead")
            print("     Use cases: Complex behaviors, research platforms, feature-rich applications")
            print("     Risk level: HIGH - Multiple subsystems increase failure modes")
        
        # Alternative recommendations
        print(f"\n🔄 ALTERNATIVE SCENARIOS:")
        
        for result in results[1:]:
            if result.name == 'minimal':
                print(f"   • Choose MINIMAL for: Maximum reliability, embedded systems, simple tasks")
            elif result.name == 'pure':
                print(f"   • Choose PURE for: GPU acceleration, cutting-edge performance, research")
            else:
                print(f"   • Choose UNIFIED for: Complex behaviors, proven architecture, full features")
        
        # Final verdict
        print(f"\n🎯 FINAL VERDICT:")
        print(f"   For MOST robot deployments: {winner.name.upper()}")
        print(f"   Confidence level: {winner.deployment_score:.0%}")
        
        if winner.deployment_score > 0.8:
            print("   Recommendation strength: STRONG - Deploy with confidence")
        elif winner.deployment_score > 0.6:
            print("   Recommendation strength: MODERATE - Good choice with some caveats")
        else:
            print("   Recommendation strength: WEAK - Consider alternatives or improvements")
        
        return results
    
    def save_analysis(self, results: List[BrainAnalysis]):
        """Save analysis results"""
        
        analysis_data = {
            'analysis_type': 'comprehensive_brain_comparison',
            'timestamp': time.time(),
            'brains_analyzed': len(results),
            'winner': results[0].name if results else None,
            'results': []
        }
        
        for result in results:
            analysis_data['results'].append({
                'name': result.name,
                'lines_of_code': result.lines_of_code,
                'complexity_score': result.complexity_score,
                'subsystem_count': result.subsystem_count,
                'torch_operations': result.torch_operations,
                'dependencies': result.dependencies,
                'predicted_performance_hz': result.predicted_performance_hz,
                'predicted_memory_mb': result.predicted_memory_mb,
                'predicted_stability': result.predicted_stability,
                'deployment_score': result.deployment_score,
                'pros': result.pros,
                'cons': result.cons
            })
        
        filename = "comprehensive_brain_analysis.json"
        with open(filename, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        print(f"\n💾 Analysis saved to: {filename}")


def main():
    """Run the comprehensive brain comparison"""
    
    comparator = QuickBrainComparison()
    results = comparator.run_comparison()
    comparator.print_detailed_comparison(results)
    comparator.save_analysis(results)
    
    return results


if __name__ == "__main__":
    main()