#!/usr/bin/env python3
"""
Parameter Sweep Tool for Data-Driven Brain Development

Enables rapid testing of different brain parameters to find optimal configurations.
"""

import sys
import os
from pathlib import Path
import json
import time
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import subprocess
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(brain_root))

@dataclass
class ParameterConfig:
    """Configuration for a parameter sweep test"""
    name: str
    force_spatial_resolution: int
    self_modification_rate: float
    energy_decay_rate: float
    consolidation_strength: float
    attention_threshold: float
    
    def to_settings_overrides(self) -> Dict:
        """Convert to settings.json overrides format"""
        return {
            "overrides": {
                "force_spatial_resolution": self.force_spatial_resolution,
                "force_device": "cpu",
                "disable_adaptation": False
            },
            "parameters": {
                "self_modification_rate": self.self_modification_rate,
                "energy_decay_rate": self.energy_decay_rate,
                "consolidation_strength": self.consolidation_strength,
                "attention_threshold": self.attention_threshold
            }
        }

@dataclass
class TestResult:
    """Result from a parameter test"""
    config: ParameterConfig
    learning_improvement: float
    efficiency_final: float
    collision_rate: float
    biological_realism: float
    test_duration: float
    error: str = None

class ParameterSweep:
    """Runs parameter sweeps for brain optimization"""
    
    def __init__(self, test_duration_minutes: int = 10):
        self.test_duration = test_duration_minutes
        self.results_dir = Path("tools/analysis/parameter_sweep_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def generate_parameter_grid(self) -> List[ParameterConfig]:
        """Generate parameter combinations to test"""
        configs = []
        
        # Test different spatial resolutions
        for resolution in [8, 16, 32]:
            # Test different self-modification rates
            for sm_rate in [0.001, 0.01, 0.1]:
                # Test different energy decay rates
                for energy_decay in [0.01, 0.05, 0.1]:
                    configs.append(ParameterConfig(
                        name=f"res{resolution}_sm{sm_rate}_ed{energy_decay}",
                        force_spatial_resolution=resolution,
                        self_modification_rate=sm_rate,
                        energy_decay_rate=energy_decay,
                        consolidation_strength=0.8,  # Keep constant for now
                        attention_threshold=0.7      # Keep constant for now
                    ))
        
        return configs
    
    def run_single_test(self, config: ParameterConfig) -> TestResult:
        """Run a single parameter test"""
        print(f"\n🧪 Testing configuration: {config.name}")
        
        # Create temporary settings file
        temp_settings = self.results_dir / f"settings_{config.name}.json"
        with open(temp_settings, 'w') as f:
            json.dump(config.to_settings_overrides(), f, indent=2)
        
        try:
            # Run validation test with custom settings
            cmd = [
                sys.executable,
                str(brain_root / "validation/embodied_learning/experiments/biological_embodied_learning.py"),
                "--duration-hours", str(self.test_duration / 60),
                "--session-minutes", str(min(5, self.test_duration)),
                "--consolidation-minutes", "2",
                "--settings", str(temp_settings),
                "--quiet"
            ]
            
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.test_duration * 60 + 60)
            duration = time.time() - start_time
            
            # Parse results
            if result.returncode == 0:
                # Extract key metrics from output
                output_lines = result.stdout.split('\n')
                
                # Look for the results JSON file path
                results_file = None
                for line in output_lines:
                    if "Results saved to:" in line:
                        results_file = line.split("Results saved to:")[-1].strip()
                        break
                
                if results_file and Path(results_file).exists():
                    with open(results_file, 'r') as f:
                        test_data = json.load(f)
                    
                    # Extract metrics
                    analysis = test_data.get('analysis', {})
                    learning = analysis.get('learning_progression', {})
                    bio_realism = analysis.get('biological_realism', {})
                    
                    # Calculate final efficiency from last session
                    sessions = test_data.get('session_results', [])
                    final_efficiency = sessions[-1]['efficiency'] if sessions else 0.0
                    
                    # Calculate average collision rate
                    avg_collision = np.mean([s['collision_rate'] for s in sessions]) if sessions else 0.0
                    
                    return TestResult(
                        config=config,
                        learning_improvement=learning.get('total_improvement', 0.0),
                        efficiency_final=final_efficiency,
                        collision_rate=avg_collision,
                        biological_realism=bio_realism.get('biological_realism_score', 0.0),
                        test_duration=duration
                    )
                
            # If we couldn't parse results, return error
            return TestResult(
                config=config,
                learning_improvement=0.0,
                efficiency_final=0.0,
                collision_rate=1.0,
                biological_realism=0.0,
                test_duration=duration,
                error=f"Failed to parse results: {result.stderr[:200]}"
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                config=config,
                learning_improvement=0.0,
                efficiency_final=0.0,
                collision_rate=1.0,
                biological_realism=0.0,
                test_duration=self.test_duration * 60,
                error="Test timeout"
            )
        except Exception as e:
            return TestResult(
                config=config,
                learning_improvement=0.0,
                efficiency_final=0.0,
                collision_rate=1.0,
                biological_realism=0.0,
                test_duration=0.0,
                error=str(e)
            )
        finally:
            # Cleanup temp settings
            if temp_settings.exists():
                temp_settings.unlink()
    
    def run_parameter_sweep(self, parallel: bool = True, max_workers: int = 4):
        """Run full parameter sweep"""
        configs = self.generate_parameter_grid()
        print(f"🚀 Running parameter sweep with {len(configs)} configurations")
        print(f"   Test duration: {self.test_duration} minutes per config")
        print(f"   Estimated total time: {len(configs) * self.test_duration / max_workers} minutes")
        
        results = []
        
        if parallel:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_config = {executor.submit(self.run_single_test, config): config 
                                  for config in configs}
                
                for future in as_completed(future_to_config):
                    result = future.result()
                    results.append(result)
                    print(f"✅ Completed: {result.config.name} "
                          f"(learning: {result.learning_improvement:.3f}, "
                          f"efficiency: {result.efficiency_final:.3f})")
        else:
            for config in configs:
                result = self.run_single_test(config)
                results.append(result)
        
        # Save results
        self.save_results(results)
        
        # Generate visualizations
        self.visualize_results(results)
        
        # Find best configuration
        best_result = self.find_best_configuration(results)
        
        return results, best_result
    
    def save_results(self, results: List[TestResult]):
        """Save sweep results to file"""
        timestamp = int(time.time())
        results_file = self.results_dir / f"sweep_results_{timestamp}.json"
        
        data = {
            "timestamp": timestamp,
            "test_duration_minutes": self.test_duration,
            "results": [
                {
                    "config": asdict(r.config),
                    "metrics": {
                        "learning_improvement": r.learning_improvement,
                        "efficiency_final": r.efficiency_final,
                        "collision_rate": r.collision_rate,
                        "biological_realism": r.biological_realism,
                        "test_duration": r.test_duration
                    },
                    "error": r.error
                }
                for r in results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
    
    def visualize_results(self, results: List[TestResult]):
        """Create visualizations of parameter sweep results"""
        # Filter out failed tests
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            print("⚠️  No valid results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Parameter Sweep Results', fontsize=16)
        
        # Extract data for plotting
        resolutions = [r.config.force_spatial_resolution for r in valid_results]
        sm_rates = [r.config.self_modification_rate for r in valid_results]
        energy_decays = [r.config.energy_decay_rate for r in valid_results]
        
        learning = [r.learning_improvement for r in valid_results]
        efficiency = [r.efficiency_final for r in valid_results]
        collisions = [r.collision_rate for r in valid_results]
        bio_realism = [r.biological_realism for r in valid_results]
        
        # Plot 1: Learning vs Resolution
        ax1 = axes[0, 0]
        for sm in set(sm_rates):
            mask = [r.config.self_modification_rate == sm for r in valid_results]
            x = [res for i, res in enumerate(resolutions) if mask[i]]
            y = [learn for i, learn in enumerate(learning) if mask[i]]
            ax1.scatter(x, y, label=f'SM Rate: {sm}', alpha=0.7)
        ax1.set_xlabel('Spatial Resolution')
        ax1.set_ylabel('Learning Improvement')
        ax1.set_title('Learning vs Resolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Efficiency vs Self-Modification Rate
        ax2 = axes[0, 1]
        ax2.scatter(sm_rates, efficiency, c=resolutions, cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Self-Modification Rate')
        ax2.set_ylabel('Final Efficiency')
        ax2.set_title('Efficiency vs Self-Modification')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(ax2.collections[0], ax=ax2)
        cbar.set_label('Resolution')
        
        # Plot 3: Collision Rate vs Energy Decay
        ax3 = axes[1, 0]
        ax3.scatter(energy_decays, collisions, c=resolutions, cmap='plasma', s=100, alpha=0.7)
        ax3.set_xlabel('Energy Decay Rate')
        ax3.set_ylabel('Collision Rate')
        ax3.set_title('Safety vs Energy Decay')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Overall Performance Heatmap
        ax4 = axes[1, 1]
        # Calculate overall score
        overall_scores = [(l + e + (1-c) + b) / 4 for l, e, c, b in 
                         zip(learning, efficiency, collisions, bio_realism)]
        
        # Create scatter plot with score as color
        scatter = ax4.scatter(sm_rates, energy_decays, c=overall_scores, 
                            s=[r*10 for r in resolutions], cmap='RdYlGn', 
                            alpha=0.7, edgecolors='black', linewidth=1)
        ax4.set_xlabel('Self-Modification Rate')
        ax4.set_ylabel('Energy Decay Rate')
        ax4.set_title('Overall Performance (size = resolution)')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        cbar2 = plt.colorbar(scatter, ax=ax4)
        cbar2.set_label('Overall Score')
        
        plt.tight_layout()
        
        # Save figure
        timestamp = int(time.time())
        fig_path = self.results_dir / f"parameter_sweep_{timestamp}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"📊 Visualization saved to: {fig_path}")
        
        # Don't show in automated context
        plt.close()
    
    def find_best_configuration(self, results: List[TestResult]) -> TestResult:
        """Find the best parameter configuration"""
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            return None
        
        # Calculate overall score for each configuration
        def overall_score(r: TestResult) -> float:
            # Weighted combination of metrics
            learning_weight = 0.3
            efficiency_weight = 0.3
            safety_weight = 0.2  # Low collision rate
            realism_weight = 0.2
            
            return (r.learning_improvement * learning_weight +
                   r.efficiency_final * efficiency_weight +
                   (1 - r.collision_rate) * safety_weight +
                   r.biological_realism * realism_weight)
        
        best = max(valid_results, key=overall_score)
        
        print(f"\n🏆 Best Configuration: {best.config.name}")
        print(f"   Resolution: {best.config.force_spatial_resolution}")
        print(f"   Self-Modification Rate: {best.config.self_modification_rate}")
        print(f"   Energy Decay: {best.config.energy_decay_rate}")
        print(f"   Learning Improvement: {best.learning_improvement:.3f}")
        print(f"   Final Efficiency: {best.efficiency_final:.3f}")
        print(f"   Collision Rate: {best.collision_rate:.3f}")
        print(f"   Biological Realism: {best.biological_realism:.3f}")
        print(f"   Overall Score: {overall_score(best):.3f}")
        
        return best


def main():
    """Run parameter sweep"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parameter sweep for brain optimization")
    parser.add_argument('--duration', type=int, default=10,
                       help='Test duration in minutes per configuration (default: 10)')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    parser.add_argument('--sequential', action='store_true',
                       help='Run tests sequentially instead of in parallel')
    
    args = parser.parse_args()
    
    sweep = ParameterSweep(test_duration_minutes=args.duration)
    
    print("🔬 Starting Parameter Sweep")
    print("=" * 60)
    
    results, best = sweep.run_parameter_sweep(
        parallel=not args.sequential,
        max_workers=args.workers
    )
    
    print("\n✅ Parameter sweep complete!")
    
    # Generate recommendations
    if best:
        print("\n📋 Recommendations:")
        print(f"1. Use spatial resolution: {best.config.force_spatial_resolution}")
        print(f"2. Set self-modification rate to: {best.config.self_modification_rate}")
        print(f"3. Set energy decay rate to: {best.config.energy_decay_rate}")
        print("\nUpdate server/settings.json with these values for optimal performance.")


if __name__ == "__main__":
    main()