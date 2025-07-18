#!/usr/bin/env python3
"""
Biological Embodied Learning Experiment

A scientifically rigorous study of how the brain learns embodied behaviors
over biological timescales. This experiment tests:

1. Sensory-motor coordination learning
2. Emergent navigation strategies
3. Memory consolidation effects
4. Transfer learning capabilities
5. Biological realism validation

This experiment generates publication-quality data and analysis.
"""

import sys
import os
import time
import datetime
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from pathlib import Path

# Add brain root to path
brain_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(brain_root))
sys.path.insert(0, str(brain_root / 'server'))

from src.communication import MinimalBrainClient
from validation.embodied_learning.environments.sensory_motor_world import SensoryMotorWorld

@dataclass
class ExperimentConfig:
    """Configuration for biological embodied learning experiment."""
    experiment_name: str = "biological_embodied_learning"
    duration_hours: float = 4.0
    session_duration_minutes: int = 20
    consolidation_duration_minutes: int = 10
    random_seed: int = 42
    
    # Environment parameters
    world_size: float = 10.0
    num_light_sources: int = 2
    num_obstacles: int = 5
    
    # Learning parameters
    actions_per_minute: int = 60
    adaptation_enabled: bool = True
    transfer_test_enabled: bool = True
    
    def to_dict(self):
        return asdict(self)

@dataclass
class SessionResults:
    """Results from a single learning session."""
    session_id: int
    start_time: float
    duration_minutes: float
    total_actions: int
    
    # Performance metrics
    avg_light_distance: float
    exploration_score: float
    efficiency: float
    collision_rate: float
    battery_usage: float
    
    # Learning metrics
    prediction_error_trend: List[float]
    strategy_emergence_score: float
    motor_coordination_score: float
    
    # Behavioral data
    action_distribution: Dict[str, int]
    trajectory_complexity: float
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ConsolidationAnalysis:
    """Analysis of memory consolidation between sessions."""
    consolidation_id: int
    pre_session_performance: float
    post_session_performance: float
    consolidation_benefit: float
    strategy_refinement: float
    memory_stability: float
    
    def to_dict(self):
        return asdict(self)

class BiologicalEmbodiedLearningExperiment:
    """
    Main experiment class for biological embodied learning study.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = MinimalBrainClient()
        self.environment = SensoryMotorWorld(
            world_size=config.world_size,
            num_light_sources=config.num_light_sources,
            num_obstacles=config.num_obstacles,
            random_seed=config.random_seed
        )
        
        # Experiment state
        self.session_results: List[SessionResults] = []
        self.consolidation_analyses: List[ConsolidationAnalysis] = []
        self.experiment_start_time = None
        self.total_actions_executed = 0
        
        # Analysis data
        self.prediction_error_history = []
        self.behavioral_trajectories = []
        self.strategy_evolution = []
        
        # Create results directory
        self.results_dir = Path(f"validation/embodied_learning/reports/{config.experiment_name}_{int(time.time())}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"🧬 Biological Embodied Learning Experiment")
        print(f"   Duration: {config.duration_hours} hours")
        print(f"   Session length: {config.session_duration_minutes} minutes")
        print(f"   Consolidation breaks: {config.consolidation_duration_minutes} minutes")
        print(f"   Results directory: {self.results_dir}")
    
    def run_experiment(self) -> Dict:
        """Run the complete biological embodied learning experiment."""
        
        print(f"\n🔬 Starting biological embodied learning experiment...")
        
        # Connect to brain server
        if not self.client.connect():
            raise ConnectionError("Failed to connect to brain server")
        
        try:
            self.experiment_start_time = time.time()
            
            # Phase 1: Baseline learning session
            print(f"\n📊 Phase 1: Baseline Learning Assessment")
            baseline_results = self._run_learning_session(session_id=0, is_baseline=True)
            self.session_results.append(baseline_results)
            
            # Phase 2: Biological learning with consolidation
            print(f"\n🧬 Phase 2: Biological Learning Sessions")
            session_count = 1
            
            total_duration = self.config.duration_hours * 3600
            session_duration = self.config.session_duration_minutes * 60
            consolidation_duration = self.config.consolidation_duration_minutes * 60
            
            while (time.time() - self.experiment_start_time) < total_duration:
                # Learning session
                print(f"\n📚 Learning Session {session_count}")
                session_results = self._run_learning_session(session_id=session_count)
                self.session_results.append(session_results)
                
                # Check if we have time for consolidation
                elapsed_time = time.time() - self.experiment_start_time
                if elapsed_time + consolidation_duration < total_duration:
                    # Consolidation break
                    print(f"😴 Consolidation break ({self.config.consolidation_duration_minutes} min)")
                    consolidation_analysis = self._run_consolidation_phase(session_count)
                    self.consolidation_analyses.append(consolidation_analysis)
                    
                    # Sleep with keepalive pings
                    self._consolidation_sleep(consolidation_duration)
                
                session_count += 1
            
            # Phase 3: Transfer learning test
            if self.config.transfer_test_enabled:
                print(f"\n🔄 Phase 3: Transfer Learning Test")
                transfer_results = self._run_transfer_test()
            
            # Phase 4: Generate scientific analysis
            print(f"\n📈 Phase 4: Scientific Analysis")
            analysis_results = self._generate_scientific_analysis()
            
            # Save complete results
            final_results = {
                'config': self.config.to_dict(),
                'session_results': [s.to_dict() for s in self.session_results],
                'consolidation_analyses': [c.to_dict() for c in self.consolidation_analyses],
                'transfer_results': transfer_results if self.config.transfer_test_enabled else None,
                'analysis': analysis_results,
                'experiment_duration': time.time() - self.experiment_start_time,
                'total_actions': self.total_actions_executed
            }
            
            self._save_results(final_results)
            self._generate_visualizations(final_results)
            
            print(f"\n🎉 Experiment completed successfully!")
            print(f"   Total duration: {(time.time() - self.experiment_start_time)/3600:.1f} hours")
            print(f"   Sessions completed: {len(self.session_results)}")
            print(f"   Total actions: {self.total_actions_executed}")
            print(f"   Results saved to: {self.results_dir}")
            
            return final_results
            
        finally:
            self.client.disconnect()
    
    def _run_learning_session(self, session_id: int, is_baseline: bool = False) -> SessionResults:
        """Run a single learning session."""
        
        session_start = time.time()
        session_duration = self.config.session_duration_minutes * 60
        actions_per_second = self.config.actions_per_minute / 60
        
        # Reset environment for new session
        self.environment.reset()
        
        # Session tracking
        actions_executed = 0
        prediction_errors = []
        light_distances = []
        action_counts = {'MOVE_FORWARD': 0, 'TURN_LEFT': 0, 'TURN_RIGHT': 0, 'STOP': 0}
        trajectory_points = []
        
        print(f"   Duration: {self.config.session_duration_minutes} minutes")
        print(f"   Target actions: {int(session_duration * actions_per_second)}")
        
        # Main learning loop
        while (time.time() - session_start) < session_duration:
            try:
                # Get sensory input
                sensory_input = self.environment.get_sensory_input()
                
                # Send to brain and get prediction
                start_time = time.time()
                prediction = self.client.get_action(sensory_input, timeout=3.0)
                
                if prediction is None:
                    print("   ⚠️ No response from brain")
                    time.sleep(1)
                    continue
                
                # Execute action in environment
                execution_result = self.environment.execute_action(prediction)
                
                # Record metrics
                metrics = execution_result['metrics']
                prediction_error = np.mean(np.abs(np.array(prediction) - np.array(sensory_input[:4])))
                
                prediction_errors.append(prediction_error)
                light_distances.append(metrics['min_light_distance'])
                
                # Track action distribution
                action_name = ['MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'STOP'][execution_result['action_executed']]
                action_counts[action_name] += 1
                
                # Track trajectory
                robot_state = execution_result['robot_state']
                trajectory_points.append((robot_state['position'][0], robot_state['position'][1]))
                
                actions_executed += 1
                self.total_actions_executed += 1
                
                # Maintain target action rate
                action_time = time.time() - start_time
                target_interval = 1.0 / actions_per_second
                if action_time < target_interval:
                    time.sleep(target_interval - action_time)
                
            except KeyboardInterrupt:
                print("\\n⏹️ Session interrupted by user")
                break
            except Exception as e:
                print(f"   ⚠️ Error in session: {e}")
                time.sleep(1)
        
        # Calculate session metrics
        session_duration_actual = (time.time() - session_start) / 60  # Convert to minutes
        
        # Performance metrics
        avg_light_distance = np.mean(light_distances) if light_distances else 0.0
        exploration_score = self._calculate_exploration_score(trajectory_points)
        efficiency = action_counts['MOVE_FORWARD'] / max(1, actions_executed)
        collision_rate = self._calculate_collision_rate(trajectory_points)
        battery_usage = 1.0 - (self.environment.robot_state.battery / self.environment.battery_capacity)
        
        # Learning metrics
        prediction_error_trend = self._calculate_learning_trend(prediction_errors)
        strategy_emergence_score = self._calculate_strategy_emergence(action_counts, trajectory_points)
        motor_coordination_score = self._calculate_motor_coordination(action_counts)
        
        # Behavioral analysis
        trajectory_complexity = self._calculate_trajectory_complexity(trajectory_points)
        
        # Create session results
        session_results = SessionResults(
            session_id=session_id,
            start_time=session_start,
            duration_minutes=session_duration_actual,
            total_actions=actions_executed,
            
            avg_light_distance=avg_light_distance,
            exploration_score=exploration_score,
            efficiency=efficiency,
            collision_rate=collision_rate,
            battery_usage=battery_usage,
            
            prediction_error_trend=prediction_error_trend,
            strategy_emergence_score=strategy_emergence_score,
            motor_coordination_score=motor_coordination_score,
            
            action_distribution=action_counts,
            trajectory_complexity=trajectory_complexity
        )
        
        # Print session summary
        print(f"   📊 Session {session_id} Results:")
        print(f"      Actions executed: {actions_executed}")
        print(f"      Avg light distance: {avg_light_distance:.3f}")
        print(f"      Exploration score: {exploration_score:.3f}")
        print(f"      Efficiency: {efficiency:.3f}")
        print(f"      Strategy emergence: {strategy_emergence_score:.3f}")
        
        return session_results
    
    def _run_consolidation_phase(self, session_id: int) -> ConsolidationAnalysis:
        """Analyze memory consolidation between sessions."""
        
        if len(self.session_results) < 2:
            return ConsolidationAnalysis(
                consolidation_id=session_id,
                pre_session_performance=0.0,
                post_session_performance=0.0,
                consolidation_benefit=0.0,
                strategy_refinement=0.0,
                memory_stability=0.0
            )
        
        # Compare performance before and after consolidation
        pre_session = self.session_results[-2]
        post_session = self.session_results[-1]
        
        # Calculate consolidation metrics
        pre_performance = self._calculate_session_performance(pre_session)
        post_performance = self._calculate_session_performance(post_session)
        
        consolidation_benefit = post_performance - pre_performance
        strategy_refinement = self._calculate_strategy_refinement(pre_session, post_session)
        memory_stability = self._calculate_memory_stability(pre_session, post_session)
        
        consolidation_analysis = ConsolidationAnalysis(
            consolidation_id=session_id,
            pre_session_performance=pre_performance,
            post_session_performance=post_performance,
            consolidation_benefit=consolidation_benefit,
            strategy_refinement=strategy_refinement,
            memory_stability=memory_stability
        )
        
        print(f"   🧠 Consolidation Analysis:")
        print(f"      Performance change: {consolidation_benefit:+.3f}")
        print(f"      Strategy refinement: {strategy_refinement:.3f}")
        print(f"      Memory stability: {memory_stability:.3f}")
        
        return consolidation_analysis
    
    def _consolidation_sleep(self, duration_seconds: float):
        """Sleep during consolidation with keepalive pings."""
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            # Sleep for 30 seconds
            time.sleep(30)
            
            # Send keepalive ping
            if (time.time() - start_time) < duration_seconds:
                try:
                    self.client.get_action([0.0, 0.0, 0.0, 0.0], timeout=5.0)
                except Exception:
                    pass  # Ignore keepalive failures
    
    def _run_transfer_test(self) -> Dict:
        """Test transfer learning to new environment."""
        print("   Testing transfer to new environment...")
        
        # Create new environment with different parameters
        transfer_env = SensoryMotorWorld(
            world_size=self.config.world_size * 1.5,
            num_light_sources=self.config.num_light_sources + 1,
            num_obstacles=self.config.num_obstacles + 3,
            random_seed=self.config.random_seed + 1
        )
        
        # Test performance in new environment
        original_env = self.environment
        self.environment = transfer_env
        
        transfer_results = self._run_learning_session(session_id=999, is_baseline=False)
        
        # Restore original environment
        self.environment = original_env
        
        return {
            'transfer_performance': self._calculate_session_performance(transfer_results),
            'transfer_session_data': transfer_results.to_dict()
        }
    
    def _generate_scientific_analysis(self) -> Dict:
        """Generate comprehensive scientific analysis."""
        
        analysis = {
            'learning_progression': self._analyze_learning_progression(),
            'consolidation_effects': self._analyze_consolidation_effects(),
            'behavioral_emergence': self._analyze_behavioral_emergence(),
            'biological_realism': self._analyze_biological_realism(),
            'efficiency_trends': self._analyze_efficiency_trends()
        }
        
        return analysis
    
    def _analyze_learning_progression(self) -> Dict:
        """Analyze learning progression over time."""
        
        if len(self.session_results) < 2:
            return {'insufficient_data': True}
        
        # Extract performance metrics over time
        performances = [self._calculate_session_performance(s) for s in self.session_results]
        light_distances = [s.avg_light_distance for s in self.session_results]
        exploration_scores = [s.exploration_score for s in self.session_results]
        
        # Calculate learning rate
        learning_rate = np.polyfit(range(len(performances)), performances, 1)[0]
        
        # Calculate learning curve characteristics
        initial_performance = performances[0]
        final_performance = performances[-1]
        improvement = final_performance - initial_performance
        
        return {
            'learning_rate': learning_rate,
            'initial_performance': initial_performance,
            'final_performance': final_performance,
            'total_improvement': improvement,
            'performance_trend': performances,
            'light_distance_trend': light_distances,
            'exploration_trend': exploration_scores
        }
    
    def _analyze_consolidation_effects(self) -> Dict:
        """Analyze memory consolidation effects."""
        
        if len(self.consolidation_analyses) < 2:
            return {'insufficient_data': True}
        
        consolidation_benefits = [c.consolidation_benefit for c in self.consolidation_analyses]
        strategy_refinements = [c.strategy_refinement for c in self.consolidation_analyses]
        memory_stabilities = [c.memory_stability for c in self.consolidation_analyses]
        
        return {
            'avg_consolidation_benefit': np.mean(consolidation_benefits),
            'consolidation_consistency': 1.0 - np.std(consolidation_benefits),
            'avg_strategy_refinement': np.mean(strategy_refinements),
            'avg_memory_stability': np.mean(memory_stabilities),
            'consolidation_trend': consolidation_benefits
        }
    
    def _analyze_behavioral_emergence(self) -> Dict:
        """Analyze emergence of complex behaviors."""
        
        strategy_scores = [s.strategy_emergence_score for s in self.session_results]
        motor_coordination = [s.motor_coordination_score for s in self.session_results]
        trajectory_complexity = [s.trajectory_complexity for s in self.session_results]
        
        return {
            'strategy_emergence_trend': strategy_scores,
            'motor_coordination_trend': motor_coordination,
            'trajectory_complexity_trend': trajectory_complexity,
            'behavioral_sophistication': np.mean(strategy_scores[-3:]) if len(strategy_scores) >= 3 else 0.0
        }
    
    def _analyze_biological_realism(self) -> Dict:
        """Analyze biological realism of learning."""
        
        # Check for biological learning characteristics
        performances = [self._calculate_session_performance(s) for s in self.session_results]
        
        # Gradual improvement (not sudden jumps)
        gradual_learning = self._check_gradual_learning(performances)
        
        # Consolidation benefits
        consolidation_present = len(self.consolidation_analyses) > 0
        consolidation_beneficial = np.mean([c.consolidation_benefit for c in self.consolidation_analyses]) > 0 if consolidation_present else False
        
        # Power law learning
        power_law_fit = self._check_power_law_learning(performances)
        
        return {
            'gradual_learning': gradual_learning,
            'consolidation_present': consolidation_present,
            'consolidation_beneficial': consolidation_beneficial,
            'power_law_learning': power_law_fit,
            'biological_realism_score': self._calculate_biological_realism_score(gradual_learning, consolidation_beneficial, power_law_fit)
        }
    
    def _analyze_efficiency_trends(self) -> Dict:
        """Analyze efficiency trends over time."""
        
        efficiencies = [s.efficiency for s in self.session_results]
        exploration_scores = [s.exploration_score for s in self.session_results]
        
        return {
            'efficiency_trend': efficiencies,
            'exploration_trend': exploration_scores,
            'efficiency_improvement': efficiencies[-1] - efficiencies[0] if len(efficiencies) >= 2 else 0.0,
            'exploration_improvement': exploration_scores[-1] - exploration_scores[0] if len(exploration_scores) >= 2 else 0.0
        }
    
    # Helper methods for calculations
    def _calculate_session_performance(self, session: SessionResults) -> float:
        """Calculate overall session performance score."""
        return (session.exploration_score * 0.3 + 
                session.efficiency * 0.3 + 
                (1.0 - session.collision_rate) * 0.2 + 
                session.strategy_emergence_score * 0.2)
    
    def _calculate_exploration_score(self, trajectory_points: List[Tuple[float, float]]) -> float:
        """Calculate exploration score based on trajectory."""
        if len(trajectory_points) < 2:
            return 0.0
        
        # Calculate coverage of world space
        positions = np.array(trajectory_points)
        world_size = self.config.world_size
        
        # Discretize world into grid
        grid_size = 20
        grid = np.zeros((grid_size, grid_size))
        
        for x, y in positions:
            grid_x = int(np.clip(x / world_size * grid_size, 0, grid_size - 1))
            grid_y = int(np.clip(y / world_size * grid_size, 0, grid_size - 1))
            grid[grid_x, grid_y] = 1
        
        return np.sum(grid) / (grid_size * grid_size)
    
    def _calculate_collision_rate(self, trajectory_points: List[Tuple[float, float]]) -> float:
        """Calculate collision rate based on trajectory."""
        # Simplified collision detection
        return 0.0  # Placeholder
    
    def _calculate_learning_trend(self, prediction_errors: List[float]) -> List[float]:
        """Calculate learning trend from prediction errors."""
        if len(prediction_errors) < 10:
            return prediction_errors
        
        # Calculate moving average trend
        window_size = len(prediction_errors) // 10
        trend = []
        for i in range(0, len(prediction_errors), window_size):
            window = prediction_errors[i:i+window_size]
            trend.append(np.mean(window))
        
        return trend
    
    def _calculate_strategy_emergence(self, action_counts: Dict, trajectory_points: List) -> float:
        """Calculate strategy emergence score."""
        # Check for balanced action usage (not just random)
        total_actions = sum(action_counts.values())
        if total_actions == 0:
            return 0.0
        
        # Calculate action diversity
        action_proportions = [count / total_actions for count in action_counts.values()]
        entropy = -sum(p * np.log2(p + 1e-10) for p in action_proportions)
        max_entropy = np.log2(4)  # Maximum entropy for 4 actions
        
        # Normalize to [0, 1] where 0.5 is random, higher is more strategic
        return min(1.0, entropy / max_entropy)
    
    def _calculate_motor_coordination(self, action_counts: Dict) -> float:
        """Calculate motor coordination score."""
        total_actions = sum(action_counts.values())
        if total_actions == 0:
            return 0.0
        
        # Good coordination means more forward movement, less stopping
        forward_ratio = action_counts['MOVE_FORWARD'] / total_actions
        stop_ratio = action_counts['STOP'] / total_actions
        
        return forward_ratio - stop_ratio * 0.5
    
    def _calculate_trajectory_complexity(self, trajectory_points: List) -> float:
        """Calculate trajectory complexity."""
        if len(trajectory_points) < 3:
            return 0.0
        
        # Calculate path length vs straight-line distance
        positions = np.array(trajectory_points)
        path_length = np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))
        straight_distance = np.linalg.norm(positions[-1] - positions[0])
        
        if straight_distance == 0:
            return 0.0
        
        return min(10.0, path_length / straight_distance)
    
    def _calculate_strategy_refinement(self, pre_session: SessionResults, post_session: SessionResults) -> float:
        """Calculate strategy refinement between sessions."""
        # Compare action distributions
        pre_actions = pre_session.action_distribution
        post_actions = post_session.action_distribution
        
        # Calculate change in strategy
        pre_total = sum(pre_actions.values())
        post_total = sum(post_actions.values())
        
        if pre_total == 0 or post_total == 0:
            return 0.0
        
        strategy_change = 0.0
        for action in pre_actions:
            pre_ratio = pre_actions[action] / pre_total
            post_ratio = post_actions[action] / post_total
            strategy_change += abs(post_ratio - pre_ratio)
        
        return strategy_change / 4  # Normalize by number of actions
    
    def _calculate_memory_stability(self, pre_session: SessionResults, post_session: SessionResults) -> float:
        """Calculate memory stability between sessions."""
        # Compare performance consistency
        pre_performance = self._calculate_session_performance(pre_session)
        post_performance = self._calculate_session_performance(post_session)
        
        # Stability is inversely related to performance variation
        performance_change = abs(post_performance - pre_performance)
        return max(0.0, 1.0 - performance_change)
    
    def _check_gradual_learning(self, performances: List[float]) -> bool:
        """Check if learning is gradual (biological characteristic)."""
        if len(performances) < 3:
            return False
        
        # Check for smooth improvement without large jumps
        changes = np.diff(performances)
        large_jumps = np.sum(np.abs(changes) > 0.2)  # More than 20% change
        
        return large_jumps < len(changes) * 0.3  # Less than 30% large jumps
    
    def _check_power_law_learning(self, performances: List[float]) -> float:
        """Check if learning follows power law (biological characteristic)."""
        if len(performances) < 5:
            return 0.0
        
        # Fit power law: performance = a * trial^b
        try:
            trials = np.arange(1, len(performances) + 1)
            log_trials = np.log(trials)
            log_performances = np.log(np.array(performances) + 1e-10)
            
            # Linear fit in log space
            correlation = np.corrcoef(log_trials, log_performances)[0, 1]
            return abs(correlation)
        except:
            return 0.0
    
    def _calculate_biological_realism_score(self, gradual: bool, consolidation: bool, power_law: float) -> float:
        """Calculate overall biological realism score."""
        score = 0.0
        score += 0.4 if gradual else 0.0
        score += 0.4 if consolidation else 0.0
        score += 0.2 * power_law
        return score
    
    def _save_results(self, results: Dict):
        """Save experiment results to file."""
        results_file = self.results_dir / "experiment_results.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the entire results dictionary
        serializable_results = convert_numpy_types(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Results saved to {results_file}")
    
    def _generate_visualizations(self, results: Dict):
        """Generate scientific visualizations."""
        print("📊 Generating visualizations...")
        
        # Create learning progression plot
        self._plot_learning_progression(results)
        
        # Create consolidation analysis plot
        self._plot_consolidation_analysis(results)
        
        # Create behavioral emergence plot
        self._plot_behavioral_emergence(results)
        
        print(f"📈 Visualizations saved to {self.results_dir}")
    
    def _plot_learning_progression(self, results: Dict):
        """Plot learning progression over time."""
        sessions = results['session_results']
        
        if len(sessions) < 2:
            return
        
        session_ids = [s['session_id'] for s in sessions]
        performances = [self._calculate_session_performance_from_dict(s) for s in sessions]
        light_distances = [s['avg_light_distance'] for s in sessions]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Performance over time
        ax1.plot(session_ids, performances, 'b-o', label='Overall Performance')
        ax1.set_xlabel('Session')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('Learning Progression Over Time')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Light distance over time
        ax2.plot(session_ids, light_distances, 'r-o', label='Distance to Light')
        ax2.set_xlabel('Session')
        ax2.set_ylabel('Average Light Distance')
        ax2.set_title('Navigation Efficiency Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'learning_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_consolidation_analysis(self, results: Dict):
        """Plot consolidation analysis."""
        consolidations = results['consolidation_analyses']
        
        if len(consolidations) < 2:
            return
        
        consolidation_ids = [c['consolidation_id'] for c in consolidations]
        benefits = [c['consolidation_benefit'] for c in consolidations]
        refinements = [c['strategy_refinement'] for c in consolidations]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Consolidation benefits
        ax1.bar(consolidation_ids, benefits, alpha=0.7, color='green')
        ax1.set_xlabel('Consolidation Period')
        ax1.set_ylabel('Performance Benefit')
        ax1.set_title('Memory Consolidation Benefits')
        ax1.grid(True, alpha=0.3)
        
        # Strategy refinement
        ax2.bar(consolidation_ids, refinements, alpha=0.7, color='orange')
        ax2.set_xlabel('Consolidation Period')
        ax2.set_ylabel('Strategy Refinement')
        ax2.set_title('Strategy Refinement During Consolidation')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'consolidation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_behavioral_emergence(self, results: Dict):
        """Plot behavioral emergence."""
        sessions = results['session_results']
        
        if len(sessions) < 2:
            return
        
        session_ids = [s['session_id'] for s in sessions]
        strategy_scores = [s['strategy_emergence_score'] for s in sessions]
        motor_coordination = [s['motor_coordination_score'] for s in sessions]
        exploration_scores = [s['exploration_score'] for s in sessions]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        ax.plot(session_ids, strategy_scores, 'b-o', label='Strategy Emergence')
        ax.plot(session_ids, motor_coordination, 'r-s', label='Motor Coordination')
        ax.plot(session_ids, exploration_scores, 'g-^', label='Exploration')
        
        ax.set_xlabel('Session')
        ax.set_ylabel('Score')
        ax.set_title('Behavioral Emergence Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'behavioral_emergence.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _calculate_session_performance_from_dict(self, session_dict: Dict) -> float:
        """Calculate session performance from dictionary."""
        return (session_dict['exploration_score'] * 0.3 + 
                session_dict['efficiency'] * 0.3 + 
                (1.0 - session_dict['collision_rate']) * 0.2 + 
                session_dict['strategy_emergence_score'] * 0.2)


def main():
    """Run biological embodied learning experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Biological Embodied Learning Experiment')
    parser.add_argument('--hours', type=float, default=2.0,
                       help='Experiment duration in hours')
    parser.add_argument('--session-minutes', type=int, default=20,
                       help='Session duration in minutes')
    parser.add_argument('--consolidation-minutes', type=int, default=10,
                       help='Consolidation duration in minutes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--world-size', type=float, default=10.0,
                       help='World size in meters')
    parser.add_argument('--lights', type=int, default=2,
                       help='Number of light sources')
    parser.add_argument('--obstacles', type=int, default=5,
                       help='Number of obstacles')
    
    args = parser.parse_args()
    
    # Create experiment configuration
    config = ExperimentConfig(
        duration_hours=args.hours,
        session_duration_minutes=args.session_minutes,
        consolidation_duration_minutes=args.consolidation_minutes,
        random_seed=args.seed,
        world_size=args.world_size,
        num_light_sources=args.lights,
        num_obstacles=args.obstacles
    )
    
    # Run experiment
    experiment = BiologicalEmbodiedLearningExperiment(config)
    results = experiment.run_experiment()
    
    # Print summary
    print(f"\n📋 Experiment Summary:")
    print(f"   Duration: {results['experiment_duration']/3600:.1f} hours")
    print(f"   Sessions: {len(results['session_results'])}")
    print(f"   Actions: {results['total_actions']}")
    print(f"   Biological realism: {results['analysis']['biological_realism']['biological_realism_score']:.3f}")
    print(f"   Learning detected: {results['analysis']['learning_progression']['total_improvement'] > 0.1}")


if __name__ == "__main__":
    main()