"""
Motivation System - Manages multiple competing drives and resolves action selection.
Provides a clean, extensible architecture for robot motivations.
"""

from typing import Dict, List, Type, Any, Optional
from dataclasses import dataclass
import importlib
import pkgutil
from .base_drive import BaseDrive, DriveContext, ActionEvaluation
from .action_proficiency import ActionMaturationSystem
from core.experience_based_actions import ExperienceBasedActionSystem


@dataclass
class MotivationResult:
    """Result of motivation system evaluation."""
    chosen_action: Dict[str, float]          # Selected motor action
    total_score: float                       # Combined motivation score
    drive_contributions: Dict[str, float]    # How much each drive contributed
    dominant_drive: str                      # Which drive had the most influence
    confidence: float                        # Overall confidence in decision
    reasoning: str                          # Human-readable explanation
    urgency: float                          # How urgent this action is
    alternative_actions: List[Dict]         # Other actions considered


class MotivationSystem:
    """
    Manages multiple robot drives and resolves action selection.
    
    This system allows for dynamic addition/removal of drives and
    provides a clean interface for multi-motivation decision making.
    """
    
    def __init__(self, world_graph=None):
        self.drives: Dict[str, BaseDrive] = {}
        self.drive_history: List[Dict] = []
        self.action_history: List[Dict] = []
        self.total_evaluations = 0
        
        # Action proficiency and maturation system
        self.world_graph = world_graph
        self.action_maturation = ActionMaturationSystem(world_graph) if world_graph else None
        self.enable_maturation = True  # Can be toggled for comparison
        
        # Experience-based action system
        self.experience_actions = ExperienceBasedActionSystem(world_graph) if world_graph else None
        self.enable_experience_actions = True  # Can be toggled for comparison
        
        # Drive-specific pain/pleasure learning (replaces global system)
        # Each drive now handles its own pain/pleasure associations
        
    def add_drive(self, drive: BaseDrive):
        """Add a drive to the motivation system."""
        if drive.name in self.drives:
            print(f"Warning: Replacing existing drive '{drive.name}'")
        
        self.drives[drive.name] = drive
        print(f"Added drive: {drive.name} (base weight: {drive.base_weight:.2f}, fluid)")
    
    def remove_drive(self, drive_name: str) -> bool:
        """Remove a drive from the motivation system."""
        if drive_name in self.drives:
            del self.drives[drive_name]
            print(f"Removed drive: {drive_name}")
            return True
        else:
            print(f"Warning: Drive '{drive_name}' not found")
            return False
    
    def get_drive(self, drive_name: str) -> Optional[BaseDrive]:
        """Get a specific drive by name."""
        return self.drives.get(drive_name)
    
    def list_drives(self) -> List[str]:
        """Get list of all active drive names."""
        return list(self.drives.keys())
    
    def evaluate_action_candidates(self, action_candidates: List[Dict[str, float]], 
                                 context: DriveContext) -> MotivationResult:
        """
        Evaluate multiple action candidates using all active drives.
        
        Args:
            action_candidates: List of potential motor actions to evaluate
            context: Current situation context
            
        Returns:
            MotivationResult with chosen action and reasoning
        """
        if not self.drives:
            raise ValueError("No drives available for evaluation")
        
        if not action_candidates:
            raise ValueError("No action candidates provided")
        
        self.total_evaluations += 1
        
        # Update all drive states
        for drive in self.drives.values():
            drive.update_drive_state(context)
        
        # Evaluate each action candidate
        candidate_results = []
        
        for action in action_candidates:
            # Get evaluation from each drive
            drive_evaluations: Dict[str, ActionEvaluation] = {}
            
            for drive_name, drive in self.drives.items():
                evaluation = drive.evaluate_action(action, context)
                
                # Get drive-specific pain/pleasure predictions
                expected_pain, expected_pleasure = drive.evaluate_action_pain_pleasure(action, context)
                evaluation.expected_pain = expected_pain
                evaluation.expected_pleasure = expected_pleasure
                
                drive_evaluations[drive_name] = evaluation
            
            # Calculate aggregated pain/pleasure bias from all drives
            pain_pleasure_bias = 0.0
            for evaluation in drive_evaluations.values():
                pain_pleasure_bias += evaluation.get_pain_pleasure_bias()
            
            # Calculate weighted total score
            total_score = 0.0
            total_weight = 0.0
            drive_contributions = {}
            max_urgency = 0.0
            
            for drive_name, evaluation in drive_evaluations.items():
                drive = self.drives[drive_name]
                contribution = evaluation.weighted_score(drive.current_weight)
                
                drive_contributions[drive_name] = contribution
                total_score += contribution
                total_weight += drive.current_weight
                max_urgency = max(max_urgency, evaluation.urgency)
            
            # Normalize score and apply pain/pleasure bias
            if total_weight > 0:
                normalized_score = total_score / total_weight
            else:
                normalized_score = 0.0
            
            # Apply pain/pleasure learning bias (can override drive preferences)
            final_score = normalized_score + (pain_pleasure_bias * 0.5)
            final_score = max(0.0, min(1.0, final_score))  # Clamp to valid range
            
            # Find dominant drive
            dominant_drive = max(drive_contributions.keys(), 
                               key=lambda d: drive_contributions[d])
            
            # Calculate overall confidence
            confidences = [eval.confidence for eval in drive_evaluations.values()]
            overall_confidence = sum(confidences) / len(confidences)
            
            # Generate combined reasoning
            dominant_eval = drive_evaluations[dominant_drive]
            reasoning = f"{dominant_drive}: {dominant_eval.reasoning}"
            
            # Add drive-specific pain/pleasure reasoning if bias was applied
            if abs(pain_pleasure_bias) > 0.1:
                bias_type = "avoidance" if pain_pleasure_bias < 0 else "seeking"
                reasoning += f" (drive-specific {bias_type}: {pain_pleasure_bias:.2f})"
                
                # Add detailed drive pain/pleasure breakdown
                drive_pain_pleasure_details = []
                for drive_name, evaluation in drive_evaluations.items():
                    if abs(evaluation.get_pain_pleasure_bias()) > 0.05:
                        drive_bias = evaluation.get_pain_pleasure_bias()
                        drive_pain_pleasure_details.append(f"{drive_name}: {drive_bias:.2f}")
                
                if drive_pain_pleasure_details:
                    reasoning += f" [{', '.join(drive_pain_pleasure_details)}]"
            
            candidate_results.append({
                'action': action,
                'score': final_score,  # Use final score with pain/pleasure bias
                'drive_contributions': drive_contributions,
                'dominant_drive': dominant_drive,
                'confidence': overall_confidence,
                'reasoning': reasoning,
                'urgency': max_urgency,
                'evaluations': drive_evaluations,
                'pain_pleasure_bias': pain_pleasure_bias
            })
        
        # Select best action
        best_result = max(candidate_results, key=lambda r: r['score'])
        
        # Create motivation result
        motivation_result = MotivationResult(
            chosen_action=best_result['action'],
            total_score=best_result['score'],
            drive_contributions=best_result['drive_contributions'],
            dominant_drive=best_result['dominant_drive'],
            confidence=best_result['confidence'],
            reasoning=best_result['reasoning'],
            urgency=best_result['urgency'],
            alternative_actions=[r['action'] for r in candidate_results if r != best_result]
        )
        
        # Log decision
        self._log_decision(motivation_result, context)
        
        return motivation_result
    
    def generate_action_candidates(self, context: DriveContext, 
                                 num_candidates: int = 5) -> List[Dict[str, float]]:
        """
        Generate diverse action candidates for evaluation.
        
        Prioritizes experience-based actions, then maturation-influenced actions,
        with fallback to static template generation.
        
        Args:
            context: Current situation context
            num_candidates: Number of action candidates to generate
            
        Returns:
            List of potential motor actions
        """
        candidates = []
        
        # Method 1: Experience-based actions (highest priority)
        if self.experience_actions and self.enable_experience_actions:
            experience_actions = self.experience_actions.generate_experience_based_actions(
                context, max(2, num_candidates // 2)  # Use half the candidates for experience
            )
            
            for exp_action in experience_actions:
                candidates.append(exp_action.motor_action)
            
            # Learn motor patterns from recent experiences
            self.experience_actions.learn_motor_patterns(context)
        
        # Method 2: Maturation-influenced actions (medium priority)
        remaining_candidates = num_candidates - len(candidates)
        if remaining_candidates > 0 and self.action_maturation and self.enable_maturation:
            maturation_actions = self.action_maturation.generate_maturation_influenced_actions(
                context, remaining_candidates
            )
            candidates.extend(maturation_actions)
        
        # Method 3: Static template actions (fallback)
        remaining_candidates = num_candidates - len(candidates)
        if remaining_candidates > 0:
            static_actions = self._generate_static_action_candidates(context, remaining_candidates)
            candidates.extend(static_actions)
        
        return candidates[:num_candidates]
    
    def _generate_static_action_candidates(self, context: DriveContext,
                                         num_candidates: int = 5) -> List[Dict[str, float]]:
        """Generate action candidates using the original static template approach."""
        import random
        
        candidates = []
        
        # Conservative action (low movement)
        candidates.append({
            'forward_motor': random.uniform(0.1, 0.3),
            'turn_motor': 0.0,
            'brake_motor': random.uniform(0.0, 0.2)
        })
        
        # Aggressive forward action
        candidates.append({
            'forward_motor': random.uniform(0.4, 0.7),
            'turn_motor': 0.0,
            'brake_motor': 0.0
        })
        
        # Turning actions (left and right)
        candidates.append({
            'forward_motor': 0.0,
            'turn_motor': random.choice([-0.6, 0.6]),
            'brake_motor': 0.0
        })
        
        # Backward action
        candidates.append({
            'forward_motor': random.uniform(-0.6, -0.3),
            'turn_motor': 0.0,
            'brake_motor': 0.0
        })
        
        # Emergency stop
        candidates.append({
            'forward_motor': 0.0,
            'turn_motor': 0.0,
            'brake_motor': 1.0
        })
        
        # Generate additional random candidates
        while len(candidates) < num_candidates:
            candidates.append({
                'forward_motor': random.uniform(-0.7, 0.7),
                'turn_motor': random.uniform(-0.7, 0.7),
                'brake_motor': random.uniform(0.0, 0.4)
            })
        
        return candidates[:num_candidates]
    
    def make_decision(self, context: DriveContext, 
                     custom_candidates: Optional[List[Dict[str, float]]] = None) -> MotivationResult:
        """
        Make a complete motivation-driven decision.
        
        Args:
            context: Current situation context
            custom_candidates: Optional custom action candidates
            
        Returns:
            MotivationResult with chosen action
        """
        # Check for natural homeostatic rest first
        if self.is_in_homeostatic_rest(context):
            rest_action = self.get_homeostatic_action(context)
            return MotivationResult(
                chosen_action=rest_action,
                total_score=0.05,  # Low score indicates rest state
                drive_contributions={'homeostatic_rest': 0.05},
                dominant_drive='homeostatic_rest',
                confidence=0.9,  # High confidence in rest decision
                reasoning='Natural homeostatic rest - all drives satisfied',
                urgency=0.0,
                alternative_actions=[]
            )
        
        # Generate action candidates if not provided
        if custom_candidates is None:
            action_candidates = self.generate_action_candidates(context)
        else:
            action_candidates = custom_candidates
        
        # Evaluate candidates
        result = self.evaluate_action_candidates(action_candidates, context)
        
        return result
    
    def _log_decision(self, result: MotivationResult, context: DriveContext):
        """Log decision for analysis and debugging."""
        decision_log = {
            'step': context.step_count,
            'chosen_action': result.chosen_action,
            'total_score': result.total_score,
            'dominant_drive': result.dominant_drive,
            'drive_contributions': result.drive_contributions,
            'confidence': result.confidence,
            'urgency': result.urgency,
            'robot_state': {
                'health': context.robot_health,
                'energy': context.robot_energy,
                'position': context.robot_position,
                'threat_level': context.threat_level
            }
        }
        
        self.action_history.append(decision_log)
        
        # Keep only recent history
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-50:]
    
    def get_motivation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the motivation system."""
        stats = {
            'total_drives': len(self.drives),
            'active_drives': list(self.drives.keys()),
            'total_evaluations': self.total_evaluations,
            'decision_history_length': len(self.action_history)
        }
        
        # Drive-specific statistics
        drive_stats = {}
        for drive_name, drive in self.drives.items():
            drive_stats[drive_name] = drive.get_drive_info()
        
        stats['drive_statistics'] = drive_stats
        
        # Recent decision patterns
        if self.action_history:
            recent_decisions = self.action_history[-10:]
            dominant_drives = [d['dominant_drive'] for d in recent_decisions]
            drive_dominance = {}
            for drive in dominant_drives:
                drive_dominance[drive] = drive_dominance.get(drive, 0) + 1
            
            stats['recent_drive_dominance'] = drive_dominance
            stats['average_recent_confidence'] = sum(d['confidence'] for d in recent_decisions) / len(recent_decisions)
            stats['average_recent_urgency'] = sum(d['urgency'] for d in recent_decisions) / len(recent_decisions)
        
        # Include current mood using recent robot state
        if self.action_history:
            # Use the most recent robot state for mood calculation
            recent_decision = self.action_history[-1]
            robot_state = recent_decision.get('robot_state', {})
            
            # Create context from recent robot state
            from .base_drive import DriveContext
            context = DriveContext(
                current_sensory=[],
                robot_health=robot_state.get('health', 1.0),
                robot_energy=robot_state.get('energy', 1.0),
                robot_position=robot_state.get('position', (0, 0)),
                robot_orientation=0,
                recent_experiences=[],
                prediction_errors=[],
                time_since_last_food=0,
                time_since_last_damage=0,
                threat_level=robot_state.get('threat_level', 'normal'),
                step_count=recent_decision.get('step', 0)
            )
            stats['mood'] = self.calculate_robot_mood(context)
        
        return stats
    
    def calculate_robot_mood(self, context) -> Dict[str, float]:
        """
        Calculate the robot's overall emotional state based on all drives.
        
        This is your brilliant aesthetic idea - the robot's "mood" as an emergent
        property of drive satisfaction, urgency, and confidence levels.
        
        Returns:
            Dictionary with mood dimensions and overall emotional state
        """
        if not self.drives:
            return {
                'overall_satisfaction': 0.0,
                'overall_urgency': 0.0, 
                'overall_confidence': 0.0,
                'mood_descriptor': 'neutral',
                'dominant_emotion': 'none'
            }
        
        # Collect mood contributions from all drives
        total_satisfaction = 0.0
        total_urgency = 0.0
        total_confidence = 0.0
        drive_moods = {}
        
        for drive_name, drive in self.drives.items():
            mood_contrib = drive.get_current_mood_contribution(context)
            drive_moods[drive_name] = mood_contrib
            
            # Weight by drive's current importance
            weight = drive.current_weight
            total_satisfaction += mood_contrib['satisfaction'] * weight
            total_urgency += mood_contrib['urgency'] * weight
            total_confidence += mood_contrib['confidence'] * weight
        
        # Normalize by total weight
        total_weight = sum(drive.current_weight for drive in self.drives.values())
        if total_weight > 0:
            avg_satisfaction = total_satisfaction / total_weight
            avg_urgency = total_urgency / total_weight
            avg_confidence = total_confidence / total_weight
        else:
            avg_satisfaction = avg_urgency = avg_confidence = 0.0
        
        # Determine mood descriptor based on satisfaction and urgency
        mood_descriptor = self._calculate_mood_descriptor(avg_satisfaction, avg_urgency, avg_confidence)
        
        # Find dominant emotion (drive with highest urgency)
        dominant_emotion = 'none'
        max_urgency = 0.0
        for drive_name, mood in drive_moods.items():
            if mood['urgency'] > max_urgency:
                max_urgency = mood['urgency']
                dominant_emotion = drive_name.lower()
        
        return {
            'overall_satisfaction': avg_satisfaction,
            'overall_urgency': avg_urgency,
            'overall_confidence': avg_confidence,
            'mood_descriptor': mood_descriptor,
            'dominant_emotion': dominant_emotion,
            'drive_moods': drive_moods
        }
    
    def _calculate_mood_descriptor(self, satisfaction: float, urgency: float, confidence: float) -> str:
        """Calculate a human-readable mood descriptor."""
        if urgency > 0.8:
            return 'panicked'
        elif urgency > 0.6:
            return 'stressed'
        elif satisfaction > 0.5:
            if confidence > 0.5:
                return 'content'
            else:
                return 'hopeful'
        elif satisfaction > 0.0:
            return 'calm'
        elif satisfaction > -0.3:
            return 'restless'
        elif satisfaction > -0.6:
            return 'frustrated'
        else:
            return 'miserable'
    
    def is_in_homeostatic_rest(self, context) -> bool:
        """
        Determine if robot is in natural homeostatic balance.
        
        Natural rest emerges when all drives have low activation - no artificial thresholds.
        """
        # Calculate total drive pressure (sum of all drive weights)
        total_drive_pressure = sum(drive.current_weight for drive in self.drives.values())
        
        # Natural rest threshold - adapts to typical drive pressure
        if not hasattr(self, 'typical_drive_pressure'):
            self.typical_drive_pressure = 0.5  # Initial estimate
            self.rest_threshold_samples = []
        
        # Update typical drive pressure (rolling average)
        self.rest_threshold_samples.append(total_drive_pressure)
        if len(self.rest_threshold_samples) > 50:
            self.rest_threshold_samples = self.rest_threshold_samples[-25:]
            self.typical_drive_pressure = sum(self.rest_threshold_samples) / len(self.rest_threshold_samples)
        
        # Natural rest when drive pressure is significantly below typical
        natural_rest_threshold = self.typical_drive_pressure * 0.2  # 20% of typical pressure
        
        return total_drive_pressure < natural_rest_threshold
    
    def get_homeostatic_action(self, context) -> Dict[str, float]:
        """
        Generate minimal action when in natural homeostatic rest.
        
        When all drives are naturally satisfied, robot takes minimal action.
        """
        return {
            'forward_motor': 0.0,  # Stay still
            'turn_motor': 0.0,     # No turning  
            'brake_motor': 0.05    # Very light brake to maintain position
        }
    
    def get_pain_pressure_summary(self, context) -> Dict[str, float]:
        """
        Your insight: behavior driven purely by pain minimization.
        
        Returns the "pain pressure" from each drive that would motivate action.
        """
        pain_pressures = {}
        
        for drive_name, drive in self.drives.items():
            mood = drive.get_current_mood_contribution(context)
            
            # Pain pressure = combination of dissatisfaction and urgency
            dissatisfaction = max(0.0, -mood['satisfaction'])  # Only negative satisfaction counts as pain
            urgency = mood['urgency']
            
            # Pain pressure motivates action to reduce this drive's pain
            pain_pressure = (dissatisfaction * 0.7) + (urgency * 0.3)
            pain_pressures[drive_name] = pain_pressure
        
        return pain_pressures
    
    def reset_system(self):
        """Reset the entire motivation system."""
        for drive in self.drives.values():
            drive.reset_drive()
        
        self.drive_history.clear()
        self.action_history.clear()
        self.total_evaluations = 0
        
        print("Motivation system reset")
    
    def load_drives_from_module(self, module_name: str = "drives"):
        """
        Automatically load all drive classes from a module.
        
        Args:
            module_name: Name of module to scan for drive classes
        """
        try:
            # Import the drives module
            drives_module = importlib.import_module(module_name)
            
            # Scan for drive classes
            for finder, name, ispkg in pkgutil.iter_modules(drives_module.__path__, drives_module.__name__ + "."):
                try:
                    module = importlib.import_module(name)
                    
                    # Look for classes that inherit from BaseDrive
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        
                        if (isinstance(attr, type) and 
                            issubclass(attr, BaseDrive) and 
                            attr != BaseDrive):
                            
                            # Instantiate and add the drive
                            drive_instance = attr()
                            self.add_drive(drive_instance)
                            
                except Exception as e:
                    print(f"Warning: Could not load drive from {name}: {e}")
                    
        except Exception as e:
            print(f"Warning: Could not load drives from module {module_name}: {e}")
    
    def record_action_outcome(self, action: Dict[str, float], 
                            prediction_accuracy: float,
                            execution_success: bool = True):
        """
        Record the outcome of an action for proficiency tracking and maturation.
        
        Args:
            action: The motor action that was executed
            prediction_accuracy: How accurate our prediction was (0.0 to 1.0)
            execution_success: Whether the action was executed successfully
        """
        if self.action_maturation:
            self.action_maturation.record_action_outcome(
                action, prediction_accuracy, execution_success
            )
    
    def get_maturation_status(self) -> Dict[str, Any]:
        """Get comprehensive maturation and proficiency status."""
        if not self.action_maturation:
            return {
                'maturation_enabled': False,
                'message': 'Action maturation system not available (no world_graph provided)'
            }
        
        status = self.action_maturation.get_maturation_status()
        status['maturation_enabled'] = self.enable_maturation
        return status
    
    def toggle_maturation(self, enabled: bool = None):
        """Toggle maturation system on/off for comparison testing."""
        if enabled is None:
            self.enable_maturation = not self.enable_maturation
        else:
            self.enable_maturation = enabled
        
        status = "enabled" if self.enable_maturation else "disabled"
        print(f"🧠 Action maturation system {status}")
    
    def toggle_experience_actions(self, enabled: bool):
        """Toggle experience-based action generation."""
        self.enable_experience_actions = enabled
        status = "enabled" if enabled else "disabled"
        print(f"🧠 Experience-based actions {status}")
    
    def get_experience_action_statistics(self) -> Dict[str, Any]:
        """Get statistics about experience-based action generation."""
        if self.experience_actions:
            return self.experience_actions.get_experience_action_statistics()
        return {
            "total_experience_actions": 0,
            "successful_experience_actions": 0,
            "experience_action_success_rate": 0.0,
            "total_motor_patterns": 0,
            "pattern_usage_stats": {},
            "avg_pattern_success_rate": 0.0
        }
    
    def get_pain_pleasure_statistics(self) -> Dict[str, Any]:
        """Get statistics about drive-specific pain/pleasure learning."""
        stats = {
            "drive_specific_learning": True,
            "total_drives": len(self.drives),
            "drive_pain_pleasure_stats": {}
        }
        
        # Get stats from each drive
        for drive_name, drive in self.drives.items():
            if hasattr(drive, 'get_pain_pleasure_statistics'):
                stats["drive_pain_pleasure_stats"][drive_name] = drive.get_pain_pleasure_statistics()
            else:
                stats["drive_pain_pleasure_stats"][drive_name] = {
                    "learning_implemented": False,
                    "associations_learned": 0
                }
        
        return stats
    
    def learn_from_action_outcome(self, action: Dict[str, float], context: DriveContext, 
                                 outcome_experiences: List[Any]):
        """
        Learn from action outcomes using drive-specific pain/pleasure evaluation.
        
        This replaces the global pain/pleasure learning with drive-specific learning.
        Each drive evaluates the outcome and learns its own associations.
        
        Args:
            action: The action that was taken
            context: The context when the action was taken
            outcome_experiences: List of experiences resulting from the action
        """
        for drive_name, drive in self.drives.items():
            # Each drive evaluates the outcome for its own domain
            total_pain = 0.0
            total_pleasure = 0.0
            
            for experience in outcome_experiences:
                # Get drive-specific pain/pleasure from the experience
                valence = drive.evaluate_experience_valence(experience, context)
                
                if valence < 0:
                    total_pain += valence
                elif valence > 0:
                    total_pleasure += valence
            
            # Normalize by number of experiences
            if outcome_experiences:
                total_pain /= len(outcome_experiences)
                total_pleasure /= len(outcome_experiences)
            
            # Let the drive learn from this outcome
            drive.learn_pain_pleasure_association(action, context, total_pain, total_pleasure)


def create_default_motivation_system(world_graph=None, world_width=40, world_height=40) -> MotivationSystem:
    """Create a motivation system with the three core drives."""
    from .curiosity_drive import CuriosityDrive
    from .survival_drive import SurvivalDrive
    from .exploration_drive import ExplorationDrive
    
    system = MotivationSystem(world_graph=world_graph)
    
    # Add the three core drives
    system.add_drive(CuriosityDrive(base_weight=0.4))
    system.add_drive(SurvivalDrive(base_weight=0.4)) 
    system.add_drive(ExplorationDrive(base_weight=0.2, world_width=world_width, world_height=world_height))
    
    return system