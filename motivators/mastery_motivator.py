"""
Mastery Drive - Seeks competence and skill development through prediction accuracy improvement.

This drive focuses on getting good at things rather than finding new things.
It optimizes for:
- Prediction accuracy improvement
- Skill refinement and efficiency
- Consistent performance
- Competence development
- Flow state achievement

The drive is satisfied by improvement and mastery, not by novelty.
"""

import math
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from .base_motivator import BaseMotivator, MotivatorContext, ActionEvaluation
from core.world_graph import WorldGraph

# Import GPU sensory predictor for predictive mastery assessment
try:
    from prediction.sensory.gpu_sensory_predictor import GPUSensoryPredictor
    from prediction.sensory.sensory_predictor import SensoryPrediction
    GPU_PREDICTION_AVAILABLE = True
except ImportError:
    GPU_PREDICTION_AVAILABLE = False


class MasteryMotivator(BaseMotivator):
    """
    Mastery Drive focused on competence and skill development.
    
    This drive seeks to improve prediction accuracy and develop competence
    in activities and environments. It is satisfied by:
    - Improving prediction accuracy
    - Developing consistent performance
    - Achieving flow states
    - Mastering challenging tasks
    
    Unlike curiosity, mastery is satisfied by repetition and refinement.
    """
    
    def __init__(self, base_weight: float = 0.25):
        super().__init__("Mastery", base_weight)
        
        # Competence tracking
        self.skill_levels = defaultdict(float)  # Skill level for different activities
        self.accuracy_history = deque(maxlen=20)  # Recent prediction accuracy
        self.performance_trends = defaultdict(list)  # Performance trends by context
        
        # Mastery progression tracking
        self.mastery_goals = {}  # Current mastery goals
        self.achieved_masteries = set()  # Completed mastery achievements
        self.current_focus_area = None  # Current area of focus
        
        # Flow state indicators
        self.flow_indicators = {
            'consistency': 0.0,  # Consistency of performance
            'challenge_level': 0.0,  # Appropriate challenge level
            'engagement': 0.0,  # Engagement with current task
            'efficiency': 0.0,  # Efficiency of actions
        }
        
        # Learning and improvement tracking
        self.improvement_rate = 0.0
        self.practice_sessions = defaultdict(int)
        self.competence_confidence = 0.0
        
        # Performance metrics
        self.total_mastery_attempts = 0
        self.successful_improvements = 0
        
        # GPU sensory predictor for predictive mastery assessment
        self.gpu_predictor = None
        self.predictive_mastery_enabled = False
        
    def initialize_gpu_predictor(self, world_graph):
        """Initialize GPU predictor for predictive mastery assessment."""
        if GPU_PREDICTION_AVAILABLE and world_graph is not None:
            try:
                self.gpu_predictor = GPUSensoryPredictor(world_graph)
                self.predictive_mastery_enabled = True
                print(f"🏆 MasteryDrive: GPU predictive mastery assessment enabled")
            except Exception as e:
                print(f"⚠️  MasteryDrive: Could not enable GPU predictor: {e}")
                self.predictive_mastery_enabled = False
        else:
            self.predictive_mastery_enabled = False
            
    def evaluate_action(self, action: Dict[str, float], context: MotivatorContext) -> ActionEvaluation:
        """
        Evaluate action based on its potential to improve competence and mastery.
        """
        self.total_evaluations += 1
        self.total_mastery_attempts += 1
        
        # Calculate mastery-related scores
        accuracy_improvement_potential = self._calculate_accuracy_improvement_potential(action, context)
        skill_development_potential = self._calculate_skill_development_potential(action, context)
        consistency_potential = self._calculate_consistency_potential(action, context)
        flow_potential = self._calculate_flow_potential(action, context)
        
        # Combine mastery scores
        mastery_score = (
            accuracy_improvement_potential * 0.35 +
            skill_development_potential * 0.3 +
            consistency_potential * 0.2 +
            flow_potential * 0.15
        )
        
        # Apply mastery focus boost
        focus_boost = self._calculate_mastery_focus_boost(action, context)
        final_score = min(1.0, mastery_score + focus_boost)
        
        # Calculate confidence based on mastery understanding
        confidence = self._calculate_mastery_confidence(
            accuracy_improvement_potential, skill_development_potential, consistency_potential
        )
        
        # Generate reasoning
        reasoning = self._generate_mastery_reasoning(
            accuracy_improvement_potential, skill_development_potential, consistency_potential, flow_potential
        )
        
        # Calculate urgency based on improvement opportunities
        urgency = self._calculate_mastery_urgency(context)
        
        return ActionEvaluation(
            drive_name=self.name,
            action_score=final_score,
            confidence=confidence,
            reasoning=reasoning,
            urgency=urgency
        )
    
    def _calculate_accuracy_improvement_potential(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate potential for improving prediction accuracy - using PREDICTIVE assessment if available.
        
        This is the core of predictive mastery: predicting how actions will improve our predictions!
        """
        # Use predictive assessment if available
        if self.predictive_mastery_enabled and self.gpu_predictor:
            return self._calculate_predictive_accuracy_improvement(action, context)
        else:
            # Fallback to reactive assessment
            return self._calculate_reactive_accuracy_improvement(action, context)
    
    def _calculate_predictive_accuracy_improvement(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Predict how much this action will improve our prediction accuracy.
        
        This is meta-prediction: using prediction to improve prediction itself!
        """
        try:
            # Predict what would happen if we take this action
            current_context = context.current_sensory[:8] if len(context.current_sensory) >= 8 else context.current_sensory
            prediction = self.gpu_predictor.predict_sensory_outcome(action, current_context)
            
            # Calculate predicted learning opportunity
            predicted_learning_value = self._calculate_predicted_learning_value(prediction, context)
            
            # Calculate predicted competence improvement
            predicted_competence_gain = self._calculate_predicted_competence_gain(prediction, context)
            
            # Calculate predicted pattern completion
            predicted_pattern_completion = self._calculate_predicted_pattern_completion(prediction, context)
            
            # Combine improvement predictions
            base_improvement = (
                predicted_learning_value * 0.4 +
                predicted_competence_gain * 0.4 +
                predicted_pattern_completion * 0.2
            )
            
            # Weight by prediction confidence
            confidence_weighted_improvement = base_improvement * prediction.confidence
            
            # Add bonus for high-confidence predictions of significant learning
            if prediction.confidence > 0.8 and base_improvement > 0.7:
                confidence_weighted_improvement += 0.1
            
            return max(0.0, min(1.0, confidence_weighted_improvement))
            
        except Exception as e:
            # Fallback to reactive assessment if prediction fails
            return self._calculate_reactive_accuracy_improvement(action, context)
    
    def _calculate_predicted_learning_value(self, prediction: SensoryPrediction, context: MotivatorContext) -> float:
        """Calculate how much learning value this predicted outcome would provide."""
        # Learning value is higher when:
        # 1. Prediction uncertainty is moderate (not too easy, not impossible)
        # 2. The experience would fill gaps in our knowledge
        # 3. The prediction confidence suggests we can learn from it
        
        # Calculate uncertainty as inverse of confidence
        uncertainty = 1.0 - prediction.confidence
        
        # Optimal learning happens at moderate uncertainty (not too easy, not impossible)
        if 0.3 < uncertainty < 0.7:
            learning_value = 0.8  # Sweet spot for learning
        elif 0.1 < uncertainty < 0.9:
            learning_value = 0.6  # Good learning opportunity
        else:
            learning_value = 0.3  # Either too easy or too hard
        
        return learning_value
    
    def _calculate_predicted_competence_gain(self, prediction: SensoryPrediction, context: MotivatorContext) -> float:
        """Calculate how much competence this predicted outcome would develop."""
        # Competence gain is higher when:
        # 1. We're practicing skills at appropriate difficulty
        # 2. The predicted outcome allows for skill refinement
        # 3. We're not repeating overly familiar patterns
        
        context_sig = self._create_context_signature(context)
        current_skill = self.skill_levels.get(context_sig, 0.0)
        
        # Use prediction confidence as a proxy for expected competence
        expected_competence = prediction.confidence
        
        # Calculate competence gain based on skill level and expected performance
        if current_skill < 0.3:
            # Beginner: high gain potential
            competence_gain = expected_competence * 0.8
        elif current_skill < 0.7:
            # Intermediate: optimal practice zone
            competence_gain = expected_competence * 1.0
        elif current_skill < 0.9:
            # Advanced: refinement opportunities
            competence_gain = expected_competence * 0.6
        else:
            # Expert: diminishing returns
            competence_gain = expected_competence * 0.2
        
        return competence_gain
    
    def _calculate_predicted_pattern_completion(self, prediction: SensoryPrediction, context: MotivatorContext) -> float:
        """Calculate how much this predicted outcome would complete patterns."""
        # Pattern completion is higher when:
        # 1. The prediction fills in missing pieces of understanding
        # 2. It connects to existing knowledge in meaningful ways
        # 3. It contributes to a coherent model of the world
        
        # Use prediction quality as a proxy for pattern completion
        pattern_completion = prediction.get_prediction_quality()
        
        # Boost if we have few experiences in this context (pattern building)
        context_sig = self._create_context_signature(context)
        experience_count = self.practice_sessions.get(context_sig, 0)
        
        if experience_count < 5:
            # Few experiences: high pattern completion value
            pattern_completion *= 1.2
        elif experience_count < 20:
            # Moderate experiences: good pattern completion
            pattern_completion *= 1.0
        else:
            # Many experiences: diminishing pattern completion
            pattern_completion *= 0.8
        
        return min(1.0, pattern_completion)
    
    def _calculate_reactive_accuracy_improvement(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """Fallback reactive accuracy improvement calculation."""
        if not context.prediction_errors:
            return 0.5  # Unknown potential
        
        # Calculate recent prediction error trend
        recent_errors = context.prediction_errors[-5:] if len(context.prediction_errors) >= 5 else context.prediction_errors
        avg_error = sum(recent_errors) / len(recent_errors)
        
        # High error = high improvement potential
        improvement_potential = avg_error
        
        # Boost potential if errors are decreasing (we're learning)
        if len(recent_errors) >= 3:
            recent_trend = recent_errors[-1] - recent_errors[-3]
            if recent_trend < 0:  # Errors decreasing
                improvement_potential += 0.2
        
        return min(1.0, improvement_potential)
    
    def _calculate_skill_development_potential(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate potential for developing skills in current context.
        
        Higher potential when:
        - We have some experience but aren't masters yet
        - Action represents practice opportunity
        - Context allows for skill refinement
        """
        # Create context signature for skill tracking
        context_sig = self._create_context_signature(context)
        current_skill = self.skill_levels.get(context_sig, 0.0)
        
        # Sweet spot for skill development: some experience but not mastery
        if current_skill < 0.3:
            return 0.7  # Good learning opportunity
        elif current_skill < 0.7:
            return 0.9  # Optimal practice zone
        elif current_skill < 0.9:
            return 0.5  # Refinement opportunity
        else:
            return 0.1  # Already mastered
    
    def _calculate_consistency_potential(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate potential for improving consistency of performance.
        
        Higher potential when:
        - Performance has been variable
        - Action could improve reliability
        - Context allows for consistency building
        """
        if not self.accuracy_history:
            return 0.5  # Unknown consistency
        
        # Calculate variance in recent performance
        recent_accuracy = list(self.accuracy_history)[-10:]  # Last 10 performances
        if len(recent_accuracy) < 3:
            return 0.5
        
        mean_accuracy = sum(recent_accuracy) / len(recent_accuracy)
        variance = sum((acc - mean_accuracy) ** 2 for acc in recent_accuracy) / len(recent_accuracy)
        
        # High variance = high potential for consistency improvement
        consistency_potential = min(1.0, variance * 5.0)  # Scale variance
        
        return consistency_potential
    
    def _calculate_flow_potential(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate potential for achieving flow state.
        
        Flow occurs when:
        - Challenge matches skill level
        - Clear goals and feedback
        - Deep engagement possible
        """
        context_sig = self._create_context_signature(context)
        current_skill = self.skill_levels.get(context_sig, 0.0)
        
        # Estimate challenge level from prediction errors
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1] if context.prediction_errors else 0.5
            challenge_level = recent_error  # High error = high challenge
        else:
            challenge_level = 0.5
        
        # Flow potential is highest when challenge matches skill
        skill_challenge_match = 1.0 - abs(current_skill - challenge_level)
        
        # Boost flow potential if we're in a focused state
        engagement_boost = self.flow_indicators.get('engagement', 0.0) * 0.3
        
        return min(1.0, skill_challenge_match + engagement_boost)
    
    def _calculate_mastery_focus_boost(self, action: Dict[str, float], context: MotivatorContext) -> float:
        """
        Calculate boost for actions that align with current mastery focus.
        """
        if not self.current_focus_area:
            return 0.0
        
        # If action is in our current focus area, provide boost
        context_sig = self._create_context_signature(context)
        if context_sig == self.current_focus_area:
            return 0.2
        
        return 0.0
    
    def _calculate_mastery_confidence(self, accuracy_potential: float, skill_potential: float, consistency_potential: float) -> float:
        """Calculate confidence in mastery evaluation."""
        # Higher confidence when potentials are clear and high
        max_potential = max(accuracy_potential, skill_potential, consistency_potential)
        potential_clarity = max_potential - min(accuracy_potential, skill_potential, consistency_potential)
        
        base_confidence = max_potential * 0.6
        clarity_bonus = potential_clarity * 0.4
        
        return min(1.0, base_confidence + clarity_bonus)
    
    def _generate_mastery_reasoning(self, accuracy_potential: float, skill_potential: float, 
                                  consistency_potential: float, flow_potential: float) -> str:
        """Generate reasoning for mastery evaluation."""
        max_potential = max(accuracy_potential, skill_potential, consistency_potential, flow_potential)
        
        if max_potential == accuracy_potential and accuracy_potential > 0.7:
            prediction_type = "predicted" if self.predictive_mastery_enabled else "reactive"
            return f"High accuracy improvement potential ({accuracy_potential:.2f}) - {prediction_type} learning opportunity"
        elif max_potential == skill_potential and skill_potential > 0.7:
            return f"High skill development potential ({skill_potential:.2f}) - practice opportunity"
        elif max_potential == consistency_potential and consistency_potential > 0.7:
            return f"High consistency improvement potential ({consistency_potential:.2f}) - reliability focus"
        elif max_potential == flow_potential and flow_potential > 0.7:
            return f"High flow potential ({flow_potential:.2f}) - optimal challenge level"
        elif max_potential > 0.5:
            return f"Good mastery opportunity ({max_potential:.2f}) - competence building"
        else:
            return f"Limited mastery potential ({max_potential:.2f}) - already competent"
    
    def _calculate_mastery_urgency(self, context: MotivatorContext) -> float:
        """Calculate urgency based on improvement opportunities."""
        # Higher urgency when there are clear improvement opportunities
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1] if context.prediction_errors else 0.0
            # High error = high urgency to improve
            error_urgency = recent_error * 0.6
        else:
            error_urgency = 0.0
        
        # Urgency also based on improvement rate
        improvement_urgency = (1.0 - self.improvement_rate) * 0.4
        
        return min(1.0, error_urgency + improvement_urgency)
    
    def update_drive_state(self, context: MotivatorContext, world_graph: Optional[WorldGraph] = None) -> float:
        """
        Update mastery drive state based on current context.
        
        This includes updating skill levels, performance tracking, and flow indicators.
        """
        # Initialize GPU predictor if world graph is available and not already initialized
        if world_graph and not self.predictive_mastery_enabled:
            self.initialize_gpu_predictor(world_graph)
        
        # Update skill levels and performance tracking
        self._update_skill_tracking(context)
        
        # Update flow indicators
        self._update_flow_indicators(context)
        
        # Update mastery goals and focus
        self._update_mastery_focus(context)
        
        # Calculate drive weight based on mastery opportunities
        mastery_opportunity = self._calculate_overall_mastery_opportunity(context)
        
        # Higher weight when there are good mastery opportunities
        self.current_weight = self.base_weight * mastery_opportunity
        
        return self.current_weight
    
    def _update_skill_tracking(self, context: MotivatorContext):
        """Update skill levels based on current performance."""
        context_sig = self._create_context_signature(context)
        
        # Update skill based on prediction accuracy
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1]
            accuracy = 1.0 - recent_error
            
            # Update skill level (moving average)
            current_skill = self.skill_levels.get(context_sig, 0.0)
            self.skill_levels[context_sig] = current_skill * 0.9 + accuracy * 0.1
            
            # Track accuracy history
            self.accuracy_history.append(accuracy)
            
            # Update improvement rate
            if len(self.accuracy_history) >= 10:
                old_avg = sum(list(self.accuracy_history)[:5]) / 5
                new_avg = sum(list(self.accuracy_history)[-5:]) / 5
                self.improvement_rate = max(0.0, new_avg - old_avg)
        
        # Update practice sessions
        self.practice_sessions[context_sig] += 1
    
    def _update_flow_indicators(self, context: MotivatorContext):
        """Update flow state indicators."""
        if not self.accuracy_history:
            return
        
        # Calculate consistency (low variance = high consistency)
        if len(self.accuracy_history) >= 5:
            recent_accuracy = list(self.accuracy_history)[-5:]
            mean_acc = sum(recent_accuracy) / len(recent_accuracy)
            variance = sum((acc - mean_acc) ** 2 for acc in recent_accuracy) / len(recent_accuracy)
            self.flow_indicators['consistency'] = max(0.0, 1.0 - variance * 2.0)
        
        # Calculate challenge level appropriateness
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1]
            context_sig = self._create_context_signature(context)
            skill_level = self.skill_levels.get(context_sig, 0.0)
            
            # Optimal challenge is slightly above skill level
            optimal_challenge = skill_level + 0.1
            challenge_appropriateness = 1.0 - abs(recent_error - optimal_challenge)
            self.flow_indicators['challenge_level'] = max(0.0, challenge_appropriateness)
        
        # Update engagement (based on consistency and appropriate challenge)
        self.flow_indicators['engagement'] = (
            self.flow_indicators['consistency'] * 0.5 +
            self.flow_indicators['challenge_level'] * 0.5
        )
    
    def _update_mastery_focus(self, context: MotivatorContext):
        """Update mastery focus based on current opportunities."""
        context_sig = self._create_context_signature(context)
        current_skill = self.skill_levels.get(context_sig, 0.0)
        
        # Focus on areas where we have some skill but aren't masters
        if 0.3 < current_skill < 0.8:
            self.current_focus_area = context_sig
        elif current_skill >= 0.8:
            # Look for new areas to focus on
            for sig, skill in self.skill_levels.items():
                if 0.3 < skill < 0.8:
                    self.current_focus_area = sig
                    break
    
    def _calculate_overall_mastery_opportunity(self, context: MotivatorContext) -> float:
        """Calculate overall mastery opportunity in current situation."""
        # Base opportunity on prediction errors and skill levels
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1]
            # Moderate error = good learning opportunity
            error_opportunity = 1.0 - abs(recent_error - 0.3)  # Peak at 0.3 error
        else:
            error_opportunity = 0.5
        
        # Skill-based opportunity
        context_sig = self._create_context_signature(context)
        skill_level = self.skill_levels.get(context_sig, 0.0)
        
        # Sweet spot for mastery development
        if skill_level < 0.3:
            skill_opportunity = 0.7  # Good learning opportunity
        elif skill_level < 0.7:
            skill_opportunity = 1.0  # Optimal mastery zone
        elif skill_level < 0.9:
            skill_opportunity = 0.6  # Refinement opportunity
        else:
            skill_opportunity = 0.2  # Already mastered
        
        # Combine opportunities
        overall_opportunity = (error_opportunity * 0.4 + skill_opportunity * 0.6)
        
        return max(0.1, overall_opportunity)  # Maintain minimum drive
    
    def _create_context_signature(self, context: MotivatorContext) -> str:
        """Create a signature for the current context for skill tracking."""
        # Combine position and basic sensory context
        pos_sig = f"{context.robot_position[0]}_{context.robot_position[1]}"
        
        # Add basic sensory context (first 4 sensors)
        sensory_sig = "_".join(f"{sensor:.1f}" for sensor in context.current_sensory[:4])
        
        return f"{pos_sig}_{sensory_sig}"
    
    def record_mastery_attempt(self, action: Dict[str, float], context: MotivatorContext, success: bool):
        """Record a mastery attempt and its outcome."""
        context_sig = self._create_context_signature(context)
        
        if success:
            self.successful_improvements += 1
            # Boost skill level for successful attempts
            current_skill = self.skill_levels.get(context_sig, 0.0)
            self.skill_levels[context_sig] = min(1.0, current_skill + 0.05)
    
    def evaluate_experience_valence(self, experience, context: MotivatorContext) -> float:
        """
        Mastery drive's pain/pleasure evaluation.
        
        Improvement and competence = PLEASURE
        Stagnation and poor performance = PAIN
        """
        # Calculate improvement in this experience
        if hasattr(experience, 'prediction_error'):
            accuracy = 1.0 - experience.prediction_error
            
            # Compare to recent performance
            if self.accuracy_history:
                recent_avg = sum(list(self.accuracy_history)[-5:]) / min(5, len(self.accuracy_history))
                improvement = accuracy - recent_avg
                
                if improvement > 0.1:
                    return 0.8  # High pleasure from significant improvement
                elif improvement > 0.05:
                    return 0.4  # Moderate pleasure from improvement
                elif improvement > -0.05:
                    return 0.1  # Slight pleasure from maintaining performance
                else:
                    return -0.5  # Pain from performance decline
            else:
                # No history - neutral
                return 0.2
        
        return 0.0
    
    def get_current_mood_contribution(self, context: MotivatorContext) -> Dict[str, float]:
        """Mastery drive's contribution to robot mood."""
        # Calculate mastery satisfaction
        context_sig = self._create_context_signature(context)
        current_skill = self.skill_levels.get(context_sig, 0.0)
        
        # Satisfaction based on skill level and improvement
        if current_skill > 0.8:
            satisfaction = 0.7  # High satisfaction from mastery
        elif current_skill > 0.5:
            satisfaction = 0.3  # Moderate satisfaction from competence
        elif self.improvement_rate > 0.1:
            satisfaction = 0.2  # Satisfaction from improvement
        else:
            satisfaction = -0.2  # Frustration from lack of progress
        
        # Urgency based on improvement opportunities
        if context.prediction_errors:
            recent_error = context.prediction_errors[-1]
            urgency = recent_error * 0.8  # High error = high urgency
        else:
            urgency = 0.3
        
        # Confidence based on skill level
        confidence = current_skill * 0.8 + 0.2
        
        return {
            'satisfaction': satisfaction,
            'urgency': urgency,
            'confidence': confidence
        }
    
    def get_mastery_stats(self) -> Dict:
        """Get detailed mastery drive statistics."""
        stats = self.get_drive_info()
        
        # Calculate average skill level
        avg_skill = sum(self.skill_levels.values()) / len(self.skill_levels) if self.skill_levels else 0.0
        
        # Calculate mastery achievements
        mastery_count = sum(1 for skill in self.skill_levels.values() if skill > 0.8)
        
        stats.update({
            'improvement_rate': self.improvement_rate,
            'average_skill_level': avg_skill,
            'mastery_achievements': mastery_count,
            'total_skills_tracked': len(self.skill_levels),
            'current_focus_area': self.current_focus_area,
            'total_mastery_attempts': self.total_mastery_attempts,
            'successful_improvements': self.successful_improvements,
            'success_rate': self.successful_improvements / max(1, self.total_mastery_attempts),
            'flow_indicators': self.flow_indicators.copy(),
            'competence_confidence': self.competence_confidence,
            'predictive_mastery_enabled': self.predictive_mastery_enabled,
            'gpu_predictor_available': self.gpu_predictor is not None
        })
        
        return stats
    
    def reset_mastery(self):
        """Reset mastery drive (useful for new environments)."""
        self.skill_levels.clear()
        self.accuracy_history.clear()
        self.performance_trends.clear()
        self.mastery_goals.clear()
        self.achieved_masteries.clear()
        self.current_focus_area = None
        self.flow_indicators = {key: 0.0 for key in self.flow_indicators}
        self.improvement_rate = 0.0
        self.practice_sessions.clear()
        self.competence_confidence = 0.0
        self.total_mastery_attempts = 0
        self.successful_improvements = 0
        self.reset_drive()