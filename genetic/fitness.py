# Modify genetic/fitness.py

# Assuming fitness is now calculated based on agent's performance on historical data
import numpy as np

class FitnessCalculator:
    """
    Calculates the fitness score for a Genetic Agent.
    In the new design, fitness is based on the agent's performance on historical data.
    """

    def __init__(self, config=None):
        """
        Initializes the fitness calculator.
        Config includes weights for different performance metrics.
        """
        self.config = config if config is not None else {}

        # The weights might need reinterpretation or new ones for data-driven fitness
        # If expected_outcome is survival (0 or 1), fitness could be accuracy, precision, etc.
        # If expected_outcome is score, fitness could be correlation or minimizing prediction error.

        # Let's redefine weights based on how well the agent's *decision* aligns with the *outcome*
        # from the historical data pair.
        # This is a simplified example based on predicting survival (0 or 1) from stat features
        # And the agent outputting an action (0-3). This needs a mapping.

        # A possible data-driven fitness: How often does the agent choose a "good" action
        # when the expected outcome was survival (1)? Or avoid "bad" actions when outcome was death (0)?
        # This requires defining what a "good" or "bad" action is relative to the outcome.

        # Let's simplify greatly: Fitness is based on the agent's ability to predict the outcome (survival).
        # The agent outputs actions (0-3). We need to map this to a survival prediction.
        # This mapping is not inherent. A simpler approach:
        # The agent's output layer (before argmax) gives scores for each action.
        # Maybe fitness relates to the score assigned to the "correct" action from the original trial?
        # But raw_data doesn't save actions taken step-by-step.

        # Let's refine based on the simplified input/target pairs:
        # Input: [score, steps, food_eaten, collided_status (0 or 1)]
        # Expected Outcome: survival_status (0 or 1)
        # Agent Output: Action (0-3)

        # Fitness calculation logic needs a clear definition in the new model.
        # Option A: Treat it as a classification task. Does the agent's *most preferred action* (highest score)
        # implicitly suggest survival (if the score for some action is high) or death (if all scores are low)?
        # This is indirect.

        # Option B: Redefine expected_outcome in inputs_targets.json to be the action taken in the original trial.
        # This requires logging actions in raw_data.json.
        # If expected_outcome is the action, then fitness is how often the agent's `argmax` output matches the expected action. This is supervised learning style.

        # Option C: Keep expected_outcome as survival (0/1). How does agent action relate to survival?
        # Maybe the *score* given by the agent's weights to the "forward" action indicates how "safe" the agent thinks the current stat state is? This is speculative.

        # Let's assume for now that the fitness is a simple score based on predicting survival,
        # where the "prediction" is derived somehow from the agent's action scores.
        # Or, more simply, fitness is the *average expected outcome* from the data pairs, weighted by how confident the agent was? This gets complex.

        # Let's use a very basic data-driven fitness:
        # Count how many times the agent's action implicitly correlates with the outcome.
        # If expected_outcome was survival (1), maybe actions 0-3 are all "good"?
        # If expected_outcome was death (0), maybe any action taken was "bad"?
        # This doesn't use the agent's *output scores*, just the chosen action.

        # Let's try a fitness based on the *confidence* in the chosen action
        # when the outcome was positive (survival = 1).
        # Fitness increases if the agent assigns a high score to *any* action when survival was 1.
        # Fitness decreases if the agent assigns high scores when survival was 0.
        # This is also quite speculative.

        # A more standard approach for this data structure: Treat it like a regression or classification task.
        # If predicting survival (0/1), maybe fitness is related to AUC or accuracy?
        # If predicting score, maybe fitness is related to R^2 or MSE?

        # Let's simplify: Assume the goal is to predict survival (1) vs death (0).
        # The agent outputs scores for 4 actions. We need to map this to a single survival prediction (e.g., sigmoid on max score?).
        # Or, assume the max score itself is an indicator of predicted "goodness".
        # Fitness = average (max_action_score * expected_outcome) over all data pairs?
        # This rewards high scores when expected_outcome is 1, and doesn't penalize when 0.
        # Needs refinement.

        # Let's use a fitness that rewards correct "classification" of survival based on the agent's highest action score.
        # If max_action_score > threshold and expected_outcome is 1 -> positive contribution to fitness.
        # If max_action_score <= threshold and expected_outcome is 0 -> positive contribution.
        # Needs a threshold. Or, map agent's action scores to a single prediction (e.g., average, max, or a small network).

        # Simpler approach: The agent's genome defines weights. Fitness is related to the *sum of weights* assigned to inputs
        # that correlate with good outcomes. This bypasses the agent's *decision process* on data. Not ideal.

        # Let's go back to the agent's output scores for actions (before argmax).
        # The agent outputs 4 scores: [score_up, score_down, score_left, score_right].
        # Expected Outcome: survival (0 or 1).
        # Fitness for a single data pair: How does the output vector relate to survival?
        # Maybe sum of scores if survived? Or max score if survived?
        # What if it didn't survive?

        # Let's define a simplified fitness that encourages high scores on actions when survival was 1,
        # and low scores when survival was 0.
        # Fitness = sum over all data pairs of (sum(agent_action_scores) * expected_outcome_survival)
        # This rewards high scores when survival=1, and ignores when survival=0. Not robust.

        # How about Mean Squared Error if predicting survival as a continuous value?
        # Agent needs to output a single survival prediction (0-1) instead of actions.
        # This requires changing the agent's output layer/structure.

        # Let's stick to the agent outputting action scores and predicting survival (0/1).
        # Assume the sum of action scores is the agent's "confidence" in the state being good.
        # Fitness for a data pair: (sum(agent_action_scores) * expected_outcome) - (sum(agent_action_scores) * (1 - expected_outcome))
        # This rewards high scores when outcome=1, penalizes high scores when outcome=0.

        # Total fitness for an agent = Average fitness over all historical data pairs.
        # Needs weights from config for different aspects if we add more metrics.
        # Let's use the simple confidence-based fitness calculation.

        self.score_weight = self.config.get('score_weight', 1.0) # Not directly used in this data-driven fitness
        self.survival_weight = self.config.get('survival_weight', 5.0) # Used to scale the data-driven fitness
        self.steps_weight = self.config.get('steps_weight', 0.01) # Not directly used
        self.entropy_score_weight = self.config.get('entropy_score_weight', 0.5) # Not directly used
        self.entropy_food_weight = self.config.get('entropy_food_weight', 0.0) # Not directly used

        # The fitness weights from the original config are less directly applicable now.
        # Let's assume a single scaling factor based on 'survival_weight'.


    # Remove the original calculate method
    # def calculate(self, game_results_stats: dict) -> float:
    #    pass # No longer used for evaluation games


    # --- New method to calculate fitness from data performance ---
    # This method is called by GeneticTrainer._evaluate_population_on_data
    def calculate_from_data(self, agent_performance_metrics: list) -> float:
        """
        Calculates the fitness score for an agent based on its performance across
        historical data pairs.

        Args:
            agent_performance_metrics: A list of dictionaries, each containing
                                       'input_vector', 'agent_decision', 'expected_outcome',
                                       and potentially agent's action scores for that input.

        Returns:
            The calculated fitness score (float).
        """
        if not agent_performance_metrics:
            return 0.0 # Return 0 fitness if no data points evaluated

        total_fitness_score = 0.0

        for metric in agent_performance_metrics:
            # Access agent's raw action scores for this data point
            # We need the scores *before* argmax in agent.decide.
            # Modify agent.decide to return both action and scores, or modify evaluation loop.
            # Let's modify _evaluate_population_on_data to pass scores as well.

            # Assuming 'agent_action_scores' is added to the metric dictionary:
            agent_action_scores = metric.get('agent_action_scores', np.zeros(4)) # Default to zeros if missing
            expected_outcome = metric.get('expected_outcome', 0.0) # Default to 0 if missing

            # Calculate a fitness contribution for this data point
            # Example: Reward high scores when survival=1, penalize when survival=0
            confidence = np.sum(agent_action_scores) # Simple sum of scores as confidence

            # Fitness contribution for this data point:
            # If expected_outcome is 1 (survived), contribute positively based on confidence
            # If expected_outcome is 0 (collided), contribute negatively based on confidence
            # Contribution = confidence * expected_outcome - confidence * (1 - expected_outcome)
            # Simplified: Contribution = confidence * (expected_outcome - (1 - expected_outcome))
            # Contribution = confidence * (2 * expected_outcome - 1)
            # If expected_outcome is 1, contribution = confidence * (2*1 - 1) = confidence * 1
            # If expected_outcome is 0, contribution = confidence * (2*0 - 1) = confidence * -1

            fitness_contribution = confidence * (2.0 * expected_outcome - 1.0)

            # Could also consider how the chosen action relates to the outcome, but that's more complex.
            # For now, simple confidence-based fitness.

            total_fitness_score += fitness_contribution

        # Normalize or scale the total fitness score.
        # Divide by the number of data points to get average contribution
        average_fitness_per_data_point = total_fitness_score / len(agent_performance_metrics)

        # Apply the survival weight as a general scaling factor (reinterpreting its role)
        final_fitness = average_fitness_per_data_point * self.survival_weight

        # Ensure fitness is a number
        if not isinstance(final_fitness, (int, float)) or np.isnan(final_fitness) or np.isinf(final_fitness):
             print(f"Warning: Calculated invalid fitness: {final_fitness}. Returning 0.0.")
             return 0.0

        return final_fitness