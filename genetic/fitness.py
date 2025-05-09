# project-root/genetic/fitness.py

import numpy as np

class FitnessCalculator:
    """
    Calculates the fitness score for a Genetic Agent based on the results
    of games played by the agent. Fitness is based on score and steps survived.
    """

    def __init__(self, config=None):
        """
        Initializes the fitness calculator with configuration for game metric weights.
        """
        self.config = config if config is not None else {}

        # --- Define weights for game metrics ---
        # These weights determine how score and steps contribute to fitness.
        # Get weights from config, default to 1.0 and 0.01 if not specified.
        self.score_weight = float(self.config.get('score_weight', 1.0))
        self.steps_weight = float(self.config.get('steps_weight', 0.01))
        # ---------------------------------------


    # --- New method to calculate fitness from game results ---
    # This method is called by GeneticTrainer._evaluate_population_by_playing
    def calculate_from_game_results(self, game_results: list) -> float:
        """
        Calculates the fitness score for an agent based on a list of game results.
        Fitness is typically a weighted sum of score and steps, averaged over games.

        Args:
            game_results: A list of dictionaries, where each dictionary represents
                          the result of a single game played by the agent (e.g.,
                          {'score': X, 'steps': Y, 'collisions': Z}).

        Returns:
            The calculated fitness score (float).
        """
        if not game_results:
            print("Warning: No game results provided for fitness calculation.")
            return -float('inf') # Return negative infinity fitness if no games were played

        total_weighted_fitness = 0.0
        num_games = len(game_results)

        for result in game_results:
            score = result.get('score', 0)
            steps = result.get('steps', 0)
            # collisions = result.get('collisions', True) # Collision status might be used for penalties if desired

            # Calculate fitness contribution for this single game
            # Example fitness: score + 0.01 * steps
            game_fitness_contribution = (score * self.score_weight) + (steps * self.steps_weight)

            # Optional: Add penalties based on game outcome or behavior
            # Example: Penalize if game ended very quickly (low steps)
            # if steps < 50:
            #     game_fitness_contribution -= 10 # Arbitrary penalty

            total_weighted_fitness += game_fitness_contribution

        # Calculate average fitness over all games played by the agent
        average_fitness = total_weighted_fitness / num_games

        # Ensure fitness is a valid number
        if not isinstance(average_fitness, (int, float)) or np.isnan(average_fitness) or np.isinf(average_fitness):
             print(f"Warning: Calculated invalid fitness: {average_fitness}. Returning -inf.")
             return -float('inf')

        return average_fitness


    # The original calculate_from_data method is no longer used in this direct GA design
    # def calculate_from_data(self, agent_predictions: np.ndarray, expected_outcomes: np.ndarray) -> float:
    #     """
    #     (Deprecated) Calculates fitness based on prediction performance on historical data.
    #     This method is not used in the new direct GA training design.
    #     """
    #     print("Warning: Calling deprecated FitnessCalculator.calculate_from_data method.")
    #     # Return -inf or raise an error as it's deprecated
    #     return -float('inf')

    # The original calculate method is also no longer used
    # def calculate(self, agent_performance_metrics: list) -> float:
    #     """
    #     (Original - Deprecated) Calculates the fitness score for an agent based on its game performance metrics.
    #     This method is not used in the new data-driven training design.
    #     """
    #     print("Warning: Calling deprecated FitnessCalculator.calculate method.")
    #     return -float('inf')
