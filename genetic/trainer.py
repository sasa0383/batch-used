# project-root/genetic/trainer.py

import json
import os
import yaml
import numpy as np
import time # Import time for stagnation check
# Import necessary components for direct evaluation
from game.environment import SnakeEnvironment # Now needed for evaluation
from .population import Population
from .fitness import FitnessCalculator
from .agent import GeneticAgent # Import GeneticAgent for type hinting and clarity


class GeneticTrainer:
    """
    Manages the Genetic Algorithm training process across generations.
    Evaluates agents by having them play games directly.
    Supports resuming training from a previous genome.
    Evolves agents using a 2-hidden-layer neural network architecture.
    Includes basic monitoring and stagnation handling.
    """

    # Modify __init__ to accept environment config and NN/feature flags
    def __init__(self, config=None, initial_genome_data: list | None = None):
        """
        Initializes the GA trainer.

        Args:
            config: Configuration dictionary for the GA (population_size, num_generations,
                    mutation_rate, crossover_rate, selection_method, etc.).
                    Includes game evaluation config, fitness config, NN architecture sizes,
                    NN/feature flags, and stagnation parameters.
            initial_genome_data: Optional list representing the genome to start the first generation from.
                                If None, the population starts randomly.
        """
        self.config = config if config is not None else {}
        self.ga_config = self.config.get('ga', {})
        self.nn_config = self.config.get('neural_network', {})
        self.game_eval_config = self.config.get('game_evaluation', {})
        self.fitness_config = self.config.get('fitness', {})
        self.stagnation_config = self.config.get('stagnation', {})


        self.population_size = self.ga_config.get('population_size', 100)
        self.num_generations = self.ga_config.get('num_generations', 1000)
        self.mutation_rate = self.ga_config.get('mutation_rate', 0.01)
        self.crossover_rate = self.ga_config.get('crossover_rate', 0.7)
        self.selection_method = self.ga_config.get('selection_method', 'tournament') # 'tournament' or 'elitism'
        self.tournament_size = self.ga_config.get('tournament_size', 5) # For tournament selection
        self.elitism_count = self.ga_config.get('elitism_count', 2) # For elitism selection


        # --- Neural Network and Feature Configuration ---
        # Ensure these match the agent.py definition
        # Based on the user's feature list, the number of input features is 39.
        self.num_features = 39 # Fixed based on the new feature list
        self.num_outputs = self.nn_config.get('output_size', 4) # Should be 4 for Snake actions
        self.hidden_size1 = self.nn_config.get('hidden_size1', 32)
        self.hidden_size2 = self.nn_config.get('hidden_size2', 16)

        self.use_he_initialization = self.nn_config.get('use_he_initialization', True)
        self.use_leaky_relu = self.nn_config.get('use_leaky_relu', True)
        self.normalize_features = self.nn_config.get('normalize_features', True) # Feature normalization flag

        # Calculate genome size based on the NN architecture and number of features
        self.genome_size = (self.num_features * self.hidden_size1) + self.hidden_size1 + \
                           (self.hidden_size1 * self.hidden_size2) + self.hidden_size2 + \
                           (self.hidden_size2 * self.num_outputs) + self.num_outputs

        if self.genome_size <= 0:
             raise ValueError(f"Calculated genome size ({self.genome_size}) is invalid.")

        print(f"Neural Network Architecture: Input={self.num_features}, Hidden1={self.hidden_size1}, Hidden2={self.hidden_size2}, Output={self.num_outputs}")
        print(f"Calculated Genome Size: {self.genome_size}")


        # --- Game Evaluation Configuration ---
        self.games_per_agent = self.game_eval_config.get('games_per_agent', 5)
        self.grid_size = self.game_eval_config.get('grid_size', 10)
        self.max_steps_per_game = self.game_eval_config.get('max_steps_per_game', 500) # Max steps to prevent infinite games
        self.render_training = self.game_eval_config.get('render_training', False) # Whether to render games during training

        # --- DEBUG PRINT ---
        print(f"DEBUG: Trainer loaded render_training config: {self.render_training}")
        # -------------------


        # --- Fitness Configuration ---
        self.fitness_calculator = FitnessCalculator(config=self.fitness_config)

        # --- Stagnation Configuration ---
        self.stagnation_generations = self.stagnation_config.get('stagnation_generations', 50)
        self.stagnation_threshold = self.stagnation_config.get('stagnation_threshold', 0.01) # Minimum improvement to reset stagnation
        self._best_fitness_history = [] # To track best fitness over generations for stagnation check
        self._last_improvement_generation = 0 # To track when the last significant improvement occurred


        # --- Population Initialization ---
        self.population = Population(
            population_size=self.population_size,
            genome_size=self.genome_size, # Pass the calculated genome size
            config=self.ga_config,
            initial_genome_data=initial_genome_data,
            num_features=self.num_features, # Pass num_features
            num_outputs=self.num_outputs, # Pass num_outputs
            hidden_size1=self.hidden_size1, # Pass hidden sizes
            hidden_size2=self.hidden_size2,
            use_he_initialization=self.use_he_initialization, # Pass NN flags
            use_leaky_relu=self.use_leaky_relu,
            normalize_features=self.normalize_features, # Pass feature normalization flag
            grid_size=self.grid_size # Pass grid size
        )


    def train(self, results_dir: str):
        """
        Runs the Genetic Algorithm training process.

        Args:
            results_dir: Directory to save training results (summaries, best genome).
        """
        print(f"Starting GA training for {self.num_generations} generations with population size {self.population_size}")
        print(f"Each agent will play {self.games_per_agent} games per generation for evaluation.")

        training_summary = [] # To store summary stats for each generation

        for generation in range(1, self.num_generations + 1):
            print(f"--- Generation {generation}/{self.num_generations} ---")

            # 1. Evaluate the population by playing games
            print(f"Evaluating population by playing {self.games_per_agent} games per agent...")
            self._evaluate_population_by_playing()

            # 2. Calculate and store generation summary
            generation_summary = self._calculate_generation_summary(generation)
            training_summary.append(generation_summary)
            print(f"Generation {generation}: Best Fitness = {generation_summary['best_fitness']:.4f}, Avg Fitness = {generation_summary['average_fitness']:.4f}, Diversity = {generation_summary['diversity']:.4f}")

            # 3. Check for stagnation
            if self._check_stagnation(generation, generation_summary['best_fitness']):
                 print(f"Training stagnated at generation {generation}. Best fitness did not improve significantly for {self.stagnation_generations} generations.")
                 break # Stop training if stagnated


            # 4. Create the next generation (selection, crossover, mutation)
            if generation < self.num_generations:
                 print("Creating next generation...")
                 self.population.create_next_generation()

        print("\nGA training finished.")
        self._save_training_results(results_dir, training_summary)


    def _evaluate_population_by_playing(self):
        """
        Evaluates each agent in the population by having it play games.
        Sets the fitness for each agent based on game results.
        """
        # Create environments for evaluation. Could potentially parallelize this.
        environments = [SnakeEnvironment(grid_size=self.grid_size, render=self.render_training, game_speed=self.game_eval_config.get('game_speed', 10)) for _ in range(self.games_per_agent)]

        for agent in self.population.get_agents():
            game_results = []
            for env in environments:
                # Reset environment for a new game
                env.reset()
                game_over = False
                steps = 0

                # Note: max_steps_per_game is read from self.game_eval_config
                while not game_over and steps < self.max_steps_per_game:
                    # Get the current state from the environment
                    current_state = env.state

                    # Agent decides the next action based on the current state features
                    action = agent.decide(current_state)

                    # Take the action in the environment
                    new_state_dict, reward, done, info = env.step(action)
                    # If you need the GameState object after step, you'd recreate it:
                    # new_state = GameState.from_dict(new_state_dict)


                    # Update game_over flag and steps
                    game_over = done
                    steps += 1 # Steps are also incremented in env.step, so be careful not to double count.
                    # Let's rely on env.state.steps for the final count in game_result

                # Collect results for this game trial
                game_result = {
                    "score": env.state.score,
                    "steps": env.state.steps, # Use steps from state object
                    "food": env.state.food_eaten_count,
                    "collision": env.state.collisions, # True if game ended due to collision
                    "termination_reason": info.get('termination_reason', 'unknown') # Capture termination reason
                }
                game_results.append(game_result)

            # Calculate agent's fitness based on the game results
            agent_fitness = self.fitness_calculator.calculate_from_game_results(game_results)
            agent.set_fitness(agent_fitness)

        # Close environments after evaluation
        for env in environments:
            env.close()


    def _calculate_generation_summary(self, generation: int) -> dict:
        """Calculates summary statistics for the current generation."""
        agents = self.population.get_agents()
        fitness_scores = [a.get_fitness() for a in agents if a.get_fitness() is not None]

        summary = {
            'generation': generation,
            'best_fitness': float(np.max(fitness_scores)) if fitness_scores else -float('inf'),
            'average_fitness': float(np.mean(fitness_scores)) if fitness_scores else -float('inf'),
            'median_fitness': float(np.median(fitness_scores)) if fitness_scores else -float('inf'),
            'std_fitness': float(np.std(fitness_scores)) if fitness_scores else 0.0,
            'diversity': float(self.population.calculate_diversity()), # Calculate diversity
            'timestamp': time.time() # Add timestamp
        }
        return summary

    def _check_stagnation(self, generation: int, current_best_fitness: float) -> bool:
        """Checks if the training has stagnated."""
        self._best_fitness_history.append(current_best_fitness)

        # Modified the condition to be <= stagnation_generations
        if len(self._best_fitness_history) <= self.stagnation_generations:
            # Not enough data yet to check for stagnation over the full period
            return False

        # Get the best fitness from 'stagnation_generations' ago
        # This is the fitness at generation (current_generation - stagnation_generations)
        # In a 0-indexed list, this is at index -(self.stagnation_generations + 1) relative to the end.
        # Example: if stagnation_generations = 3, current generation history has length 4 (indices 0, 1, 2, 3).
        # We compare generation 3 (index -1) with generation 0 (index -4).
        # -(3 + 1) = -4. So index -(self.stagnation_generations + 1) is correct if list length >= stagnation_generations + 1
        # The condition above ensures list length is at least stagnation_generations + 1.
        fitness_past = self._best_fitness_history[-(self.stagnation_generations + 1)]


        # Check for significant improvement
        improvement = current_best_fitness - fitness_past

        # Using an absolute difference threshold for simplicity.
        # Reset last improvement generation if there's a significant improvement.
        if improvement > self.stagnation_threshold:
             self._last_improvement_generation = generation
             #print(f"DEBUG: Improvement {improvement:.4f} > threshold {self.stagnation_threshold:.4f}. Resetting stagnation at generation {generation}.")


        # Stagnation occurs if the last significant improvement was more than stagnation_generations ago
        # and we have recorded enough generations to check the full period.
        # This condition means: current_generation - last_improvement_generation >= stagnation_generations
        is_stagnated = (generation - self._last_improvement_generation) >= self.stagnation_generations

        # Add debug print to see stagnation check logic
        #print(f"DEBUG: Gen {generation}, Last Impr Gen {self._last_improvement_generation}, Stagnation Gens {self.stagnation_generations}. Is Stagnated: {is_stagnated}")


        return is_stagnated


    def _save_training_results(self, results_dir: str, training_summary: list):
        """Saves the training summary and best agent's genome."""
        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Save training summary
        summary_path = os.path.join(results_dir, "training_summary.json")
        try:
            # Ensure numpy floats are converted to standard Python floats for JSON
            summary_serializable = [{k: float(v) if isinstance(v, np.floating) else (int(v) if isinstance(v, np.integer) else v) for k, v in summary_entry.items()} for summary_entry in training_summary] # Changed s to summary_entry for clarity
            with open(summary_path, 'w') as f:
                json.dump(summary_serializable, f, indent=4)
            print(f"Training summary saved to {summary_path}")
        except IOError as e:
             print(f"Error saving training summary to {summary_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred saving training summary: {e}")


        # Save best agent's genome
        best_agent = self.population.get_best_agent()
        if best_agent and best_agent.get_genome() is not None:
            genome_path = os.path.join(results_dir, "best_genome.json")
            try:
                # Convert numpy array genome to a list for JSON serialization
                with open(genome_path, 'w') as f:
                    json.dump(best_agent.get_genome().tolist(), f, indent=4)
                print(f"Best genome saved to {genome_path}")
            except IOError as e:
                 print(f"Error saving best genome to {genome_path}: {e}")
            except Exception as e:
                 print(f"An unexpected error occurred saving best genome: {e}")


        # Save scores (fitness scores from the last generation)
        scores_path = os.path.join(results_dir, "scores.json")
        try:
            # Ensure scores are standard Python floats
            last_gen_scores = [float(a.get_fitness()) if a and a.get_fitness() is not None else -float('inf') for a in self.population.get_agents()]
            with open(scores_path, 'w') as f:
                json.dump(last_gen_scores, f, indent=4)
            print(f"Last generation scores saved to {scores_path}")
        except IOError as e:
             print(f"Error saving last generation scores to {scores_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred saving last generation scores: {e}")