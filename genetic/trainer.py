# project-root/genetic/trainer.py

import json
import os
import yaml
import numpy as np
from .population import Population
from .fitness import FitnessCalculator
from .agent import GeneticAgent # Import GeneticAgent for type hinting and clarity

# Need to import the game environment if it's still used anywhere (e.g., for initial batch in runner)
# But the trainer itself no longer uses it for evaluation in the new design.
# from game.environment import SnakeEnvironment # Removed direct dependency in trainer


class GeneticTrainer:
    """
    Manages the Genetic Algorithm training process across generations.
    In the new design, evaluates agents based on processed statistical data.
    Supports initializing from a previous genome.
    Evolves agents using a 1-hidden-layer neural network architecture.
    """

    def __init__(self, config=None, initial_genome_data: list | None = None):
        """
        Initializes the GA trainer.

        Args:
            config: Configuration dictionary for the GA (population_size, num_generations,
                    mutation_rate, crossover_rate, selection_method, etc.).
                    Includes fitness config, hidden_size, and implicit genome size info
                    (derived from expected features and outputs).
            initial_genome_data: Optional list representing the genome to start the first generation from.
                                If None, the population starts randomly.
        """
        self.config = config if config is not None else {}
        self.initial_genome_data = initial_genome_data

        self.population_size = int(self.config.get('population_size', 100))
        self.num_generations = int(self.config.get('num_generations', 10))

        self.population_config = self.config.get('population', {})
        self.fitness_config = self.config.get('fitness', {})

        # --- Define the neural network architecture sizes ---
        # Number of input features (from statistics/bayesian.py)
        self.num_features = 27 # <-- Matches the number of features in statistics/bayesian.py
        # Number of output values (predicting a single score)
        self.num_outputs = 1 # <-- Agent outputs a single value

        # Hidden layer size (from config, default to 16)
        self.hidden_size = int(self.config.get('hidden_size', 16))
        # ----------------------------------------------------

        # --- Calculate Genome size based on the NN architecture ---
        # Genome size = (Input * Hidden) + Hidden + (Hidden * Output) + Output
        self.genome_size = (self.num_features * self.hidden_size) + self.hidden_size + \
                           (self.hidden_size * self.num_outputs) + self.num_outputs
        # ----------------------------------------------------------

        # Check if genome_size is explicitly set in config (should be ignored in favor of calculated size)
        if 'genome_size' in self.config and int(self.config['genome_size']) != self.genome_size:
             print(f"Warning: Ignoring 'genome_size' ({self.config['genome_size']}) in config. Calculated size based on NN architecture is {self.genome_size}.")


        # Pass initial_genome_data, num_features, num_outputs, AND hidden_size to the Population
        self.population = Population(
            population_size=self.population_size,
            genome_size=self.genome_size, # Use the calculated genome_size
            config=self.population_config,
            initial_genome_data=self.initial_genome_data, # Pass the loaded genome data
            num_features=self.num_features, # <-- Pass num_features
            num_outputs=self.num_outputs, # <-- Pass num_outputs
            hidden_size=self.hidden_size # <-- Pass hidden_size
        )

        # The FitnessCalculator is currently not used by _evaluate_population_on_data,
        # as fitness is calculated directly as -MSE in the trainer.
        # self.fitness_calculator = FitnessCalculator(config=self.fitness_config)


        self.generation_history = [] # To store summary data per generation


    # The set_evaluation_environment method is no longer needed in the new design
    # def set_evaluation_environment(self, env):
    #     """Sets the game environment to be used for agent evaluation."""
    #     self._evaluation_environment = env


    def train(self, processed_stats_path: str, results_dir: str):
        """
        Runs the GA training process for a specified number of generations.
        Evaluates agents based on processed statistical data from processed_stats_path.

        Args:
            processed_stats_path: Path to the inputs_targets.json file containing
                                  processed statistical data (input/target pairs).
            results_dir: Directory to save training results (summary, best genome).
        """
        # Load processed statistical data for evaluation
        try:
            with open(processed_stats_path, 'r') as f:
                processed_data_pairs = json.load(f)
            print(f"Loaded {len(processed_data_pairs)} data pairs from {processed_stats_path} for training.")
        except FileNotFoundError:
            print(f"Error: Processed statistical data file not found at {processed_stats_path}. Cannot train.")
            return
        except json.JSONDecodeError:
            print(f"Error: could not decode JSON from {processed_stats_path}")
            return
        except Exception as e:
             print(f"An unexpected error occurred loading data from {processed_stats_path}: {e}")
             return

        if not processed_data_pairs:
             print("Warning: Processed statistical data is empty. Cannot train.")
             return

        # --- Ensure the number of features in the data matches the agent's expectation ---
        if processed_data_pairs:
             first_input_vector = processed_data_pairs[0].get('input_vector')
             if first_input_vector is not None and len(first_input_vector) != self.num_features:
                  print(f"Error: Feature vector size mismatch in loaded data ({len(first_input_vector) if first_input_vector is not None else 'None'}) vs expected ({self.num_features}). Cannot train.")
                  print(f"Please ensure the number of features calculated in statistics/bayesian.py::process_historical_data matches genetic/trainer.py::num_features ({self.num_features}).")
                  return

        print(f"Starting GA training for {self.num_generations} generations with population size {self.population_size}")
        print(f"Neural Network Architecture: Input={self.num_features}, Hidden={self.hidden_size}, Output={self.num_outputs}")
        print(f"Calculated Genome Size: {self.genome_size}")


        for generation in range(self.num_generations):
            print(f"--- Generation {generation + 1}/{self.num_generations} ---")

            # Evaluate each agent using the processed statistical data
            self._evaluate_population_on_data(processed_data_pairs)

            # Record generation summary
            best_agent = self.population.get_best_agent()
            if not self.population.get_agents() or all(a.get_fitness() is None for a in self.population.get_agents()):
                 avg_fitness = 0.0
                 best_fitness = -float('inf') # Use negative infinity as initial best fitness
                 best_genome = None
            else:
                 # Filter out None fitness values before calculating mean
                 valid_fitness_scores = [a.get_fitness() for a in self.population.get_agents() if a.get_fitness() is not None]
                 avg_fitness = np.mean(valid_fitness_scores) if valid_fitness_scores else 0.0
                 best_fitness = best_agent.get_fitness() if best_agent and best_agent.get_fitness() is not None else -float('inf')
                 best_genome = best_agent.get_genome().tolist() if best_agent and best_agent.get_genome() is not None else None # Save best genome as list


            generation_summary = {
                'generation': generation + 1,
                'best_fitness': float(best_fitness), # Ensure float
                'average_fitness': float(avg_fitness), # Ensure float
                'best_genome': best_genome # Save best genome as list
            }
            self.generation_history.append(generation_summary)

            print(f"Generation {generation + 1}: Best Fitness = {generation_summary['best_fitness']:.4f}, Avg Fitness = {generation_summary['average_fitness']:.4f}")


            # Create the next generation (unless it's the last generation)
            if generation < self.num_generations - 1:
                self.population.create_next_generation()


        # Training finished
        print("GA training complete.")

        # Save final results
        self._save_training_results(results_dir)


    # --- Method to evaluate population based on processed data ---
    def _evaluate_population_on_data(self, data_pairs: list):
        """
        Evaluates the fitness of each agent based on processed statistical data pairs.
        Fitness is calculated as the negative Mean Squared Error (-MSE) between
        the agent's predicted score and the actual score for each data pair.

        Args:
            data_pairs: A list of dictionaries, each containing 'input_vector' and 'expected_outcome' (actual score).
        """
        agents = self.population.get_agents()
        if not agents:
            print("Warning: Population is empty, cannot evaluate on data.")
            return

        num_data_pairs = len(data_pairs)
        if num_data_pairs == 0:
             print("Warning: No data pairs provided for evaluation. Setting fitness to -inf for all agents.")
             for agent in agents:
                  agent.set_fitness(-float('inf'))
             return

        # Evaluate each agent on the entire dataset of input/target pairs
        for i, agent in enumerate(agents):
            # print(f"  Evaluating Agent {i+1}/{len(agents)} on data...") # Optional debug

            total_squared_error = 0.0
            valid_data_points = 0 # Count data points successfully processed for this agent

            for data_pair in data_pairs:
                input_vector = data_pair.get('input_vector')
                expected_outcome = data_pair.get('expected_outcome') # Actual final score from the trial

                # Basic validation of data pair
                if input_vector is None or expected_outcome is None:
                     print(f"Warning: Skipping data pair due to missing 'input_vector' or 'expected_outcome'.")
                     continue

                # Ensure the input vector size matches the agent's expectation
                if len(input_vector) != agent.num_features:
                     print(f"Warning: Data pair input vector size mismatch ({len(input_vector)}) vs expected ({agent.num_features}). Skipping this data pair for Agent {i+1}.")
                     continue # Skip this data pair if sizes don't match

                try:
                    # Agent decides output (predicted score) based on the input vector
                    # The agent's decide method now expects the statistical feature vector (size 27)
                    # and should return a single float (predicted score).
                    predicted_score = agent.decide(np.array(input_vector)) # Pass input as numpy array

                    # Ensure the output is a single number
                    if not isinstance(predicted_score, (int, float)):
                         print(f"Warning: Agent.decide did not return a single number for data pair. Got {type(predicted_score)}. Skipping fitness calculation for this data pair.")
                         # Assign a very high error for this specific data pair
                         error = float('inf')
                    else:
                         # Calculate squared error for this data pair
                         error = (predicted_score - expected_outcome) ** 2
                         valid_data_points += 1 # Increment if processed successfully

                except Exception as e:
                     print(f"Error during agent.decide for data pair: {e}. Assigning high error for this data pair.")
                     error = float('inf') # Assign a very high error on error
                     # Do NOT increment valid_data_points here

                total_squared_error += error # Accumulate squared error

            # Calculate Mean Squared Error (MSE)
            if valid_data_points == 0:
                 mse = float('inf') # Avoid division by zero
                 print(f"Warning: No valid data points processed for Agent {i+1}. MSE is infinite.")
            elif total_squared_error == float('inf'):
                 mse = float('inf') # If any single error was infinite
            else:
                 mse = total_squared_error / valid_data_points


            # Fitness is the negative of the MSE (higher fitness = lower error)
            # Less negative is better. Initial random agents will have high MSE (low fitness).
            agent_fitness = -mse

            # Set the calculated fitness for the agent
            agent.set_fitness(agent_fitness)

        # print("Population evaluation on data finished.") # Optional debug


    # The _evaluate_population method (for game-based evaluation) is no longer used in the new design
    # def _evaluate_population(self, initial_stat_features: dict):
    #    pass # This method is replaced by _evaluate_population_on_data


    def _save_training_results(self, results_dir: str):
        """Saves the training history and best genome."""
        # The results_dir is already created by the executor
        # os.makedirs(results_dir, exist_ok=True) # No need to recreate here

        # Save generation history (summary)
        summary_path = os.path.join(results_dir, "summary.yaml")
        try:
            with open(summary_path, 'w') as f:
                # Ensure history data is serializable (e.g., convert numpy arrays in best_genome to lists)
                serializable_history = []
                for entry in self.generation_history:
                    serializable_entry = entry.copy()
                    if isinstance(serializable_entry.get('best_genome'), np.ndarray):
                        serializable_entry['best_genome'] = serializable_entry['best_genome'].tolist()
                    serializable_history.append(serializable_entry)

                yaml.dump(serializable_history, f, indent=4)
            print(f"Training summary saved to {summary_path}")
        except ImportError:
            print("Warning: PyYAML not installed. Cannot save training summary.")
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
             print(f"Error saving scores to {scores_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred saving scores: {e}")

