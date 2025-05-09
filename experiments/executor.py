# project-root/experiments/executor.py

import yaml
import os
import json
import time # For timestamp in run folder name
import hashlib # Import hashlib for generating a hash
from .runner import ExperimentRunner
# from .model_selector import ModelSelector # ModelSelector used by Runner now

class ExperimentExecutor:
    """
    Manages the overall experiment execution process.
    Reads configuration, sets up results directories, and runs the experiment runner.
    Handles parameter tracking and batch versioning (via run_XXX folders).
    Generates a parameter key based on relevant hyperparameters for direct GA training.
    Uses a hash for a shorter, more robust parameter key directory name.
    """

    def __init__(self, config_path: str):
        """
        Initializes the executor by loading the experiment configuration.

        Args:
            config_path: Path to the experiment configuration YAML file.
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)

        self.experiment_name = self.config.get('experiment_name', 'default_experiment')
        self.results_base_dir = "results" # Base directory for all experiment results

    def _load_config(self, config_path: str) -> dict:
        """Loads the experiment configuration from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Configuration loaded successfully from {config_path}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error loading configuration from {config_path}: {e}")
        except Exception as e:
             raise RuntimeError(f"An unexpected error occurred loading config: {e}")


    def _generate_param_key(self, config: dict) -> str:
        """
        Generates a unique string key from the configuration parameters using a hash.
        This creates a shorter, more robust folder name for the parameter set.
        The full configuration is saved within the run directory for details.
        Includes parameters relevant to the direct GA training setup.
        """
        # Create a dictionary of parameters that define a unique run configuration
        # This should include all hyperparameters and relevant settings that differentiate runs.
        param_dict = {}
        # Include core run parameters relevant to game environment and evaluation
        param_dict['grid_size'] = config.get('grid_size', 10)
        param_dict['games_per_agent'] = config.get('games_per_agent', 5)


        # Include AI hyperparameters (from the specific AI section or flat)
        ai_model_name = config.get('ai_model', 'genetic')
        ai_config = config.get(ai_model_name, {}) # Specific AI config section

        param_dict['population_size'] = ai_config.get('population_size', config.get('population_size', 100))
        param_dict['num_generations'] = ai_config.get('num_generations', config.get('num_generations', 10))
        param_dict['mutation_rate'] = ai_config.get('mutation_rate', config.get('mutation_rate', 0.03)) # Updated default
        param_dict['crossover_rate'] = ai_config.get('crossover_rate', config.get('crossover_rate', 0.9))
        param_dict['elite_count'] = ai_config.get('elite_count', config.get('elite_count', 5)) # Updated default
        param_dict['selection_method'] = ai_config.get('selection_method', config.get('selection_method', 'tournament'))
        param_dict['tournament_size'] = ai_config.get('tournament_size', config.get('tournament_size', 5))
        param_dict['topk_k'] = ai_config.get('topk_k', config.get('topk_k', 10))

        # Include Neural Network Architecture parameters and flags
        param_dict['hidden_size1'] = ai_config.get('hidden_size1', config.get('hidden_size1', 32)) # Updated default
        param_dict['hidden_size2'] = ai_config.get('hidden_size2', config.get('hidden_size2', 16)) # Updated default
        param_dict['use_he_initialization'] = ai_config.get('use_he_initialization', config.get('use_he_initialization', True))
        param_dict['use_leaky_relu'] = ai_config.get('use_leaky_relu', config.get('use_leky_relu', True)) # Corrected typo LeakyReLU
        param_dict['normalize_features'] = ai_config.get('normalize_features', config.get('normalize_features', True))


        # Include Fitness weights (for game results)
        fitness_config = ai_config.get('fitness', config.get('fitness', {}))
        param_dict['score_weight'] = fitness_config.get('score_weight', config.get('score_weight', 1.0))
        param_dict['steps_weight'] = fitness_config.get('steps_weight', config.get('steps_weight', 0.01)) # Updated default


        # Include Stagnation parameters
        param_dict['stagnation_threshold'] = ai_config.get('stagnation_threshold', config.get('stagnation_threshold', 0.001))
        param_dict['stagnation_generations'] = ai_config.get('stagnation_generations', config.get('stagnation_generations', 20))
        param_dict['stagnation_mutation_boost'] = ai_config.get('stagnation_mutation_boost', config.get('stagnation_mutation_boost', 0.1))


        # Sort keys to ensure consistent hashing regardless of dictionary order
        sorted_params = sorted(param_dict.items())

        # Convert the sorted parameters to a string representation
        # Use json.dumps for consistent string formatting, handle floats explicitly if needed
        # Using a custom lambda for float formatting within dumps for consistency
        param_string = json.dumps(sorted_params, sort_keys=True, default=lambda x: f"{x:.4f}" if isinstance(x, float) else x)


        # Generate a hash of the parameter string
        # Using SHA256 for a reasonably short and unique hash
        param_hash = hashlib.sha256(param_string.encode('utf-8')).hexdigest()

        # Use a portion of the hash as the parameter key for the directory name
        # Using the first 16 characters of the hash should be unique enough for most cases.
        param_key = param_hash[:16]


        print(f"Generated parameter key (hash prefix): {param_key}")
        return param_key

    def execute(self):
        """Starts the experiment execution."""
        print(f"Executing experiment: {self.experiment_name}")

        # Generate the parameter key for the results folder
        param_key = self._generate_param_key(self.config)
        param_results_dir = os.path.join(self.results_base_dir, self.experiment_name, param_key)

        # Create the run-specific directory (batch versioning)
        # Use a timestamp or incremental counter. Timestamp is simpler for unique runs.
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir_name = f"run_{timestamp}"
        current_run_results_dir = os.path.join(param_results_dir, run_dir_name)

        print(f"Results will be saved to: {current_run_results_dir}")

        # Ensure the run directory exists
        try:
            os.makedirs(current_run_results_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {current_run_results_dir}: {e}")
            print("Please check if the path is too long or contains invalid characters.")
            return # Stop execution if directory creation fails
        except Exception as e:
             print(f"An unexpected error occurred creating directory {current_run_results_dir}: {e}")
             return


        # Save the config used for this specific run in the run directory for reproducibility
        run_config_path = os.path.join(current_run_results_dir, "config.yaml")
        try:
            with open(run_config_path, 'w') as f:
                yaml.dump(self.config, f, indent=4)
            print(f"Run configuration saved to {run_config_path}")
        except IOError as e:
             print(f"Error saving run configuration: {e}")
        except Exception as e:
             print(f"An unexpected error occurred saving run config: {e}")


        # Instantiate and run the experiment runner
        try:
            # Pass the full config to the runner
            runner = ExperimentRunner(config=self.config, param_results_dir=param_results_dir) # <--- Pass full config and param_results_dir

            runner.run(results_dir=current_run_results_dir)
        except Exception as e:
            print(f"Error during experiment run: {e}")
            # Optionally log the error or clean up the directory
            import traceback
            traceback.print_exc() # Print traceback for debugging


        print(f"Experiment execution finished for run: {run_dir_name}")

# Example usage (would be in main.py):
# config_file = "experiments/configs/exp_genetic.yaml"
# executor = ExperimentExecutor(config_file)
# executor.execute()
