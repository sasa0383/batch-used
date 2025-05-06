# project-root/experiments/executor.py

import yaml
import os
import json
import time # For timestamp in run folder name
from .runner import ExperimentRunner
# from .model_selector import ModelSelector # ModelSelector is used by Runner now

class ExperimentExecutor:
    """
    Manages the overall experiment execution process.
    Reads configuration, sets up results directories, and runs the experiment runner.
    Handles parameter tracking and batch versioning (via run_XXX folders).
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

    def _generate_param_key(self, config: dict) -> str:
        """
        Generates a unique string key from the configuration parameters.
        Used to create a parameter-specific folder name.
        Excludes non-parameter keys like 'experiment_name', 'stat_model', 'ai_model'.
        Should include hyperparameters relevant to the models and batch size.
        """
        # Filter out keys that aren't hyperparameters or defining run parameters
        # This requires knowing which keys *are* parameters.
        # Based on exp_genetic.yaml: batch_size, learning_rate, momentum, population_size, num_generations
        # and model-specific configs like bayesian: { ... }

        param_dict = {}
        # Include core run parameters
        param_dict['batch_size'] = config.get('batch_size', 50)

        # Include AI hyperparameters (from the specific AI section or flat)
        ai_model_name = config.get('ai_model', 'genetic')
        ai_config = config.get(ai_model_name, {}) # Specific AI config section
        param_dict['population_size'] = ai_config.get('population_size', config.get('population_size', 100)) # Check specific section first, then flat
        param_dict['num_generations'] = ai_config.get('num_generations', config.get('num_generations', 10))
        param_dict['mutation_rate'] = ai_config.get('mutation_rate', config.get('mutation_rate', 0.01)) # These might be in population config
        param_dict['crossover_rate'] = ai_config.get('crossover_rate', config.get('crossover_rate', 0.9)) # These might be in population config
        param_dict['elite_count'] = ai_config.get('elite_count', config.get('elite_count', 0)) # These might be in population config

        # Include Stat model parameters (from the specific Stat section)
        stat_model_name = config.get('stat_model', 'bayesian')
        stat_config = config.get(stat_model_name, {}) # Specific Stat config section
        # Include relevant stat parameters, e.g., from the 'bayesian' section
        if stat_model_name == 'bayesian':
             param_dict['use_entropy'] = stat_config.get('use_entropy', True)
             param_dict['confidence_interval'] = stat_config.get('confidence_interval', 95)

        # Include other potentially relevant parameters (e.g., game specific if applicable)
        param_dict['grid_size'] = config.get('grid_size', 10)
        # Add fitness weights if they are parameters that define the experiment run's context
        param_dict['score_weight'] = config.get('score_weight', 1.0)
        param_dict['survival_weight'] = config.get('survival_weight', 2.0)
        param_dict['steps_weight'] = config.get('steps_weight', 0.01)
        param_dict['entropy_score_weight'] = config.get('entropy_score_weight', 0.5)


        # Sort keys to ensure consistent string generation
        sorted_params = sorted(param_dict.items())

        # Create key string
        param_key_parts = []
        for key, value in sorted_params:
            # Format values for the key (handle floats, booleans, ints)
            if isinstance(value, float):
                # Format floats to avoid issues with precision in folder names
                param_key_parts.append(f"{key}_{value:.4f}".replace('.', '_'))
            elif isinstance(value, bool):
                 param_key_parts.append(f"{key}_{str(value).lower()}")
            else:
                param_key_parts.append(f"{key}_{value}")

        param_key = "_".join(param_key_parts)
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
        os.makedirs(current_run_results_dir, exist_ok=True)

        # Save the config used for this specific run in the run directory for reproducibility
        run_config_path = os.path.join(current_run_results_dir, "config.yaml")
        try:
            with open(run_config_path, 'w') as f:
                yaml.dump(self.config, f, indent=4)
            print(f"Run configuration saved to {run_config_path}")
        except IOError as e:
             print(f"Error saving run configuration: {e}")


        # Instantiate and run the experiment runner
        try:
            runner = ExperimentRunner(config=self.config, param_results_dir=param_results_dir) # <--- Add param_results_dir here

            runner.run(results_dir=current_run_results_dir)
        except Exception as e:
            print(f"Error during experiment run: {e}")
            # Optionally log the error or clean up the directory

        print(f"Experiment execution finished for run: {run_dir_name}")

# Example usage (would be in main.py):
# config_file = "experiments/configs/exp_genetic.yaml"
# executor = ExperimentExecutor(config_file)
# executor.execute()