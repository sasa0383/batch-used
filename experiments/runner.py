# project-root/experiments/runner.py

import os
import json
import time
import yaml
import numpy as np
import hashlib # Import hashlib for generating a hash
# Import necessary components
from game.environment import SnakeEnvironment # Now used by trainer
# from statistics.interface import StatisticalModelInterface # Not used for training data anymore
from genetic.trainer import GeneticTrainer
# from experiments.model_selector import ModelSelector # ModelSelector used by executor


class ExperimentRunner:
    """
    Orchestrates a single experiment run.
    Handles setting up the AI training (Genetic Trainer) which now evaluates agents
    by playing games directly.
    Supports resuming training from a previous genome.
    """

    def __init__(self, config: dict, param_results_dir: str):
        """
        Initializes the runner with experiment configuration and parameter directory.

        Args:
            config: The configuration dictionary for the experiment run.
                    Includes game, ai model configurations, NN architecture sizes,
                    NN/feature flags, and stagnation parameters.
            param_results_dir: The directory for this specific parameter set (results/{experiment}/{param_key}/).
                               Used to find previous runs for resuming.
        """
        self.config = config
        self.param_results_dir = param_results_dir

        # Get game configuration to pass to the trainer
        self.grid_size = int(self.config.get('grid_size', 10))
        self.render_game = self.config.get('render', False)
        self.game_speed = int(self.config.get('game_speed', 10))
        self.games_per_agent = int(self.config.get('games_per_agent', 5)) # Get games_per_agent

        self.resume_training = self.config.get('resume_training', False)
        self.initial_genome_data = None # Will store loaded genome if resuming

        # --- If resuming, find and load the best genome ---
        if self.resume_training:
            # This method must be defined in this class
            self.initial_genome_data = self._find_and_load_latest_genome() # <-- Call to the method
            if self.initial_genome_data is None:
                 print("Warning: resume_training is True, but could not find a genome to load. Starting training from scratch.")
                 self.resume_training = False # Revert to non-resuming if loading fails
            else:
                 print("Resume training activated: Loaded genome data.")


        # Extract configs for sub-components
        # self.stat_model_name = self.config.get('stat_model', 'bayesian') # Not used for training data anymore
        self.ai_model_name = self.config.get('ai_model', 'genetic')

        # Ensure the AI model exists in the configuration
        if self.ai_model_name not in self.config and self.ai_model_name != 'genetic':
             print(f"Warning: Specific config for AI model '{self.ai_model_name}' not found in config. Using default.")


        # We don't instantiate the StatisticalModelInterface for training data anymore.
        # stat_model_specific_config = self.config.get(self.stat_model_name, {})
        ai_model_specific_config = self.config.get(self.ai_model_name, {})

        # Pass relevant configs to the specific AI model config (GeneticTrainer)
        ai_model_specific_config['population_size'] = int(self.config.get('population_size', 100))
        ai_model_specific_config['num_generations'] = int(self.config.get('num_generations', 10))
        ai_model_specific_config['mutation_rate'] = float(self.config.get('mutation_rate', 0.03)) # Updated default
        ai_model_specific_config['crossover_rate'] = float(self.config.get('crossover_rate', 0.9))
        ai_model_specific_config['selection_method'] = self.config.get('selection_method', 'tournament')
        ai_model_specific_config['tournament_size'] = int(self.config.get('tournament_size', 5))
        ai_model_specific_config['topk_k'] = int(self.config.get('topk_k', 10))
        ai_model_specific_config['elite_count'] = int(self.config.get('elite_count', 5)) # Updated default

        # Pass the NN architecture parameters and flags
        ai_model_specific_config['hidden_size1'] = int(self.config.get('hidden_size1', 32)) # Updated default
        ai_model_specific_config['hidden_size2'] = int(self.config.get('hidden_size2', 16)) # Updated default
        ai_model_specific_config['use_he_initialization'] = self.config.get('use_he_initialization', True)
        ai_model_specific_config['use_leaky_relu'] = self.config.get('use_leaky_relu', True)
        ai_model_specific_config['normalize_features'] = self.config.get('normalize_features', True)


        # Pass stagnation parameters to the AI model config
        ai_model_specific_config['stagnation_threshold'] = float(self.config.get('stagnation_threshold', 0.001))
        ai_model_specific_config['stagnation_generations'] = int(self.config.get('stagnation_generations', 20))
        ai_model_specific_config['stagnation_mutation_boost'] = float(self.config.get('stagnation_mutation_boost', 0.1))

        # Pass game evaluation config to the AI model config
        ai_model_specific_config['games_per_agent'] = self.games_per_agent
        ai_model_specific_config['grid_size'] = self.grid_size
        ai_model_specific_config['render'] = self.render_game
        ai_model_specific_config['game_speed'] = self.game_speed


        ai_model_specific_config['population'] = {
             'mutation_rate': ai_model_specific_config.get('mutation_rate'),
             'crossover_rate': ai_model_specific_config.get('crossover_rate'),
             'selection_method': ai_model_specific_config.get('selection_method'),
             'elite_count': ai_model_specific_config.get('elite_count'),
             'tournament_size': ai_model_specific_config.get('tournament_size'),
             'topk_k': ai_model_specific_config.get('topk_k'),
             # Pass NN/feature flags to population config as well
             'hidden_size1': ai_model_specific_config.get('hidden_size1'),
             'hidden_size2': ai_model_specific_config.get('hidden_size2'),
             'use_he_initialization': ai_model_specific_config.get('use_he_initialization'),
             'use_leaky_relu': ai_model_specific_config.get('use_leaky_relu'),
             'normalize_features': ai_model_specific_config.get('normalize_features'),
        }

        # Fitness weights configuration (now used by FitnessCalculator for game results)
        ai_model_specific_config['fitness'] = {
            'score_weight': float(self.config.get('score_weight', 1.0)),
            'steps_weight': float(self.config.get('steps_weight', 0.01)), # Updated default
        }


        # Instantiate the AI Model (Genetic Trainer)
        # The trainer will now manage game environments for evaluation.
        # Pass the AI model specific config which now includes all necessary parameters.
        self.ai_model = GeneticTrainer(config=ai_model_specific_config, initial_genome_data=self.initial_genome_data)


    def _find_and_load_latest_genome(self) -> list | None:
        """
        Finds the latest run folder in the parameter directory and loads the best_genome.json.
        Returns the genome data (as a list) or None if not found.
        This method is defined within ExperimentRunner.
        """
        print(f"Attempting to find latest genome in {self.param_results_dir} for resuming...")
        # List all items in the parameter directory
        try:
            # Filter for directories that look like run folders (e.g., starting with 'run_')
            run_folders = [d for d in os.listdir(self.param_results_dir)
                           if os.path.isdir(os.path.join(self.param_results_dir, d)) and d.startswith('run_')]
        except FileNotFoundError:
            print(f"Parameter directory not found: {self.param_results_dir}. No previous runs to resume from.")
            return None

        if not run_folders:
            print(f"No previous run folders found in {self.param_results_dir} for resuming.")
            return None

        # Sort run folders by name (which includes timestamp) to find the latest completed run
        # Assuming folder names are like run_YYYYMMDD-HHMMSS
        run_folders.sort()
        latest_run_folder_name = run_folders[-1]
        latest_run_path = os.path.join(self.param_results_dir, latest_run_folder_name)
        genome_file_path = os.path.join(latest_run_path, "best_genome.json")

        if not os.path.exists(genome_file_path):
            print(f"Best genome file not found in latest run folder: {genome_file_path}. Cannot resume.")
            return None

        # Load the genome data
        try:
            with open(genome_file_path, 'r') as f:
                genome_data = json.load(f)
            # Basic check: ensure loaded data is a list
            if not isinstance(genome_data, list):
                 print(f"Warning: Loaded genome data from {genome_file_path} is not a list. Cannot resume.")
                 return None

            print(f"Successfully loaded genome from: {genome_file_path}")
            return genome_data # Return the loaded list/data
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading genome from {genome_file_path}: {e}")
            return None
        except Exception as e:
             print(f"An unexpected error occurred loading genome: {e}")
             return None


    def run(self, results_dir: str):
        """
        Executes a single experiment run according to the direct GA plan.

        Args:
            results_dir: The specific run_XXX directory for saving this run's specific outputs (config, final genome, etc.).
        """
        print(f"Starting experiment run in {results_dir}")
        # results_dir is already created by the executor

        # --- Phase 1: Train AI Model by Playing Games ---
        # The trainer now handles the game playing and evaluation loop internally.
        print("Starting AI training via direct game evaluation...")

        # The train method in genetic/trainer.py is modified to handle game playing
        self.ai_model.train(
            results_dir=results_dir, # Save this run's GA training results (summary, final genome) here
        )
        # Note: No processed_stats_path is passed anymore.


        print("Experiment run finished.")

        # No environment to close here, as the trainer manages them per generation evaluation.
        # If you had a separate environment for a final evaluation game after training, you'd close it here.


    # Removed historical data handling methods as per Phase 1 plan
    # def _run_game_batch(self) -> list: ...
    # def _load_historical_raw_data(self) -> list: ...
    # def _save_historical_raw_data(self, data: list): ...

    def _save_data(self, data, path: str):
        """
        (Original) Saves data (e.g., raw results of the current batch) to a JSON file
        within the current run_XXX directory.
        This is kept for logging purposes if needed, but the main training data flow changes.
        """
        # ... (remains the same)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            # print(f"Data saved to {path}") # Moved print outside or adjusted
        except IOError as e:
            print(f"Error saving data to {path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred saving data: {e}")

