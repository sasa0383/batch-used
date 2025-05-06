# project-root/experiments/runner.py

import os
import json
import time
import yaml
import numpy as np
# Import necessary components
from game.environment import SnakeEnvironment
from statistics.interface import StatisticalModelInterface
from genetic.trainer import GeneticTrainer
# from experiments.model_selector import ModelSelector # ModelSelector used by executor


class ExperimentRunner:
    """
    Orchestrates a single experiment run.
    Handles running game batches, statistical processing, and AI training.
    Supports resuming training from a previous genome and managing historical data.
    Now trains a Genetic Algorithm with a Neural Network agent.
    """

    def __init__(self, config: dict, param_results_dir: str):
        """
        Initializes the runner with experiment configuration and parameter directory.

        Args:
            config: The configuration dictionary for the experiment run.
                    Includes game, stat, and ai model configurations, including NN hidden_size.
            param_results_dir: The directory for this specific parameter set (results/{experiment}/{param_key}/).
                               Used to find previous runs for resuming.
        """
        self.config = config
        self.param_results_dir = param_results_dir

        self.batch_size = int(self.config.get('batch_size', 50))
        self.grid_size = int(self.config.get('grid_size', 10))

        self.resume_training = self.config.get('resume_training', False)
        self.initial_genome_data = None # Will store loaded genome if resuming

        # Define the path for historical data storage (central location)
        self.historical_data_dir = os.path.join("results", "historical") # Central historical storage
        self.historical_raw_data_path = os.path.join(self.historical_data_dir, "raw_data.json")
        self.historical_processed_stats_path = os.path.join(self.historical_data_dir, "processed_stats.json") # As per redesign plan
        self.historical_inputs_targets_path = os.path.join(self.historical_data_dir, "inputs_targets.json") # As per redesign plan
        self.historical_stat_summary_path = os.path.join(self.historical_data_dir, "summary_historical_stat.yaml") # For historical stats summary


        # Extract configs for sub-components
        self.stat_model_name = self.config.get('stat_model', 'bayesian')
        self.ai_model_name = self.config.get('ai_model', 'genetic')

        # Ensure the stat model and AI model exist in the configuration
        if self.stat_model_name not in self.config and self.stat_model_name != 'bayesian':
             print(f"Warning: Specific config for stat model '{self.stat_model_name}' not found in config. Using default.")
        if self.ai_model_name not in self.config and self.ai_model_name != 'genetic':
             print(f"Warning: Specific config for AI model '{self.ai_model_name}' not found in config. Using default.")


        stat_model_specific_config = self.config.get(self.stat_model_name, {})
        ai_model_specific_config = self.config.get(self.ai_model_name, {})

        # Pass flat GA/Fitness/Population configs to the specific AI model config
        ai_model_specific_config['population_size'] = self.config.get('population_size', 100)
        ai_model_specific_config['num_generations'] = int(self.config.get('num_generations', 10))
        ai_model_specific_config['mutation_rate'] = self.config.get('mutation_rate', 0.01)
        ai_model_specific_config['crossover_rate'] = self.config.get('crossover_rate', 0.9)
        ai_model_specific_config['selection_method'] = self.config.get('selection_method', 'tournament')
        ai_model_specific_config['tournament_size'] = self.config.get('tournament_size', 5) # Added tournament_size
        ai_model_specific_config['topk_k'] = self.config.get('topk_k', 10) # Added topk_k
        ai_model_specific_config['elite_count'] = int(self.config.get('elite_count', 0))

        # Pass the NEW hidden_size parameter from the main config to the AI model config
        ai_model_specific_config['hidden_size'] = int(self.config.get('hidden_size', 16)) # <-- Pass hidden_size


        ai_model_specific_config['population'] = {
             'mutation_rate': ai_model_specific_config.get('mutation_rate'),
             'crossover_rate': ai_model_specific_config.get('crossover_rate'),
             'selection_method': ai_model_specific_config.get('selection_method'),
             'elite_count': ai_model_specific_config.get('elite_count'),
             'tournament_size': ai_model_specific_config.get('tournament_size'), # Pass tournament_size
             'topk_k': ai_model_specific_config.get('topk_k'), # Pass topk_k
             # Add hidden_size to population config as well, although trainer passes it directly
             'hidden_size': ai_model_specific_config.get('hidden_size'),
        }

        # Fitness weights configuration (might be less relevant for data-driven MSE fitness but kept for compatibility)
        ai_model_specific_config['fitness'] = {
            'score_weight': self.config.get('score_weight', 1.0),
            'survival_weight': self.config.get('survival_weight', 5.0),
            'steps_weight': self.config.get('steps_weight', 0.02),
            'entropy_score_weight': self.config.get('entropy_score_weight', 0.5),
            'entropy_food_weight': self.config.get('entropy_food_weight', 0.0),
        }

        # --- If resuming, find and load the best genome ---
        if self.resume_training:
            # This method must be defined in this class
            self.initial_genome_data = self._find_and_load_latest_genome() # <-- Call to the method
            if self.initial_genome_data is None:
                 print("Warning: resume_training is True, but could not find a genome to load. Starting training from scratch.")
                 self.resume_training = False # Revert to non-resuming if loading fails
            else:
                 print("Resume training activated: Loaded genome data.")


        # Instantiate the environment (used for running initial batch only in the new design)
        render_game = self.config.get('render', False)
        game_speed = self.config.get('game_speed', 10)
        # The environment is primarily for generating raw data now. It's not used in the GA training loop.
        self.environment = SnakeEnvironment(grid_size=self.grid_size, render=render_game, game_speed=game_speed)


        # Instantiate the Statistical Model Interface
        # The stat model will now process historical data to produce input/target pairs
        stat_interface_config = {'stat_model': self.stat_model_name, **stat_model_specific_config}
        self.stat_model = StatisticalModelInterface(config=stat_interface_config)

        # Instantiate the AI Model (Genetic Trainer)
        # The trainer will now train on processed statistical data
        # Pass the AI model specific config which now includes 'hidden_size'
        self.ai_model = GeneticTrainer(config=ai_model_specific_config, initial_genome_data=self.initial_genome_data)
        # The trainer does NOT need the environment for evaluation games in the new design.
        # The environment is only used by the runner to generate the initial batch of data.
        # self.ai_model.set_evaluation_environment(self.environment) # This line should be removed


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


    def run(self, results_dir: str):
        """
        Executes a single experiment run according to the new data-driven plan.

        Args:
            results_dir: The specific run_XXX directory for saving this run's specific outputs (config, final genome, etc.).
        """
        print(f"Starting experiment run with batch size {self.batch_size} in {results_dir}")
        # results_dir is already created by the executor

        # --- Step A: Run Batch of Games & Reuse Previous Batch Data ---
        # This batch adds new data to the historical pool
        print("Running a new batch of games to add to historical data pool...")
        current_batch_raw_data = self._run_game_batch() # Run the new batch using the environment

        # Ensure the historical data directory exists
        os.makedirs(self.historical_data_dir, exist_ok=True)

        # Load existing historical raw data
        historical_raw_data = self._load_historical_raw_data() # Method defined in this class

        # Append the current batch data
        historical_raw_data.extend(current_batch_raw_data)

        # Save the updated historical raw data
        self._save_historical_raw_data(historical_raw_data) # Method defined in this class

        # --- Step B (Part 1): Process Historical Data with Statistical Model ---
        # The stat model now processes the cumulative historical data to produce input/target pairs
        print("Processing cumulative historical raw data with statistical model...")
        # The stat model interface needs to process the cumulative file path and save to historical processed path
        # This method should be defined in statistics/interface.py
        # It should call the appropriate processing method in the selected stat model (e.g., BayesianModel)
        processed_data_list = self.stat_model.process_batch_historical(
            historical_raw_data_path=self.historical_raw_data_path,
            output_path=self.historical_inputs_targets_path, # Save input/target pairs here
            summary_path=self.historical_stat_summary_path # Save historical stats summary here
        )
        # Note: The output_path now points to the inputs_targets.json file.
        # The stat_model.process_batch_historical should save the list of {'input_vector', 'expected_outcome'}

        if not processed_data_list:
             print("Warning: No processed data pairs generated from historical data. Cannot proceed with training.")
             self.environment.close_render()
             return # Stop if no data was processed


        # --- Step B (Part 2) & C: Train AI Model using Processed Statistical Data ---
        # The trainer will now load processed stats (input/target pairs) from the historical path
        print("Starting AI training using historical processed statistical data...")
        # The train method needs to accept the path to the inputs_targets.json file
        # It will evaluate agents on this data, not via environment games.
        self.ai_model.train(
            processed_stats_path=self.historical_inputs_targets_path, # Path to inputs_targets.json
            results_dir=results_dir, # Save this run's GA training results (summary, final genome) here
        )
        # Note: The train method in genetic/trainer.py is modified to handle this


        print("Experiment run finished.")

        # Close any open visualizations (from the initial batch run)
        # The environment is only used for the initial batch generation now.
        self.environment.close_render()


    def _run_game_batch(self) -> list:
        """
        Runs a batch of games using the environment and collects raw results.
        This batch's data is added to the historical data pool.
        Agents are NOT evaluated here for GA fitness in the new design.
        """
        print(f"Running a batch of {self.batch_size} games to generate new data...")
        raw_batch_results = []

        for i in range(self.batch_size):
            # Added progress print to see batch execution
            if (i + 1) % 10 == 0 or i == 0 or i == self.batch_size - 1:
                print(f"  Generating data from game {i+1}/{self.batch_size}...")
            # Run a single game episode with agent=None (random actions) for data generation
            # Pass max_steps to prevent infinite games
            # The environment's run_game returns the raw trial data
            trial_result = self.environment.run_game(agent=None, max_steps=2000)
            raw_batch_results.append(trial_result)

        print("Game data generation batch finished.")
        return raw_batch_results

    # --- New methods for handling historical data ---
    def _load_historical_raw_data(self) -> list:
        """Loads cumulative historical raw data from file."""
        historical_data = []
        if not os.path.exists(self.historical_raw_data_path):
            print(f"Historical raw data file not found at {self.historical_raw_data_path}. Starting historical data fresh.")
            return historical_data # Return empty list if file doesn't exist
        try:
            with open(self.historical_raw_data_path, 'r') as f:
                data = json.load(f)
                # Ensure loaded data is a list
                if isinstance(data, list):
                    historical_data = data
                    print(f"Loaded {len(historical_data)} records from historical raw data.")
                else:
                    print(f"Warning: Historical raw data file at {self.historical_raw_data_path} contains non-list data. Starting historical data fresh.")

            return historical_data

        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading historical raw data from {self.historical_raw_data_path}: {e}. Starting historical data fresh.")
            return [] # Return empty list on error
        except Exception as e:
             print(f"An unexpected error occurred loading historical raw data: {e}. Starting historical data fresh.")
             return []


    def _save_historical_raw_data(self, data: list):
        """Saves cumulative historical raw data to file."""
        # Ensure the historical directory exists
        os.makedirs(self.historical_data_dir, exist_ok=True)
        try:
            with open(self.historical_raw_data_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved {len(data)} records to historical raw data at {self.historical_raw_data_path}")
        except IOError as e:
            print(f"Error saving historical raw data to {self.historical_raw_data_path}: {e}")
        except Exception as e:
             print(f"An unexpected error occurred saving historical raw data: {e}")


    def _save_data(self, data, path: str):
        """
        (Original) Saves data (e.g., raw results of the current batch) to a JSON file
        within the current run_XXX directory.
        This is kept for logging the raw data of each specific run.
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

