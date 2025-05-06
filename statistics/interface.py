# project-root/statistics/interface.py

import os
import json
import yaml # Need yaml for saving summary
import numpy as np  # Fix: Import numpy as np
# Import the specific stat models here
from .bayesian import BayesianModel # Import the Bayesian model class

class StatisticalModelInterface:
    """
    Interface and dispatcher for different statistical preprocessing models.
    Handles processing both single batch data (for logging) and cumulative historical data (for AI training).
    """
    # Map model names from config to their respective classes
    STAT_MODELS = {
        "bayesian": BayesianModel,
        # Add other statistical models here as they are implemented
    }

    def __init__(self, config=None):
        """
        Initializes the interface with configuration.

        Args:
            config: A dictionary containing the configuration for the stat model.
                    Should include 'stat_model' name and potentially model-specific configs.
        """
        self.config = config if config is not None else {}
        self.selected_model_name = self.config.get('stat_model')
        self.model_instance = None # Will be instantiated when process is called or explicitly initialized

        if self.selected_model_name not in self.STAT_MODELS:
            raise ValueError(f"Statistical model '{self.selected_model_name}' not found in available models.")

        # Instantiate the selected model
        model_class = self.STAT_MODELS[self.selected_model_name]
        # Pass model-specific config if available (e.g., self.config['bayesian'])
        model_config = self.config.get(self.selected_model_name, {})
        try:
            self.model_instance = model_class(config=model_config)
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate statistical model '{self.selected_model_name}': {e}")


    # --- Original process_batch (kept for logging latest batch stats in run_XXX folder) ---
    def process_batch(self, raw_data_path: str, output_path: str, summary_path: str):
        """
        Processes a single batch's raw data using the selected statistical model's
        original batch processing method. Saves output to stat_output.json and summary.yaml
        within the specific run_XXX folder.
        """
        if self.model_instance is None:
             raise RuntimeError("Statistical model not initialized.")

        print(f"Processing batch data from {raw_data_path} using {self.selected_model_name} model...")
        # Call the original batch processing method (e.g., BayesianModel.process)
        # This method is expected to return a dictionary of summary statistics.
        try:
            processed_data = self.model_instance.process(raw_data_path)
        except AttributeError:
            print(f"Error: Statistical model '{self.selected_model_name}' does not have a 'process' method for single batch processing.")
            processed_data = {} # Return empty if method doesn't exist
        except Exception as e:
            print(f"Error during single batch processing: {e}")
            processed_data = {}


        if not processed_data:
            print("Warning: Statistical model returned empty processed data for batch.")

        # Save processed data to output_path (e.g., stat_output.json in run_XXX)
        try:
            with open(output_path, 'w') as f:
                # Ensure data is JSON serializable (e.g., convert numpy types if necessary)
                # A simple conversion to float for all numeric values might work, or use a custom encoder
                # For simplicity, let's try converting known numpy types if they exist
                # This is a basic attempt; a robust solution might need a custom JSON encoder
                def convert_numpy(obj):
                    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                        return int(obj)
                    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, (np.ndarray,)): # Convert arrays to lists
                         return obj.tolist()
                    # Add other types as needed
                    raise TypeError('Unknown type: ' + str(obj))

                # Use the default encoder, but if it fails, try with a custom one
                try:
                    json.dump(processed_data, f, indent=4)
                except TypeError:
                     print("Attempting to save with custom numpy encoder...")
                     json.dump(processed_data, f, indent=4, default=convert_numpy)


            print(f"Processed statistical output saved to {output_path}")
        except IOError as e:
            print(f"Error saving processed data to {output_path}: {e}")

        # Save summary to summary_path (e.g., summary_stat.yaml in run_XXX)
        try:
            with open(summary_path, 'w') as f:
                yaml.dump(processed_data, f, indent=4) # Saving processed data as summary for simplicity
            print(f"Statistical summary saved to {summary_path}")
        except ImportError:
            print("Warning: PyYAML not installed. Cannot save summary.yaml.")
        except IOError as e:
            print(f"Error saving summary to {summary_path}: {e}")


    # --- New method to process historical cumulative data ---
    def process_batch_historical(self, historical_raw_data_path: str, output_path: str, summary_path: str):
        """
        Processes cumulative historical raw data using the selected statistical model's
        historical processing method. Saves output to a designated historical location
        (e.g., inputs_targets.json).

        Args:
            historical_raw_data_path: Path to the cumulative historical_raw_data.json file.
            output_path: Path where the cumulative processed data (e.g., inputs_targets.json) should be saved.
            summary_path: Path where the historical statistical summary should be saved (e.g., summary_historical_stat.yaml).
        """
        if self.model_instance is None:
             raise RuntimeError("Statistical model not initialized.")

        print(f"Processing cumulative historical data from {historical_raw_data_path} using {self.selected_model_name} model...")

        # --- CORRECTED CALL: Call the new historical processing method ---
        # This method is expected to return a list of dictionaries (input/target pairs).
        try:
            processed_data_list = self.model_instance.process_historical_data(historical_raw_data_path)
        except AttributeError:
             raise AttributeError(f"Statistical model '{self.selected_model_name}' does not have a 'process_historical_data' method required for historical processing.")
        except Exception as e:
             print(f"Error during historical data processing: {e}")
             processed_data_list = [] # Return empty list on error


        if not processed_data_list:
            print("Warning: Statistical model returned empty processed data list for historical data.")
            # Ensure processed_data_list is an empty list even if empty
            processed_data_list = []


        # Save processed historical data (list of input/target pairs) to the output path
        try:
            # Ensure the directory for historical outputs exists (e.g., results/historical/)
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w') as f:
                 # The data returned by process_historical_data should already be JSON serializable (lists, floats, ints, strings, dicts)
                 # If it still contains numpy types, the conversion needs to happen in BayesianModel.process_historical_data
                 json.dump(processed_data_list, f, indent=4)

            print(f"Processed historical statistical output saved to {output_path}")
        except IOError as e:
            print(f"Error saving processed historical data to {output_path}: {e}")
            # This is a critical error for the new design
        except TypeError as e:
             print(f"TypeError during JSON saving of historical data: {e}. Data might contain non-JSON serializable types.")
             # You might want to inspect processed_data_list here to see the types

        # Save historical summary (can be a summary of the historical data)
        # The processed_data_list is a list of pairs, not a summary dict.
        # We could calculate a summary of the historical data here before saving summary_path.
        # For simplicity, let's save a basic count for the summary file.
        historical_summary = {"total_records_processed": len(processed_data_list)}
        # You could add more summary stats about the historical data here if needed
        # e.g., mean score across all historical games, etc.

        try:
            summary_dir = os.path.dirname(summary_path)
            os.makedirs(summary_dir, exist_ok=True)
            with open(summary_path, 'w') as f:
                yaml.dump(historical_summary, f, indent=4)
            print(f"Historical statistical summary saved to {summary_path}")
        except ImportError:
            print("Warning: PyYAML not installed. Cannot save historical summary.yaml.")
        except IOError as e:
            print(f"Error saving historical summary to {summary_path}: {e}")

        # Note: This method processes the cumulative historical file.
        # Its output (inputs_targets.json) is used by the trainer.
        # It returns the list of processed data (input/target pairs).
        return processed_data_list # Return the list of input/target pairs
