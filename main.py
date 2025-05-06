# project-root/main.py

import sys
import os
import yaml # <-- Added import yaml

# Add the project root to the Python path to allow importing modules
# Assuming main.py is at the project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ExperimentExecutor
from experiments.executor import ExperimentExecutor

def main():
    """
    Main entry point for the experiment framework.
    Loads the default configuration and starts the experiment execution.
    """
    # Define the path to the default configuration file
    default_config_path = os.path.join("experiments", "configs", "exp_genetic.yaml")

    # Check if the default config file exists
    if not os.path.exists(default_config_path):
        print(f"Error: Default configuration file not found at {default_config_path}")
        print("Please ensure 'experiments/configs/exp_genetic.yaml' exists.")
        sys.exit(1) # Exit if config is missing

    print(f"Starting experiment using config: {default_config_path}")

    try:
        # Create an ExperimentExecutor instance
        executor = ExperimentExecutor(config_path=default_config_path)

        # Execute the experiment
        executor.execute()

    except FileNotFoundError as e:
        print(f"Execution failed: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Execution failed due to config error: {e}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during execution
        print(f"An unexpected error occurred during execution: {e}")
        # In a real application, you might want more detailed logging
        sys.exit(1)

    print("Main execution finished.")

if __name__ == "__main__":
    # Ensure results directory exists at the project root if it doesn't
    if not os.path.exists("results"):
        os.makedirs("results")

    main()