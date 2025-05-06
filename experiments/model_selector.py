# project-root/experiments/model_selector.py

# Placeholder imports for future model classes
# These will be actual classes when genetic/trainer.py and statistics/interface.py/bayesian.py are implemented
# from genetic.trainer import GeneticTrainer
# from statistics.interface import StatisticalModelInterface # or specific models like BayesianModel

class ModelSelector:
    """Selects AI and Statistical model classes based on configuration."""

    # Map model names from config to their respective Python classes
    # Add new models here as they are implemented
    AI_MODELS = {
        # "genetic": GeneticTrainer # Example mapping
    }

    STAT_MODELS = {
        # "bayesian": BayesianModel # Example mapping
    }

    def __init__(self):
        # To avoid circular imports or issues during initial setup,
        # we can dynamically import or rely on later registration.
        # For now, we'll assume the classes are available when get_models is called
        # or import them within the method if needed.
        pass # No complex initialization needed here

    def get_models(self, ai_model_name: str, stat_model_name: str):
        """
        Returns instances of the selected AI and Statistical model classes.

        Args:
            ai_model_name: The name of the AI model from the config.
            stat_model_name: The name of the statistical model from the config.

        Returns:
            A tuple (ai_model_instance, stat_model_instance).

        Raises:
            ValueError: If the requested model name is not found.
        """
        # Dynamic import or lazy loading could go here if models are large
        # For simplicity, let's add direct imports assuming minimal complexity for now
        try:
            # Import model classes when needed to avoid early dependency issues
            from genetic.trainer import GeneticTrainer
            from statistics.bayesian import BayesianModel # Assuming Bayesian is the first stat model

            # Update the mappings once classes are imported
            self.AI_MODELS["genetic"] = GeneticTrainer
            self.STAT_MODELS["bayesian"] = BayesianModel

        except ImportError as e:
            print(f"Error importing model classes: {e}")
            # Handle gracefully, maybe specific error for which model failed

        ai_model_class = self.AI_MODELS.get(ai_model_name)
        stat_model_class = self.STAT_MODELS.get(stat_model_name)

        if not ai_model_class:
            raise ValueError(f"AI model '{ai_model_name}' not found.")
        if not stat_model_class:
            raise ValueError(f"Statistical model '{stat_model_name}' not found.")

        # Instantiate the models. Configuration will be passed later,
        # possibly during an .initialize() or directly in __init__
        # depending on how the classes are designed.
        # For now, assume __init__ takes relevant config parts or no args.
        # The config parsing (executor.py) will be responsible for passing the right args.
        try:
            ai_instance = ai_model_class() # Model classes will need __init__ methods
            stat_instance = stat_model_class()
            return ai_instance, stat_instance
        except Exception as e:
             raise RuntimeError(f"Failed to instantiate models: {e}")


# Example usage (would be in executor.py):
# selector = ModelSelector()
# ai_model, stat_model = selector.get_models("genetic", "bayesian")
# ai_model.initialize(ai_config) # Assuming models have an init method
# stat_model.initialize(stat_config)