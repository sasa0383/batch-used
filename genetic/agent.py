# project-root/genetic/agent.py

import numpy as np
# Removed the import that caused the circular dependency:
# from .population import Population # No longer needed here

class GeneticAgent:
    """
    Represents a single agent in the Genetic Algorithm population.
    Its "genome" is a set of weights for a 1-hidden-layer neural network.
    The agent predicts a single value (e.g., score) using this network.
    """

    # Modify __init__ to accept hidden_size
    def __init__(self, num_features: int, num_outputs: int, hidden_size: int):
        """
        Initializes a Genetic Agent with a random genome representing NN weights.

        Args:
            num_features: The number of input features the agent expects (input layer size).
            num_outputs: The number of output values the agent produces (output layer size, should be 1).
            hidden_size: The number of neurons in the hidden layer.
        """
        self.num_features = num_features
        self.num_outputs = num_outputs # Should be 1 for predicting a single value
        self.hidden_size = hidden_size

        # --- Define the structure and calculate genome size for the NN ---
        # Input layer to Hidden layer (W1, b1)
        self.w1_size = self.num_features * self.hidden_size
        self.b1_size = self.hidden_size
        # Hidden layer to Output layer (W2, b2)
        self.w2_size = self.hidden_size * self.num_outputs
        self.b2_size = self.num_outputs

        # Total genome size is the sum of all weights and biases
        self.genome_size = self.w1_size + self.b1_size + self.w2_size + self.b2_size
        # -----------------------------------------------------------------

        # Validate calculated genome size
        if self.genome_size <= 0:
             raise ValueError(f"""
Calculated genome size is invalid: {self.genome_size}.
Check num_features ({self.num_features}), hidden_size ({self.hidden_size}), and num_outputs ({self.num_outputs}).
Ensure these values are positive and correctly passed from trainer.py.
""")


        # Initialize the genome (all weights and biases concatenated) randomly between -1 and 1
        self.genome = 2 * np.random.rand(self.genome_size) - 1

        # Store fitness
        self._fitness: float | None = None # Initialize fitness to None or 0.0

    def decide(self, features: np.ndarray) -> float:
        """
        Uses the agent's genome (NN weights) to make a decision based on input features.
        Performs a forward pass through the 1-hidden-layer neural network.

        Args:
            features: A numpy array of input features (shape: (num_features,)).

        Returns:
            A single float value representing the agent's prediction (e.g., predicted score).
        """
        # Ensure features are a numpy array and have the correct size
        if not isinstance(features, np.ndarray):
             features = np.array(features)

        # Ensure the input feature vector size matches the expected size
        if features.shape[0] != self.num_features:
             raise ValueError(f"Input feature vector size mismatch. Expected {self.num_features}, got {features.shape[0]}.")

        # --- Split the 1D genome into NN weights and biases ---
        # Use slicing based on the calculated sizes
        w1_end = self.w1_size
        b1_end = w1_end + self.b1_size
        w2_end = b1_end + self.w2_size
        b2_end = w2_end + self.b2_size # Should equal self.genome_size

        w1 = self.genome[0:w1_end].reshape(self.num_features, self.hidden_size)
        b1 = self.genome[w1_end:b1_end] # Bias vector for hidden layer
        w2 = self.genome[b1_end:w2_end].reshape(self.hidden_size, self.num_outputs)
        b2 = self.genome[w2_end:b2_end] # Bias vector for output layer
        # -----------------------------------------------------

        # --- Perform the forward pass ---
        # Input layer to Hidden layer
        # z1 = x * W1 + b1
        # x is (num_features,), W1 is (num_features, hidden_size)
        # np.dot(features, w1) results in (hidden_size,)
        hidden_layer_input = np.dot(features, w1) + b1

        # Activation function (ReLU)
        hidden_layer_output = np.maximum(0, hidden_layer_input) # ReLU

        # Hidden layer to Output layer
        # z2 = hidden_output * W2 + b2
        # hidden_output is (hidden_size,), W2 is (hidden_size, num_outputs)
        # np.dot(hidden_layer_output, w2) results in (num_outputs,)
        output_layer_input = np.dot(hidden_layer_output, w2) + b2

        # Output layer activation (linear, as we're predicting a score)
        predicted_value_np = output_layer_input # Output is already the predicted value

        # Ensure the output is a single float value (since num_outputs is 1)
        predicted_value = float(predicted_value_np.item())

        return predicted_value


    def get_genome(self) -> np.ndarray:
        """Returns the agent's genome (concatenated weights and biases)."""
        return self.genome

    def set_genome(self, genome: np.ndarray):
        """Sets the agent's genome (concatenated weights and biases)."""
        if genome is None or genome.shape != (self.genome_size,):
             raise ValueError(f"Attempted to set genome with invalid shape. Expected ({self.genome_size},), got {genome.shape if genome is not None else 'None'}.")
        self.genome = genome

    def get_fitness(self) -> float | None:
        """Returns the agent's fitness score."""
        return self._fitness

    def set_fitness(self, fitness: float):
        """Sets the agent's fitness score."""
        # Ensure fitness is a number before setting
        if not isinstance(fitness, (int, float)) or np.isnan(fitness) or np.isinf(fitness):
             print(f"Warning: Attempted to set fitness with non-numeric or invalid value: {fitness} ({type(fitness)}). Ignoring.")
             return # Do not set invalid fitness

        self._fitness = fitness

    # Optional: Add a method to get weights/biases separately for inspection
    def get_weights(self) -> dict:
        """Returns weights and biases as separate numpy arrays."""
        w1_end = self.w1_size
        b1_end = w1_end + self.b1_size
        w2_end = b1_end + self.w2_size
        # b2_end = w2_end + self.b2_size # Should equal self.genome_size

        w1 = self.genome[0:w1_end].reshape(self.num_features, self.hidden_size)
        b1 = self.genome[w1_end:b1_end]
        w2 = self.genome[b1_end:w2_end].reshape(self.hidden_size, self.num_outputs)
        b2 = self.genome[w2_end:] # Slice till the end for b2

        return {
            'W1': w1,
            'b1': b1,
            'W2': w2,
            'b2': b2
        }
