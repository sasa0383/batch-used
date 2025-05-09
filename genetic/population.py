# project-root/genetic/population.py

import numpy as np
import random
from .agent import GeneticAgent # Import GeneticAgent class

class Population:
    """
    Manages the population of Genetic Agents and applies GA operations.
    Supports initializing the first generation from a previous genome.
    Agents are evaluated by playing games directly.
    Now supports agents with a 2-hidden-layer neural network and NN/feature flags.
    """

    # Modify __init__ to accept two hidden layer sizes and NN/feature flags
    def __init__(self, population_size: int, genome_size: int, config=None, initial_genome_data: list | None = None,
                 num_features: int = 0, num_outputs: int = 4, hidden_size1: int = 32, hidden_size2: int = 16,
                 use_he_initialization: bool = True, use_leaky_relu: bool = True, normalize_features: bool = True, grid_size: int = 10):
        """
        Initializes the population.

        Args:
            population_size: The number of agents in the population.
            genome_size: The size of each agent's genome (number of concatenated NN weights/biases).
                         This should be calculated based on the NN architecture.
            config: Optional configuration for GA operations (e.g., mutation_rate).
            initial_genome_data: Optional list representing the genome to start the first generation from.
                                If None, the population starts randomly.
            num_features: The number of input features agent expects.
            num_outputs: The number of output values agent produces.
            hidden_size1: The number of neurons in the first hidden layer.
            hidden_size2: The number of neurons in the second hidden layer.
            use_he_initialization: Whether to use He Initialization for weights.
            use_leaky_relu: Whether to use LeakyReLU activation in hidden layers.
            normalize_features: Whether to apply normalization to input features.
            grid_size: The size of the game grid, used for normalization in the Agent.
        """
        self.population_size = population_size
        self.genome_size = genome_size # Store the passed-in genome size
        self.config = config if config is not None else {}
        self.mutation_rate = self.config.get('mutation_rate', 0.01)
        self.crossover_rate = self.config.get('crossover_rate', 0.7)
        self.selection_method = self.config.get('selection_method', 'tournament')
        self.tournament_size = self.config.get('tournament_size', 5)
        self.elitism_count = self.config.get('elitism_count', 2)


        # Store NN and feature configuration to pass to agents
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.use_he_initialization = use_he_initialization
        self.use_leaky_relu = use_leaky_relu
        self.normalize_features = normalize_features
        self.grid_size = grid_size # Store grid size


        self.agents: list[GeneticAgent] = [] # List to hold agent instances
        self.initial_genome_data = initial_genome_data # Store initial genome data


        # Initialize the population
        if self.initial_genome_data is not None and len(self.initial_genome_data) == self.genome_size:
             print(f"Initializing population from provided genome data with size {len(self.initial_genome_data)}")
             self._initialize_from_genome(self.initial_genome_data)
        else:
             print(f"Initializing population randomly with genome size {self.genome_size}")
             self._initialize_random_population()

        if len(self.agents) != self.population_size:
             raise RuntimeError(f"Population size mismatch after initialization. Expected {self.population_size}, got {len(self.agents)}.")

        # Sort agents by fitness (descending) initially if some have non-None fitness (e.g., loaded from data)
        # Otherwise, sorting will happen after the first evaluation.
        self.agents.sort(key=lambda agent: agent.get_fitness() if agent.get_fitness() is not None else -float('inf'), reverse=True)


    def _initialize_random_population(self):
        """Initializes the population with agents having random genomes."""
        self.agents = []
        for _ in range(self.population_size):
            agent = GeneticAgent(
                num_features=self.num_features, # Pass num_features
                num_outputs=self.num_outputs, # Pass num_outputs
                hidden_size1=self.hidden_size1, # Pass hidden sizes
                hidden_size2=self.hidden_size2,
                use_he_initialization=self.use_he_initialization, # Pass NN flags
                use_leaky_relu=self.use_leaky_relu,
                normalize_features=self.normalize_features, # Pass feature normalization flag
                grid_size=self.grid_size # Pass grid size
            )
            # The agent's genome is initialized within its constructor based on the provided sizes
            if agent.genome_size != self.genome_size:
                 raise RuntimeError(f"Agent initialized with incorrect genome size. Expected {self.genome_size}, got {agent.genome_size}.")

            self.agents.append(agent)


    def _initialize_from_genome(self, initial_genome: list):
        """Initializes the population by duplicating a single provided genome."""
        initial_genome_np = np.array(initial_genome)
        if initial_genome_np.shape[0] != self.genome_size:
             raise ValueError(f"Initial genome data size mismatch. Expected {self.genome_size}, got {initial_genome_np.shape[0]}.")

        self.agents = []
        for _ in range(self.population_size):
             agent = GeneticAgent(
                num_features=self.num_features, # Pass num_features
                num_outputs=self.num_outputs, # Pass num_outputs
                hidden_size1=self.hidden_size1, # Pass hidden sizes
                hidden_size2=self.hidden_size2,
                use_he_initialization=self.use_he_initialization, # Pass NN flags
                use_leaky_relu=self.use_leaky_relu,
                normalize_features=self.normalize_features, # Pass feature normalization flag
                grid_size=self.grid_size # Pass grid size
             )
             agent.set_genome(initial_genome_np.copy()) # Set the provided genome
             if agent.genome_size != self.genome_size:
                  raise RuntimeError(f"Agent initialized from genome with incorrect genome size. Expected {self.genome_size}, got {agent.genome_size}.")

             self.agents.append(agent)


    def create_next_generation(self):
        """
        Selects parents, performs crossover and mutation to create the next generation.
        Replaces the current population with the new generation.
        """
        # Sort agents by fitness (descending) before selection
        self.agents.sort(key=lambda agent: agent.get_fitness() if agent.get_fitness() is not None else -float('inf'), reverse=True)

        next_generation_agents: list[GeneticAgent] = []

        # Elitism: Carry over the best agents directly to the next generation
        # Ensure elitism_count does not exceed population size
        num_elites = min(self.elitism_count, self.population_size)
        for i in range(num_elites):
            elite_agent = self.agents[i]
            # Create a new agent instance for the next generation to avoid modifying the current one
            new_elite = GeneticAgent(
                num_features=self.num_features, num_outputs=self.num_outputs,
                hidden_size1=self.hidden_size1, hidden_size2=self.hidden_size2,
                use_he_initialization=self.use_he_initialization, use_leaky_relu=self.use_leaky_relu,
                normalize_features=self.normalize_features, grid_size=self.grid_size
            )
            new_elite.set_genome(elite_agent.get_genome().copy()) # Copy the genome
            next_generation_agents.append(new_elite)


        # Fill the rest of the next generation using selection, crossover, and mutation
        while len(next_generation_agents) < self.population_size:
            # Selection: Choose parents based on fitness
            parent1 = self._select_parent()
            parent2 = self._select_parent() # Could be the same as parent1

            # Crossover: Combine genomes of parents
            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = self._crossover(parent1.get_genome(), parent2.get_genome())
            else:
                # No crossover, children are direct copies of parents
                child1_genome = parent1.get_genome().copy() if parent1.get_genome() is not None else np.array([])
                child2_genome = parent2.get_genome().copy() if parent2.get_genome() is not None else np.array([])

            # Mutation: Introduce random changes to children's genomes
            mutated_child1_genome = self._mutate(child1_genome)
            mutated_child2_genome = self._mutate(child2_genome)


            # Create new agent instances for the children and add to the next generation
            child1 = GeneticAgent(
                num_features=self.num_features, num_outputs=self.num_outputs,
                hidden_size1=self.hidden_size1, hidden_size2=self.hidden_size2,
                use_he_initialization=self.use_he_initialization, use_leaky_relu=self.use_leaky_relu,
                normalize_features=self.normalize_features, grid_size=self.grid_size
            )
            child1.set_genome(mutated_child1_genome) # Set the mutated genome
             # Ensure genome size matches after mutation (should be handled in _mutate)
            if child1.genome_size != self.genome_size:
                 print(f"Warning: Child1 genome size mismatch after mutation. Expected {self.genome_size}, got {child1.genome_size}. Attempting to resize or skipping.")
                 # Depending on how _mutate handles empty/invalid genomes, this might not be strictly needed.
                 # If _mutate guarantees returning a genome of self.genome_size, this check can be removed.
                 # For safety, let's ensure we only add if size is correct.
                 if mutated_child1_genome.shape[0] == self.genome_size:
                      next_generation_agents.append(child1)
                 else:
                      print("Skipping Child1 due to incorrect genome size after mutation.")
            else:
                 next_generation_agents.append(child1)


            # Add child2 if population size is not yet reached
            if len(next_generation_agents) < self.population_size:
                child2 = GeneticAgent(
                    num_features=self.num_features, num_outputs=self.num_outputs,
                    hidden_size1=self.hidden_size1, hidden_size2=self.hidden_size2,
                    use_he_initialization=self.use_he_initialization, use_leaky_relu=self.use_leaky_relu,
                    normalize_features=self.normalize_features, grid_size=self.grid_size
                )
                child2.set_genome(mutated_child2_genome) # Set the mutated genome
                # Ensure genome size matches after mutation
                if child2.genome_size != self.genome_size:
                     print(f"Warning: Child2 genome size mismatch after mutation. Expected {self.genome_size}, got {child2.genome_size}. Attempting to resize or skipping.")
                     if mutated_child2_genome.shape[0] == self.genome_size:
                          next_generation_agents.append(child2)
                     else:
                          print("Skipping Child2 due to incorrect genome size after mutation.")
                else:
                     next_generation_agents.append(child2)


        self.agents = next_generation_agents # Replace the old population with the new generation


    def _select_parent(self) -> GeneticAgent:
        """Selects a parent using the specified selection method."""
        if self.selection_method == 'tournament':
            # Tournament Selection
            tournament_members = random.sample(self.agents, min(self.tournament_size, self.population_size))
            # Select the agent with the best fitness from the tournament members
            # Handle potential None fitness by treating None as negative infinity
            best_in_tournament = max(tournament_members, key=lambda agent: agent.get_fitness() if agent.get_fitness() is not None else -float('inf'))
            return best_in_tournament
        else: # Default to Elitism-based selection (can be just picking from top performers)
            # Simple Elitism-based selection: Pick randomly from the top percentage or top N
            # For simplicity here, let's pick randomly from the entire population (effectively random selection if elitism_count is small)
            # A more proper implementation would select from the top performers based on their rank/fitness.
            # Given we already sorted and carried over elites, random selection from the *entire* population is a simple way
            # to get parents for the remaining slots, albeit not strictly based on 'elitism' beyond the direct carry-over.
            # Let's stick to a simple random choice for now for the remaining slots after elites are handled.
             print(f"Warning: Selection method '{self.selection_method}' not recognized. Using random selection.")
             return random.choice(self.agents)


    def _crossover(self, genome1: np.ndarray | None, genome2: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Performs single-point crossover between two parent genomes."""
        if genome1 is None or genome2 is None or genome1.shape[0] != self.genome_size or genome2.shape[0] != self.genome_size:
             print(f"Warning: Attempted crossover with invalid genome(s). Genome1 size: {genome1.shape[0] if genome1 is not None else 'None'}, Genome2 size: {genome2.shape[0] if genome2 is not None else 'None'}. Expected size: {self.genome_size}. Returning copies of valid genomes or random new genomes.")
             # Return copies of valid genomes, or new random genomes if inputs were None/invalid size
             valid_g1 = genome1.copy() if genome1 is not None and genome1.shape[0] == self.genome_size else (2 * np.random.rand(self.genome_size) - 1)
             valid_g2 = genome2.copy() if genome2 is not None and genome2.shape[0] == self.genome_size else (2 * np.random.rand(self.genome_size) - 1)
             return valid_g1, valid_g2


        crossover_point = random.randint(0, self.genome_size - 1)

        child1_genome = np.concatenate((genome1[:crossover_point], genome2[crossover_point:]))
        child2_genome = np.concatenate((genome2[:crossover_point], genome1[crossover_point:]))

        # Ensure children genomes have the correct size after concatenation
        if child1_genome.shape[0] != self.genome_size or child2_genome.shape[0] != self.genome_size:
             print(f"Error: Crossover resulted in incorrect genome sizes. Child1 size: {child1_genome.shape[0]}, Child2 size: {child2_genome.shape[0]}. Expected size: {self.genome_size}.")
             # As a fallback, return new random genomes if crossover failed unexpectedly
             return 2 * np.random.rand(self.genome_size) - 1, 2 * np.random.rand(self.genome_size) - 1


        return child1_genome, child2_genome

    def _mutate(self, genome: np.ndarray | None) -> np.ndarray:
        """Applies random mutation to the genome."""
        if genome is None or genome.shape[0] != self.genome_size:
            print(f"Warning: Attempted to mutate an invalid genome. Genome size: {genome.shape[0] if genome is not None else 'None'}. Expected size: {self.genome_size}. Returning a new random genome.")
            # Create a new random genome with the expected size
            return 2 * np.random.rand(self.genome_size) - 1

        mutated_genome = genome.copy()

        # Create a boolean mask of the same size as the genome
        mutation_mask = np.random.rand(self.genome_size) < self.mutation_rate

        noise_size = np.sum(mutation_mask)
        if noise_size > 0:
             # Apply Gaussian noise centered at 0
             # Standard deviation of 0.1 is a common choice, adjust as needed
             mutation_noise = np.random.normal(0, 0.1, size=noise_size)
             mutated_genome[mutation_mask] += mutation_noise

        return mutated_genome

    def get_best_agent(self) -> GeneticAgent | None:
        """Returns the agent with the highest fitness in the current population."""
        if not self.agents:
            return None
        # Sort to ensure the best agent is at the beginning
        self.agents.sort(key=lambda agent: agent.get_fitness() if agent.get_fitness() is not None else -float('inf'), reverse=True)
        return self.agents[0]

    def get_agents(self) -> list[GeneticAgent]:
        """Returns the list of agents in the population."""
        return self.agents

    def calculate_diversity(self) -> float:
        """
        Calculates the genetic diversity of the population.
        Using average pairwise Euclidean distance between genomes as a metric.
        """
        if len(self.agents) < 2:
             return 0.0 # Diversity is 0 with less than 2 agents

        genomes = [agent.get_genome() for agent in self.agents if agent.get_genome() is not None]
        if not genomes:
             return 0.0 # No valid genomes

        # Ensure all genomes have the expected size before calculating distance
        valid_genomes = [g for g in genomes if g.shape[0] == self.genome_size]
        if len(valid_genomes) < 2:
             return 0.0

        # Calculate average pairwise Euclidean distance
        total_distance = 0.0
        num_pairs = 0
        for i in range(len(valid_genomes)):
            for j in range(i + 1, len(valid_genomes)):
                total_distance += np.linalg.norm(valid_genomes[i] - valid_genomes[j])
                num_pairs += 1

        return total_distance / num_pairs if num_pairs > 0 else 0.0