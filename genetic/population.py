# project-root/genetic/population.py

import numpy as np
import random
from .agent import GeneticAgent

class Population:
    """
    Manages the population of Genetic Agents and applies GA operations.
    Supports initializing the first generation from a previous genome.
    Agents are evaluated on data, and their genomes represent NN weights.
    """

    # Modify __init__ to accept hidden_size
    def __init__(self, population_size: int, genome_size: int, config=None, initial_genome_data: list | None = None, num_features: int = 0, num_outputs: int = 1, hidden_size: int = 16):
        """
        Initializes the population.

        Args:
            population_size: The number of agents in the population.
            genome_size: The size of each agent's genome (number of concatenated NN weights/biases).
                         This should be calculated as (num_features * hidden_size) + hidden_size + (hidden_size * num_outputs) + num_outputs.
            config: Optional configuration for GA operations (e.g., mutation_rate).
            initial_genome_data: Optional list representing the genome to start the first generation from.
                                If None, the population starts randomly.
            num_features: The number of input features agents expect.
            num_outputs: The number of output values agents produce.
            hidden_size: The number of neurons in the hidden layer.
        """
        self.population_size = population_size
        self.genome_size = genome_size # This should be the calculated size from trainer.py
        self.config = config if config is not None else {}
        self.initial_genome_data = initial_genome_data

        # Store num_features, num_outputs, and hidden_size to pass to agents
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_size = hidden_size

        # Validate that genome_size passed from trainer matches the calculated size
        calculated_genome_size = (self.num_features * self.hidden_size) + self.hidden_size + \
                                 (self.hidden_size * self.num_outputs) + self.num_outputs

        if self.genome_size != calculated_genome_size:
             # This is a critical mismatch, likely an error in trainer's calculation or config
             raise ValueError(f"""
Population initialized with genome_size ({self.genome_size}) that does not match
calculated size from NN architecture ({self.num_features} input, {self.hidden_size} hidden, {self.num_outputs} output) = {calculated_genome_size}.
Please ensure genome_size is correctly calculated and passed from trainer.py.
""")
             # If we wanted to be less strict, we could print a warning and use the calculated size:
             # print(f"Warning: Population initialized with genome_size ({self.genome_size}) that doesn't match calculated size ({calculated_genome_size}). Using calculated size.")
             # self.genome_size = calculated_genome_size


        # GA Parameters from config
        self.mutation_rate = self.config.get('mutation_rate', 0.01)
        self.crossover_rate = self.config.get('crossover_rate', 0.9)
        self.selection_method = self.config.get('selection_method', 'tournament')
        self.topk_k = int(self.config.get('topk_k', max(1, population_size // 10)))
        self.tournament_size = int(self.config.get('tournament_size', 3))
        self.elite_count = int(self.config.get('elite_count', 0))

        # --- Initialize the population ---
        self.agents = []
        if self.initial_genome_data is not None:
            # --- Initialize from loaded genome ---
            if len(self.initial_genome_data) != self.genome_size:
                 print(f"Error: Loaded genome size ({len(self.initial_genome_data)}) does not match expected genome size ({self.genome_size}). Starting randomly.")
                 # Fallback to random initialization if genome size is wrong
                 self._initialize_random_population()
            else:
                print(f"Initializing population from loaded genome.")
                # Create the first agent with the loaded genome
                # Ensure genome data is a numpy array when setting
                loaded_genome_np = np.array(self.initial_genome_data)

                # Add elite agent if elite_count is desired even when loading
                # The elite agent will have the loaded genome without initial mutation
                if self.elite_count > 0:
                     # Pass num_features, num_outputs, and hidden_size to GeneticAgent
                     elite_agent = GeneticAgent(self.num_features, self.num_outputs, self.hidden_size)
                     elite_agent.set_genome(loaded_genome_np.copy()) # Copy genome
                     self.agents.append(elite_agent)
                     # Adjust population size to fill for the rest
                     fill_size = self.population_size - 1
                else:
                     # If no elitism when loading, fill the whole population
                     fill_size = self.population_size

                # Fill the rest of the population by mutating the loaded genome
                # Apply mutation rate defined in config to create variation in the first generation
                for _ in range(fill_size):
                     # Pass num_features, num_outputs, and hidden_size to GeneticAgent
                     new_agent = GeneticAgent(self.num_features, self.num_outputs, self.hidden_size)
                     # Start with a copy of the loaded genome and apply mutation
                     mutated_genome = loaded_genome_np.copy()
                     mutated_genome = self._mutate(mutated_genome) # Apply mutation
                     new_agent.set_genome(mutated_genome)
                     self.agents.append(new_agent)

                # Ensure we don't exceed population size
                self.agents = self.agents[:self.population_size]


        else:
            # --- Initialize randomly as before ---
            print("Initializing population randomly.")
            self._initialize_random_population()

        # Ensure the population size is correct after initialization
        if len(self.agents) != self.population_size:
            print(f"Warning: Population size mismatch after initialization. Expected {self.population_size}, got {len(self.agents)}. Adjusting.")
            # This should ideally not happen with the logic above, but as a safeguard
            self.agents = self.agents[:self.population_size] # Trim if too large
            while len(self.agents) < self.population_size: # Add random agents if too small
                 # Pass num_features, num_outputs, and hidden_size to GeneticAgent
                 self.agents.append(GeneticAgent(self.num_features, self.num_outputs, self.hidden_size))


    # Modify _initialize_random_population to pass num_features, num_outputs, and hidden_size
    def _initialize_random_population(self):
         """Helper to create a population of random agents."""
         # Pass num_features, num_outputs, and hidden_size to GeneticAgent
         self.agents = [GeneticAgent(self.num_features, self.num_outputs, self.hidden_size) for _ in range(self.population_size)]


    def evaluate(self, fitness_calculator):
        """
        Placeholder for evaluation. The trainer handles running games
        and setting fitness based on results.
        """
        pass


    def get_agents(self) -> list[GeneticAgent]:
        """Returns the list of agents in the population."""
        return self.agents


    def get_best_agent(self) -> GeneticAgent | None:
        """Returns the agent with the highest fitness in the current population."""
        if not self.agents:
            return None
        # Use a small epsilon or check for None/NaN fitness values if issues arise with max()
        # Filter out agents with None fitness before finding the max
        valid_agents = [agent for agent in self.agents if agent and agent.get_fitness() is not None]
        if not valid_agents:
             return None # No agents with valid fitness found

        return max(valid_agents, key=lambda agent: agent.get_fitness())


    def create_next_generation(self):
        """
        Creates the next generation of agents using selection, crossover, and mutation.
        Assumes agents have been evaluated and their fitness scores are set.
        """
        if not self.agents or self.population_size == 0:
            print("Warning: Cannot create next generation, population is empty or size is zero.")
            return

        # Sort agents by fitness in descending order
        # Ensure sorting handles None fitness values gracefully (put them at the end)
        sorted_agents = sorted(self.agents, key=lambda agent: agent.get_fitness() if agent and agent.get_fitness() is not None else -float('inf'), reverse=True)

        next_generation = []

        # Elitism: Carry over the top N agents unchanged
        for i in range(self.elite_count):
             if i < len(sorted_agents):
                  # Create a new agent instance with the genome of the elite agent
                  # Pass num_features, num_outputs, and hidden_size to GeneticAgent
                  elite_agent = GeneticAgent(self.num_features, self.num_outputs, self.hidden_size)
                  elite_agent.set_genome(sorted_agents[i].get_genome().copy()) # Copy genome
                  next_generation.append(elite_agent)


        # Fill the rest of the population using selection and reproduction
        # Need to ensure we select from agents that have been evaluated (fitness is not None)
        selectable_agents = [a for a in sorted_agents if a and a.get_fitness() is not None]


        while len(next_generation) < self.population_size:
            # Selection - Need at least 2 agents available to select from for parents
            if len(selectable_agents) < 2:
                 # If not enough valid agents for selection, fill rest with random agents
                 print(f"Warning: Not enough selectable agents ({len(selectable_agents)}) for reproduction. Filling remainder with random agents.")
                 while len(next_generation) < self.population_size:
                      # Pass num_features, num_outputs, and hidden_size to GeneticAgent
                      next_generation.append(GeneticAgent(self.num_features, self.num_outputs, self.hidden_size))
                 break # Exit the while loop

            # Select parents from the list of selectable agents
            parent1 = self._select_parent(selectable_agents)
            parent2 = self._select_parent(selectable_agents)

            # Ensure parents are valid agents before proceeding
            if parent1 is None or parent2 is None:
                 print("Warning: Parent selection returned None. Filling remaining population with random agents.")
                 while len(next_generation) < self.population_size:
                      # Pass num_features, num_outputs, and hidden_size to GeneticAgent
                      next_generation.append(GeneticAgent(self.num_features, self.num_outputs, self.hidden_size))
                 break # Exit the while loop


            # Crossover
            if random.random() < self.crossover_rate:
                child1_genome, child2_genome = self._crossover(parent1.get_genome(), parent2.get_genome())
            else:
                # No crossover, children are copies of parents' genomes
                child1_genome = parent1.get_genome().copy()
                child2_genome = parent2.get_genome().copy()


            # Mutation
            mutated_child1_genome = self._mutate(child1_genome)
            mutated_child2_genome = self._mutate(child2_genome)

            # Create new agent instances for the children
            # Pass num_features, num_outputs, and hidden_size to GeneticAgent
            child1 = GeneticAgent(self.num_features, self.num_outputs, self.hidden_size)
            child1.set_genome(mutated_child1_genome)

            # Pass num_features, num_outputs, and hidden_size to GeneticAgent
            child2 = GeneticAgent(self.num_features, self.num_outputs, self.hidden_size)
            child2.set_genome(mutated_child2_genome)

            # Add children to the next generation, up to population size
            if len(next_generation) < self.population_size:
                next_generation.append(child1)
            if len(next_generation) < self.population_size:
                next_generation.append(child2)


        # Replace the old population with the new generation
        self.agents = next_generation[:self.population_size] # Ensure size is exactly population_size


    def _select_parent(self, agents_pool: list[GeneticAgent]) -> GeneticAgent | None:
        """Selects a parent agent from a pool of agents based on the configured selection method."""
        if not agents_pool:
            return None

        # Ensure agents in the pool have valid fitness before selection
        valid_agents_pool = [a for a in agents_pool if a and a.get_fitness() is not None]
        if not valid_agents_pool:
             return None # Cannot select if no agents with valid fitness

        if self.selection_method == 'topk':
            # Select randomly from the top k agents in the provided pool
            # The pool (selectable_agents) is already sorted descending by fitness from create_next_generation
            k = min(self.topk_k, len(valid_agents_pool))
            if k <= 0:
                 # Fallback to random selection from the pool if k is invalid
                 return random.choice(valid_agents_pool) if valid_agents_pool else None

            top_k_pool = valid_agents_pool[:k]
            return random.choice(top_k_pool) if top_k_pool else None


        elif self.selection_method == 'tournament':
            # Select N agents randomly from the pool and pick the best among them
            tournament_size = min(self.tournament_size, len(valid_agents_pool))
            if tournament_size <= 0:
                 # Fallback to random selection from the pool if tournament size is invalid
                 return random.choice(valid_agents_pool) if valid_agents_pool else None

            try:
                tournament_agents = random.sample(valid_agents_pool, tournament_size)
            except ValueError:
                print("Warning: Could not select enough agents for tournament. Using all available in pool.")
                tournament_agents = valid_agents_pool

            if not tournament_agents:
                 return None

            # Use a small epsilon or check for None/NaN fitness values if issues arise with max()
            return max(tournament_agents, key=lambda agent: agent.get_fitness())

        else:
            print(f"Warning: Unknown selection method '{self.selection_method}'. Falling back to tournament.")
            self.selection_method = 'tournament'
            return self._select_parent(valid_agents_pool)


    def _crossover(self, genome1: np.ndarray, genome2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies one-point crossover to create two new genomes."""
        if genome1 is None or genome2 is None or genome1.shape != genome2.shape or genome1.shape[0] < 1:
             print("Warning: Cannot perform crossover, genomes are None, have incompatible shapes or are too short. Returning copies.")
             # Return copies if possible, otherwise empty arrays
             g1_copy = genome1.copy() if genome1 is not None else np.array([])
             g2_copy = genome2.copy() if genome2 is not None else np.array([])
             return g1_copy, g2_copy


        crossover_point = random.randint(0, self.genome_size - 1)

        child1_genome = np.concatenate((genome1[:crossover_point], genome2[crossover_point:]))
        child2_genome = np.concatenate((genome2[:crossover_point], genome1[crossover_point:]))

        return child1_genome, child2_genome

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        """Applies random mutation to the genome."""
        if genome is None or genome.shape[0] == 0:
            print("Warning: Attempted to mutate an empty or None genome. Returning a new random genome.")
            # Create a new random genome with the expected size
            return 2 * np.random.rand(self.genome_size) - 1

        mutated_genome = genome.copy()

        mutation_mask = np.random.rand(self.genome_size) < self.mutation_rate
        noise_size = np.sum(mutation_mask)
        if noise_size > 0:
             # Apply Gaussian noise centered at 0
             mutation_noise = np.random.normal(0, 0.1, size=noise_size) # Standard deviation 0.1
             mutated_genome[mutation_mask] += mutation_noise

             # Optional: Clamp genome values to a certain range if needed
             # mutated_genome = np.clip(mutated_genome, -10, 10) # Example clamping

        return mutated_genome
