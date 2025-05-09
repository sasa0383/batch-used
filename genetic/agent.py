import numpy as np
import math
# Assuming the game package is structured correctly and accessible
from game.actions import ACTION_DELTAS, ACTION_MAP_INV, UP, DOWN, LEFT, RIGHT
from game.state import GameState # Import GameState for type hinting in helper functions
from statistics.normalization import min_max_normalize, standardize

class GeneticAgent:
    """
    Represents a single agent in the Genetic Algorithm population.
    Its "genome" is a set of weights for a 2-hidden-layer neural network.
    The agent predicts action probabilities/scores using this network based on game state features.
    Trained via direct game evaluation.
    """

    def __init__(self, num_features: int, num_outputs: int, hidden_size1: int, hidden_size2: int,
                 use_he_initialization: bool = True, use_leaky_relu: bool = True, normalize_features: bool = True, grid_size: int = 10):
        """
        Initializes a Genetic Agent with a genome representing NN weights.

        Args:
            num_features: The number of input features the agent expects (input layer size).
            num_outputs: The number of output values the agent produces (output layer size, should be 4 for actions).
            hidden_size1: The number of neurons in the first hidden layer.
            hidden_size2: The number of neurons in the second hidden layer.
            use_he_initialization: Whether to use He Initialization for weights.
            use_leaky_relu: Whether to use LeakyReLU activation in hidden layers.
            normalize_features: Whether to apply normalization to input features.
            grid_size: The size of the game grid, used for normalization and some feature calculations.
        """
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.use_he_initialization = use_he_initialization
        self.use_leaky_relu = use_leaky_relu
        self.normalize_features = normalize_features
        self.grid_size = grid_size

        # Define the structure and calculate genome size for the 2-Layer NN
        self.w1_shape = (self.num_features, self.hidden_size1)
        self.b1_shape = (self.hidden_size1,)
        self.w1_size = self.num_features * self.hidden_size1
        self.b1_size = self.hidden_size1

        self.w2_shape = (self.hidden_size1, self.hidden_size2)
        self.b2_shape = (self.hidden_size2,)
        self.w2_size = self.hidden_size1 * self.hidden_size2
        self.b2_size = self.hidden_size2

        self.w3_shape = (self.hidden_size2, self.num_outputs)
        self.b3_shape = (self.num_outputs,)
        self.w3_size = self.hidden_size2 * self.num_outputs
        self.b3_size = self.num_outputs

        self.genome_size = self.w1_size + self.b1_size + \
                           self.w2_size + self.b2_size + \
                           self.w3_size + self.b3_size

        if self.genome_size <= 0 or self.num_features <= 0 or self.num_outputs <= 0 or self.hidden_size1 <= 0 or self.hidden_size2 <= 0:
             raise ValueError(f"""
Calculated genome size ({self.genome_size}) or NN dimensions are invalid.
Check num_features ({self.num_features}), num_outputs ({self.num_outputs}),
hidden_size1 ({self.hidden_size1}), and hidden_size2 ({self.hidden_size2}).
Ensure these values are positive and correctly passed from trainer.py.
""")

        self.genome = self._initialize_genome()
        self._fitness: float | None = None

        # --- Feature Index Mapping (for get_features_from_state) ---
        # This maps meaningful feature names to their index in the input vector.
        # Update this dictionary to include all the new features.
        self.feature_indices = {
            # 1. Snake State Features (6 features)
            'is_moving_up': 0, 'is_moving_down': 1, 'is_moving_left': 2, 'is_moving_right': 3,
            'normalized_snake_length': 4,
            'normalized_turns_without_scoring': 5,

            # 2. Food Location Features (11 features)
            'is_food_up': 6, 'is_food_down': 7, 'is_food_left': 8, 'is_food_right': 9,
            'normalized_relative_x_distance_to_food': 10, 'normalized_relative_y_distance_to_food': 11,
            'food_near_north_wall': 12, 'food_near_south_wall': 13, 'food_near_east_wall': 14, 'food_near_west_wall': 15,
            'food_in_corner': 16,

            # 3. Danger Detection Features (9 features)
            'danger_straight': 17, 'danger_right': 18, 'danger_left': 19,
            'normalized_distance_to_north_wall': 20, 'normalized_distance_to_south_wall': 21,
            'normalized_distance_to_east_wall': 22, 'normalized_distance_to_west_wall': 23,
            'self_intersection_ahead': 24,
            'normalized_body_parts_nearby': 25,

            # 4. Path Planning Features (9 features)
            'normalized_head_x_position': 26, 'normalized_head_y_position': 27,
            'open_space_ratio': 28,
            'normalized_free_adjacent_cells': 29,
            'border_navigation': 30,
            'navigation_quadrant_ne': 31, 'navigation_quadrant_nw': 32, 'navigation_quadrant_se': 33, 'navigation_quadrant_sw': 34,

            # 5. Strategic Features (4 features)
            'safe_approach_available': 35,
            'corner_approach_required': 36,
            'food_between_snake_and_wall': 37,
            'normalized_space_after_food': 38,

            # Total features = 39
        }
        self.num_features = len(self.feature_indices)

        # Re-calculate genome size with the updated num_features
        self.genome_size = (self.num_features * self.hidden_size1) + self.hidden_size1 + \
                           (self.hidden_size1 * self.hidden_size2) + self.hidden_size2 + \
                           (self.hidden_size2 * self.num_outputs) + self.num_outputs

        if self.num_features != len(self.feature_indices):
             raise ValueError(f"Internal error: Mismatch between calculated num_features ({self.num_features}) and feature_indices size ({len(self.feature_indices)}).")

        # Re-initialize genome with the new size
        self.genome = self._initialize_genome()

    def _initialize_genome(self) -> np.ndarray:
        """Initializes the genome using random or He Initialization based on the calculated genome_size."""
        genome = np.empty(self.genome_size)

        if self.use_he_initialization:
            scale_w1 = math.sqrt(2.0 / self.num_features)
            scale_w2 = math.sqrt(2.0 / self.hidden_size1)
            scale_w3 = math.sqrt(2.0 / self.hidden_size2)

            w1 = np.random.randn(self.num_features, self.hidden_size1) * scale_w1
            w2 = np.random.randn(self.hidden_size1, self.hidden_size2) * scale_w2
            w3 = np.random.randn(self.hidden_size2, self.num_outputs) * scale_w3

            b1 = np.zeros(self.hidden_size1)
            b2 = np.zeros(self.hidden_size2)
            b3 = np.zeros(self.num_outputs)

        else:
            w1 = 2 * np.random.rand(self.num_features, self.hidden_size1) - 1
            b1 = 2 * np.random.rand(self.hidden_size1) - 1
            w2 = 2 * np.random.rand(self.hidden_size1, self.hidden_size2) - 1
            b2 = 2 * np.random.rand(self.hidden_size2) - 1
            w3 = 2 * np.random.rand(self.hidden_size2, self.num_outputs) - 1
            b3 = 2 * np.random.rand(self.num_outputs) - 1

        genome_parts = [w1.flatten(), b1.flatten(), w2.flatten(), b2.flatten(), w3.flatten(), b3.flatten()]
        genome = np.concatenate(genome_parts)

        if genome.shape[0] != self.genome_size:
             raise RuntimeError(f"Genome initialization error: Concatenated size ({genome.shape[0]}) does not match expected size ({self.genome_size}).")

        return genome


    def get_features_from_state(self, state: GameState) -> np.ndarray:
        """
        Extracts the new feature vector from the current game state.

        Args:
            state: The current GameState object.

        Returns:
            A numpy array representing the feature vector (shape: (num_features,)).
        """
        if not hasattr(state, 'snake') or not hasattr(state, 'food') or not hasattr(state, 'grid_size') or \
           not hasattr(state, 'direction') or not hasattr(state, 'turns_without_score'):
             print("Error: Invalid state object passed to get_features_from_state. Missing required attributes.")
             return np.zeros(self.num_features)

        snake_body = state.snake
        food_pos = state.food
        grid_size = state.grid_size
        head_pos = snake_body[0] if snake_body else (0, 0)
        current_direction = state.direction
        turns_without_score = state.turns_without_score
        snake_length = len(snake_body)

        features = np.zeros(self.num_features)

        # 1. Snake State Features
        features[self.feature_indices['is_moving_up']] = 1.0 if current_direction == 'UP' else 0.0
        features[self.feature_indices['is_moving_down']] = 1.0 if current_direction == 'DOWN' else 0.0
        features[self.feature_indices['is_moving_left']] = 1.0 if current_direction == 'LEFT' else 0.0
        features[self.feature_indices['is_moving_right']] = 1.0 if current_direction == 'RIGHT' else 0.0

        max_possible_length = grid_size * grid_size
        features[self.feature_indices['normalized_snake_length']] = float(snake_length) / max_possible_length if max_possible_length > 0 else 0.0

        grid_area = grid_size * grid_size
        features[self.feature_indices['normalized_turns_without_scoring']] = float(turns_without_score) / grid_area if grid_area > 0 else 0.0

        # 2. Food Location Features
        features[self.feature_indices['is_food_up']] = 1.0 if food_pos[0] < head_pos[0] and food_pos[1] == head_pos[1] else 0.0
        features[self.feature_indices['is_food_down']] = 1.0 if food_pos[0] > head_pos[0] and food_pos[1] == head_pos[1] else 0.0
        features[self.feature_indices['is_food_left']] = 1.0 if food_pos[1] < head_pos[1] and food_pos[0] == head_pos[0] else 0.0
        features[self.feature_indices['is_food_right']] = 1.0 if food_pos[1] > head_pos[1] and food_pos[0] == head_pos[0] else 0.0

        features[self.feature_indices['normalized_relative_x_distance_to_food']] = (food_pos[1] - head_pos[1]) / (grid_size - 1) if grid_size > 1 else 0.0
        features[self.feature_indices['normalized_relative_y_distance_to_food']] = (food_pos[0] - head_pos[0]) / (grid_size - 1) if grid_size > 1 else 0.0

        features[self.feature_indices['food_near_north_wall']] = 1.0 if food_pos[0] == 0 else 0.0
        features[self.feature_indices['food_near_south_wall']] = 1.0 if food_pos[0] == grid_size - 1 else 0.0
        features[self.feature_indices['food_near_east_wall']] = 1.0 if food_pos[1] == grid_size - 1 else 0.0
        features[self.feature_indices['food_near_west_wall']] = 1.0 if food_pos[1] == 0 else 0.0
        features[self.feature_indices['food_in_corner']] = 1.0 if (food_pos[0] == 0 or food_pos[0] == grid_size - 1) and (food_pos[1] == 0 or food_pos[1] == grid_size - 1) else 0.0

        # 3. Danger Detection Features
        current_dir_int = ACTION_MAP_INV.get(current_direction)
        danger_straight = 0.0
        danger_right = 0.0
        danger_left = 0.0

        if current_dir_int is not None:
            straight_delta = ACTION_DELTAS.get(current_dir_int)
            if straight_delta and (self._is_wall_ahead(head_pos, straight_delta, grid_size) or self._is_self_collision_ahead(head_pos, straight_delta, snake_body)):
                danger_straight = 1.0

            right_turn_delta = self._get_relative_direction_delta(current_dir_int, 'RIGHT')
            if right_turn_delta and (self._is_wall_ahead(head_pos, right_turn_delta, grid_size) or self._is_self_collision_ahead(head_pos, right_turn_delta, snake_body)):
                 danger_right = 1.0

            left_turn_delta = self._get_relative_direction_delta(current_dir_int, 'LEFT')
            if left_turn_delta and (self._is_wall_ahead(head_pos, left_turn_delta, grid_size) or self._is_self_collision_ahead(head_pos, left_turn_delta, snake_body)):
                 danger_left = 1.0

        features[self.feature_indices['danger_straight']] = danger_straight
        features[self.feature_indices['danger_right']] = danger_right
        features[self.feature_indices['danger_left']] = danger_left

        features[self.feature_indices['normalized_distance_to_north_wall']] = float(head_pos[0]) / (grid_size - 1) if grid_size > 1 else 0.0
        features[self.feature_indices['normalized_distance_to_south_wall']] = float(grid_size - 1 - head_pos[0]) / (grid_size - 1) if grid_size > 1 else 0.0
        features[self.feature_indices['normalized_distance_to_east_wall']] = float(grid_size - 1 - head_pos[1]) / (grid_size - 1) if grid_size > 1 else 0.0
        features[self.feature_indices['normalized_distance_to_west_wall']] = float(head_pos[1]) / (grid_size - 1) if grid_size > 1 else 0.0

        features[self.feature_indices['self_intersection_ahead']] = float(self._is_self_intersection_ahead(head_pos, current_dir_int, snake_body))

        features[self.feature_indices['normalized_body_parts_nearby']] = float(self._count_body_parts_nearby(head_pos, snake_body, range=2)) / max_possible_length if max_possible_length > 0 else 0.0

        # 4. Path Planning Features
        features[self.feature_indices['normalized_head_x_position']] = float(head_pos[1]) / (grid_size - 1) if grid_size > 1 else 0.0
        features[self.feature_indices['normalized_head_y_position']] = float(head_pos[0]) / (grid_size - 1) if grid_size > 1 else 0.0

        features[self.feature_indices['open_space_ratio']] = float(self._calculate_open_space_ratio(snake_body, grid_size))
        features[self.feature_indices['normalized_free_adjacent_cells']] = float(self._count_free_adjacent_cells(head_pos, snake_body, grid_size)) / 4.0 # Max 4 adjacent cells

        features[self.feature_indices['border_navigation']] = 1.0 if (head_pos[0] == 0 or head_pos[0] == grid_size - 1 or head_pos[1] == 0 or head_pos[1] == grid_size - 1) else 0.0

        center_x = grid_size // 2
        center_y = grid_size // 2
        features[self.feature_indices['navigation_quadrant_ne']] = 1.0 if head_pos[0] <= center_y and head_pos[1] >= center_x else 0.0
        features[self.feature_indices['navigation_quadrant_nw']] = 1.0 if head_pos[0] <= center_y and head_pos[1] < center_x else 0.0
        features[self.feature_indices['navigation_quadrant_se']] = 1.0 if head_pos[0] > center_y and head_pos[1] >= center_x else 0.0
        features[self.feature_indices['navigation_quadrant_sw']] = 1.0 if head_pos[0] > center_y and head_pos[1] < center_x else 0.0


        # 5. Strategic Features
        features[self.feature_indices['safe_approach_available']] = float(self._is_safe_approach_available(head_pos, food_pos, snake_body, grid_size))
        features[self.feature_indices['corner_approach_required']] = float(self._is_corner_approach_required(head_pos, food_pos, snake_body, grid_size))
        features[self.feature_indices['food_between_snake_and_wall']] = float(self._is_food_between_snake_and_wall(head_pos, food_pos, snake_body, grid_size))
        features[self.feature_indices['normalized_space_after_food']] = float(self._calculate_space_after_food(head_pos, food_pos, snake_body, grid_size)) / max_possible_length if max_possible_length > 0 else 0.0


        # Normalization is applied within the feature calculation where specified in the feature list.
        # If additional global normalization was needed, it would go here.

        if features.shape[0] != self.num_features:
             print(f"Error: Extracted feature vector size mismatch. Expected {self.num_features}, got {features.shape[0]}.")
             return np.zeros(self.num_features)

        return features

    # --- Helper functions for new features ---

    def _get_relative_direction_delta(self, current_dir_int: int, turn: str):
        """Calculates the direction delta for turning 'turn' (RIGHT or LEFT) from current_dir_int."""
        turn_mapping = {
            UP: (ACTION_DELTAS[RIGHT], ACTION_DELTAS[LEFT]),
            DOWN: (ACTION_DELTAS[LEFT], ACTION_DELTAS[RIGHT]),
            LEFT: (ACTION_DELTAS[UP], ACTION_DELTAS[DOWN]),
            RIGHT: (ACTION_DELTAS[DOWN], ACTION_DELTAS[UP]),
        }
        if current_dir_int in turn_mapping:
            right_delta, left_delta = turn_mapping[current_dir_int]
            if turn == 'RIGHT':
                return right_delta
            elif turn == 'LEFT':
                return left_delta
        return None

    def _is_wall_ahead(self, head_pos: tuple[int, int], direction_delta: tuple[int, int], grid_size: int):
        """Checks if moving one step in a direction results in wall collision."""
        new_head_y = head_pos[0] + direction_delta[0]
        new_head_x = head_pos[1] + direction_delta[1]
        return not (0 <= new_head_y < grid_size and 0 <= new_head_x < grid_size)

    def _is_self_collision_ahead(self, head_pos: tuple[int, int], direction_delta: tuple[int, int], snake_body: list[tuple[int, int]]):
        """Checks if moving one step in a direction results in self collision."""
        if not snake_body: return False

        new_head_y = head_pos[0] + direction_delta[0]
        new_head_x = head_pos[1] + direction_delta[1]
        new_head = (new_head_y, new_head_x)

        if len(snake_body) > 1 and new_head == snake_body[-1]:
             return False

        return new_head in snake_body[:-1]

    def _is_self_intersection_ahead(self, head_pos: tuple[int, int], current_dir_int: int | None, snake_body: list[tuple[int, int]]):
        """Checks if the snake body is directly in front of the head in the current direction."""
        if not snake_body or len(snake_body) < 2 or current_dir_int is None:
            return False

        straight_delta = ACTION_DELTAS.get(current_dir_int)
        if straight_delta:
            next_cell_y = head_pos[0] + straight_delta[0]
            next_cell_x = head_pos[1] + straight_delta[1]
            next_cell = (next_cell_y, next_cell_x)
            return next_cell in snake_body[:-1]

        return False

    def _count_body_parts_nearby(self, head_pos: tuple[int, int], snake_body: list[tuple[int, int]], range: int = 2):
        """Counts the number of body segments within a given Manhattan distance range from the head."""
        if not snake_body: return 0

        head_y, head_x = head_pos
        count = 0
        for i, (body_y, body_x) in enumerate(snake_body):
            if i == 0: continue
            distance = abs(head_y - body_y) + abs(head_x - body_x)
            if distance <= range:
                count += 1
        return count

    def _calculate_open_space_ratio(self, snake_body: list[tuple[int, int]], grid_size: int):
        """Calculates the ratio of available free spaces to total grid spaces."""
        total_spaces = grid_size * grid_size
        occupied_spaces = len(snake_body)
        free_spaces = total_spaces - occupied_spaces
        return float(free_spaces) / total_spaces if total_spaces > 0 else 0.0

    def _count_free_adjacent_cells(self, head_pos: tuple[int, int], snake_body: list[tuple[int, int]], grid_size: int):
        """Counts the number of adjacent cells (up, down, left, right) that are not occupied by the snake or walls."""
        head_y, head_x = head_pos
        free_count = 0
        for dy, dx in ACTION_DELTAS.values():
            next_y, next_x = head_y + dy, head_x + dx
            if 0 <= next_y < grid_size and 0 <= next_x < grid_size and (next_y, next_x) not in snake_body:
                free_count += 1
        return free_count

    def _is_safe_approach_available(self, head_pos: tuple[int, int], food_pos: tuple[int, int], snake_body: list[tuple[int, int]], grid_size: int):
        """
        Checks if there is at least one adjacent cell to the food that the snake can move into
        without immediate collision after eating the food. Simplified.
        """
        food_y, food_x = food_pos
        potential_snake_body = [food_pos] + snake_body[:-1] if snake_body else [food_pos]

        for dy, dx in ACTION_DELTAS.values():
            approach_y, approach_x = food_y + dy, food_x + dx
            approach_cell = (approach_y, approach_x)

            if 0 <= approach_y < grid_size and 0 <= approach_x < grid_size and approach_cell not in potential_snake_body:
                 return True

        return False

    def _is_corner_approach_required(self, head_pos: tuple[int, int], food_pos: tuple[int, int], snake_body: list[tuple[int, int]], grid_size: int):
        """Checks if the food is in a location where the snake might need to use a 'cornering' strategy."""
        food_y, food_x = food_pos
        adjacent_to_wall_count = 0
        if food_y == 0 or food_y == grid_size - 1:
            adjacent_to_wall_count += 1
        if food_x == 0 or food_x == grid_size - 1:
            adjacent_to_wall_count += 1

        return adjacent_to_wall_count >= 2

    def _is_food_between_snake_and_wall(self, head_pos: tuple[int, int], food_pos: tuple[int, int], snake_body: list[tuple[int, int]], grid_size: int):
        """Checks if the food is located such that the snake is between the food and a wall, potentially trapping it."""
        head_y, head_x = head_pos
        food_y, food_x = food_pos

        if head_y == food_y:
            if food_x > head_x:
                if food_x == grid_size - 1: return True
            elif food_x < head_x:
                if food_x == 0: return True

        if head_x == food_x:
            if food_y > head_y:
                if food_y == grid_size - 1: return True
            elif food_y < head_y:
                if food_y == 0: return True

        return False

    def _calculate_space_after_food(self, head_pos: tuple[int, int], food_pos: tuple[int, int], snake_body: list[tuple[int, int]], grid_size: int):
        """
        A simplified heuristic to estimate the free space available immediately after eating food.
        Counts free adjacent cells to the food location that are NOT the cell the snake came from.
        """
        food_y, food_x = food_pos
        free_adjacent_to_food_count = 0

        cell_snake_came_from = snake_body[1] if len(snake_body) > 1 else None

        for dy, dx in ACTION_DELTAS.values():
            next_y, next_x = food_y + dy, food_x + dx
            next_cell = (next_y, next_x)

            if 0 <= next_y < grid_size and 0 <= next_x < grid_size and \
               next_cell not in snake_body and \
               next_cell != cell_snake_came_from:
                free_adjacent_to_food_count += 1

        return free_adjacent_to_food_count


    def decide(self, state: GameState) -> int:
        """
        Uses the agent's genome (NN weights) to decide the next action based on the current game state.
        Performs a forward pass through the 2-hidden-layer neural network.
        Returns the index of the action with the highest predicted value.

        Args:
            state: The current GameState object.

        Returns:
            An integer representing the chosen action (0: UP, 1: DOWN, 2: LEFT, 3: RIGHT).
        """
        features = self.get_features_from_state(state)

        if not isinstance(features, np.ndarray) or features.shape[0] != self.num_features:
             print(f"Error: Invalid feature vector received by decide. Expected numpy array of size {self.num_features}, got {type(features)} with shape {features.shape if isinstance(features, np.ndarray) else 'N/A'}.")
             return np.random.choice([UP, DOWN, LEFT, RIGHT])

        w1_end = self.w1_size
        b1_end = w1_end + self.b1_size
        w2_end = b1_end + self.w2_size
        b2_end = w2_end + self.b2_size
        w3_end = b2_end + self.w3_size

        w1 = self.genome[0:w1_end].reshape(self.w1_shape)
        b1 = self.genome[w1_end:b1_end]
        w2 = self.genome[b1_end:w2_end].reshape(self.w2_shape)
        b2 = self.genome[w2_end:b2_end]
        w3 = self.genome[b2_end:w3_end].reshape(self.w3_shape)
        b3 = self.genome[w3_end:]

        hidden1_layer_input = np.dot(features, w1) + b1

        if self.use_leaky_relu:
             hidden1_layer_output = np.maximum(0.01 * hidden1_layer_input, hidden1_layer_input)
        else:
             hidden1_layer_output = np.maximum(0, hidden1_layer_input)

        hidden2_layer_input = np.dot(hidden1_layer_output, w2) + b2

        if self.use_leaky_relu:
             hidden2_layer_output = np.maximum(0.01 * hidden2_layer_input, hidden2_layer_input)
        else:
             hidden2_layer_output = np.maximum(0, hidden2_layer_input)

        output_layer_input = np.dot(hidden2_layer_output, w3) + b3
        action_preferences = output_layer_input

        chosen_action = np.argmax(action_preferences)

        valid_actions = [UP, DOWN, LEFT, RIGHT]
        if chosen_action not in valid_actions:
             print(f"Warning: Agent decided an invalid action index ({chosen_action}). Falling back to random action.")
             chosen_action = np.random.choice(valid_actions)

        return int(chosen_action)

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
        if not isinstance(fitness, (int, float)) or np.isnan(fitness) or np.isinf(fitness):
             print(f"Warning: Attempted to set fitness with non-numeric or invalid value: {fitness} ({type(fitness)}). Ignoring.")
             return
        self._fitness = fitness

    def get_weights(self) -> dict:
        """Returns weights and biases as separate numpy arrays."""
        w1_end = self.w1_size
        b1_end = w1_end + self.b1_size
        w2_end = b1_end + self.w2_size
        b2_end = w2_end + self.b2_size
        w3_end = b2_end + self.w3_size

        w1 = self.genome[0:w1_end].reshape(self.w1_shape)
        b1 = self.genome[w1_end:b1_end]
        w2 = self.genome[b1_end:w2_end].reshape(self.w2_shape)
        b2 = self.genome[w2_end:b2_end]
        w3 = self.genome[b2_end:w3_end].reshape(self.w3_shape)
        b3 = self.genome[w3_end:]

        return {
            'W1': w1,
            'b1': b1,
            'W2': w2,
            'b2': b2,
            'W3': w3,
            'b3': b3
        }