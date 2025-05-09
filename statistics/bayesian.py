# project-root/statistics/bayesian.py

import json
import numpy as np
from scipy.stats import entropy as scipy_entropy
import math # Import math for distance calculation
# Corrected import: Use absolute import from the game package
from game.actions import ACTION_DELTAS, ACTION_MAP_INV, UP, DOWN, LEFT, RIGHT # Import ACTION_DELTAS and other constants for calculations

# Note: The role of this model changes from calculating *batch summaries*
# to calculating *per-trial features and target outcomes* from historical data.

class BayesianModel:
    """
    Processes raw game data to extract features and target outcomes for AI input.
    In the new design, this calculates features and a target vector per historical game trial.
    Also retains the original batch processing method for logging purposes.
    """

    def __init__(self, config=None):
        """
        Initializes the Bayesian model.
        Config might include parameters for feature extraction or normalization.
        """
        self.config = config if config is not None else {}
        # Keep any relevant config like normalization settings if added later
        # self.use_entropy = self.config.get('use_entropy', False) # Might still be relevant for entropy features

        # --- Feature Index Mapping (should match agent.py) ---
        # This is needed to construct the feature vector in the correct order.
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

        # Expected size of the input feature vector after processing
        self.expected_input_vector_size = self.num_features # Should be 39

    def process_historical_data(self, historical_raw_data: list) -> list[dict]:
        """
        Processes cumulative historical raw game data to extract features and
        determine target outcomes for each game trial.

        Args:
            historical_raw_data: A list of dictionaries, where each dictionary
                                 represents the raw result of a single game trial,
                                 including the 'final_state'.

        Returns:
            A list of dictionaries, where each dictionary contains:
            'input_vector': A numpy array of features extracted from the game state.
            'target_vector': A numpy array representing the target outcome (e.g., winning action).
                             For this GA design, the target vector might be less direct,
                             perhaps related to the final game outcome or a metric.
                             For now, let's focus on extracting features. The 'target'
                             concept here might need refinement based on how this
                             BayesianModel's output is intended to be used if not
                             for supervised learning of actions.
                             Given the GA trains by playing, the Bayesian model's
                             output here is likely for analysis or a different training
                             approach. We will create a placeholder target, perhaps
                             based on whether the game was won (food eaten > 0) or not.
        """
        processed_data = []

        for trial_data in historical_raw_data:
            final_state_dict = trial_data.get('final_state')
            score = trial_data.get('score', 0)
            steps = trial_data.get('steps', 0)

            if final_state_dict is None:
                print(f"Warning: Skipping historical trial data due to missing 'final_state'. Trial data: {trial_data}")
                continue

            # Extract features from the final state dictionary using the helper
            input_vector = self._extract_features_from_state_dict(final_state_dict)

            if input_vector is None or input_vector.shape[0] != self.expected_input_vector_size:
                 print(f"Warning: Skipping historical trial due to invalid feature vector size. Expected {self.expected_input_vector_size}, got {input_vector.shape[0] if input_vector is not None else 'None'}.")
                 continue

            # Define a simple placeholder target vector for now.
            # In a supervised learning context, this would be the 'correct' action.
            # For GA analysis, it could be a vector indicating success/failure or related metrics.
            # Let's use a simple binary target: 1 if score > 0 (food was eaten), 0 otherwise.
            # This is NOT a behavioral cloning target. It's just an example target.
            target_outcome = 1.0 if score > 0 else 0.0
            # You might want a more complex target based on the specific analysis needed.
            # For a classification task (e.g., predict if a state leads to success),
            # the target could be 1 if the game eventually reached a high score/won, 0 otherwise.
            # For now, a single value target based on scoring food is a simple starting point.
            target_vector = np.array([target_outcome], dtype=float) # Target vector size 1


            processed_data.append({
                'input_vector': input_vector,
                'target_vector': target_vector,
                'raw_trial_data': trial_data # Keep raw data reference if needed
            })

        # Optional: Apply normalization to the collected input_vectors if needed
        # This would require collecting all input vectors first, calculating global
        # min/max or mean/std_dev, and then normalizing each vector.
        # For now, normalization is handled within the feature extraction for specific features.

        print(f"Processed {len(processed_data)} historical game trials.")

        return processed_data


    def _extract_features_from_state_dict(self, state_dict: dict) -> np.ndarray | None:
        """
        Extracts the new feature vector from a game state dictionary.
        This logic mirrors the get_features_from_state in agent.py but operates on a dict.
        """
        if not all(k in state_dict for k in ['snake', 'food', 'grid_size', 'direction', 'turns_without_score']):
             print("Error: Invalid state dictionary passed to _extract_features_from_state_dict. Missing required keys.")
             return None

        snake_body = state_dict['snake']
        food_pos = state_dict['food']
        grid_size = state_dict['grid_size']
        head_pos = snake_body[0] if snake_body else (0, 0)
        current_direction = state_dict['direction']
        turns_without_score = state_dict['turns_without_score']
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

        max_possible_length = grid_size * grid_size
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
        # These helper functions need to work with dictionary data.
        # They were originally designed for GameState object in agent.py.
        # We can adapt them or recreate the necessary info from the dict.
        # Let's create equivalent helper functions within BayesianModel.

        features[self.feature_indices['safe_approach_available']] = float(self._is_safe_approach_available(head_pos, food_pos, snake_body, grid_size))
        features[self.feature_indices['corner_approach_required']] = float(self._is_corner_approach_required(head_pos, food_pos, snake_body, grid_size))
        features[self.feature_indices['food_between_snake_and_wall']] = float(self._is_food_between_snake_and_wall(head_pos, food_pos, snake_body, grid_size))
        features[self.feature_indices['normalized_space_after_food']] = float(self._calculate_space_after_food(head_pos, food_pos, snake_body, grid_size)) / max_possible_length if max_possible_length > 0 else 0.0


        if features.shape[0] != self.num_features:
             print(f"Error: Extracted feature vector size mismatch in BayesianModel. Expected {self.num_features}, got {features.shape[0]}.")
             return None

        return features


    # --- Helper functions for feature extraction from state dictionary ---
    # These mirror the helper functions in agent.py but take state components directly or from a dict.

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


    def process_batch_data(self, batch_raw_data: list) -> dict:
        """
        Processes a single batch of raw game data to calculate summary statistics.
        This method is likely used for logging/monitoring the performance of a batch
        of games, rather than generating training data for the GA agent itself.

        Args:
            batch_raw_data: A list of dictionaries, where each dictionary represents
                            the raw result of a single game trial.

        Returns:
            A dictionary containing summary statistics for the batch.
        """
        if not batch_raw_data:
            return {"message": "No batch data to process."}

        scores = [trial.get('score', 0) for trial in batch_raw_data]
        steps = [trial.get('steps', 0) for trial in batch_raw_data]
        food_counts = [trial.get('food_eaten_count', 0) for trial in batch_raw_data]
        collisions = [trial.get('collision', False) for trial in batch_raw_data] # Assuming 'collision' key exists

        summary_stats = {
            'num_games_in_batch': len(batch_raw_data),
            'mean_score': float(np.mean(scores)) if scores else 0.0,
            'median_score': float(np.median(scores)) if scores else 0.0,
            'max_score': float(np.max(scores)) if scores else 0.0,
            'mean_steps': float(np.mean(steps)) if steps else 0.0,
            'mean_food_eaten': float(np.mean(food_counts)) if food_counts else 0.0,
            'collision_rate': float(np.mean(collisions)) if collisions else 0.0 # Mean of boolean is proportion True
        }

        # Optional: Add entropy of scores if desired
        # This requires scipy, so keep it optional or handle the import gracefully
        try:
             from scipy.stats import entropy as scipy_entropy
             if scores and len(scores) > 1:
                 score_counts = {}
                 for score in scores:
                     score_counts[score] = score_counts.get(score, 0) + 1
                 total_games = len(scores)
                 if total_games > 0:
                     # Create a list of counts corresponding to unique scores
                     counts = list(score_counts.values())
                     # Normalize counts to probabilities
                     probabilities = [count / total_games for count in counts]
                     try:
                         # Use base=2 for entropy in bits
                         summary_stats['entropy_score'] = float(scipy_entropy(probabilities, base=2))
                     except ValueError:
                          # Handle cases like all probabilities being 0 or 1 (entropy is 0)
                         summary_stats['entropy_score'] = 0.0
                     except Exception as e:
                          print(f"Error calculating entropy: {e}")
                          summary_stats['entropy_score'] = 0.0 # Assign 0.0 on error
                 else:
                     summary_stats['entropy_score'] = 0.0
             else:
                 summary_stats['entropy_score'] = 0.0 # Entropy is 0 or undefined for 0 or 1 game

        except ImportError:
             print("Warning: scipy not installed. Cannot calculate entropy.")
             summary_stats['entropy_score'] = None # Set to None if scipy is not available
        except Exception as e:
             print(f"An unexpected error occurred during entropy calculation: {e}")
             summary_stats['entropy_score'] = None


        # Ensure all values in the dictionary are standard Python types before returning
        # This is crucial for JSON serialization by the interface or executor
        for key, value in summary_stats.items():
             if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, bool, np.bool_)):
                  summary_stats[key] = int(value) if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)) else bool(value)
             elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                  summary_stats[key] = float(value)
             # Add checks for other numpy types if necessary

        return summary_stats