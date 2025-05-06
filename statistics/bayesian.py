# project-root/statistics/bayesian.py

import json
import numpy as np
from scipy.stats import entropy as scipy_entropy
import math # Import math for distance calculation
# Corrected import: Use absolute import from the game package
from game.actions import ACTION_DELTAS # Import ACTION_DELTAS for directional checks

# Note: The role of this model changes from calculating *batch summaries*
# to calculating *per-trial features and outcomes* from historical data.

class BayesianModel:
    """
    Processes raw game data to extract features for AI input.
    In the new design, this calculates features per historical game trial.
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

    # --- Helper function to check for wall collision one step ahead ---
    def _is_wall_ahead(self, head_pos, direction_delta, grid_size):
        """Checks if moving one step in a direction results in wall collision."""
        new_head_y = head_pos[0] + direction_delta[0]
        new_head_x = head_pos[1] + direction_delta[1]

        # Wall collision check
        if not (0 <= new_head_y < grid_size and 0 <= new_head_x < grid_size):
            return True
        return False

    # --- Helper function to check for self collision one step ahead ---
    def _is_self_collision_ahead(self, head_pos, direction_delta, snake_body):
        """Checks if moving one step in a direction results in self collision."""
        new_head_y = head_pos[0] + direction_delta[0]
        new_head_x = head_pos[1] + direction_delta[1]
        new_head = (new_head_y, new_head_x)

        # Check against the body excluding the current head
        if new_head in snake_body:
             # If the new head is the same as the tail, it's not a collision because the tail moves
             if len(snake_body) > 1 and new_head == snake_body[-1]:
                 return False
             return True
        return False

    # --- Helper function to check if food is in a straight line in a direction ---
    def _is_food_in_direction(self, head_pos, food_pos, direction_delta):
        """Checks if food is in a straight line from head in the given cardinal direction."""
        head_y, head_x = head_pos
        food_y, food_x = food_pos
        dy, dx = direction_delta

        if dx != 0 and dy == 0: # Horizontal direction (Left or Right)
            if head_y == food_y: # Must be in the same row
                if dx > 0 and food_x > head_x: return True # Food to the right
                if dx < 0 and food_x < head_x: return True # Food to the left
        elif dy != 0 and dx == 0: # Vertical direction (Up or Down)
            if head_x == food_x: # Must be in the same column
                if dy > 0 and food_y > head_y: return True # Food down
                if dy < 0 and food_y < head_y: return True # Food up

        return False

    # --- Helper function for directional distance to food ---
    def _directional_distance_to_food(self, head_pos, food_pos, direction_delta, grid_size):
        """Calculates Manhattan distance to food in a cardinal direction, capped at grid size."""
        head_y, head_x = head_pos
        food_y, food_x = food_pos
        dy, dx = direction_delta

        distance = -1 # Indicate food not in this exact cardinal line

        if dx != 0 and dy == 0: # Horizontal (Left or Right)
            if head_y == food_y:
                if dx > 0 and food_x >= head_x: # Food is to the right or at head
                    distance = food_x - head_x
                elif dx < 0 and food_x <= head_x: # Food is to the left or at head
                    distance = head_x - food_x
        elif dy != 0 and dx == 0: # Vertical (Up or Down)
            if head_x == food_x:
                 if dy > 0 and food_y >= head_y: # Food is down or at head
                    distance = food_y - head_y
                 elif dy < 0 and food_y <= head_y: # Food is up or at head
                    distance = head_y - food_y

        # If food is not in that exact line, maybe return max distance or -1?
        # Let's return -1 if not in the direct cardinal line for this feature.
        if distance == -1:
            return float(distance) # Return -1.0

        # Cap distance at approximate grid boundary if needed (already handled by calculation logic)
        # But we can ensure it doesn't exceed grid bounds just in case, although the logic above should prevent this.
        # Example: dist up cannot be more than head_y
        if dy < 0: distance = min(distance, head_y) # Distance up capped by rows above
        elif dy > 0: distance = min(distance, grid_size - 1 - head_y) # Distance down capped by rows below
        elif dx < 0: distance = min(distance, head_x) # Distance left capped by columns left
        elif dx > 0: distance = min(distance, grid_size - 1 - head_x) # Distance right capped by columns right


        return float(distance)


    # --- New method to process historical raw data per trial ---
    def process_historical_data(self, historical_raw_data_path: str) -> list:
        """
        Processes cumulative historical raw game data to create input/target pairs
        for AI training. Extracts statistical-like features for each trial.

        Args:
            historical_raw_data_path: Path to the cumulative historical_raw_data.json file.

        Returns:
            A list of dictionaries, where each dictionary contains an 'input_vector'
            (statistical features for a trial) and an 'expected_outcome'.
        """
        input_target_pairs = []

        try:
            with open(historical_raw_data_path, 'r') as f:
                raw_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading historical raw data from {historical_raw_data_path} for processing: {e}")
            return [] # Return empty list on error

        if not raw_data:
            print("Warning: Historical raw data is empty.")
            return []

        print(f"Processing {len(raw_data)} historical game trials...")

        # --- Calculate Overall Survival Rate (Feature 4.1) ---
        total_games = len(raw_data)
        survival_count = sum(1 for trial in raw_data if not trial.get('collisions', True))
        overall_survival_rate = float(survival_count / total_games) if total_games > 0 else 0.0


        # Iterate through each trial's results in the historical data
        for trial in raw_data:
            # Extract raw metrics for this trial
            score = trial.get('score', 0)
            steps = trial.get('steps', 0)
            food_eaten = trial.get('food', 0)
            collided = trial.get('collisions', True) # True if collided

            # --- Get the final state data logged in environment.py ---
            final_state = trial.get('final_state')

            if final_state is None:
                print(f"Warning: 'final_state' missing for a trial. Skipping this trial.")
                continue # Skip trials where final state wasn't logged

            # Extract data from the final state
            snake_body = final_state.get('snake', [])
            food_pos = final_state.get('food', (0, 0))
            grid_size = final_state.get('grid_size', 10) # Get grid size from state
            final_head_pos = tuple(final_state.get('snake', [(-1,-1)])[0]) # Get final head pos as tuple
            final_direction_str = final_state.get('direction', 'RIGHT') # Get final direction

            # Handle case where snake body might be empty (game ended instantly?)
            if not snake_body or final_head_pos == (-1,-1):
                 print(f"Warning: Invalid snake body or head position in final_state for a trial. Skipping feature calculation for this trial.")
                 continue # Skip this trial if state is invalid

            # --- Calculate New Features from Final State and Logged Data ---

            # Features from previous snippet:
            snake_length = len(snake_body)
            distance_to_food = abs(final_head_pos[1] - food_pos[1]) + abs(final_head_pos[0] - food_pos[0]) # Manhattan distance
            time_since_last_food = final_state.get('time_since_last_food', steps)
            turns_without_score = final_state.get('turns_without_score', steps)

            # --- New Features Added in the Second Snippet ---

            # Direction deltas for checks
            delta_up = (-1, 0)
            delta_down = (1, 0)
            delta_left = (0, -1)
            delta_right = (0, 1)

            # 1. Is Wall Ahead (4 directions: U, D, L, R) - Based on final state
            is_wall_ahead_up = float(self._is_wall_ahead(final_head_pos, delta_up, grid_size))
            is_wall_ahead_down = float(self._is_wall_ahead(final_head_pos, delta_down, grid_size))
            is_wall_ahead_left = float(self._is_wall_ahead(final_head_pos, delta_left, grid_size))
            is_wall_ahead_right = float(self._is_wall_ahead(final_head_pos, delta_right, grid_size))

            # 2. Is Self Collision Ahead (4 directions: U, D, L, R) - Based on final state
            is_self_collision_ahead_up = float(self._is_self_collision_ahead(final_head_pos, delta_up, snake_body))
            is_self_collision_ahead_down = float(self._is_self_collision_ahead(final_head_pos, delta_down, snake_body))
            is_self_collision_ahead_left = float(self._is_self_collision_ahead(final_head_pos, delta_left, snake_body))
            is_self_collision_ahead_right = float(self._is_self_collision_ahead(final_head_pos, delta_right, snake_body))

            # 3. Is Food Ahead (4 directions: U, D, L, R) - Based on final state
            is_food_ahead_up = float(self._is_food_in_direction(final_head_pos, food_pos, delta_up))
            is_food_ahead_down = float(self._is_food_in_direction(final_head_pos, food_pos, delta_down))
            is_food_ahead_left = float(self._is_food_in_direction(final_head_pos, food_pos, delta_left))
            is_food_ahead_right = float(self._is_food_in_direction(final_head_pos, food_pos, delta_right))

            # 4. Overall Survival Rate (calculated once before the loop)
            # This feature is the same for all data points in this batch

            # 5. Directional Distance to Food (4 directions: U, D, L, R) - Based on final state
            dist_to_food_up = self._directional_distance_to_food(final_head_pos, food_pos, delta_up, grid_size)
            dist_to_food_down = self._directional_distance_to_food(final_head_pos, food_pos, delta_down, grid_size)
            dist_to_food_left = self._directional_distance_to_food(final_head_pos, food_pos, delta_left, grid_size)
            dist_to_food_right = self._directional_distance_to_food(final_head_pos, food_pos, delta_right, grid_size)


            # --- Define the statistical features for AI input ---
            # This is the input vector for the Genetic Agent.
            # Total features = 6 (original) + 4 (prev snippet) + 17 (this snippet) = 27
            input_vector = [
                float(score),       # Current Feature 1: Final score
                float(steps),       # Current Feature 2: Steps taken
                float(food_eaten),  # Current Feature 3: Food eaten
                float(1.0 if not collided else 0.0), # Current Feature 4: Survival status (1.0 if survived, 0.0 if collided)
                float(score) / max(1.0, float(steps)), # Current Feature 5: Score per step
                float(food_eaten) / max(1.0, float(steps)), # Current Feature 6: Food per step
                # Features from previous snippet:
                float(snake_length), # Feature 7
                float(distance_to_food), # Feature 8 (Manhattan)
                float(time_since_last_food), # Feature 9
                float(turns_without_score), # Feature 10
                # --- Add the New Features (17 total) ---
                is_wall_ahead_up, # Feature 11
                is_wall_ahead_down, # Feature 12
                is_wall_ahead_left, # Feature 13
                is_wall_ahead_right, # Feature 14
                is_self_collision_ahead_up, # Feature 15
                is_self_collision_ahead_down, # Feature 16
                is_self_collision_ahead_left, # Feature 17
                is_self_collision_ahead_right, # Feature 18
                is_food_ahead_up, # Feature 19
                is_food_ahead_down, # Feature 20
                is_food_ahead_left, # Feature 21
                is_food_ahead_right, # Feature 22
                overall_survival_rate, # Feature 23 (Same for all trials in this run)
                dist_to_food_up, # Feature 24
                dist_to_food_down, # Feature 25
                dist_to_food_left, # Feature 26
                dist_to_food_right, # Feature 27
                # --------------------------
                # Add more features here following the same pattern,
                # calculating them from the trial data or final_state.
            ]
            # The size of this input_vector is now 27.


            # --- Define the expected outcome/target for this trial ---
            # We'll keep the final score as the target for this example.
            expected_outcome = float(score) # Target: Final score

            # Ensure all values are standard Python types before appending
            # This is handled by converting to float above.
            input_target_pairs.append({
                'input_vector': input_vector,
                'expected_outcome': float(expected_outcome)
            })

        print(f"Generated {len(input_target_pairs)} input/target pairs with {len(input_target_pairs[0]['input_vector']) if input_target_pairs else 0} features each.")
        return input_target_pairs


    # --- Original process method (kept for compatibility/logging) ---
    def process(self, raw_data_path: str) -> dict:
        # ... (This method remains the same, it's for single batch summary)
        """
        (Original) Processes a single batch's raw data to calculate summary statistics.
        This method's output (summary stats) is NOT used for AI training input
        in the new data-driven design. It's called by StatisticalModelInterface.process_batch
        to save summary stats in the run_XXX folder.
        """
        print(f"Warning: Calling original batch process method from BayesianModel. Its output is NOT used for AI training input in the new design.")
        summary_stats = {}
        try:
            with open(raw_data_path, 'r') as f:
                raw_data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading raw data from {raw_data_path} for batch processing: {e}")
            return summary_stats # Return empty dict on error

        if not raw_data:
            return summary_stats

        scores = [trial.get('score', 0) for trial in raw_data]
        steps = [trial.get('steps', 0) for trial in raw_data]
        collisions = [trial.get('collisions', True) for trial in raw_data] # True if collided

        # Calculate summary statistics using NumPy, then convert to standard Python types
        if scores:
            summary_stats['mean_score'] = float(np.mean(scores))
            summary_stats['median_score'] = float(np.median(scores))
            summary_stats['max_score'] = float(np.max(scores))
        if steps:
            summary_stats['mean_steps'] = float(np.mean(steps))
            summary_stats['median_steps'] = float(np.median(steps))
            summary_stats['max_steps'] = float(np.max(steps))
        if collisions:
            # Calculate survival rate (proportion of games where collision was False)
            survival_count = sum(1 for c in collisions if not c)
            total_games = len(collisions)
            summary_stats['survival_rate'] = float(survival_count / total_games) if total_games > 0 else 0.0
            # Calculate collision rate (proportion of games where collision was True)
            collision_count = sum(1 for c in collisions if c)
            summary_stats['collision_rate'] = float(collision_count / total_games) if total_games > 0 else 0.0


        # Add entropy calculation if desired for the summary
        # Note: Entropy requires a distribution, which is tricky for small batches.
        # This is a basic example.
        if scores and len(scores) > 1:
             score_counts = {}
             for score in scores:
                 score_counts[score] = score_counts.get(score, 0) + 1
             total_games = len(scores)
             if total_games > 0:
                 probabilities = [count / total_games for count in score_counts.values()]
                 try:
                     summary_stats['entropy_score'] = float(scipy_entropy(probabilities, base=2))
                 except ValueError: # Handle cases like all probabilities being 0 or 1
                     summary_stats['entropy_score'] = 0.0
                 except Exception as e:
                      print(f"Error calculating entropy: {e}")
                      summary_stats['entropy_score'] = 0.0 # Assign 0.0 on error
             else:
                 summary_stats['entropy_score'] = 0.0


        # Ensure all values in the dictionary are standard Python types before returning
        # This is crucial for JSON serialization by the interface
        for key, value in summary_stats.items():
             if isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                  summary_stats[key] = int(value)
             elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                  summary_stats[key] = float(value)
             # Add other type conversions if necessary


        return summary_stats