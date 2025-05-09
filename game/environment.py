# project-root/game/environment.py

import random
import time
import json
import numpy as np
# Assuming state.py and actions.py are in the same game directory
from .actions import ACTION_DELTAS, ACTION_MAP, ACTION_MAP_INV, UP, DOWN, LEFT, RIGHT
from .state import GameState
# Ensure this is imported if render is used
try:
    from .visualizer import GameVisualizer
except ImportError:
    GameVisualizer = None # Handle cases where visualizer might not be available


class SnakeEnvironment:
    """Manages the Snake game environment and game loop."""

    def __init__(self, grid_size=10, render=False, game_speed=10):
        """
        Initializes the game environment.

        Args:
            grid_size: Size of the square grid.
            render: Whether to visualize the game.
            game_speed: Steps per second if rendering.
        """
        self.grid_size = grid_size
        self.state = GameState(grid_size) # Initialize the GameState object
        self.render = render
        self.game_speed = game_speed
        self.visualizer = None # Initialize visualizer to None

        # --- Initialize visualizer directly in __init__ if rendering is enabled ---
        if self.render and GameVisualizer is not None:
             try:
                 print("Attempting to initialize game visualizer...") # Debug print
                 self.visualizer = GameVisualizer(grid_size=self.grid_size)
                 print("Game visualizer initialized successfully.") # Debug print
             except Exception as e:
                  print(f"Warning: Failed to initialize visualizer: {e}. Rendering disabled.")
                  self.render = False # Disable rendering if initialization fails
                  self.visualizer = None
        elif self.render and GameVisualizer is None:
             print("Warning: GameVisualizer class not available (matplotlib not installed?). Rendering disabled.")
             self.render = False
        # --------------------------------------------------------------------------


    def reset(self):
        """Resets the game environment to the initial state."""
        self.state = GameState(self.grid_size) # Create a fresh GameState instance
        if self.visualizer:
            self.visualizer.reset() # Call the visualizer's reset method
        # No need to call _ensure_visualizer_initialized() here anymore
        return self.state.to_dict()


    def step(self, action: int):
        """
        Takes an action in the environment and updates the state.

        Args:
            action: An integer representing the action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT).

        Returns:
            A tuple: (new_state_dict, reward, done, info)
            - new_state_dict: The state after the action.
            - reward: Reward gained from the action.
            - done: True if the episode has ended, False otherwise.
            - info: Dictionary with additional information (e.g., collision type).
        """
        # Check state.game_over, not self.game_over
        if self.state.game_over:
            # If game is already over, taking a step does nothing
            return self.state.to_dict(), 0, True, {"message": "Game over"}

        self.state.steps += 1
        # --- Update temporal counters ---
        self.state.time_since_last_food += 1
        self.state.turns_without_score += 1
        # --------------------------------

        reward = -0.01 # Small penalty for each step to encourage efficiency

        # Get current head position and direction
        head_x, head_y = self.state.snake[0]
        current_direction_str = self.state.direction
        current_direction_int = ACTION_MAP_INV.get(current_direction_str)

        # Calculate new head position based on action
        new_head = (head_x, head_y) # Initialize new_head
        new_direction_str = current_direction_str # Initialize new_direction_str

        if action in ACTION_DELTAS:
            # Prevent reversing direction instantly
            reverse_directions = {
                UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT
            }
            if current_direction_int is not None and action != reverse_directions.get(current_direction_int, -1):
                # Valid move (not reversing)
                dy, dx = ACTION_DELTAS[action]
                new_head = (head_x + dy, head_y + dx)
                new_direction_str = ACTION_MAP[action]
                self.state.direction = new_direction_str
            else:
                 # Attempted reversal or invalid current_direction_int, keep moving in current direction delta
                 # Find the delta for the current direction string
                 current_dir_delta = ACTION_DELTAS.get(current_direction_int) if current_direction_int is not None else (0,0)
                 dy, dx = current_dir_delta
                 new_head = (head_x + dy, head_y + dx)
                 # Direction string remains the same

        else:
            # Invalid action integer
            self.state.game_over = True # Set game over on the state object
            self.state.collisions = True # Set collisions on the state object
            reward = -1 # Large penalty
            # If visualization is enabled, update it to show the final state before returning
            if self.visualizer: # Check before using visualizer
                 self.visualizer.update(self.state)
            return self.state.to_dict(), reward, True, {"message": "Invalid action", "collision_type": "invalid_action"}


        # Check for collision with walls
        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            self.state.game_over = True # Set game over on the state object
            self.state.collisions = True # Set collisions on the state object
            reward = -1 # Penalty for hitting wall
            # If visualization is enabled, update it to show the final state before returning
            if self.visualizer: # Check before using visualizer
                 self.visualizer.update(self.state)
            return self.state.to_dict(), reward, True, {"message": "Wall collision", "collision_type": "wall"}

        # Check for collision with self (body)
        snake_body_to_check = self.state.snake[:] # Copy the list
        if new_head in snake_body_to_check:
             self.state.game_over = True # Set game over on the state object
             self.state.collisions = True # Set collisions on the state object
             reward = -1 # Penalty for hitting self
             # If visualization is enabled, update it to show the final state before returning
             if self.visualizer: # Check before using visualizer
                 self.visualizer.update(self.state)
             return self.state.to_dict(), reward, True, {"message": "Self collision", "collision_type": "self"}


        # Check for food
        if new_head == self.state.food:
            self.state.snake.insert(0, new_head) # Add head, tail stays (grows)
            self.state.score += 1 # Score increases
            self.state.food_eaten_count += 1 # Track food eaten
            reward = 1 # Reward for eating food
            self.state.food = self._place_food() # Place new food
            # --- Reset temporal counters on scoring ---
            self.state.time_since_last_food = 0
            self.state.turns_without_score = 0
            # ------------------------------------------
        else:
            # Move snake: add new head, remove tail
            self.state.snake.insert(0, new_head)
            self.state.snake.pop()


        # Game ends if score reaches a certain limit or steps exceed limit (optional)
        # Currently, only ends on collision or invalid action.

        # Check if game is over (only by collision or invalid action in this version)
        done = self.state.game_over # Get done status from the state object

        info = {"message": "Step successful", "collision_type": None}
        if done:
             info["message"] = "Game finished"

        # Update visualizer AFTER state is updated, BEFORE returning
        if self.visualizer: # Check before using visualizer
             self.visualizer.update(self.state)
             if self.game_speed > 0:
                 time.sleep(1 / self.game_speed)

        return self.state.to_dict(), reward, done, info


    def _place_food(self):
        """Places food at a random location not occupied by the snake."""
        # This method accesses self.state.snake and self.grid_size correctly
        while True:
            food_pos = (random.randint(0, self.grid_size - 1),
                        random.randint(0, self.grid_size - 1))
            if food_pos not in self.state.snake:
                return food_pos

    def get_state(self):
        """Returns the current game state dictionary."""
        return self.state.to_dict()

    def render_game(self):
         """Explicitly trigger a render update if not already happening."""
         if self.visualizer: # Check before using visualizer
              self.visualizer.update(self.state)

    # --- Added the close method to SnakeEnvironment ---
    def close(self):
        """Closes the visualization window if a visualizer exists."""
        if self.visualizer:
            try:
                print("Closing visualization window...") # Debug print
                self.visualizer.close()
                print("Visualization window closed.") # Debug print
            except Exception as e:
                print(f"Warning: Could not close visualization window: {e}")
        # No other resources to close in this basic environment

    # --- Removed the _ensure_visualizer_initialized helper function ---
    # It's no longer needed with the direct initialization in __init__


    def get_observation(self):
        """
        Generates observations/features from the current state for the AI.
        This provides a simple grid representation.
        """
        # Using numpy for array creation
        observation = np.zeros((self.grid_size, self.grid_size), dtype=int)
        # Mark snake body
        for segment in self.state.snake:
            # Added boundary check for safety
            if 0 <= segment[0] < self.grid_size and 0 <= segment[1] < self.grid_size:
                observation[segment[0], segment[1]] = 1 # Snake body (row, col)
        # Mark snake head
        head = self.state.snake[0]
        # Added boundary check for safety
        if 0 <= head[0] < self.grid_size and 0 <= head[1] < self.grid_size:
             observation[head[0], head[1]] = 2 # Snake head (row, col)
        # Mark food
        food = self.state.food
        # Added boundary check for safety
        if 0 <= food[0] < self.grid_size and 0 <= food[1] < self.grid_size:
            observation[food[0], food[1]] = 3 # Food (row, col)

        return observation


    def _get_agent_features(self, state_dict):
        """
        Generates a feature vector from the state dictionary for the agent.
        This is a simple example based on common Snake AI features.
        The exact features should match what the GA agent's genome/weights expect.
        NOTE: This method is *not* used for training the data-driven GA in the current design.
        Features for training are generated in statistics/bayesian.py from logged historical data.
        This method might be used by a live-playing agent.
        """
        grid_size = state_dict['grid_size']
        snake = state_dict['snake']
        food = state_dict['food']
        if not snake: # Handle empty snake case
            head_x, head_y = (0, 0)
        else:
            head_x, head_y = snake[0]


        features = []

        # Feature 1 & 2: Distance to food
        food_dx = food[1] - head_y
        food_dy = food[0] - head_x
        features.extend([food_dx, food_dy])

        # Feature 3-6: Danger in 4 directions (UP, DOWN, LEFT, RIGHT relative to grid)
        # Check one step ahead
        danger_up = 0
        danger_down = 0
        danger_left = 0
        danger_right = 0

        # Check cell UP (row - 1)
        next_cell_up = (head_x - 1, head_y)
        if next_cell_up[0] < 0 or next_cell_up in snake:
            danger_up = 1

        # Check cell DOWN (row + 1)
        next_cell_down = (head_x + 1, head_y)
        if next_cell_down[0] >= grid_size or next_cell_down in snake:
            danger_down = 1

        # Check cell LEFT (col - 1)
        next_cell_left = (head_x, head_y - 1)
        if next_cell_left[1] < 0 or next_cell_left in snake:
            danger_left = 1

        # Check cell RIGHT (col + 1)
        next_cell_right = (head_x, head_y + 1)
        if next_cell_right[1] >= grid_size or next_cell_right in snake:
            danger_right = 1

        features.extend([danger_up, danger_down, danger_left, danger_right])

        # Feature 7-10: Current direction (one-hot encoded)
        direction_one_hot = [0, 0, 0, 0] # UP, DOWN, LEFT, RIGHT
        current_direction = state_dict.get('direction', 'RIGHT') # Use get with default for safety
        if current_direction == 'UP': direction_one_hot[0] = 1
        elif current_direction == 'DOWN': direction_one_hot[1] = 1
        elif current_direction == 'LEFT': direction_one_hot[2] = 1
        elif current_direction == 'RIGHT': direction_one_hot[3] = 1
        features.extend(direction_one_hot)

        # Feature 11-14: Is food in direction (UP, DOWN, LEFT, RIGHT) relative to snake head?
        # This checks if food is in the same row/col in that direction, ignoring obstacles
        food_dir_up = 1 if food[0] < head_x and food[1] == head_y else 0
        food_dir_down = 1 if food[0] > head_x and food[1] == head_y else 0
        food_dir_left = 1 if food[1] < head_y and food[0] == head_x else 0
        food_dir_right = 1 if food[1] > head_y and food[0] == head_x else 0
        features.extend([food_dir_up, food_dir_down, food_dir_left, food_dir_right])

        # Total features is 14 (Note: This is the OLD feature set, not the new 39)
        # This method is marked as NOT used for the GA training, so it's okay it's different.

        return np.array(features)


    def run_game(self, agent=None, max_steps=2000):
        """
        Runs a single game episode.

        Args:
            agent: An optional AI agent object with a decide(features) method.
                   If None, actions are random. This agent would use REAL-TIME game features.
                   In the data generation phase (usually agent=None), random actions are used.
                   The GA agent being trained does *not* use this method for evaluation.
            max_steps: Maximum steps before ending the game (to prevent infinite loops).

        Returns:
            A dictionary containing the raw results of the game.
        """
        # Ensure visualizer is initialized if rendering is enabled before the game starts
        # This call is needed here if run_game is used independently of the trainer loop
        # if self.render: # Check render flag before attempting initialization
        #     self._ensure_visualizer_initialized() # Removed helper, rely on __init__

        self.reset() # Reset the environment state and visualizer
        done = False
        steps = 0

        # Optional: Initial render pause
        # This happens inside the __init__ if visualizer is created there
        # if self.visualizer and self.game_speed > 0:
        #      self.visualizer.update(self.state) # Draw initial state
        #      time.sleep(1 / self.game_speed) # Pause at start

        while not done and steps < max_steps:
            # In the data-driven approach, the agent (if present) would typically
            # be a random agent taking actions to generate diverse data for training.
            # The step logic handles state updates, rewards, and checks for game over.
            # We do NOT collect step-by-step data here in this version, only final trial data
            # and possibly some temporal data logged within the state per step.

            # current_state_dict = self.get_state() # Not strictly needed here if only final state is logged

            if agent:
                # If an agent is provided (e.g., a random agent for data generation), use its decision
                # Note: This agent is NOT the GA agent being trained on historical data.
                # The features passed here would be the *real-time game state* features if used.
                # Currently, we use random actions for data generation simplicity.
                # features = self._get_agent_features(current_state_dict) # Use state features
                # action = agent.decide(features) # If agent decision is used
                 action = random.choice(list(ACTION_DELTAS.keys())) # Keep random for data gen simplicity
            else:
                # If no agent, take random actions for baseline batch run
                action = random.choice(list(ACTION_DELTAS.keys()))

            # Step the environment, which also handles visualizer update per step
            new_state_dict, reward, done, info = self.step(action)
            steps += 1 # This steps count is local to run_game, rely on state.steps for game duration

        # Game finished or max steps reached
        # Collect raw data for this trial
        raw_trial_data = {
            "score": self.state.score,
            "steps": self.state.steps, # Use steps from state object
            "food": self.state.food_eaten_count,
            "collisions": self.state.collisions, # True if collided
            # --- Log the final state dictionary ---
            "final_state": self.state.to_dict()
            # --------------------------------------
        }

        # Final visualizer update after game ends
        if self.visualizer: # Check before using visualizer
             self.visualizer.update(self.state) # Update to show final state (e.g., collision)
             # Add a slightly longer pause at the end of the game
             time.sleep(1.5) # Pause briefly on game over or max steps

        return raw_trial_data