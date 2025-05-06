# project-root/game/state.py

import random

class GameState:
    """Represents the state of the Snake game."""

    def __init__(self, grid_size=10):
        """Initializes the game state."""
        self.grid_size = grid_size
        # Snake starts at the center, length 3, moving right
        initial_head = (grid_size // 2, grid_size // 2)
        self.snake = [
            initial_head,
            (initial_head[0], initial_head[1] - 1),
            (initial_head[0], initial_head[1] - 2)
        ]
        # Initial direction (matching initial body)
        # Using string representation for simplicity, mapping later
        self.direction = 'RIGHT' # Corresponds to ACTION_MAP in actions.py
        self.food = self._place_food()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.collisions = False # Track if game ended due to collision
        self.food_eaten_count = 0 # Track total food items eaten

        # --- New Temporal Attributes ---
        self.time_since_last_food = 0
        self.turns_without_score = 0
        # -------------------------------

    def _place_food(self):
        """Places food at a random location not occupied by the snake."""
        while True:
            food_pos = (random.randint(0, self.grid_size - 1),
                        random.randint(0, self.grid_size - 1))
            if food_pos not in self.snake:
                return food_pos

    def to_dict(self):
        """Converts the current state to a dictionary."""
        return {
            'grid_size': self.grid_size,
            'snake': self.snake,
            'food': self.food,
            'direction': self.direction,
            'score': self.score,
            'steps': self.steps,
            'game_over': self.game_over,
            'collisions': self.collisions,
            'food_eaten_count': self.food_eaten_count,
            # --- Add new attributes to dict ---
            'time_since_last_food': self.time_since_last_food,
            'turns_without_score': self.turns_without_score
            # ----------------------------------
        }

    @classmethod
    def from_dict(cls, state_dict):
        """Creates a GameState object from a dictionary."""
        state = cls(state_dict['grid_size'])
        state.snake = [tuple(p) for p in state_dict['snake']] # Ensure tuples
        state.food = tuple(state_dict['food']) # Ensure tuple
        state.direction = state_dict['direction']
        state.score = state_dict['score']
        state.steps = state_dict['steps']
        state.game_over = state_dict['game_over']
        state.collisions = state_dict['collisions']
        state.food_eaten_count = state_dict['food_eaten_count']
        # --- Load new attributes from dict ---
        state.time_since_last_food = state_dict.get('time_since_last_food', 0) # Use .get for backwards compatibility
        state.turns_without_score = state_dict.get('turns_without_score', 0) # Use .get
        # -------------------------------------
        return state

# Note: This file defines the state structure and basic initialization/serialization.
# The game logic (applying actions, checking collisions, etc.) will be in environment.py