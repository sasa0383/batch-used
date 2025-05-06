# project-root/game/visualizer.py

# Using matplotlib for visualization as it's often simpler than pygame for basic grids
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numpy as np # Added import for potential use, although not strictly needed for this fix

class GameVisualizer:
    """Visualizes the Snake game state."""

    def __init__(self, grid_size):
        """Initializes the visualizer."""
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(1, 1, figsize=(grid_size/2, grid_size/2)) # Adjust figure size
        self.ax.set_xlim([0, grid_size])
        self.ax.set_ylim([0, grid_size])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks(range(grid_size + 1))
        self.ax.set_yticks(range(grid_size + 1))
        self.ax.grid(True)
        self.ax.set_title("Snake Game")
        self.ax.invert_yaxis() # Set y-axis to start from top (row 0)
        plt.ion() # Turn on interactive mode
        plt.show(block=False) # Show the plot without blocking execution

        # Initialize empty lists/objects for drawing elements
        self._snake_segments = []
        self._food_patch = None

    def reset(self):
        """
        Resets the visualizer to prepare for a new game.
        Clears the current plot.
        """
        # Clear the axes to remove the previous game's drawing
        self.ax.clear()
        # Re-apply basic plot settings after clearing
        self.ax.set_xlim([0, self.grid_size])
        self.ax.set_ylim([0, self.grid_size])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks(range(self.grid_size + 1))
        self.ax.set_yticks(range(self.grid_size + 1))
        self.ax.grid(True)
        self.ax.set_title("Snake Game - New Game") # Update title
        self.ax.invert_yaxis() # Re-invert y-axis

        # Redraw immediately to show the empty grid
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            # Handle cases where the window might have been closed manually
            print(f"Warning: Could not draw during visualizer reset: {e}")
            # If drawing fails, the window might be closed, disable render?
            # This is more complex error handling. For now, just print warning.
            pass


    def update(self, state):
        """
        Updates the visualization based on the current game state.

        Args:
            state: The GameState object.
        """
        # Clear previous drawing (already done by reset or first update of a game)
        # If update is called multiple times per game, clearing here is necessary
        self.ax.clear() # Clear previous frame
        # Re-apply basic plot settings after clearing for the new frame
        self.ax.set_xlim([0, self.grid_size])
        self.ax.set_ylim([0, self.grid_size])
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks(range(self.grid_size + 1))
        self.ax.set_yticks(range(self.grid_size + 1))
        self.ax.grid(True)
        self.ax.set_title(f"Snake Game - Score: {state.score} - Steps: {state.steps} - {state.direction}") # Update title with game info
        self.ax.invert_yaxis() # Re-invert y-axis

        # Draw snake
        for i, (r, c) in enumerate(state.snake):
             # Plotting (col, row) for grid (row, col)
             rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor='black', facecolor='green')
             if i == 0: # Head
                  rect = patches.Rectangle((c, r), 1, 1, linewidth=1, edgecolor='black', facecolor='darkgreen')
             self.ax.add_patch(rect)

        # Draw food
        food_r, food_c = state.food
        food_patch = patches.Rectangle((food_c, food_r), 1, 1, linewidth=1, edgecolor='black', facecolor='red')
        self.ax.add_patch(food_patch)

        # Redraw the figure
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            # Handle cases where the window might have been closed manually
            # In a real application, you might want to catch this and stop rendering
            # For now, just print a warning.
            print(f"Warning: Could not draw during visualizer update: {e}")
            # A more robust solution would check if plt.get_fignums() contains self.fig.number


    def close(self):
        """Closes the visualization window."""
        try:
            plt.close(self.fig)
            print("Visualization window closed.")
        except Exception as e:
            print(f"Warning: Could not close visualization window: {e}")

# Example usage (would be in environment.py or an external loop):
# visualizer = GameVisualizer(grid_size=10)
# # In environment.reset():
# # visualizer.reset()
# # In environment.step() or run_game loop:
# # visualizer.update(current_game_state)
# # time.sleep(1/game_speed)
# # After experiment finished (in runner):
# # visualizer.close()