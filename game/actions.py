# project-root/game/actions.py

# Define discrete actions for the Snake game
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Optional: A mapping for easier use or printing
ACTION_MAP = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT"
}

# Optional: Inverse mapping
ACTION_MAP_INV = {
    "UP": UP,
    "DOWN": DOWN,
    "LEFT": LEFT,
    "RIGHT": RIGHT
}

# Optional: Coordinate changes for each action (dy, dx)
ACTION_DELTAS = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1)
}