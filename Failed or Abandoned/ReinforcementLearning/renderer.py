import pyglet
from pyglet.gl import glViewport
import random
import math
from model import *

# --- CONFIGURATION ---
WINDOW_WIDTH = 1000
SIDEBAR_PERCENT = 0.25
SPLIT_X = int(WINDOW_WIDTH * (1.0 - SIDEBAR_PERCENT))

# --- GRID PARAMETERS ---
GRID_N = 5
GRID_SIDE_LENGTH = SPLIT_X
CELL_SIZE = GRID_SIDE_LENGTH / GRID_N
WINDOW_HEIGHT = GRID_SIDE_LENGTH

# Colors
COLOR_BG      = (15, 15, 15)
COLOR_GRID    = (40, 40, 40)
COLOR_SIDEBAR = (25, 25, 25)
COLOR_ACCENT  = (240, 240, 240)
COLOR_TEXT    = (150, 150, 150)
COLOR_WALL    = (240, 240, 240)
COLOR_START   = (0, 200, 255)
COLOR_GOAL    = (255, 50, 50)
COLOR_AGENT   = (255, 215, 0)   # Gold color for the agent

# Setup
config = pyglet.gl.Config(sample_buffers=1, samples=16, depth_size=24, double_buffer=True)
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, "Basic Reinforcement Learning.", config=config)

# Batches
cells_batch = pyglet.graphics.Batch()
grid_batch = pyglet.graphics.Batch()
ui_batch = pyglet.graphics.Batch()
# ... existing batches ...
# New batch specifically for the Q-value text
labels_batch = pyglet.graphics.Batch()

# Storage for the label objects so they don't get garbage collected
q_value_labels = {} # Using a dict for easy (row, col) access

# Storage
active_cells = []
grid_lines = []
sidebar_elements = []

# --- HELPER FUNCTIONS ---
def rebuild_cells():
    """Reads WORLD and creates colored Rectangles."""
    active_cells.clear()
    for x in range(GRID_N):
        for y in range(GRID_N):
            value = WORLD[x][y] # WORLD is [row][col]
            
            cell_color = None
            if value == 1: cell_color = COLOR_WALL
            elif value == 2: cell_color = COLOR_START
            elif value == 3: cell_color = COLOR_GOAL
            
            if cell_color:
                # Map Row/Col to Screen X/Y
                # x (in loop) = row, y (in loop) = col
                px = y * CELL_SIZE
                py = (GRID_N - 1 - x) * CELL_SIZE # Flip Y so row 0 is at top
                
                rect = pyglet.shapes.Rectangle(px, py, CELL_SIZE, CELL_SIZE, 
                                               color=cell_color, batch=cells_batch)
                active_cells.append(rect)

def create_grid_lines():
    grid_lines.clear()
    for i in range(GRID_N + 1):
        pos = i * CELL_SIZE
        grid_lines.append(pyglet.shapes.Line(pos, 0, pos, GRID_SIDE_LENGTH, color=COLOR_GRID, batch=grid_batch))
        grid_lines.append(pyglet.shapes.Line(0, pos, GRID_SIDE_LENGTH, pos, color=COLOR_GRID, batch=grid_batch))
    
    sidebar_elements.append(pyglet.shapes.Line(SPLIT_X, 0, SPLIT_X, WINDOW_HEIGHT, color=COLOR_ACCENT, batch=ui_batch))

# --- INITIAL SETUP ---
rebuild_cells()
create_grid_lines()

# Sidebar UI
sidebar_elements.append(pyglet.shapes.Rectangle(SPLIT_X, 0, WINDOW_WIDTH - SPLIT_X, WINDOW_HEIGHT, color=COLOR_SIDEBAR, batch=ui_batch))
info_label = pyglet.text.Label("Iterations: 0", 
                               font_name="Google Sans Code NF", font_size=11,
                               x=SPLIT_X + 20, y=WINDOW_HEIGHT - 40, 
                               color=(200, 200, 200, 255), batch=ui_batch, multiline=True,
                               width=WINDOW_WIDTH - SPLIT_X - 40)

# --- AGENT SETUP ---
agent = Agent(pos=(0,0))

def create_q_labels():
    """Initializes a text label for every cell in the grid."""
    q_value_labels.clear()
    
    for row in range(GRID_N):
        for col in range(GRID_N):
            # Calculate center of the cell
            px = col * CELL_SIZE + (CELL_SIZE / 2)
            py = (GRID_N - 1 - row) * CELL_SIZE + (CELL_SIZE / 2)
            
            label = pyglet.text.Label(
                "",
                font_name="JetBrainsMono NF",
                font_size=8,  # Keep it small
                x=px, y=py,
                anchor_x='center', anchor_y='center',
                color=(100, 100, 100, 255), # Dim text by default
                batch=labels_batch
            )
            q_value_labels[(row, col)] = label

# Call this in your INITIAL SETUP section
create_q_labels()

# Create the Agent Sprite (The yellow circle)
# We place it in 'grid_batch' so it renders on top of the cells but below the UI
agent_sprite = pyglet.shapes.Circle(x=0, y=0, radius=CELL_SIZE/3, color=COLOR_AGENT, batch=grid_batch)

def update_agent_visuals():
    """Syncs the sprite position with the agent's logical position."""
    # Agent pos is (row, col)
    row, col = agent.pos
    
    # Map to screen pixels
    # Center X = col * size + half_size
    px = col * CELL_SIZE + (CELL_SIZE / 2)
    # Center Y = (flipped_row) * size + half_size
    py = (GRID_N - 1 - row) * CELL_SIZE + (CELL_SIZE / 2)
    
    agent_sprite.x = px
    agent_sprite.y = py
    
    # Update text
    info_label.text = f"""
    Iter: {agent.ITERATIONS}
    Steps Left: {agent.STEPS_AVAILBLE}
    Successes: {agent.SUCCESS}"""

def update_cell_label(row, col):
    """Updates the text of a single cell to show max Q-value."""
    if (row, col) not in q_value_labels: return

    # Get all 8 values for this cell
    probs = agent.q_table[row][col]
    
    # Find the max probability (Confidence)
    max_prob = np.max(probs)
    
    # Update the text
    label = q_value_labels[(row, col)]
    # Map indices 0-7 to arrows
    ARROWS = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖']

    # Inside update_cell_label:
    best_action_idx = np.argmax(probs)
    arrow = ARROWS[best_action_idx]
    label.text = f"{arrow}" # Multiline label
    
    # Optional: Highlight highly confident cells with brighter text
    if max_prob > 0.5:
        label.color = (255, 255, 255, 255) # Bright White
        label.bold = True
    else:
        label.color = (100, 100, 100, 255) # Dim Grey


# Initialize visual position
update_agent_visuals()

# --- GAME LOOP ---

def update(dt):
    """
    Runs every frame. 
    1. Moves the agent logic.
    2. Updates the screen.
    """
    old_row, old_col = agent.pos
    
    # 1. Run Logic
    if not agent.COMPLETED:
        agent.move()
        print(agent.q_table)
        update_cell_label(old_row, old_col)
    else:
        # If completed (hit goal or ran out of steps), reset immediately
        agent.reset()
        
        # Optional: Clear the trail in WORLD if you want to avoid cyan clutter
        # rebuild_cells() 
    
    # 2. Update Visuals
    update_agent_visuals()
    print(agent.q_table)

# Schedule the update. 
# 0.05 = 20 times per second. Change to 0.01 for super fast, or 0.2 for slow.
pyglet.clock.schedule_interval(update, 0.05)


@window.event
def on_key_press(symbol, modifiers):
    if symbol == pyglet.window.key.SPACE:
        rebuild_cells()
    # Manual step for debugging
    if symbol == pyglet.window.key.RIGHT:
        agent.move()
        update_agent_visuals()

@window.event
def on_resize(width, height):
    phys_w, phys_h = window.get_framebuffer_size()
    glViewport(0, 0, phys_w, phys_h)
    return False

@window.event
def on_draw():
    window.clear()
    pyglet.gl.glClearColor(15/255, 15/255, 15/255, 1.0)
    cells_batch.draw()
    grid_batch.draw()
    ui_batch.draw()
    labels_batch.draw()

pyglet.app.run()