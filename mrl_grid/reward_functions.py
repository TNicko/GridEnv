# Description: Individual Rewards functions for the grid world 
# environment to be summmed together in the environments main reward function.

from mrl_grid.world import Wall

REWARD_MAP = {
    'illegal': -0.5,
    'new': 1,
    'move': -0.05,
    'wait': -0.1,
    'collision': -20,
    'goal': 100,
}

def get_illegal_move_reward(new_pos, cols, rows):
    """Returns reward for moving to an illegal position as well as a done flag if in test mode"""
    x, y = new_pos
    if not (0 <= x < cols and 0 <= y < rows):
        return REWARD_MAP['illegal'], True
    return 0, False

def get_collision_reward(agent, new_pos, done, world, test_mode):
    """Returns reward for colliding with another agent or wall as well as a done flag if in test mode"""
    other_agent = world.check_agent(new_pos)
    if other_agent or world.check_wall(new_pos):
        agent.collided = True
        if other_agent:
            other_agent.collided = True
        if not test_mode:
            done = True
        return REWARD_MAP['collision'], True, done
    return 0, False, done

def get_new_cell_reward(agent, new_pos, done, world, visited_counter, goal_reward_assigned):
    """Returns reward for visiting a cell that has not been seen before as well as a done flag if all cells have been visited"""
    reward = 0
    if not world.is_cell_visited(new_pos):
        reward += REWARD_MAP['new']
        world.cell_visited(new_pos, agent)
        visited_counter = 0

        if world.all_cells_visited():
            if not goal_reward_assigned:
                reward += REWARD_MAP['goal']
                goal_reward_assigned = True
            done = True
    return reward, done, visited_counter, goal_reward_assigned

def get_seen_cell_reward(agent, new_pos, world, visited_counter):
    """Returns reward for visiting a cell that has been seen before"""
    reward = 0
    if world.is_cell_visited(new_pos):
        visited_counter += 1
    return reward, visited_counter

def get_movement_cost(agent, new_pos):
    return REWARD_MAP['move'] if new_pos != agent.pos else 0

def get_wait_cost(agent, new_pos):
    return REWARD_MAP['wait'] if new_pos == agent.pos else 0

def get_exploration_reward(agent, new_pos, world, view_radius):
    """Returns reward for moving to location that has more unexplored cells in the local area"""
    current_unexplored_count = count_local_unexplored_cells(agent.pos, world, view_radius)
    new_unexplored_count = count_local_unexplored_cells(new_pos, world, view_radius)

    if new_unexplored_count > current_unexplored_count:
        return 0.5
    elif new_unexplored_count < current_unexplored_count:
        return -0.5
    return 0

def get_revisit_penalty(new_pos, world):
    """Returns incremented penalty for revisiting a cell that has been seen before"""
    seen_cell = world.get_cell(new_pos)
    seen_cell.seen_counter += 1
    return -0.5 * seen_cell.seen_counter

def get_adjacent_seen_cell_reward(agent, new_pos, world):
    """Returns reward for moving to a new cell that is adjacent to a wall, boundary or a visited cell"""
    x, y = new_pos
    adjacent_cells = [
        (x-1, y),
        (x+1, y),
        (x, y-1),
        (x, y+1)
    ]

    # Remove old agent pos from list
    adjacent_cells = [cell for cell in adjacent_cells if cell != agent.pos]

    for cell_pos in adjacent_cells:
        cell = world.get_cell(cell_pos)
        if world.is_cell_visited(cell_pos) or isinstance(cell, Wall):
            return 1
    return 0



# Helper functions
# --------------------------------------------------------------------------------------------

def count_local_unexplored_cells(pos, world, view_radius):
    """Count the number of unexplored cells in the agent's view."""
    x, y = pos
    unexplored_count = 0

    for i in range(-view_radius, view_radius + 1):
        for j in range(-view_radius, view_radius + 1):
            world_x, world_y = x + i, y + j
            if not world.is_cell_visited((world_x, world_y)):
                unexplored_count += 1

    return unexplored_count
    