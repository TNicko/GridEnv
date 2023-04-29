import gym.spaces
import numpy as np
from mrl_grid.render import WorldRenderer
from mrl_grid.world import World, Wall, Agent

FPS = 1

REWARD_MAP = {
    'illegal': -0.5,
    'new': 1,
    'move': -0.05,
    'wait': -0.1,
    'collision': -20,
    'goal': 100,
}

class GridEnv(gym.Env):
    def __init__(self, grid_map: list[list[int]], n_channels: int, view_radius: int, traversal_limit_factor: float = None):

        self.test_mode = False # Check if test mode is on
            
        self.grid_map = np.asarray(grid_map)
        self.cols, self.rows = self.grid_map.shape
        self.grid_size = self.cols * self.rows
        self.channels = n_channels

        self.view_radius = view_radius
        self.traversal_limit_factor = traversal_limit_factor
        self.visited_counter = 0 # count number of times agent has visited a cell in a row

        self.world = self._initialise_world()
        self.n_agents = len(self.world.agents)

        self.shared_reward = False
        self.goal_reward_assigned = False

        self.nA = 5 # no of actions (up, down, left, right, wait)
        self.action_space = gym.spaces.MultiDiscrete([self.nA] * self.n_agents)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(self.n_agents, 2 * self.view_radius + 1, 2 * self.view_radius + 1, self.channels),
            dtype=np.float32
        )

        # Rendering
        self.window = None
        self.fps = FPS

    def _initialise_world(self):
        world = World(self.rows, self.cols)

        num_agents = 0
        for row in range(self.rows):
            for col in range(self.cols):
                pos = (col, row)

                # Initialise agents in grid
                if self.grid_map[col, row] == 1:
                    agent_id = num_agents
                    agent = Agent(agent_id, pos)
                    agent.collided = False
                    world.agents.append(agent)
                    num_agents += 1
                    world.cells += 1

                    # mark cell on grid as visited
                    world.cell_visited(pos, agent)

                # Initialise walls in grid
                if self.grid_map[col, row] == 2:
                    wall = Wall(pos)
                    world.walls.append(wall)
                
                if self.grid_map[col, row] == 0:
                    world.cells += 1

        return world
    
    def _get_obs(self, agent):
        # Get the agent's current position
        x, y = agent.pos

        # Initialize the observation array with the agent's view size
        obs = np.zeros((2 * self.view_radius + 1, 2 * self.view_radius + 1, self.channels))

        # Iterate over the agent's view area
        for i in range(-self.view_radius, self.view_radius + 1):
            for j in range(-self.view_radius, self.view_radius + 1):
                # Calculate the world coordinates for the current cell in the agent's view
                world_x, world_y = x + i, y + j

                # Check if the current cell is outside the world's bounds
                if not (0 <= world_y < self.world.rows and 0 <= world_x < self.world.cols):
                    # Set the corresponding observation value to -1 for cells outside the world
                    obs[self.view_radius + i, self.view_radius + j, :] = -1
                    continue

                # Check if the current cell is visited by any agent
                if (world_x, world_y) in [cell.pos for cell in self.world.seen_cells]:
                    # Set the corresponding observation value to 1 in channel 1 (Visited cells)
                    obs[self.view_radius + i, self.view_radius + j, 1] = 1

                # Check if the current cell contains a wall
                if (world_x, world_y) in [wall.pos for wall in self.world.walls]:
                    # Set the corresponding observation value to 1 in the new channel (channel 3)
                    obs[self.view_radius + i, self.view_radius + j, 3] = 1

                # Iterate over all agents in the world
                for a in self.world.agents:
                    # Check if an agent is in the current cell
                    if a.pos == (world_x, world_y):
                        # If it's the current agent, set the observation value to 1 in channel 0
                        if a == agent:
                            obs[self.view_radius + i, self.view_radius + j, 0] = 1
                        # If it's another agent, set the observation value to 1 in channel 2
                        else:
                            obs[self.view_radius + i, self.view_radius + j, 2] = 1

        return obs
    
    def action_conversion(self, action_n):
        action = []
        for i in range(self.n_agents):
            action.append(action_n[i])
        return action

    def get_centralized_state(self, state):
        return np.concatenate(state, axis=1)
    
    def get_centralized_reward(self, reward_n):
        return np.sum(reward_n)
    
    def reward_conversion(self, reward):
        reward_n = [reward] * self.n_agents
        return reward_n
            
    def step(self, action_n):
        assert len(action_n) == len(self.world.agents)
        action = self.action_conversion(action_n)
        done = False
        # set action for each agent
        reward_n = []
        for i, agent in enumerate(self.world.agents):
            agent.steps_taken += 1
            new_pos = agent.get_new_pos(action[i])
            reward, updated_pos, done = self._get_reward(agent, new_pos, done)
            reward_n.append(reward)
            self.world.add_trail(agent.pos, updated_pos, agent)
            agent.pos = updated_pos

        # all agents get total reward in cooperative case
        reward = self.get_centralized_reward(reward_n)
        if self.shared_reward:
            reward_n = self.reward_conversion(reward)

        # Check if traversal limit has been reached
        if self.traversal_limit_factor and self.traversal_limit_factor != 0:
            traversal_limit = round(self.traversal_limit_factor * self.world.cells)
            if self.visited_counter >= traversal_limit:
                done = True

        # Compute next observations for each agent
        state_n = []
        for agent in self.world.agents:
            state_n.append(self._get_obs(agent))
        state = np.stack(state_n, axis=0)
        info = self._get_info()
        return state, reward, done, info
    
    def _count_local_unexplored_cells(self, pos):
        x, y = pos
        unexplored_count = 0

        for i in range(-self.view_radius, self.view_radius + 1):
            for j in range(-self.view_radius, self.view_radius + 1):
                world_x, world_y = x + i, y + j
                if not self.world.is_cell_visited((world_x, world_y)):
                    unexplored_count += 1

        return unexplored_count
    
    def _adjacent_seen_cell(self, pos, old_pos):
        x, y = pos
        adjacent_cells = [
            (x-1, y),
            (x+1, y),
            (x, y-1),
            (x, y+1)
        ]

        # Remove old agent pos from list
        adjacent_cells = [cell for cell in adjacent_cells if cell != old_pos]

        for cell_pos in adjacent_cells:
            cell = self.world.get_cell(cell_pos)
            if self.world.is_cell_visited(cell_pos) or isinstance(cell, Wall):
                return True
        return False

    def _get_reward(self, agent, new_pos, done):
        x, y = new_pos
        reward = 0
        updated_pos = new_pos

        # Illegal move outside of grid boundary
        if not (0 <= x < self.world.cols and 0 <= y < self.world.rows):
            reward += REWARD_MAP['illegal']
            updated_pos = agent.pos
        else:
            # Check if the new position is occupied by another agent or an obstacle
            other_agent = self.world.check_agent(new_pos)
            if other_agent:
                agent.collided = True
                other_agent.collided = True
                reward += REWARD_MAP['collision']
                updated_pos = agent.pos
                if not self.test_mode:
                    done = True
            elif self.world.check_wall(new_pos):
                agent.collided = True
                reward += REWARD_MAP['collision']
                updated_pos = agent.pos
                if not self.test_mode:
                    done = True
            else:
                # moved to new grid cell
                if not self.world.is_cell_visited(new_pos):
                    reward += REWARD_MAP['new']
                    self.world.cell_visited(new_pos, agent)
                    self.visited_counter = 0

                    # All of the grid explored
                    if self.world.all_cells_visited():
                        if not self.goal_reward_assigned:  # Check if the goal reward is already assigned
                            reward += REWARD_MAP['goal']
                            self.goal_reward_assigned = True  # Mark the goal reward as assigned
                        done = True

                    # --- UPDATE 1 --- #

                    # Calculate the number of unexplored cells around the current position and the new position
                    # current_unexplored_count = self._count_local_unexplored_cells(agent.pos)
                    # new_unexplored_count = self._count_local_unexplored_cells(new_pos)
                    
                    # # Encourage the agent to move towards areas with more unexplored cells
                    # if new_unexplored_count > current_unexplored_count:
                    #     reward += 0.5
                    # elif new_unexplored_count < current_unexplored_count:
                    #     reward -= 0.5

                    # --- UPDATE 3 --- #
                    # Reward for moving to an empty cell next to a seen cell, wall, or boundary 
                    if self._adjacent_seen_cell(new_pos, agent.pos):
                        reward += 1 

                # moved to seen grid cell
                else:
                    self.visited_counter += 1

                    # --- UPDATE 2 --- #
                    # seen_cell = self.world.get_cell(new_pos)
                    # seen_cell.seen_counter += 1
                    # reward -= 0.5 * seen_cell.seen_counter

                # movement cost
                if new_pos != agent.pos:
                    reward += REWARD_MAP['move']

                # wait cost
                if new_pos == agent.pos:
                    reward += REWARD_MAP['wait']
            
        
        return reward, updated_pos, done
    
    def _get_info(self):
        overall_coverage, individual_coverage = self.world.get_coverage()
        steps_taken = self.world.get_steps_taken()

        info = {
            "total_coverage": overall_coverage,
            "collision":  any(agent.collided for agent in self.world.agents),
            "traversal_limit_reached": self.visited_counter >= round(self.traversal_limit_factor * self.world.cells),
            "agents": []
        }

        for agent_id in individual_coverage:
            agent_info = {
                "name": f"Agent {agent_id}",
                "coverage": individual_coverage[agent_id],
                "steps_taken": steps_taken[agent_id]
            }
            info["agents"].append(agent_info)

        return info
        
    def reset(self):
        self.window = None
        self.world = self._initialise_world()
        self.visited_counter = 0
        initial_states = []
        for agent in self.world.agents:
            obs = self._get_obs(agent)
            initial_states.append(obs)
        
        initial_state = np.stack(initial_states, axis=0)
        return initial_state

    def render(self, mode='human', episode=None):
        if mode == "human":
            self.render_gui()
        if mode == "image":
            self.render_image(episode)

    def render_gui(self):
        if self.window == None:
            self.window = WorldRenderer("Grid world", self.world, fps=self.fps)
            self.window.show() 

        filters = []
        self.window.render(filters=filters)

    def render_image(self, episode):
        if self.window == None:
            self.window = WorldRenderer("Grid world", self.world, fps=self.fps)
        self.window.render_image(episode)

    def close(self):
        if self.window:
            self.window.close()
        return
