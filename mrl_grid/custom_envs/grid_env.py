import gym.spaces
import numpy as np
from mrl_grid.render import WorldRenderer
from mrl_grid.world import World, Wall, Agent
from mrl_grid.reward_functions import (get_illegal_move_reward, get_collision_reward, get_new_cell_reward,
                              get_seen_cell_reward, get_movement_cost, get_wait_cost,
                              get_exploration_reward, get_revisit_penalty, get_adjacent_seen_cell_reward)

FPS = 20 # frames per second for rendered environment

class MultiGridEnv(gym.Env):
    """A multi-agent environment class for gridworld navigation task with partial observability."""
    def __init__(self, grid_map: list[list[int]], n_channels: int, view_radius: int, traversal_limit_factor: float = None):
        """
        Parameters:
            grid_map (list[list[int]]): a list of lists containing integers that represent the grid world. The integers
                                        represent different objects on the grid: 0 for empty cells, 1 for agent cells, and 
                                        2 for wall cells.
            n_channels (int): the number of channels in the observation space.
            view_radius (int): the radius of the agent's observation area.
            traversal_limit_factor (float): a factor that determines the maximum number of cells an agent can visit before
                                            the episode terminates. If None, there is no traversal limit.
        """

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
        """Initialize the world object based on the grid map."""
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
        """Get the observation/state for a given agent."""
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
        """Convert the list of actions for each agent into a single list."""
        action = []
        for i in range(self.n_agents):
            action.append(action_n[i])
        return action

    def get_centralized_state(self, state):
        """Convert the list of observations for each agent into a centralized state."""
        return np.concatenate(state, axis=1)
    
    def get_centralized_reward(self, reward_n):
        """Calculate the total reward for all agents in the cooperative setting."""
        return np.sum(reward_n)
    
    def reward_conversion(self, reward):
        """Convert a single reward into a list of rewards for each agent."""
        reward_n = [reward] * self.n_agents
        return reward_n
            
    def step(self, action_n):
        """Take a step in the environment."""
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

    def _get_reward(self, agent, new_pos, done):
        reward = 0
        updated_pos = new_pos

        # Check for illegal moves (outside the grid boundary)
        illegal_reward, illegal_move = get_illegal_move_reward(new_pos, self.world.cols, self.world.rows)
        if illegal_move:
            reward += illegal_reward
            updated_pos = agent.pos
        else:
            # Check for collisions with other agents or walls
            collision_reward, collision_occurred, done = get_collision_reward(agent, new_pos, done, self.world, self.test_mode)
            if collision_occurred:
                reward += collision_reward
                updated_pos = agent.pos
            else:
                # Check for moving to a new cell and update the visited state
                new_cell_reward, done, self.visited_counter, self.goal_reward_assigned = get_new_cell_reward(agent, new_pos, done, self.world, self.visited_counter, self.goal_reward_assigned)
                reward += new_cell_reward

                # Check for moving to a previously seen cell
                seen_cell_reward, self.visited_counter = get_seen_cell_reward(agent, new_pos, self.world, self.visited_counter)
                if self.world.is_cell_visited(new_pos):
                    # Update 2: Penalize revisiting a cell
                    # revisit_penalty = get_revisit_penalty(new_pos)
                    # seen_cell_reward += revisit_penalty
                    pass
                reward += seen_cell_reward

                # Calculate movement cost
                movement_cost = get_movement_cost(agent, new_pos)
                reward += movement_cost

                # Calculate wait cost (if agent remains in the same position)
                wait_cost = get_wait_cost(agent, new_pos)
                reward += wait_cost

                # Update 1: Encourage the agent to move towards areas with more unexplored cells
                # exploration_reward = get_exploration_reward(agent, new_pos, self.world, self.view_radius)
                # reward += exploration_reward

                # Update 3: Reward for moving to an empty cell next to a seen cell, wall, or boundary
                adjacent_cell_reward = get_adjacent_seen_cell_reward(agent, new_pos, self.world)
                reward += adjacent_cell_reward

        return reward, updated_pos, done
    
    def _get_info(self):
        """Return the information about the environment."""
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
        """Render the environment in a GUI window."""
        if self.window == None:
            self.window = WorldRenderer("Grid world", self.world, fps=self.fps)
            self.window.show() 

        self.window.render()

    def render_image(self, episode):
        """Render the environment as an image at the current step."""
        if self.window == None:
            self.window = WorldRenderer("Grid world", self.world, fps=self.fps)
        self.window.render_image(episode)

    def close(self):
        if self.window:
            self.window.close()
        return