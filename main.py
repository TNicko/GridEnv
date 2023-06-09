from mrl_grid.custom_envs.grid_env import MultiGridEnv
import mrl_grid.maps as maps
from mrl_grid.random_action_runner import RandomActionRunner

grid_name = "16x20 Room1"
grid_map = maps.THREE_AGENT_MAPS[grid_name]

episodes = 5
view_area = 1
traversal_limit_factor = 1
render = True
n_split = 1

env = MultiGridEnv(grid_map, view_area, traversal_limit_factor)
env.test_mode = True

model = RandomActionRunner(env, episodes, n_split, render)
model.run()

env.close()