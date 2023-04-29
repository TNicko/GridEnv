from mrl_grid.custom_envs.grid_env import MultiGridEnv
import mrl_grid.maps as maps
from mrl_grid.nna import NNA

grid_name = "16x20 Room1"
grid_map = maps.THREE_AGENT_MAPS[grid_name]

episodes = 5
n_channels = 4
view_area = 1
traversal_limit_factor = 1
render = True
n_split = 1

env = MultiGridEnv(grid_map, n_channels, view_area, traversal_limit_factor)

nna = NNA(env, 5, n_split, render)
env.test_mode = True
nna.run()

env.close()