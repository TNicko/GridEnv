import numpy as np

class Env:
    def __init__(self, env, episodes, n_split, render):
        self.env = env
        self.episodes = episodes
        self.n_split = n_split
        self.render = render

    def print_episode(self, n, reward, info, epsilon=None):

        data_string = "Episode: " + str(n).rjust(3) + "\n"
    
        # Print Info for current episode. Each key in info dict should be on new line
        for key, value in info.items():
            if key == 'agents':
                for i, agent in enumerate(value):
                    data_string += " | " + agent['name'] + f": Reward = {round(reward, 2)}" + f"| Coverage = {agent['coverage']}%" + f"| Steps Taken = {agent['steps_taken']}"+ "\n"
                continue

            data_string += " | " + key + ": " + str(value).rjust(4) + "\n"

        if epsilon != None:
            data_string += " | epsilon: " + str(epsilon).rjust(4)

        print(data_string)