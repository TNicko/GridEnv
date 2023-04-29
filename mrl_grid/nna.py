from mrl_grid.env import Env

class NNA(Env):
    def __init__(self, env, episodes, n_split, render):
        super().__init__(env, episodes, n_split, render)

    def run(self):
        for n in range(self.episodes):
            episode_reward = 0
            done = False
            states = self.env.reset()
            
            while not done:
                if self.render: self.env.render() # Render grid 
                
                actions = self.select_actions()
                next_states, reward, done, info = self.env.step(actions) # Take step in env
                states = next_states
                episode_reward += reward

            if n % self.n_split == 0 or n == self.episodes-1:

                self.print_episode(n, episode_reward, info)

    def select_actions(self):
        action = self.env.action_space.sample() # Choose random available action
        return action


