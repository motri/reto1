from deustorl.common import *
from deustorl.helpers import TensorboardLogger
import numpy as np

class DoubleSarsa:
    def __init__(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table1 = QTable(n_states, n_actions)
        self.q_table2 = QTable(n_states, n_actions)
        self.env = env

    def learn(self, policy, n_steps:int=100, discount_rate=1, lr=0.01, lrdecay=1.0, n_episodes_decay=100, tb_episode_period=100, verbose=False):
        obs, _ = self.env.reset()
        selected_action = policy(self.get_combined_q_values(obs))

        tblogger = TensorboardLogger("DoubleSARSA_(dr=" + str(discount_rate) +"-lr=" + str(lr) + "-lrdecay=" + str(lrdecay) + "e"+ str(n_episodes_decay) + ")" , episode_period=tb_episode_period)

        n_episodes = 0
        episode_reward = 0
        episode_steps = 0

        for n in range(n_steps):
            previous_obs = obs
            previous_action = selected_action
            obs, reward, terminated, truncated, _ = self.env.step(selected_action)

            episode_reward += reward
            episode_steps += 1

            if verbose:
                self.env.render()

            selected_action = policy(self.get_combined_q_values(obs))

            # Randomly choose which Q-table to update
            if np.random.random() < 0.5:
                active_q = self.q_table1
                target_q = self.q_table2
            else:
                active_q = self.q_table2
                target_q = self.q_table1

            td = reward + discount_rate * target_q[obs][selected_action]

            active_q[previous_obs][previous_action] += lr * (td - active_q[previous_obs][previous_action])

            if terminated or truncated:
                tblogger.log(episode_reward, episode_steps)
                if verbose:
                    print(self.get_combined_q_values())
                # reset the environment and reinitialize trajectory
                if verbose:
                    print("--- EPISODE STARTS ---")
                episode_reward = 0
                episode_steps = 0
                obs, _ = self.env.reset()
                selected_action = policy(self.get_combined_q_values(obs))
                
                # lrdecay update
                n_episodes += 1
                if n_episodes % n_episodes_decay == 0:
                    lr *= lrdecay

    def get_combined_q_values(self, state=None):
        if state is None:
            return (self.q_table1 + self.q_table2) / 2
        return (self.q_table1[state] + self.q_table2[state]) / 2
