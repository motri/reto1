import os
import gymnasium as gym
import random

from deustorl.common import *
from deustorl.common import max_policy
from deustorl.q_sarsa import Q_SARSA
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning
from deustorl.doublesarsa import DoubleSarsa
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

os.system("rm -rf ./logs/")

env_name = "FrozenLake-v1"
env = gym.make(env_name, desc=generate_random_map(size=8), is_slippery=True)
seed = 3
random.seed(seed)
env.reset(seed=seed)

n_steps = 500_000
epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
                                            
algo = Sarsa(env)
print("Testing SARSA")
algo.learn(epsilon_greedy_policy, n_steps)

algo = QLearning(env)
print("Testing Q-Learning")
algo.learn(epsilon_greedy_policy, n_steps)

algo = ExpectedSarsa(env)
print("Testing Expected SARSA")
algo.learn(epsilon_greedy_policy, n_steps)

algo = Q_SARSA(env)
print("Testing Q_SARSA")
algo.learn(epsilon_greedy_policy, n_steps)

algo = DoubleSarsa(env)
print("Testing DoubleSARSA")
algo.learn(epsilon_greedy_policy, n_steps)
