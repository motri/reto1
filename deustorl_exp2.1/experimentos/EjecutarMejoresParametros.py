import gymnasium as gym
import random
from deustorl.optimizador_optuna import OptunaOptimizer 
import json
import os
from deustorl.common import EpsilonGreedyPolicy, evaluate_policy
from deustorl.q_sarsa import Q_SARSA
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning
from deustorl.doublesarsa import DoubleSarsa

class LearnConParams:
    def __init__(self, env, algo):
        self.env = env
        self.algo_name = algo
        self.params = self._load_params()

    def _load_params(self):
        # Path to the best trial parameters JSON file
        params_path = f"./optuna/{self.algo_name}/best_trial.json"
        
        # Check if the file exists
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"Parameter file for {self.algo_name} not found at {params_path}")
        
        # Load parameters from the JSON file
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        return params

    def _instantiate_algo(self):
        # Instantiate the correct algorithm based on algo_name
        if self.algo_name == "q_sarsa":
            algo = Q_SARSA(self.env)
        elif self.algo_name == "sarsa":
            algo = Sarsa(self.env)
        elif self.algo_name == "esarsa":
            algo = ExpectedSarsa(self.env)
        elif self.algo_name == "qlearning":
            algo = QLearning(self.env)
        elif self.algo_name == "doublesarsa":
            algo = DoubleSarsa(self.env)
        else:
            raise ValueError(f"Unknown algorithm: {self.algo_name}")
        
        return algo

    def execute_learning(self):
        # Get algorithm instance
        algo = self._instantiate_algo()
        
        # Extract parameters from the loaded JSON
        discount_rate = self.params.get("discount_rate", 1.0)
        lr = self.params.get("learning_rate", 0.01)
        lr_decay = self.params.get("learning_rate_decay", 1.0)
        lr_episodes_decay = self.params.get("lr_episodes_decay", 100)
        epsilon = self.params.get("epsilon", 0.1)
        n_steps = 200_000  # Set a default value or load from params if desired

        # Define the epsilon-greedy policy
        epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)

        # Run the learning process
        algo.learn(
            policy=epsilon_greedy_policy,
            n_steps=n_steps,
            discount_rate=discount_rate,
            lr=lr,
            lrdecay=lr_decay,
            n_episodes_decay=lr_episodes_decay
        )

        # Evaluate the learned policy
        avg_reward, avg_steps = evaluate_policy(
            algo.env, 
            algo.q_table1 if self.algo_name == "q_sarsa" or self.algo_name == "doublesarsa" else algo.q_table, 
            max_policy, 
            n_episodes=100)
        
        # Return evaluation metrics for reporting or further analysis
        return avg_reward, avg_steps

if __name__ == "__main__":
    # Creamos el entorno frozen lake con tamaño 8x8 y slippery
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, desc=None, map_name="8x8", is_slippery=True)
    seed = 3
    random.seed(seed)
    env.reset(seed=seed)

    # Creamos los directorios para resultados
    os.system("rm -rf ./mejores_params/logs/")


    # List of algorithms to test
    algorithms = ["q_sarsa", "sarsa", "esarsa", "qlearning","doublesarsa"]

    # Run optimization for each algorithm
    for algo in algorithms:
        learnconparams = LearnConParams(env,algo)
        learnconparams.execute_learning()

    # Close environment after optimization
    env.close()