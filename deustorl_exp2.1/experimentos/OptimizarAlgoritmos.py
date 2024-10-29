import os
import gymnasium as gym
import random
from deustorl.optimizador_optuna import OptunaOptimizer 


if __name__ == "__main__":
    # Creamos el entorno frozen lake con tamaño 8x8 y slippery
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, desc=None, map_name="8x8", is_slippery=True)
    seed = 3
    random.seed(seed)
    env.reset(seed=seed)

    # Creamos los directorios para resultados
    os.system("rm -rf ./logs/")
    os.system("mkdir -p ./optuna/")

    # List of algorithms to test
    algorithms = ["q_sarsa", "sarsa", "esarsa", "qlearning","doublesarsa"]

    # Run optimization for each algorithm
    for algo in algorithms:
        optimizer = OptunaOptimizer(env, study_name=f"{algo}")
        optimizer.optimize()

    # Close environment after optimization
    env.close()
