import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random
from deustorl.optimizador_optuna import OptunaOptimizer 


if __name__ == "__main__":
    # Creamos el entorno frozen lake con tamaño 16x16 y slippery
    env_name = "FrozenLake-v1"
    env = gym.make(env_name, desc=generate_random_map(size=9), is_slippery=True)
    seed = 3
    random.seed(seed)
    env.reset(seed=seed)

    # Creamos los directorios para resultados
    os.system("rm -rf ./logs/")
    os.system("mkdir -p ./optuna/")

    # Lista de algoritmo para ejecutar
    algorithms = ["q_sarsa", "sarsa", "esarsa", "qlearning","doublesarsa"]

    # Ejecutamos la optimizacion para cada algoritmo
    for algo in algorithms:
        optimizer = OptunaOptimizer(env, study_name=f"{algo}")
        optimizer.optimize()

   
    env.close()
