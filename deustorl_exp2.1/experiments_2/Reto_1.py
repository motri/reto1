import os
import gymnasium as gym
import random
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json

from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from deustorl.common import *
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning
from deustorl.helpers import DiscretizedObservationWrapper
from deustorl.common import EpsilonGreedyPolicy
from deustorl.q_sarsa import Q_SARSA



# Definición de la función objetivo de Optuna
def objective(trial):
    # Selección del algoritmo y definición de hiperparámetros
    algo_name = trial.suggest_categorical("algo_name", ["q_sarsa", "sarsa", "esarsa", "qlearning"])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.9, 1.0, step=0.01)
    lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay", [100, 1_000, 10_000])
    discount_rate = trial.suggest_float("discount_rate", 0.8, 1.0, step=0.05)
    epsilon = trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
    
    n_steps = 200_000

    # Creación de instancias de los algoritmos según el nombre seleccionado
    if algo_name == "q_sarsa":
        algo = Q_SARSA(env)
    elif algo_name == "sarsa":
        algo = Sarsa(env)
    elif algo_name == "esarsa":
        algo = ExpectedSarsa(env)
    elif algo_name == "qlearning":
        algo = QLearning(env)

    # Política de epsilon-greedy
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    algo.learn(epsilon_greedy_policy, n_steps=n_steps, discount_rate=discount_rate, lr=lr, lrdecay=lr_decay, n_episodes_decay=lr_episodes_decay)

    # Evaluación de la política aprendida
    avg_reward, avg_steps = evaluate_policy(algo.env, algo.q_table1 if algo_name == "q_sarsa" else algo.q_table, max_policy, n_episodes=100)
    
    return avg_reward

if __name__ == "__main__":
    # Configuración del entorno de CartPole y wrapper de discretización
    env_name = "CartPole-v1"
    env = DiscretizedObservationWrapper(gym.make(env_name), n_bins=10)
    seed = 3
    random.seed(seed)
    env.reset(seed=seed)

    # Preparación de directorios de logs
    os.system("rm -rf ./logs/")
    os.system("mkdir -p ./optuna/")

    # Configuración del estudio de Optuna
    storage_file = "sqlite:///optuna/optuna.db"
    study_name = "cartpole"
    full_study_dir_path = f"optuna/{study_name}"
    tpe_sampler = TPESampler(seed=seed)  # Reproducibilidad
    study = optuna.create_study(sampler=tpe_sampler, direction="maximize", study_name=study_name, storage=storage_file, load_if_exists=True)
    n_trials = 10  # Normalmente se usaría 50 o 100

    # Inicio de la optimización de hiperparámetros
    print(f"Searching for the best hyperparameters in {n_trials} trials...")
    study.optimize(objective, n_trials=n_trials)

    # Guardado de los mejores hiperparámetros
    best_trial = study.best_trial
    best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
    os.system(f"mkdir -p {full_study_dir_path}")
    best_trial_file = open(f"{full_study_dir_path}/best_trial.json", "w")
    best_trial_file.write(best_trial_params)
    best_trial_file.close()

    # Generación de gráficos de Optuna
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{full_study_dir_path}/optimization_history.html")
    fig = optuna.visualization.plot_contour(study)
    fig.write_html(f"{full_study_dir_path}/contour.html")
    fig = optuna.visualization.plot_slice(study)
    fig.write_html(f"{full_study_dir_path}/slice.html")
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_html(f"{full_study_dir_path}/param_importances.html")

    env.close()
