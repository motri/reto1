import os
import optuna
import json
from optuna.samplers import TPESampler
from deustorl.doublesarsa import DoubleSarsa
from deustorl.common import EpsilonGreedyPolicy, evaluate_policy, max_policy
from deustorl.q_sarsa import Q_SARSA
from deustorl.sarsa import Sarsa
from deustorl.expected_sarsa import ExpectedSarsa
from deustorl.qlearning import QLearning

class OptunaOptimizer:
    def __init__(self, env, study_name, storage_path="sqlite:///optuna/optuna.db", seed=3):
        # Set up environment, study name, and storage path
        self.env = env
        self.study_name = study_name
        self.storage_path = storage_path
        self.seed = seed
        self.study = self._create_study()

    def _create_study(self):
        # Create Optuna study with TPE sampler for hyperparameter optimization
        tpe_sampler = TPESampler(seed=self.seed)  # For reproducibility
        study = optuna.create_study(
            sampler=tpe_sampler,
            direction="maximize",
            study_name=self.study_name,
            storage=self.storage_path,
            load_if_exists=True
        )
        return study

    def objective(self, trial):
        # Define hyperparameter search space
        algo_name = self.study_name
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        lr_decay = trial.suggest_float("learning_rate_decay", 0.9, 1.0, step=0.01)
        lr_episodes_decay = trial.suggest_categorical("lr_episodes_decay", [100, 1_000, 10_000])
        discount_rate = trial.suggest_float("discount_rate", 0.8, 1.0, step=0.05)
        epsilon = trial.suggest_float("epsilon", 0.0, 0.4, step=0.05)
        
        n_steps = 200_000  # Fixed number of steps for learning

        # Instantiate the algorithm based on algo_name
        if algo_name == "q_sarsa":
            algo = Q_SARSA(self.env)
        elif algo_name == "sarsa":
            algo = Sarsa(self.env)
        elif algo_name == "esarsa":         
            algo = ExpectedSarsa(self.env)
        elif algo_name == "qlearning":          
            algo = QLearning(self.env)
        elif algo_name == "doublesarsa":
            algo = DoubleSarsa(self.env)

        # Epsilon-greedy policy for exploration-exploitation balance
        epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
        algo.learn(
            epsilon_greedy_policy,
            n_steps=n_steps,
            discount_rate=discount_rate,
            lr=lr,
            lrdecay=lr_decay,
            n_episodes_decay=lr_episodes_decay
        )

        # Evaluate policy performance and return average reward
        avg_reward, avg_steps = evaluate_policy(
            algo.env, 
            algo.q_table2 if algo_name == "q_sarsa" or algo_name == "doublesarsa" else algo.q_table, 
            max_policy, 
            n_episodes=100
        )
        return avg_reward

    def optimize(self, n_trials=10):
        # Optimize hyperparameters over n_trials and save results
        print(f"Searching for the best hyperparameters in {n_trials} trials...")
        self.study.optimize(self.objective, n_trials=n_trials)

        # Save best trial parameters to JSON file
        best_trial = self.study.best_trial
        best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
        study_dir = f"../experimentos/optuna/{self.study_name}"
        os.makedirs(study_dir, exist_ok=True)
        with open(f"{study_dir}/best_trial.json", "w") as f:
            f.write(best_trial_params)

        # Generate Optuna visualizations and save them
        self._generate_visualizations(study_dir)

    def _generate_visualizations(self, study_dir):
        # Generate and save Optuna visualizations
        fig = optuna.visualization.plot_optimization_history(self.study)
        fig.write_html(f"{study_dir}/optimization_history.html")
        
        fig = optuna.visualization.plot_contour(self.study)
        fig.write_html(f"{study_dir}/contour.html")
        
        fig = optuna.visualization.plot_slice(self.study)
        fig.write_html(f"{study_dir}/slice.html")
        
        fig = optuna.visualization.plot_param_importances(self.study)
        fig.write_html(f"{study_dir}/param_importances.html")
