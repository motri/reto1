from deustorl.common import *
from deustorl.common import QTable
from deustorl.helpers import TensorboardLogger

class Q_SARSA:
    def __init__(self, env):
        # Inicializamos dos tablas Q: una para Q-Learning y otra para SARSA
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table1 = QTable(n_states, n_actions)  # Tabla Q para Q-Learning
        self.q_table2 = QTable(n_states, n_actions)  # Tabla Q para SARSA
        self.env = env

    def learn(self, policy, n_steps=100, discount_rate=1, lr=0.01, lrdecay=1.0, n_episodes_decay=100, tb_episode_period=100, verbose=False):
        # Inicia el ambiente y selecciona la acción inicial
        obs, _ = self.env.reset()
        alpha = 0.7  # Empezamos con un 70% de peso para SARSA
        selected_action = np.argmax(self.combine_q_values(obs, alpha=alpha))

        # Inicialización del registro en Tensorboard
        tblogger = TensorboardLogger("Q-SARSA (dr=" + str(discount_rate) + "-lr=" + str(lr) + "-lrdecay=" + str(lrdecay) + "e" + str(n_episodes_decay) + ")", episode_period=tb_episode_period)

        n_episodes = 0
        episode_reward = 0
        episode_steps = 0

        # Bucle de aprendizaje principal
        for n in range(n_steps):
            # Guardamos el estado y acción anterior para actualización
            previous_obs = obs
            previous_action = selected_action
            obs, reward, terminated, truncated, _ = self.env.step(selected_action)

            # Acumulamos recompensa y pasos para estadísticas
            episode_reward += reward
            episode_steps += 1

            # Selección de la próxima acción usando la política epsilon-greedy y una combinación ponderada de Q
            combined_q_values = self.combine_q_values(obs, alpha=alpha)
            selected_action = np.argmax(combined_q_values)

            # Actualización de la Tabla Q1 con Q-Learning
            max_qvalue1 = max(self.q_table1[obs])
            td1 = reward + discount_rate * max_qvalue1
            self.q_table1[previous_obs][previous_action] += lr * (td1 - self.q_table1[previous_obs][previous_action])

            # Actualización de la Tabla Q2 con SARSA
            td2 = reward + discount_rate * self.q_table2[obs][selected_action]
            self.q_table2[previous_obs][previous_action] += lr * (td2 - self.q_table2[previous_obs][previous_action])

            # Reinicio al terminar episodio
            if terminated or truncated:
                tblogger.log(episode_reward, episode_steps)
                episode_reward = 0
                episode_steps = 0
                obs, _ = self.env.reset()
                selected_action = np.argmax(self.combine_q_values(obs, alpha=alpha))
                
                # Actualización de la tasa de aprendizaje y alpha por episodios
                n_episodes += 1
                if n_episodes % n_episodes_decay == 0:
                    lr *= lrdecay
                    # Reducir progresivamente alpha, sin bajar de 0.3 para conservar exploración
                    alpha = max(0.3, alpha - 0.05)

    def combine_q_values(self, state, alpha=0.5):
        # Combinación ponderada de ambas tablas Q, usando alpha como peso
        return alpha * self.q_table1[state] + (1 - alpha) * self.q_table2[state]
