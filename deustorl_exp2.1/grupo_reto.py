import gymnasium as gym
from deustorl.common import EpsilonGreedyPolicy
from deustorl.sarsaOptuna import SarsaOptuna, optimize_sarsa
from deustorl.expectedsarsaOptuna import ExpectedSarsaOptuna, optimize_expected_sarsa
from deustorl.dublesarsaOptuna import DoubleSarsaOptuna, optimize_double_sarsa
from deustorl.qlearningOptuna import QLearningOptuna, optimize_qlearning


if __name__ == "__main__":
    
    # Optimizar SARSA
    #print("Optimizando Sarsa...")
    #best_params = optimize_sarsa(n_trials=100)
    #print("Optimización de Sarsa completada")

    # Ejecutar SARSA con los mejores parámetros
    #print("Ejecutando Sarsa con los mejores parámetros...")
    #env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    #policy = EpsilonGreedyPolicy(best_params['epsilon'])
    #agent = SarsaOptuna(env)
    #agent.learn(
        #policy,
        #n_steps=100000,
        #discount_rate=best_params['discount_rate'],
        #lr=best_params['lr'],
        #lrdecay=best_params['lrdecay'],
        #n_episodes_decay=best_params['n_episodes_decay'],
        #tb_episode_period=100,
        #verbose=False
    #)
    #print("Ejecución de Sarsa con parámetros optimizados completada")

        
    # Optimizar EXPECTED SARSA
    #print("Optimizando Expected Sarsa...")
    #best_params = optimize_expected_sarsa(n_trials=100)
    #print("Optimización de Expected Sarsa completada")

    # Ejecutar EXPECTED SARSA con los mejores parámetros
    #print("Ejecutando Expected Sarsa con los mejores parámetros...")
    #env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    #policy = EpsilonGreedyPolicy(best_params['epsilon'])
    #agent = ExpectedSarsaOptuna(env)
    #agent.learn(
        #policy,
        #n_steps=100000,
        #discount_rate=best_params['discount_rate'],
        #lr=best_params['lr'],
        #lrdecay=best_params['lrdecay'],
        #n_episodes_decay=best_params['n_episodes_decay'],
        #tb_episode_period=100,
        #verbose=False
    #)
    #print("Ejecución de Expected Sarsa con parámetros optimizados completada")

    # Optimizar DOUBLE SARSA
    #print("Optimizando Double Sarsa...")
    #best_params = optimize_double_sarsa(n_trials=100)
    #print("Optimización de Double Sarsa completada")

    # Ejecutar DOUBLE SARSA con los mejores parámetros
    #print("Ejecutando Double Sarsa con los mejores parámetros...")
    #env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    #policy = EpsilonGreedyPolicy(best_params['epsilon'])
    #agent = DoubleSarsaOptuna(env)
    #agent.learn(
        #policy,
        #n_steps=100000,
        #discount_rate=best_params['discount_rate'],
        #lr=best_params['lr'],
        #lrdecay=best_params['lrdecay'],
        #n_episodes_decay=best_params['n_episodes_decay'],
        #tb_episode_period=100,
        #verbose=False
    #)
    #print("Ejecución de Double Sarsa con parámetros optimizados completada")

    # Optimizar Q LEARNING
    print("Optimizando Q Learning...")
    best_params = optimize_qlearning(n_trials=100)
    print("Optimización de Q Learning completada")

    # Ejecutar Q LEARNING con los mejores parámetros
    print("Ejecutando Q Learning con los mejores parámetros...")
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    policy = EpsilonGreedyPolicy(best_params['epsilon'])
    agent = QLearningOptuna(env)
    agent.learn(
        policy,
        n_steps=100000,
        discount_rate=best_params['discount_rate'],
        lr=best_params['lr'],
        lrdecay=best_params['lrdecay'],
        n_episodes_decay=best_params['n_episodes_decay'],
        tb_episode_period=100,
        verbose=False
    )
    print("Ejecución de Q Learning con parámetros optimizados completada")