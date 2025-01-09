import gymnasium as gym
import random
import numpy as np

# Initialise the environment
env = gym.make("CliffWalking-v0", render_mode="ansi")
espilon = 0.1

def next_action(state:int, q_table: np.ndarray, epsilon: float = 0):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

qtable = np.zeros((env.observation_space.n, env.action_space.n))

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = next_action(observation, qtable, epsilon=epsilon)

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()

print()
