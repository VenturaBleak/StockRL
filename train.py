"""
This file contains the training loop for the agent.

The train() function implements the following steps:
- Reset the environment at the start of each episode.
- Choose an action based on the current state.
- Perform the chosen action and observe the next state, reward, and done flag.
- Learn from the experience by updating the agent's network.
- Repeat until the episode is done.
"""

def train(agent, env, episodes=1000):
    epsilon = 1.0
    for episode in range(episodes):
        state = env.reset()  # Resetting the environment at the start of each episode
        observation = env.observation_space()  # Fetch the observation
        done = False
        while not done:
            action = agent.choose_action(observation, epsilon)
            next_state, reward, done, log_df = env.step(action)
            next_observation = env.observation_space()
            agent.learn(observation, action, reward, next_observation)
            observation = next_observation
        epsilon *= 0.995
        if episode % 100 == 0:
            print(f"Episode {episode} completed.", log_df)