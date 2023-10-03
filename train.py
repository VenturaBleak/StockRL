"""
This file contains the training loop for the agent.

The train() function implements the following steps:
- Initialize the agent.
- Initialize the environment.
- Loop over episodes.
- Loop over steps.
- Choose an action.
- Perform the action.
- Update the agent.
- Update the state.
- Update the episode.
"""

def train(agent, env, episodes=1000):
    epsilon = 1.0
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            print("next_state", next_state)
            agent.learn(state, action, reward, next_state)
            state = next_state
        epsilon *= 0.995  # decay epsilon, so that the agent chooses less random actions over time
        if episode % 100 == 0:
            print(f"Episode {episode} completed.")