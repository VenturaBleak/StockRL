"""
This file contains the training loop for the agent.

The train() function implements the following steps:
- Reset the environment at the start of each episode.
- Choose an action using the agent's epsilon-greedy policy.
- Perform the chosen action and observe the next state, reward, done, and log_df.
- Store the experience in the replay buffer.
- Learn from a batch of experiences.
- Repeat until the episode is done and update epsilon.
"""

def train(agent, env, episodes=1000):
    epsilon = 1.0
    for episode in range(episodes):
        state = env.reset()  # Resetting the environment at the start of each episode
        observation = env.observation_space()  # Fetch the observation
        done = False
        while not done:
            action = agent.choose_action(observation, epsilon)
            next_state, reward, done, log_dict = env.step(action)
            next_observation = env.observation_space()

            # Store experience in the replay buffer
            agent.store(observation, action, reward, next_observation)

            # Learn from a batch of experiences
            agent.learn()

            observation = next_observation
        epsilon *= 0.995
        if episode % 1 == 0:
            formatted_log = {
                "Steps Elapsed": log_dict["Steps Elapsed"],
                'Model Updates': agent.total_updates,
                'LR': agent.get_learning_rate(),
                'Action Distribution': env.total_actions,
                'Epsilon': f"{epsilon:.4f}",
                'Buffer Size': len(agent.buffer),
                "Portf Value": f"{log_dict['Portfolio Value']:.2f}",
                "Savings Investment": f"{log_dict['Savings Investment']:.2f}",
                "Stock Investment": f"{log_dict['Stock Investment']:.2f}",
                "IRR": f"{log_dict['IRR']*100:.2f}%"
            }
            print(f"Episode {episode} completed.", formatted_log)