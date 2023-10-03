import torch
import torch.optim as optim
import numpy as np
from model import DQN

class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=1e-3):
        # Define the agent's network: DQN(n_observations, n_actions)
        self.q_network = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def choose_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def learn(self, state, action, reward, next_state):
        state_values = list(state.values())  # Convert dict to list
        print("state_values", state_values)
        next_state_values = list(next_state.values())  # Convert dict to list

        state_tensor = torch.tensor(state_values, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state_values, dtype=torch.float32)

        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.int64)

        q_values = self.q_network(state_tensor)
        with torch.no_grad():
            q_values_next = self.q_network(next_state_tensor)
        q_value = q_values[action_tensor]
        q_value_next = torch.max(q_values_next).item()
        # 0.99 is the discount factor, that is, how much we care about future rewards
        expected_q_value = reward_tensor + 0.99 * q_value_next

        loss = self.criterion(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()