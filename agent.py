import torch
import torch.optim as optim
import numpy as np
from model import DQN

class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=1e-3, device=torch.device("cpu")):
        self.device = device

        self.q_network = DQN(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

    def choose_action(self, observation, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1])
        else:
            observation_tensor = torch.tensor(list(observation.values()), dtype=torch.float32).to(self.device)
            q_values = self.q_network(observation_tensor)
            return torch.argmax(q_values).item()

    def learn(self, observation, action, reward, next_observation):
        observation_tensor = torch.tensor(list(observation.values()), dtype=torch.float32).to(self.device)
        next_observation_tensor = torch.tensor(list(next_observation.values()), dtype=torch.float32).to(self.device)

        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.int64).to(self.device)

        q_values = self.q_network(observation_tensor)
        with torch.no_grad():
            q_values_next = self.q_network(next_observation_tensor)
        q_value = q_values[action_tensor]
        q_value_next = torch.max(q_values_next).item()

        # 0.99 is the discount factor, that is, how much we care about future rewards
        expected_q_value = reward_tensor + 0.99 * q_value_next

        loss = self.criterion(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()