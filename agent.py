import torch
import torch.optim as optim
import numpy as np
import random
from model import DQN
import torchinfo

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_dim, output_dim, learning_rate=1e-3, device=torch.device("cpu"), buffer_capacity=10000,
                 batch_size=64):
        self.device = device
        self.q_network = DQN(input_dim, output_dim).to(self.device)

        # debug the following line
        torchinfo.summary(self.q_network, input_size=(input_dim,))

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()

        # Initialize Replay Buffer
        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size

    def choose_action(self, observation, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0, 1])
        else:
            observation_tensor = torch.tensor(list(observation.values()), dtype=torch.float32).to(self.device)
            q_values = self.q_network(observation_tensor)
            return torch.argmax(q_values).item()

    def store(self, observation, action, reward, next_observation):
        observation = torch.tensor(list(observation.values())).float().to(self.device)
        next_observation = torch.tensor(list(next_observation.values())).float().to(self.device)
        self.buffer.push(observation, action, reward, next_observation)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        actions = torch.tensor(actions).long().to(self.device)

        # Compute current q-values
        q_values = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute next q-values
        with torch.no_grad():
            q_values_next = self.q_network(next_states).max(1)[0]

        # Calculate expected Q value
        expected_q_values = rewards + 0.99 * q_values_next

        # Compute the loss
        loss = self.criterion(q_values, expected_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()