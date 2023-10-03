# StockRL - An AI Based RL Trading Agent Repository

This repository consists of the implementation of a deep reinforcement learning (DRL) agent for stock trading. The agent interacts with a trading environment, makes decisions on where to invest, and learns from its actions to improve its performance.

## File Descriptions

### `environment.py`

The `environment.py` script defines the `TradingEnvironment` class which simulates a trading scenario for the agent. This class is an adaptation of the OpenAI Gym's environment class and provides the framework where the agent can interact, perform actions, and receive feedback.

- **`reset()`**: Resets the environment to its initial state and returns the initial state.
- **`step(action)`**: Executes the specified action and returns the next state, reward, the completion status, and logging details.
- **`action_space()`**: Returns the action space of the environment, which comprises two actions: investing in savings or in stocks.
- **`observation_space()`**: Offers the current state of the environment, detailing features like the quarter of the year, adjusted closing prices, and their moving averages.

### `agent.py`

The `agent.py` script introduces both the `ReplayBuffer` and the `DQNAgent` classes:

- **`ReplayBuffer`**: It's a data structure used for storing past experiences, allowing the agent to learn from previous actions.
- **`DQNAgent`**: Represents the DRL agent. This agent uses a deep Q-network to evaluate actions, choose them based on an epsilon-greedy policy, store experiences in the replay buffer, and learn by updating its Q-values from batches of experiences.

### `train.py`

In `train.py`, the main training loop for the agent is established. The `train()` function orchestrates the agent's interactions with the environment, storage of experiences, and learning from these experiences. The function also manages the agent's exploration vs. exploitation balance through the epsilon decay process.

### `main.py`

The `main.py` script serves as the entry point for running the entire simulation. This script:
1. Sets the device preference (CPU or CUDA).
2. Loads the stock data.
3. Initializes the trading environment.
4. Establishes the DQNAgent with the appropriate input and output dimensions.
5. Triggers the training process for the agent.

## Additional Files

- `model.py`: This script holds the neural network architecture used for evaluating Q-values in the DRL agent.
- `data.py`: Responsible for preprocessing and managing data-related tasks.
- `retrieve_savings_data.py`: A script to fetch and preprocess the savings rate data.
- `retrieve_stock_data.py`: A script to source and preprocess the stock data.

## Usage

To run the agent's training process, simply execute the `main.py` script. Make sure all the required dependencies are installed, and the necessary data files are available in the data directory.

---

For more detailed insights or specific instructions, delve into the comments within each individual script.
