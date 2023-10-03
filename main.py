from agent import DQNAgent
from train import train
from environment import TradingEnvironment

import pandas as pd
import os
import torch

if __name__ == "__main__":
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data
    stock_data = pd.read_csv(os.path.join("data", "processed_train_data.csv"), index_col=0, parse_dates=True)

    # create the environment
    env = TradingEnvironment(stock_data, device=device)

    # retrieve the dimensions of the observation and action spaces
    input_dim = len(env.observation_space())
    output_dim = len(env.action_space())
    print(f"Input dimension: {input_dim};", f"Output dimension: {output_dim}")

    # create the agent
    agent = DQNAgent(input_dim, output_dim, device=device)

    # train the agent
    train(agent, env, episodes=1000)