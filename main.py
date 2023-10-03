from agent import DQNAgent
from train import train
from environment import TradingEnvironment

import pandas as pd
import os

if __name__ == "__main__":
    stock_data = pd.read_csv(os.path.join("data", "processed_train_data.csv"), index_col=0, parse_dates=True)
    env = TradingEnvironment(stock_data)

    input_dim = len(env.observation_space())
    print("input_dim", input_dim)
    output_dim = len(env.action_space())
    print("output_dim", output_dim)

    agent = DQNAgent(input_dim, output_dim)
    train(agent, env, episodes=1000)