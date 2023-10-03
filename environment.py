"""
This file contains the TradingEnvironment class, which is used to create a trading environment for the agent.

The TradingEnvironment class is based on the OpenAI Gym environment class. It implements the following methods:
- reset(): Reset the environment to its initial state and return the initial state.
- step(action): Perform the given action and return the next state, reward, done, and log_df.
- action_space(): Return the action space of the environment.
- observation_space(): Return the observation space of the environment.
"""
import pandas as pd
import os


class TradingEnvironment:
    def __init__(self, stock_data, initial_portfolio_value=10000, look_back_days=30):
        self.stock_data = stock_data
        self.look_back_days = look_back_days
        self.initial_portfolio_value = initial_portfolio_value
        self.portfolio_value = initial_portfolio_value
        self.reset()

    def reset(self):
        self.portfolio_value = self.initial_portfolio_value
        self.current_step = self.look_back_days
        self.stock_quantity = 0
        self.asset_class = 'A'  # Initially invest in asset class A
        self.done = False
        return self.get_state()

    def get_state(self):
        state = self.stock_data.iloc[self.current_step].to_dict()
        state['position'] = 0 if self.stock_quantity == 0 else 1
        state['portfolio_value'] = self.portfolio_value if self.stock_quantity == 0 else self.stock_quantity * state['Adj Close']
        return state

    def get_observation(self):
        observation = self.get_state()
        observation['Adj_Close_MA'] = self.stock_data['Adj Close'].iloc[self.current_step-self.look_back_days:self.current_step].mean()
        observation['Volume_MA'] = self.stock_data['Volume'].iloc[self.current_step-self.look_back_days:self.current_step].mean()
        return observation

    def step(self, action):
        prev_portfolio_value = self.get_state()['portfolio_value']

        # Action 0: Invest in A (Savings)
        if action == 0:
            if self.asset_class != 'A':
                self.portfolio_value += self.stock_quantity * self.stock_data['Adj Close'].values[self.current_step]
                self.stock_quantity = 0
                self.asset_class = 'A'
            savings_rate_annual = self.stock_data['Savings_Rate'].iloc[self.current_step]
            savings_rate_daily = (1 + savings_rate_annual / 100) ** (1 / 252) - 1
            self.portfolio_value += self.portfolio_value * savings_rate_daily

        # Action 1: Invest in B (Stock)
        elif action == 1:
            if self.asset_class != 'B':
                self.portfolio_value += self.stock_quantity * self.stock_data['Adj Close'].values[self.current_step]
                self.stock_quantity = 0
                self.asset_class = 'B'
            self.stock_quantity += self.portfolio_value / self.stock_data['Adj Close'].values[self.current_step]
            self.portfolio_value = 0

        # current_step += 1
        self.current_step += 1

        # This indicates, if we are at the end of the observation period
        if self.current_step == len(self.stock_data) - 1:
            self.done = True

        # Calculate reward
        current_portfolio_value = self.get_state()['portfolio_value']
        reward = current_portfolio_value - prev_portfolio_value

        # Calculate returns, for human interpretabilty
        daily_return = reward / prev_portfolio_value
        if self.current_step >= 5:
            weekly_return = (self.stock_data['Adj Close'].values[self.current_step] - self.stock_data['Adj Close'].values[self.current_step - 5]) / self.stock_data['Adj Close'].values[self.current_step - 5]
        else:
            weekly_return = 0
        if self.current_step >= 252:
            yearly_return = (self.stock_data['Adj Close'].values[self.current_step] - self.stock_data['Adj Close'].values[self.current_step - 252]) / self.stock_data['Adj Close'].values[self.current_step - 252]
        else:
            yearly_return = 0

        # Calculate the financial return, for human interpretabilty
        financial_return = (current_portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value

        # Create log dataframe
        investment_asset_class_A = self.portfolio_value if self.asset_class == 'A' else 0
        investment_asset_class_B = self.stock_quantity * self.stock_data['Adj Close'].values[
            self.current_step] if self.asset_class == 'B' else 0

        log_df = pd.DataFrame({
            'Step': [self.current_step],
            'Savings Investment': [investment_asset_class_A],
            'Stock Investment': [investment_asset_class_B],
            'Daily Return': [daily_return],
            'Weekly Return': [weekly_return],
            'Yearly Return': [yearly_return],
            'Portfolio Value': [current_portfolio_value],
            'Financial Return': [financial_return]
        })

        return self.get_state(), reward, self.done, log_df
        # log the information in a 1 row df: step, asset A investment (savings), asset B investment (ticker), daily return, weekly return, yearly return, portfolio value, financial_return

    def action_space(self):
        """Return the action space of the environment.
        The action space is represented by a list with the following actions:
        - 0: Invest in A
        - 1: Invest in B
        """
        return [0, 1]

    def observation_space(self):
        """Return the observation space of the environment.
        The observation space is represented by a dictionary with the following keys:
        - Quarter: Current quarter of the year.
        - Adj_Close_MA: Moving average of adjusted closing prices for the last n days.
        - Adj Close: Adjusted close price for the current day.
        """
        return {
            'Quarter': self.stock_data['Quarter'].iloc[self.current_step],
            'Adj Close': self.stock_data['Adj Close'].iloc[self.current_step],
            'Adj_Close_MA': self.stock_data['Adj Close'].iloc[
                            self.current_step - self.look_back_days:self.current_step].mean()
        }

    # To be used for future adaptations
    # def observation_space(self):
    #     """Return the observation space of the environment.
    #     The observation space is represented by a dictionary with the following keys:
    #     - stock_data: A numpy array of adjusted closing prices for the last n days
    #     - position: The current position of the agent (0 = no position, 1 = long position)
    #     - portfolio_value: The current value of the portfolio
    #     """
    #     return {
    #         'Weekday': self.stock_data['Weekday'].iloc[self.current_step],
    #         'Month': self.stock_data['Month'].iloc[self.current_step],
    #         'Quarter': self.stock_data['Quarter'].iloc[self.current_step],
    #         'Day_Sine': self.stock_data['Day_Sine'].iloc[self.current_step],
    #         'Day_Cos': self.stock_data['Day_Cos'].iloc[self.current_step],
    #         'Month_Sine': self.stock_data['Month_Sine'].iloc[self.current_step],
    #         'Month_Cos': self.stock_data['Month_Cos'].iloc[self.current_step],
    #         'Open': self.stock_data['Open'].iloc[self.current_step],
    #         'High': self.stock_data['High'].iloc[self.current_step],
    #         'Low': self.stock_data['Low'].iloc[self.current_step],
    #         'Close': self.stock_data['Close'].iloc[self.current_step],
    #         'Adj Close': self.stock_data['Adj Close'].iloc[self.current_step],
    #         'Volume': self.stock_data['Volume'].iloc[self.current_step],
    #         'Adj_Close_MA': self.stock_data['Adj Close'].iloc[
    #                         self.current_step - self.look_back_days:self.current_step].mean(),
    #         'Volume_MA': self.stock_data['Volume'].iloc[
    #                      self.current_step - self.look_back_days:self.current_step].mean(),
    #         'position': 0 if self.stock_quantity == 0 else 1,
    #         'portfolio_value': self.portfolio_value,
    #         'Savings_Rate': self.stock_data['Savings_Rate'].iloc[self.current_step]
    #     }

    # Placeholder structures for future adaptations
    def multiple_stocks_structure(self):
        pass

    def fractional_investment_structure(self):
        pass

if __name__ == "__main__":
    stock_data = pd.read_csv(os.path.join("data", "processed_train_data.csv"), index_col=0, parse_dates=True)
    env = TradingEnvironment(stock_data)
    state = env.reset()
    action = 0
    next_state, reward, done, log_df = env.step(action)
    print(log_df)
    action = 1
    next_state, reward, done, log_df = env.step(action)
    print(log_df)
    action = 1
    next_state, reward, done, log_df = env.step(action)
    print(log_df)
    action = 1
    next_state, reward, done, log_df = env.step(action)
    print(log_df)
