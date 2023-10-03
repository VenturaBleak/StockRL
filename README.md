<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>

<h1>Trading Agent Repository</h1>

<p>This repository contains the necessary components to build and train a deep reinforcement learning agent for stock trading.</p>

<h2>File Descriptions:</h2>
<ul>
        <strong>agent.py</strong>
        <p>This file contains the implementation of the deep Q-learning agent and the replay buffer used for experience replay.</p>

        <strong>data.py</strong>
        <p>Handles data preprocessing and other data-related tasks to make the stock data usable by the agent and environment.</p>

        <strong>environment.py</strong>
        <p>Describes the trading environment. This is where the agent interacts, makes decisions, and receives feedback in the form of rewards or penalties. It's based on the OpenAI Gym environment class.</p>

        <strong>main.py</strong>
        <p>The entry point for running the training loop. It initializes the environment and agent, and then proceeds with the training process.</p>

        <strong>model.py</strong>
        <p>Contains the neural network architecture for the Q-learning algorithm.</p>

        <strong>retrieve_savings_data.py</strong>
        <p>A script responsible for fetching and possibly preprocessing the savings rate data.</p>

        <strong>retrieve_stock_data.py</strong>
        <p>A script that fetches stock data. This is essential for the environment to provide the agent with stock prices and other related information.</p>

        <strong>train.py</strong>
        <p>Contains the training loop where the agent learns by interacting with the environment over multiple episodes.</p>
    
</ul>

<p>For more detailed information on each component or any further instructions, please refer to the comments inside each script.</p>

</body>
</html>
