import numpy as np
import pandas as pd

class rlalgorithm:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        # Design: Q-Table representation
        # I used a pandas DataFrame to store state-action values. 
        # This makes it easy to name rows after maze coordinates and columns after actions. 
        # Using .mean() and .max() on these rows simplifies the math needed for Expected SARSA.
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = "Expected SARSA"

    def check_state_exist(self, state):

        # Design: Handling New/Unseen States
        # We start with an empty table to save memory. 
        # We only add a square to the table when the agent actually visits it. 
        # This ensures we don't waste computer space on walls or parts of the maze the agent never sees.
        if str(state) not in self.q_table.index:
            new_row = pd.DataFrame([[0]*len(self.actions)], columns=self.q_table.columns, index=[str(state)])
            self.q_table = pd.concat([self.q_table, new_row])

    def choose_action(self, observation, **kwargs):
        self.check_state_exist(observation)
        # Epsilon-greedy action selection
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[str(observation), :]
            #picking the best known move 
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def learn(self, s, a, r, s_, **kwargs):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[str(s), a]
        
        # Design: Expected SARSA Update Logic
        # Unlike Q-Learning (which is a total optimist) or SARSA (which is very cautious), Expected SARSA is a 'Realist.' 
        # It calculates the average value of the next state by looking at the probability of taking every possible action.
        if s_ != 'terminal':
            q_values_next = self.q_table.loc[str(s_), :]
            n_actions = len(self.actions)
            # CHALLENGE: The math must account for the 10% chance (epsilon) of moving randomly.
            # We take the weighted average: (Prob of best move * its value) + (Prob of random moves * their average value).
            # This makes the updates more stable and less 'jumpy' than standard SARSA.
           
            expected_value = (q_values_next.max() * (1 - self.epsilon)) + (q_values_next.mean() * self.epsilon)
            q_target = r + self.gamma * expected_value
        else:
            # At the Goal or Pit, there is no 'next' expectation to calculate.
            q_target = r
            
        self.q_table.loc[str(s), a] += self.lr * (q_target - q_predict)
        return s_, self.choose_action(s_) # Return next state and action for compatibility with the main loop.