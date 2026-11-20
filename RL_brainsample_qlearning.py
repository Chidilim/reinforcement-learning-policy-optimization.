import numpy as np
import pandas as pd

class rlalgorithm:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        # Design: Q-Table representation
        # I have represented the Q-table as a pandas DataFrame. 
        # This allows easy indexing using the String based representations of the states (coordinates) as rows and actions as columns. 
        # The built-in methods like .max() and .idxmax() make it easy to simplift the Bellman update logic.
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.display_name = "QLearning"

    def check_state_exist(self, state):
        # Design: Handling New/Unseen States
        # Instead of making a massive table for every single square in the maze (even the ones we never visit), we starting with an empty table. 
        # We only add a new square to our memory the moment the agent actually steps into it. This saves a lot of computer memory by not wasting space on areas of the map that are blocked off or unreachable.
        if str(state) not in self.q_table.index:
            new_row = pd.DataFrame([[0]*len(self.actions)], columns=self.q_table.columns, index=[str(state)])
            self.q_table = pd.concat([self.q_table, new_row])

    def choose_action(self, observation, **kwargs):
        self.check_state_exist(observation)

        # Epsilon-greedy action selection
        if np.random.uniform() < self.epsilon:
            action = np.random.choice(self.actions) # Choosing a random action
        else:
            state_action = self.q_table.loc[str(observation), :]
            # Choose the best action for current state and randomly break ties if multiple actions have the same max Q-value
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def learn(self, s, a, r, s_, **kwargs):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[str(s), a]

        # Design: Q-Learning Update Logic (Bellman Equation)
        # The update is 'Off-Policy,' meaning the agent is an optimist. When it 
        # looks at the next square (s_), it assumes it will take the absolute best 
        # action possible in the future, even if it is currently moving randomly to explore. 

        # CHALLENGE: Terminal States ('End of the Road'). If the agent hits a pit or the goal, the episode ends. 
        # There is no 'next move' to look forward to, so we set the future reward to zero and only update the table using the immediate points gained or lost.

        if s_ != 'terminal':
            # Q-Learning update: Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s',a')) - Q(s,a)]
            q_target = r + self.gamma * self.q_table.loc[str(s_), :].max()
        else:
            q_target = r
        
        self.q_table.loc[str(s), a] += self.lr * (q_target - q_predict)
        
        # Return the next state and the next action to maintain compatibility with the update loop in run_main.py.
        return s_, self.choose_action(s_)