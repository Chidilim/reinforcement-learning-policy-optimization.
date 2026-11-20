
import numpy as np
import pandas as pd
import random
from collections import defaultdict
#Used a random seed to ensure reproducibility. 
random.seed(42)

DEBUG=1

def debug(debuglevel, msg, **kwargs):
    if debuglevel <= DEBUG:
        if 'printNow' in kwargs:
            if kwargs['printNow']:
                print(msg) 
        else:
            print(msg) 


class rlalgorithm:
    def __init__(self, actions, gamma =0.9,alpha=0.01,epsilon =0.1, *args, **kwargs):
        """Your code goes here"""
        #pass
        self.gamma=gamma
        self.alpha=alpha
        self.epsilon = epsilon
        self.display_name="SARSA"
        
        # Q is implemented as dictionary of dictionaries. 
        # Defaultdict is used so that if a state has not been seen yet, it is automatically created .
        # So need for pre-inilaizition of all states, more effcient.
        self.Q= defaultdict(lambda:defaultdict(float)) 
        
        self.actions=actions
        self.num_actions = len(actions)

        
        debug(1, 'Init new RL Algorithm {}: |A|={} A={} gamma={}'.format(self.display_name,self.num_actions, self.actions, self.gamma))

    def choose_action(self, observation, **kwargs):
        """Your code goes here"""
        next_state = observation
        #Use e-greedy as the policy to select the next action
        if random.random()< self.epsilon:
            return random.choice(self.actions)
        else:
            return max(self.actions, key= lambda a:self.Q[next_state][a])
        
        
       

    def learn(self, s, a, r, s_, **kwargs):
        """Your code goes here"""
        a_ = self.choose_action(s_)
        #Updating Q(s,a) value using the choosen a_ 
        self.Q[s][a] += self.alpha*(r + (self.gamma*self.Q[s_][a_]) - self.Q[s][a])
        s = s_
        a= a_
        return s,a 
        
        
