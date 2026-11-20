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
    def __init__(self, actions, gamma =0.9, alpha =0.1, epsilon =0.1, lam=0.8, tao=0.01, *args, **kwargs):
        
        self.gamma=gamma
        self.alpha=alpha
        self.lam = lam
        self.epsilon = epsilon
        self.tao = tao #threshold 
        self.display_name="TD-Lambda"
        
        # Q is implemented as dictionary with the keys as a tuple of a state and action pair (i.e , (s,a))
        # Defaultdict is used so that if a state has not been seen yet, it is automatically created .
        # So need for pre-inilaizition of all states, more effcient.
        self.Q= defaultdict(float) 
        
        # Elig is implemented as dictionary with the keys as a tuple of a state and action pair (i.e , (s,a))
        # Defaultdict is used so that if a state has not been seen yet, it is automatically created .
        # So need for pre-inilaizition of all states, more effcient.
        self.elig = defaultdict(float) 
        
        self.actions=actions
        self.num_actions = len(actions)


    def choose_action(self, observation, **kwargs):
       next_state = observation
       #Use e-greedy as the policy to select the next action
       if random.random()< self.epsilon:
           return random.choice(self.actions)
       else:
            return max(self.actions, key= lambda a:self.Q[next_state, a])
        
        

    def learn(self, s, a, r, s_, **kwargs):
        #EligTrace learn algorithim was implemented using Backward view using the SARSA(lambda)
        a_ = self.choose_action(s_)
        self.elig[s,a] += 1 

        error = r + (self.gamma*self.Q[s_,a_]) - self.Q[s,a]
    
        #Computing the decay outside of the for loop to avoid recalculation for increased effeciency 
        decay = self.gamma * self.lam
        
        #Computing the alpha *error value  outside of the for loop to avoid recalculation for increased effeciency 
        alpha_error = error * self.alpha

        #To improve effeiciency, only states whoese trace value is greater than tao will be update. 
        # We will check that if trace value is less than tao, then the trace value for the state-action pair will be set to zero
        # as such it's Q(S,A) value will not be updated in the next step. This helps eliminate the double for loop which is needed to iterate all the states. 
        
        #get all the states that are etraces is greater than tao
        #then we compute the Q value and then the new trace value
        #if the new trace value is less than tao we set it to zero


        viable_keys = [k for k, v in self.elig.items() if v > self.tao]

        for key in viable_keys:
            self.Q[key] += alpha_error * self.elig[key]
            if self.elig[key]*decay <=self.tao:
                self.elig[key] = 0
            else: 
                self.elig[key] = self.elig[key]*decay
        
        s = s_
        a= a_
        return s,a 
        
