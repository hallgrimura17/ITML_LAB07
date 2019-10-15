import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

ls = (10,10,10,10,2,3,4,5,6,7,8,9,11)
phase = ['init', 'player', 'dealer', 'terminal']
def calcProb(x):
    if x == 10:
        return 4/13
    else:
        return 1/13

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    def __init__(self):
        self.computeStates()

    # discount factor
    discountFactor = 0.9

    # Return the start state.
    def startState(self): raise NotImplementedError('Override me')

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError('Override me')

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', reward = r, prob = p(s', r | s, a)
    # If state is a terminal state, return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError('Override me')

    # Compute set of states reachable from startState.  Helper function
    # to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        print('%d reachable states' % len(self.states))
        # print(self.states)
        
class BlackjackMDP(MDP):

    # the discount factor for future rewards
    discountFactor = 0.9 # TODO: set this to the correct value
 
    # Return the start state.
    def startState(self):
        return ('init', 0, False, 0, False)

    # Return set of actions possible from |state|.
    def actions(self, state):
        if state[0] == 'player':
            return ['Hit', 'Stand']
        elif state[0] == 'init' or state[0] == 'dealer':
            return ['Noop']
        else:
            return []

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', reward = r, prob = p(s', r | s, a)
    # If state is a terminal state, return the empty list.
    def succAndProbReward(self, state, action):
        results = []
        nextstate = ()
        if state[0] == 'init':
            for x in ls:
                for y in ls:
                    playerValue = x + y
                    playerUsableAce = x == 11 or y == 11
                    if playerValue > 21:
                        playerValue -= 10
                    for z in ls:
                        dealerValue = z
                        dealerUsableAce = z == 11
                        prob1 = calcProb(x)
                        prob2 = calcProb(y)
                        prob3 = calcProb(z)
                        nextstate = ('player', playerValue, playerUsableAce, dealerValue, dealerUsableAce)
                        results.append((nextstate, prob1*prob2*prob3, 0))
        elif state[0] == 'player':
            if action == 'Stand':
                nextstate = ('dealer',) + state[1:]
                p = 1.0
                reward = 0
                return [(nextstate, p, reward)]
            if action == 'Hit':
                reward = 0
                playerUsableAce = state[2]
                for x in ls:
                    phv = x + state[1]
                    p = calcProb(x)
                    if phv > 21 and state[2]:
                        phv -= 10
                        playerUsableAce = x == 11
                    elif phv > 21 and x == 11:
                        phv -= 10
                        playerUsableAce = False
                    if phv > 21:
                        # nextstate = ('terminal', phv, playerUsableAce, state[3], state[4])
                        nextstate = ('terminal', 0, False, 0, False)
                        reward = -1
                    else:
                        nextstate = ('dealer', phv, playerUsableAce, state[3], state[4])
                    results.append((nextstate, p, reward))
        elif state[0] == 'dealer':
            if state[3] > 16:
                # nextstate = ('terminal',) + state[1:]
                nextstate = ('terminal', 0, False, 0, False)
                p = 1.0
                reward = 0
                if state[1] > state[3]:
                    reward = 1
                elif state[1] < state[3]:
                    reward = -1
                return [(nextstate,p,reward)]
            if state[3] <= 16:
                dealerUsableAce = state[4]
                reward = 0
                for x in ls:
                    p = calcProb(x)
                    dhv = x + state[1]
                    if dhv > 21 and state[2]:
                        dhv -= 10
                        playerUsableAce = x == 11
                    elif dhv > 21 and x == 11:
                        dhv -= 10
                        dealerUsableAce = False
                    if dhv > 21:
                        # nextstate = ('terminal', state[1], state[2], dhv, dealerUsableAce)
                        nextstate = ('terminal', 0, False, 0 , False)
                        reward = 1
                    else:
                        nextstate = ('player', state[1], state[2], dhv, dealerUsableAce)
                    results.append((nextstate,p,reward))     
        return results
        
def computeQ(state, action ,mdp, v):
    lis = []
    for x in mdp.succAndProbReward(state, action):
        lis.append(x[1] * (x[2] + mdp.discountFactor * v[x[0]]))
    return max(lis)

mdp = BlackjackMDP()
v = {}
for state in mdp.states:
    v[state] = 0

delta_bound = 0.0001
i = 0
converged = False
while not converged:
    delta = 0
    for state in mdp.states:
        qmax = -100000
        for action in mdp.actions(state):
            qmax = max(qmax, computeQ(state, action, mdp, v))
            # if action == 'Noop':
            #     qmax  = 0
            # else:
            #     qmax = max(qmax, computeQ(state, action, mdp, v))
        delta = max(delta, abs(v[state] - qmax))
        v[state] = qmax
    i += 1
    print("iteration: ", i, " delta: ", delta)
    converged = (delta < delta_bound)

pi = {}
for state in mdp.states:
    qmax = -100000
    for action in mdp.actions(state):
        q = computeQ(state, action , mdp, v)
        # if action == 'Noop':
        #     q = 0
        # else: 
        #     q = computeQ(state, action , mdp, v)
        if q > qmax:
            pi[state] = (q, action)
            qmax = q


# for i in pi:
#     print(pi[i])

# sorted_by_second = sorted(mdp.states, key=lambda state: state[1])
# for i in mdp.states:
#     print(i, "\t", v[i])
d = [[0]*21]*21
# dd = np.ndarray(shape=(21,21), dtype=float, order='F')
dd = np.full(shape=(22,22), fill_value=0.0, dtype=float, order='F')
for state in pi:
    # print(state, "\t", v[state], "\t", pi[state])
    # dd[state[3]][state[1]] = v[state]
    dd[state[3]][state[1]] = pi[state][0]

sns.heatmap(data=dd, annot=True, cbar=False)
# plt.xlim(1.5, 11.5)
# plt.ylim(3.5, 21.5)
plt.ylim(-0.5, 22.5)
# plt.xticks(list(range(2,12)))
# plt.yticks(list(range(4,22)))
plt.xlabel("Player hand value")
plt.ylabel("Dealer hand value")
plt.title("Policy")
plt.show()



dd = np.full(shape=(22,22), fill_value=0.0, dtype=float, order='F')
for state in v:
    dd[state[3]][state[1]] = v[state]

sns.heatmap(data=dd, annot=True, cbar=False, cmap="Blues")
# plt.xlim(1.5, 11.5)
# plt.ylim(3.5, 21.5)
plt.ylim(-0.5, 22.5)
# plt.xticks(list(range(2,12)))
# plt.yticks(list(range(4,22)))
plt.xlabel("Player hand value")
plt.ylabel("Dealer hand value")
plt.title("Reward")
plt.show()
"""
for each state we need to have the hand value of the player and dealer, if either of them have a usable ace, the actions are only from the view of the player which are hit or stand or wait, which is not the same as doing nothing. successor states for the player either involve the dealers turn with addition to dragging more cards or a terminal state. the successor states of a dealer 
"""