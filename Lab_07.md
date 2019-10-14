# Introduction to Machine Learning - Lab 07
| Lab7 group 0_             |                     |
| ---                       | ---                 |
| Baldur Þór Haraldsson     | baldurh17@ru.is     |
| Egill Torfason            | egilltor17@ru.is    |
| Hallgrímur Snær Andrésson | hallgrimura17@ru.is |

1. Modeling the environment _(Total points: 50)_ \
   For this task, assume that the game is played with a deck consisting of infinitely many copies of each of the 52 cards. That is, even after having seen a card, the probability of drawing a specific card is still 1/52.
   1. (a) _(20 points)_ \
    How would you describe a state of the problem? What are the possible actions, successor states, rewards and transition probabilities? Try to reduce the state space as much as possible (combine states whose values and best action do not differ), but make sure to keep the Markov property.

        ...

   2. (b) _(30 points)_ \
    Implement your model of the problem by filling out the relevant functions in BlackjackMDP in lab7.py. Test your model by generating some simulations of random game play and checking whether the resulting states make sense. How many reachable states does your model have?

        ...

2. Value Iteration (Total points: 50)
   1. (a) _(25 points)_ \
    Implement Value Iteration for an arbitrary MDP (any object derived from the MDP class in the code). Decide on an appropriate stopping criterion and run your code on the Blackjack MDP.

        ...

   2. (b) _(10 points)_ \
    What is the expected outcome of playing Blackjack with the optimal policy? Who will win the the long run, the player or the dealer? Why?

        ...

   3. (c) _(15 points)_ \
    Visualize the resulting value function and policy. For example, you could use a table similar to the one on Wikipedia, depending on your choice of state space. 

        ...

   4. (d) _(5 bonus points)_ \
    Suppose the player had an additional action ”Double down”. Upon ”Double down” the player will get exactly one more card and end his turn. He will also double his initial bet, that is, win or loose double the normal amount. Show how having this additional action changes the optimal policy and expected outcome of the game.
        
        ...

   5. (e) _(10 bonus points)_ \
    Playing with an infinite deck of cards is somewhat unrealistic. How does the value of the game change, if we assume to only play with one deck of cards (52 cards)? How many different states do you need to consider in this case?

        ...
