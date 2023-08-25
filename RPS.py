# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 14:25:52 2023

@author: Lillemoen
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from RPS_game import mrugesh, abbey, quincy, kris

EPISODES = 60000
LEARNING_RATE = 0.5  # learning rate
GAMMA = 0.96
KEPT_MOVES = 3

STATES = np.power(3,KEPT_MOVES) # example state would be ['R', 'P', 'P'] for
# KEPT_MOVES = 3
ACTIONS = 3 # R, P, or S

letter_to_value = {'R': 0, 'P': 1, 'S': 2}
moves = ["R", "P", "S"]

def letter_mapping(sequence_array):
    """
    Function uses a base-3 three system to create a unique base-10 integer reprenting a
    sequence of R, P and S characters. The mapping is {'R': 0, 'P': 1, 'S': 2},
    which is defined globally.

    Parameters
    ----------
    sequence_array : List of characters
        List of singular letters that should be either R, P or S

    Raises
    ------
    ValueError
        If letter if not a R, P or S

    Returns
    -------
    base_10_value : int
        An integer unique to an ordered sequence of R, P and S's.

    """
    base_10_value = 0

    for i, letter in enumerate(sequence_array):
        if letter in letter_to_value:
            base_10_value += letter_to_value[letter] * (3 ** i)
        else:
            raise ValueError(f"Invalid letter: {letter}")

    return base_10_value

def reverse_letter_mapping(base_10_value): #not particularly useful but might
# be in future
    """
    Function provides the inverse mapping of letter_mapping function above.
    It takes a base-10 integer and returns the correspondng sequence of R, P 
    and S characters.

    Parameters
    ----------
    base_10_value : int
        Integer representing a sequence of R, P and S characters

    Returns
    -------
    sequence_array : List of characters
        List of singular letters of either R, P or S

    """
    value_to_letter = {0: 'R', 1: 'P', 2: 'S'}
    sequence_array = []

    for _ in range(KEPT_MOVES):
        base_10_value, remainder = divmod(base_10_value, 3)
        letter = value_to_letter[remainder]
        sequence_array.insert(0, letter)  # Insert at the beginning to maintain order

    return np.array(sequence_array)

def calculate_reward(p1_play, p2_play):
    """
    Function implements the rules of rock, paper, scissors with player 1 being
    rewarded for winning

    Parameters
    ----------
    p1_play : char
        R, P or S representing player 1's move
    
    p2_play : char
        R, P or S representing player 2's move

    Returns
    -------
    reward : float
        Either 0, 0.5 or 1 for a loss, draw or win respectively
    """
    if p1_play == p2_play:
        return 0.5
    elif (p1_play == "P" and p2_play == "R") or (
            p1_play == "R" and p2_play == "S") or (p1_play == "S"
                                                   and p2_play == "P"):
        return 1
    elif p2_play == "P" and p1_play == "R" or p2_play == "R" and p1_play == "S" or p2_play == "S" and p1_play == "P":
        return 0

def create_Q_table(epsilon = 0.9):
    """
    Creates a Q-table based off the Q-learning algorithm on playing games of 
    rock, paper, scissors against 4 different bots.

    Parameters
    ----------
    epsilon : float, optional
        The starting amount of randomness in moves. 1 corresponds to complete
        randomness while 0 uses Q-table values only. The default is 0.9.

    Returns
    -------
    Q : numpy array
        The Q-table of shape (STATES, ACTIONS)
    rewards : list
        The rewards after each successive game

    """
    Q = np.zeros((STATES, ACTIONS))
    rewards = []
    opponent_history = ["R"] * KEPT_MOVES #I only want last KEPT_MOVES to limit number of total states
    my_history = [""] #opponent wants my complete history and can deal with empty history
    opponents = [mrugesh, abbey, quincy, kris]
    counter = [0] #need this so that quincy can keep following previous goes even if not called
    
    for episode in range(EPISODES):
      state = letter_mapping(opponent_history)
      opponent_index = random.randint(0, 3) #we have to train for all opponents
      #simultaneously. If we knew our chosen opponent before each test we could
      # create a Q-table just for them, which give a much higher success rate
      if opponent_index == 2:
        opponent_play = opponents[opponent_index](my_history[-1], counter) #quincy
        #quincy increases the counter for us
      elif opponent_index == 3:
        opponent_play = opponents[opponent_index](my_history[-1]) #kris
        counter[0] += 1
      else:
        opponent_play = opponents[opponent_index](my_history[-1], my_history)
        counter[0] += 1
    
      if np.random.uniform(0, 1) < epsilon: #good to start with lots of random
      #moves and slowly decrease amount of random moves
        my_play = moves[random.randint(0,2)]
      else:
        my_play = moves[np.argmax(Q[state, :])]
    
      reward = calculate_reward(my_play, opponent_play)
      my_history.append(my_play)
      opponent_history.append(opponent_play)
      opponent_history.pop(0) #We can't keep everything otherwise STATES too big
      next_state = letter_mapping(opponent_history)
    
      action = letter_to_value[my_play]
      # Key Q-table learning algorithm:
      Q[state, action] = Q[state, action] + LEARNING_RATE * (reward + GAMMA *
                                  np.max(Q[next_state, :]) - Q[state, action])
    
      rewards.append(reward)
      epsilon -= 0.00005
      
    return Q, rewards

def player(prev_play, opponent_history=[]):
    """
    Represents our learned player that uses our Q-table to make choices.
    A slightly unusual implementation due to the constraints of the set-up
    having to conform with all other bots, i.e taking "" characters and having
    only one required argument.

    Parameters
    ----------
    prev_play : char
        Either R, P or S
    opponent_history : List of characters
        Gets added to everytime we call the function

    Returns
    -------
    guess : char
        Player's guess depending on Q-table and opponent history

    """
    if prev_play == "":
        prev_play = "R" #Q-table function can't deal with no letter
    
    opponent_history.append(prev_play)
    
    if len(opponent_history) <= KEPT_MOVES: #needed for right at the start as
        guess = moves[random.randint(0,2)] #need a state of at least 3 characters
    else:
        recent_opponent_history = opponent_history[-KEPT_MOVES :]
        state = letter_mapping(recent_opponent_history)
        guess = moves[np.argmax(Q[state, :])] #Substitute for Q_best for better results
    
    return guess

def display_learning_process(rewards):
    """
    Displays our average reward and also a plot of rewards through the learning
    process. Useful for tweaking model but not for final test.

    Parameters
    ----------
    rewards : List
        The rewards after each successive game of our Q-learning process

    Returns
    -------
    None.

    """
    print(f"Average reward: {sum(rewards)/len(rewards)}:")
    
    #we can plot the training progress and see how the agent improved  
    avg_rewards = list(map(lambda i: sum(rewards[i:i+1000])/len(rewards[i:i+1000]),
                           range(0, len(rewards), 1000)))
    
    plt.plot(avg_rewards)
    plt.ylabel('average reward')
    plt.xlabel('episodes (100\'s)')
    plt.show()
    
Q, rewards = create_Q_table() # define variable here so don't have to calculate
#it everytime player is called plus I can't change the argument set-up of player
# to put Q in. We shall then import Q variable on main

#display_learning_process(rewards)

#everytime we run the Q-learning algorithm we seem to create quite different
#Q-tables. This is the best one I created. Need to investigate how we can limit
#this phenomena.
Q_best = np.array([[15.83983959, 13.57831271, 13.49668239],
       [15.95511533, 13.59366605, 13.44702273],
       [13.74423434, 13.73012614, 16.57841526],
       [17.69431652, 13.59760718, 13.56315442],
       [16.0579863 , 13.6173776 , 13.58937593],
       [16.42862034, 13.7620882 , 13.48780741],
       [17.25271025, 13.66626841, 13.62432283],
       [16.76540338, 13.58244316, 13.58443472],
       [13.55466796, 13.58616044, 17.34826412],
       [13.51587381, 16.93906979, 13.72682469],
       [13.56224067, 13.52685776, 17.24345494],
       [13.62017267, 17.5279344 , 13.63333252],
       [13.58951039, 13.56957279, 16.4881354 ],
       [13.46458654, 16.96803223, 13.48221647],
       [13.57329388, 13.70864453, 16.30666949],
       [13.61512232, 17.42318014, 13.51439346],
       [13.86386538, 13.56017063, 17.02078313],
       [13.90721864, 13.83015018, 16.86928286],
       [16.57906308, 13.5063943 , 13.40517972],
       [13.5753127 , 13.57051851, 17.05704107],
       [13.50501995, 13.65898955, 17.46124994],
       [13.66935761, 16.5528343 , 13.60488409],
       [13.66342942, 17.09415897, 13.60428418],
       [17.14113476, 13.48289227, 13.7236864 ],
       [16.99872828, 13.47575143, 13.59131993],
       [17.21147492, 13.60427073, 13.46475155],
       [16.49451001, 13.67036326, 13.60405676]])


