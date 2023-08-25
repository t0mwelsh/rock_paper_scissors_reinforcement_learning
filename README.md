# rock_paper_scissors_reinforcement_learning
Python files that define an environment in which a Q-learning algorithm is deployed to learn how to beat 4 separate bots and then tested against these to check win-rate. 

Note that the Q-learning algorithm creates quite different Q-tables on each run so code will not pass all the tests everytime. Therefore, I have provided a Q-best variable, which is the best Q-table I have been able to create from just running it a few times. To use this instead, change line 205 in RPS.py to Q_best as opposed to Q. Note also that this player is trained against all 4 bots simultaneouly so if you just wanted to have a player that beats one particular bot near-100% of the time then change line 140 in RPS.py from random.randint(0, 3) to a specific integer in [0,3], with the integer representing the chosen index of opponents = [mrugesh, abbey, quincy, kris].

I wrote the RPS.py file (where the model was made) but all other files (testing and gameplay) were provided by freeCodeCamp.
