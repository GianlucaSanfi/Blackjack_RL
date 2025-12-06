# Blackjack_RL
Reinforcement Learning project for AI & ML course, based on Blackjack-v1 environment by Gymnasium.<br>
Implements standard Deep Q-Learning Neural Network (DQN) and Tabluar Q-Learning algorithm for non deterministic environments.<br>
Aim of the project is to evaluate the two methods (DQN and TabularQL) varying the hyperparameters of the Reinforcement Learning models: Learning rate, Discount factor, number of episodes, epsilon-greedy param variation.<br>
<br>

DEPENDENCIES:<br>
pip install gymnasium==0.28.* numpy matplotlib tqdm
<br>

TABULAR Q-Learning<br> 
USE in src/tabularQL:<br>
    python TabularQL.py {--episodes [NUM_EPISODES] --alpha [ALPHA] --gamma [GAMMA] --epsilon [EPSILON] --epsilon-decay [E_DECAY]}
<br>

Deep Q-Learning Neural Network (DQN)<br>
USE in src/DQN:<br>
    python DQN.py {--episodes [NUM_EPISODES] --alpha [ALPHA] --gamma [GAMMA] --epsilon [EPSILON] --epsilon-decay [E_DECAY] --replay-size [BUFFER_SIZE] --batch-size [BATCH_SIZE]}
   
<br>
COMPARATOR of results:<br>
USE: python comparator.py<br><br>
Generates an image plotting the best scoring settings of the model.<br>
Used to estimate the best performance over inserted metrics based on win, loss, draw rates. 

