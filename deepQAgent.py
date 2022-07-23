from collections import deque
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import math

"""
The structure of the Q-learning algorithm is based on code provided here:
https://medium.com/@ts1829/solving-mountain-car-with-q-learning-b77bf71b1de2
but with modifications to operate more efficiently using Keras
(i.e. batch updates instead of online updates)
"""

class DQNAgent:
    """
    A Deep Q-Learning Agent.

    Given the size of the state space and action space, creates a neural
    network to approximate the Q-Learning table.

    This basic framework should work for any RL problem, but each new
    problem will likely need a different neural network model.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.990
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.history = []

    def _build_model(self):
        """
        Construct a neural network model using keras.

        We need outputs to be both negative and positive, so use a linear
        activation function.
        """
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=SGD(learning_rate=self.learning_rate))
        model.summary()
        return model

    def save_weights(self, filename="model.h5"):
        """
        Saves the network weights to a file.
        """
        self.model.save_weights(filename)

    def load_weights(self, filename="model.h5"):
        """
        Reloads the network weights from a file.
        """
        self.model.load_weights(filename)

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the given experience in the memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def epsilon_greedy_act(self, state):
        """
        Given a state, chooses whether to explore or to exploit based
        on the self.epsilon probability.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return self.greedy_act(state)

    def greedy_act(self, state):
        """
        Given a state, chooses the action with the best value.
        """
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, iterations):
        """
        Selects a random sample of experiences from the memory to
        train on using batch updates.
        """
        for i in range(iterations):
            minibatch = random.sample(self.memory, batch_size)

            states = np.asarray([state[0] for state, action, reward, \
                                 next_state, done in minibatch])
            nextStates = np.asarray([next_state[0] for state, action, \
                                     reward, next_state, done in minibatch])
            rewards = np.asarray([reward for state, action, reward, \
                                  next_state, done in minibatch])
            actions = np.asarray([action for state, action, reward, \
                                  next_state, done in minibatch])
            notdone = np.asarray([not(done) for state, action, reward, \
                                  next_state, done in minibatch]).astype(int)
            nextVals = np.amax(self.model.predict(nextStates), axis=1)
            targets =  rewards + (nextVals * notdone * self.gamma)
            targetFs = self.model.predict(states)
            for i in range(len(minibatch)):
                targetFs[i, actions[i]] = targets[i]
            self.model.fit(states, targetFs, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def plot_history(self, show=True):
        """
        Plots the rewards per episode over number of episodes.
        If show is True, displays the plot. Saves plot to a file.
        """
        plt.figure(1)
        plt.plot(self.history)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig("RewardByEpisode.png")
        if show:
            plt.show()

    def reward(self, state):
        """
        Custom reward function to provide more positive feedback for Lunar Lander.
        """
        reward = 0
        if state[6] == 1:
            reward += 100
        if state[7] == 1:
            reward += 100
        #if landed between flags add 100 or more - how to get location of flags?
        #punish for distance away from goal - similar to built-in
        reward += -10*math.sqrt(state[0]*state[0] + state[1]*state[1])
        #punish for not pointing feet downward - similar to built-in
        reward += -10*abs(state[4])
        return reward

    def train(self, env, episodes, steps, batchSize, batchIterations,\
              envName):
        """
        Train the agent in the environment env for the given number
        of episodes, where each episode is at most steps long.  Use
        the batchSize and batchIterations when calling replay. The
        agent should follow the epsilon-greedy policy.

        Returns: None
        Side Effects:
        -Every 50 episodes saves the agent's weights to a filename
         of the form: envName_episode_#.h5, it must have the h5 extension
        -Updates self.history with total rewards received per episode
        -Prints a summary of reward received each episode
        """
        #raise NotImplementedError("TODO")
        if envName == "CartPole":
            self.history = []
            for i in range(episodes):
                state = env.reset()
                state = np.reshape(state, [1, self.state_size])
                filename = envName + "_episode_" + str(i) + ".h5"
                if i%50 == 0:
                    self.save_weights(filename)
                total_reward = 0
                for j in range(steps):
                    action = self.epsilon_greedy_act(state)
                    next_state, reward, done, info = env.step(action)
                    total_reward += reward
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
                self.history.append(total_reward)
                if len(self.memory) > batchSize:
                    self.replay(batchSize, batchIterations)
                print("Episode %d, total reward: %f" %(i+1, total_reward))
        if envName == "LunarLander":
            self.history = []
            for i in range(episodes):
                state = env.reset()
                state = np.reshape(state, [1, self.state_size])
                filename = envName + "_episode_" + str(i) + ".h5"
                if i%50 == 0:
                    self.save_weights(filename)
                total_reward = 0
                for j in range(steps):
                    action = self.epsilon_greedy_act(state)
                    next_state, reward, done, info = env.step(action)
                    reward = self.reward(next_state)
                    total_reward += reward
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.remember(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
                self.history.append(total_reward)
                if len(self.memory) > batchSize:
                    self.replay(batchSize, batchIterations)
                print("Episode %d, total reward: %f" %(i+1, total_reward))

    def test(self, env, episodes, steps):
        """
        Test the agent in the environment env for the given number
        of episodes, where each episode is a most steps long, when
        the agent follows its greedy policy.

        Returns: None
        Side Effects:
        -Renders the environment to see the agent in action.
        -Prints a summary of reward received each episode.
        """
        #raise NotImplementedError("TODO")
        self.history = []
        for i in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            for j in range(steps):
                env.render()
                action = self.greedy_act(state)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                next_state = np.reshape(next_state, [1, self.state_size])
                state = next_state
                if done:
                    break
            self.history.append(total_reward)
            print("Episode %d, total reward: %f" %(i, total_reward))
