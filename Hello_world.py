'''
Author: sbg, https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/
First program using keras-rl to perform reinforced learning.
Keras-rl is a interface that uses keras to make a simple interface for 
creating agents for the OpenAI gym (see gym.openAI.com). 
OpenAIGym comes with a set of pre-defined environment making RL easy to implement, especially using the pre-defined environments.

This will make a simple "cartpole" pole balancing program
'''

import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Nadam

from rl.agents import DQNAgent, DDPGAgent
from rl.policy import EpsGreedyQPolicy, GreedyQPolicy
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


# Variables
ENV_NAME = 'Pendulum-v0'
gym.undo_logger_setup()

# get the environment and extract observations and actions
env = gym.make(ENV_NAME)
seed = 123
np.random.seed(seed)
env.seed(seed)
#nb_actions = env.action_space.n #for CartPole
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

#setup model
actorModel = Sequential()
actorModel.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actorModel.add(Dense(30))
actorModel.add(Activation('tanh'))
actorModel.add(Dense(20))
actorModel.add(Activation('tanh'))
actorModel.add(Dense(nb_actions))
actorModel.add(Activation('linear'))
print(actorModel.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape)
observation_flat = Flatten()(observation_input)
#TODO: fix this
x = Concatenate()([action_input, observation_flat])
x = Dense(30)(x)
x = Activation('tanh')(x)
x = Dense(20)(x)
x = Activation('tanh')(x)
x = Dense(nb_actions)(x)
x = Activation('linear')(x)
criticModel = Model(inputs=[action_input, observation_input], outputs=x)
print(criticModel.summary())

#setup policy and memory
memory = SequentialMemory(limit=50000, window_length=1)
#setup agent, using defined keras model alog with the policy and actions from above

#Discrete actions:
policy = EpsGreedyQPolicy()
testPolicy = GreedyQPolicy()
#agent = DQNAgent(model=actorModel, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10, policy=policy, test_policy=testPolicy)

#continuous actions:
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(actor=actorModel, critic=criticModel, nb_actions=nb_actions, memory=memory, nb_steps_warmup_actor=100, nb_steps_warmup_critic=100,
                 critic_action_input=action_input, random_process=random_process)

#compile model
agent.compile(Nadam(lr=1e-3, clipnorm=0.1), metrics=['mae'])

# Okay, now it's time to learn something! 
# We visualize the training here for show, but this slows down training quite a lot. 
agent.fit(env, nb_steps=50000, visualize=True, verbose=2)

#TEST!
#blockingVar = input('Press a key!: ')
agent.test(env, nb_episodes=5, visualize=True)