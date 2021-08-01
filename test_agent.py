import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

# env = gym.make('Pendulum-v0').unwrapped
env = gym.make('LunarLanderContinuous-v2').unwrapped

agent = Agent(alpha=0.0001, beta=0.001,
              input_dims=env.observation_space.shape, tau=0.001,
              batch_size=64, fc1_dims=400, fc2_dims=300,
              n_actions=env.action_space.shape[0])

# load optimized policy
agent.load_models()
agent.actor.eval()

print(env.observation_space.shape[0])
print(env.action_space.shape[0])

state = env.reset() 

episodes = 5
for e in range(episodes):
    state = env.reset()  # get observations
    done = False  # reset done
    reward_ep = 0
    i = 0
    while not done:
        env.render()
        action = agent.choose_action(state)  # predict action. The action is a value between -2.0 and 2.0, representing the amount of left or right force on the pendulum
        state, reward, done, _ = env.step(action)  # execute action
        i += 1
        reward_ep += reward
        if done or i == 200:
            break
    print(e, f'{i:.3f},{reward_ep:2f}')
env.close()

# For Pendulum
# There are three observation: Cos(theta), sine(theta) that represents the angle of the pendulum and its angular velocity. 
# Theta is normalized between -pi and pi. Therefore, the lowest cost is -(π2 + 0.1*82 + 0.001*22) = -16.2736044, and the highest cost is 0
# The precise equation for reward is: -theta2 + 0.1*theta_dt2 +0.001*action2.

# For LunarLanderContinuous-V2
# Action is two floats [main engine, left-right engines].
# Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
# Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
"""
For LunarLanderContinuous-V2
Args:
    env: The environment
    s (list): The state. Attributes:
                s[0] is the horizontal coordinate
                s[1] is the vertical coordinate
                s[2] is the horizontal speed
                s[3] is the vertical speed
                s[4] is the angle
                s[5] is the angular speed
                s[6] 1 if first leg has contact, else 0
                s[7] 1 if second leg has contact, else 0
returns:
        a: The heuristic to be fed into the step function defined above to determine the next step and reward.

The landing pad is always at coordinates (0,0). The coordinates are the first two numbers in the state vector.
Reward for moving from the top of the screen to the landing pad and zero speed is about 100..140 points.
If the lander moves away from the landing pad it loses reward. The episode finishes if the lander crashes or
comes to rest, receiving an additional -100 or +100 points. Each leg with ground contact is +10 points.
Firing the main engine is -0.3 points each frame. Firing the side engine is -0.03 points each frame.
Solved is 200 points."""