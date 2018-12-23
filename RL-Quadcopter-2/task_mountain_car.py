import numpy as np
import math


class TaskMountainCar():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, env):

        self.env = env
        self.action_repeat = 3

        self.state_size = self.action_repeat * np.prod(env.observation_space.shape)
        self.action_low = env.action_space.low[0]
        self.action_high = env.action_space.high[0]
        self.action_size = np.prod(env.action_space.shape)

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        total_reward = 0
        obs = []
        for _ in range(self.action_repeat):
            ob, reward, done, info = self.env.step(action)
            total_reward += reward
            obs.append(ob)
        next_state = np.concatenate(obs)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        ob = self.env.reset()
        state = np.concatenate([ob] * self.action_repeat)
        return state
