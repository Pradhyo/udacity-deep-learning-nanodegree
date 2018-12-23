import numpy as np
from physics_sim import PhysicsSim
import math


class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""

    def __init__(self, init_pose=None, init_velocities=None, init_angle_velocities=None,
                 runtime=5., target_pose=None):

        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities,
                              init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.init_z = init_pose[2]
        self.target_pose = target_pose


    def get_reward(self):
        """Uses current pose of sim to return reward."""

        """Takeoff"""
        # Punish upto -1 each for moving away from x, y coordinates
        punish_x = np.tanh(abs(self.sim.pose[0] - self.target_pose[0]))
        punish_y = np.tanh(abs(self.sim.pose[1] - self.target_pose[1]))

        # Similar to above but with more weightage since with this more important
        # This is positive when current position is > initial position and negative when not
        reward_z = 3*np.tanh(self.sim.pose[2] - self.init_z)

        # Punish upto -1 for rotating
        punish_rot1 = np.tanh(abs(self.sim.pose[3]))
        punish_rot2 = np.tanh(abs(self.sim.pose[4]))
        punish_rot3 = np.tanh(abs(self.sim.pose[5]))
        reward = reward_z - punish_x - punish_y - punish_rot1 - punish_rot2 - punish_rot3
        return reward

        """Hover"""
        """
        punishments = []
        dist = abs(np.linalg.norm(self.target_pose[:3] - self.sim.pose[:3]))

        # Reward 10 for staying within a distance of 3
        if dist < 3:
            punishments.append(-10)
        # Punish upto -1 for going beyond a distance of 3
        else:
            punishments.append(np.tanh(dist))

        # Punish upto -1 each for rotating or having any velocity
        punishments.append(np.tanh(np.linalg.norm(self.sim.pose[3:6])))
        punishments.append(np.tanh(np.linalg.norm(self.sim.v)))
        reward = -sum(punishments)
        return reward
        """


        """Unused experiments"""


        # curr_values = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))
        # # pose_distance = abs(np.linalg.norm(self.sim.pose - self.target_pose[:6]))
        # # v_distance = abs(np.linalg.norm(self.sim.v - self.target_pose[6:9]))
        # distance = abs(np.linalg.norm(curr_values - self.target_pose))
        # # print (curr_values, self.target_pose)
        # reward = np.tanh(distance)
        # return -reward

        # Zdist = abs(self.target_pose[2]-self.sim.pose[2])
        # We want minimum change in direction of x and y and making z close to the target z
        # # so we punish for moving in directions x and y and give rewards to direction in z

        # reward = 1.-.003*(abs(self.sim.pose[:3] - self.target_pose[:3])).sum()
        # return reward

        # Trying sum of squares of difference between current value and target values didn't work too well
        # curr_values = np.concatenate((self.sim.pose, self.sim.v, self.sim.angular_v))
        # reward = 0
        # for _curr_value, _target_value in zip(curr_values, self.target_pose):
        #     reward += (_curr_value - _target_value)**2
        # # print(curr_values)
        # print (" -" + str(reward))
        # # print ("\n\n")
        # return -reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds)
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        return state
