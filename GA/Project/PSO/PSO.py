import time

import numpy as np
import random

import torch

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from GA.Project.lower_level import LowerLevel


class PSO:

    def __init__(self, instance, c_soc = 1.49445, c_cog = 1.49445, w = 0.8, lower_eps=10**(-12),
                 pop_size=None, device=None, reuse_p=False):

        self.device = device
        self.instance = instance

        self.n_paths = self.instance.n_paths
        self.pop_size = pop_size

        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        self.lower = LowerLevel(self.instance, lower_eps, device=device, M=self.M, reuse_p=reuse_p)

        self.c_soc = c_soc
        self.c_cog = c_cog
        self.w = w

        self.obj_val = 0
        self.times = []

    def update_position(self, position, velocity, boundaries):
        new_position = position + velocity
        for i, b in enumerate(boundaries):
            # 0 and 1 are the boundary limits, so the action depends on which one the particle is pass
            if new_position[i] < b[0]:
                new_position[i] = b[0]
                velocity[i] = - np.random.random() * velocity[i]  # the velocity direction is changed
            elif new_position[i] > b[1]:
                new_position[i] = b[1]
                velocity[i] = - np.random.random() * velocity[i]
        return new_position

    def update_velocity(self, position, velocity, global_best, local_best, max_velocities):
        n = len(velocity)
        r1 = np.random.random(n)
        r2 = np.random.random(n)
        social_component = self.c_soc * r1 * (global_best - position)
        cognitive_component = self.c_cog * r2 * (local_best - position)
        inertia = self.w * velocity
        new_velocity = inertia + social_component + cognitive_component
        # check we are not going above the maximum velocity:
        for i, v in enumerate(max_velocities):
            if np.abs(new_velocity[i]) < v[0]:
                new_velocity[i] = np.sign(new_velocity[i]) * v[0]
            elif np.abs(new_velocity[i]) > v[1]:
                new_velocity[i] = np.sign(new_velocity[i]) * v[1]
        return new_velocity

    def run_newpso(self, swarm_size, boundaries, max_velocities, n_iter, fit):
        m = len(boundaries)
        positions = [np.array([np.random.random() * (b[1] - b[0]) + b[0] for b in boundaries])
                     for i in range(0, swarm_size)]
        velocities = [np.array([np.random.choice([-1, 1]) * np.random.uniform(v[0], v[1])
                                for v in max_velocities]) for i in range(0, swarm_size)]
        local_best = positions  # initially (at the first generation)
        global_best = min(positions, key=fit)
        hist = [positions]
        # updating:
        for i in range(0, n_iter):
            velocities = [self.update_velocity(p, v, global_best, lb, max_velocities)
                          for p, v, lb in zip(positions, velocities, local_best)]
            positions = [self.update_position(p, v, boundaries) for p, v in zip(positions, velocities)]
            local_best = [min([p, lb], key=fit) for p, lb in zip(positions, local_best)]
            global_best = min([min(positions, key=fit), global_best], key=fit)
            hist.append(positions)
        return global_best, hist

    def fitness_evaluation(self, population):
        population = torch.repeat_interleave(torch.from_numpy(np.array(population)).unsqueeze(0),
                                             repeats=self.instance.n_od, dim=0)
        return - self.lower.eval(torch.from_numpy(np.array(population)))