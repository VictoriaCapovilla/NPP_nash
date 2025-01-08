import time

import numpy as np

from GA.Project.lower_level import LowerLevel


class PSO:

    def __init__(self, instance, c_soc = 1.49445, c_cog = 1.49445, w = 0.8, lower_eps=10**(-12), save=False):

        self.save = save
        self.instance = instance

        # network data
        self.n_od = self.instance.n_od
        self.n_paths = self.instance.n_paths

        # calculate individuals maximum value
        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        # initialize the Lower Level
        self.lower = LowerLevel(self.instance, lower_eps, M=self.M)

        # parameters
        self.c_soc = c_soc
        self.c_cog = c_cog
        self.w = w

        if self.save:
            self.data_individuals = []
            self.data_fit = []
            self.times = []

    def update_position(self, position, velocity):
        new_position = position + velocity
        for i in range(0, self.n_paths):
            if new_position[i] < 0:
                new_position[i] = 0
                velocity[i] = - np.random.random() * velocity[i]    # the velocity direction change
            elif new_position[i] > self.M:
                new_position[i] = self.M
                velocity[i] = - np.random.random() * velocity[i]    # the velocity direction change
        return new_position

    def update_velocity(self, position, velocity, global_best, local_best, max_velocities):
        n = len(velocity)
        r1 = np.random.random(n)
        r2 = np.random.random(n)
        social_component = self.c_soc * r1 * (global_best - position)
        cognitive_component = self.c_cog * r2 * (local_best - position)
        inertia = self.w * velocity
        new_velocity = inertia + social_component + cognitive_component
        # check the particle is not going outside the maximum velocity boundaries:
        for i, v in enumerate(max_velocities):
            if np.abs(new_velocity[i]) < v[0]:
                new_velocity[i] = np.sign(new_velocity[i]) * v[0]
            elif np.abs(new_velocity[i]) > v[1]:
                new_velocity[i] = np.sign(new_velocity[i]) * v[1]
        return new_velocity

    def fitness_evaluation(self, individual):
        # adapting the individual to Lower Level requirements
        individual = np.transpose(np.reshape(np.repeat(np.array(individual), repeats=self.n_od),
                                             (self.n_od, self.n_paths)))
        return self.lower.eval(individual)

    def run_PSO(self, n_iter, swarm_size, max_velocity=None):
        if self.save:
            self.times.append(time.time())
        if max_velocity is None:
            max_velocity = [[0.001, self.M / 3]]
        # expand the velocity bounds to each dimension
        max_velocities = max_velocity * self.n_paths
        positions = [np.array([np.random.random() * self.M for _ in range(0, self.n_paths)])
                     for i in range(0, swarm_size)]
        ind = [(x, self.fitness_evaluation(x)) for x in positions]
        velocities = [np.array([np.random.choice([-1, 1]) * np.random.uniform(v[0], v[1])
                                for v in max_velocities]) for i in range(0, swarm_size)]
        local_best = positions
        local = ind
        global_best, global_fit = max(ind, key=lambda t: t[1])
        if self.save:
            self.data_individuals.append(global_best)
            self.data_fit.append(float(global_fit))
        # updating:
        for i in range(0, n_iter):
            velocities = [self.update_velocity(p, v, global_best, lb, max_velocities)
                          for p, v, lb in zip(positions, velocities, local_best)]
            positions = [self.update_position(p, v) for p, v in zip(positions, velocities)]
            ind = [(x, self.fitness_evaluation(x)) for x in positions]
            local = [max([p, lb], key=lambda t: t[1]) for p, lb in zip(ind, local)]
            global_best, global_fit = max([max(ind, key=lambda t: t[1]), (global_best, global_fit)], key=lambda t: t[1])
            if self.save:
                self.data_individuals.append(global_best)
                self.data_fit.append(float(global_fit))
                self.times.append(time.time())

        print("PSO best solution:", global_best)
        print("Whose fitness is:", global_fit)

        if self.save:
            self.times = np.array(self.times)
            self.times = list(self.times - self.times[0])
