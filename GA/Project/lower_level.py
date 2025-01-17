import time

import numpy as np

from Instance.instance import Instance


class LowerLevel:

    def __init__(self, instance: Instance, eps, M):

        # set require grad False
        self.travel_time = instance.travel_time

        self.n_od = instance.n_od

        self.alpha = instance.alpha
        self.beta = instance.beta

        self.total_paths = instance.n_paths + 1
        self.q = np.zeros_like(self.travel_time)
        q_p = np.transpose(np.reshape(np.repeat(instance.q_p, repeats=self.n_od), (self.n_od, instance.n_paths)))
        self.q[:, :-1] = q_p
        self.q[:, -1] = instance.q_od

        self.n_users = (np.array([instance.n_users for _ in range(self.total_paths)])).T

        self.costs = np.zeros((instance.n_od, self.total_paths))

        self.eps = eps

        self.K = (self.travel_time[:, :] * (1 + self.alpha * (self.n_users[:, 0].sum() / self.q[:, :])
                                            ** self.beta)).max() + M

        self.m_new = np.zeros_like(self.q)
        self.m_old = np.zeros_like(self.q)

        self.p_old = None

    def compute_probs(self, T):
        self.costs[:, : -1] = T

        # initial probabilities
        p_old = np.ones((self.n_od, self.total_paths)) / self.total_paths
        p_new = p_old

        # payoff we want to maximize
        prod = self.n_users * p_new
        self.m_old = (self.K - self.travel_time * (
                1 + self.alpha * (np.reshape(np.repeat(prod.sum(axis=0), repeats=self.n_od),
                                             (self.n_od, self.total_paths)) / self.q) ** self.beta) - self.costs)
        self.m_old[:, -1] = self.K - self.travel_time[:, -1] * (
                                1 + self.alpha * (prod[:, -1] / self.q[:, -1]) ** self.beta)
        self.m_new = self.m_old

        star = False
        iter = 0
        while (np.abs(p_old - p_new) > self.eps).any() or not star:
            p_old = p_new
            self.m_old = self.m_new
            star = True

            # average payoff
            m_average = (self.m_old * p_old).sum(axis=1)
            m_average = np.reshape(np.repeat(m_average, repeats=p_old.shape[1]), (self.n_od, self.total_paths))

            # updated probabilities
            p_new = p_old * self.m_old / m_average

            # updated payoff
            prod = self.n_users * p_new
            self.m_new = (self.K - self.travel_time * (
                    1 + self.alpha * (np.reshape(np.repeat(prod.sum(axis=0), repeats=self.n_od),
                                                 (self.n_od, self.total_paths)) / self.q) ** self.beta) - self.costs)
            self.m_new[:, -1] = self.K - self.travel_time[:, -1] * (
                        1 + self.alpha * (prod[:, -1] / self.q[:, -1]) ** self.beta)

            iter += 1
        print(p_new)
        return p_old

    def compute_fitness(self, probs):
        fitness = (self.costs[:, :-1] * probs[:, :-1] * self.n_users[:, :-1]).sum()
        return fitness

    def eval(self, T):
        probs = self.compute_probs(T)
        fit = self.compute_fitness(probs)
        return fit
