import time

import numpy as np
import torch

from Instance.instance import Instance


class LowerLevel:

    def __init__(self, instance: Instance, eps, device, M, reuse_p=False):

        # set require grad False
        self.reuse_p = reuse_p
        self.device = device
        self.travel_time = torch.tensor(instance.travel_time).to(self.device)

        self.n_od = instance.n_od

        self.alpha = instance.alpha
        self.beta = instance.beta

        self.total_paths = instance.n_paths + 1
        self.q = torch.zeros_like(self.travel_time).to(self.device)
        q_p = torch.repeat_interleave(torch.tensor(instance.q_p, device=self.device).unsqueeze(0),
                                      repeats=self.n_od, dim=0)
        self.q[:, :-1] = q_p
        self.q[:, -1] = torch.tensor(instance.q_od, device=self.device)

        self.n_users = torch.tensor(np.array([instance.n_users for _ in range(self.total_paths)]), device=self.device).T

        self.costs = torch.zeros((instance.n_od, self.total_paths)).to(self.device)

        self.eps = eps

        self.K = (self.travel_time[:, :] * (1 + self.alpha * (self.n_users[:, 0].sum() / self.q[:, :])
                                            ** self.beta)).max() + M

        self.m_new = torch.zeros_like(self.q)
        self.m_old = torch.zeros_like(self.q)

        if self.reuse_p:
            self.p_old = torch.ones((self.n_od, self.total_paths), device=self.device) / self.total_paths
        else:
            self.p_old = None

        self.data_individuals = []
        self.data_fit = []
        self.n_iter = []
        self.data_time = []

        self.total_time = []

    def compute_probs(self, T):
        self.data_individuals.append(T[0].detach().cpu().numpy())

        self.costs[:, : -1] = T

        # initial probabilities
        if not self.reuse_p:
            p_old = torch.ones((self.n_od, self.total_paths), device=self.device) / self.total_paths
        else:
            p_old = self.p_old

        p_new = p_old

        # payoff we want to maximize
        prod = self.n_users * p_new
        self.m_old = self.K - self.travel_time * (
                      1 + self.alpha * (torch.repeat_interleave(prod.sum(dim=0).unsqueeze(0), repeats=self.n_od, dim=0)
                                        / self.q) ** self.beta) - self.costs
        self.m_old[:, -1] = self.K - self.travel_time[:, -1] * (
                                1 + self.alpha * (prod[:, -1] / self.q[:, -1]) ** self.beta)
        self.m_new = self.m_old

        star = False
        iter = 0
        while (torch.abs(p_old - p_new) > self.eps).any() or not star:
            p_old = p_new
            self.m_old = self.m_new
            star = True

            # average payoff
            m_average = (self.m_old * p_old).sum(dim=1).unsqueeze(1)
            m_average = torch.repeat_interleave(m_average, repeats=p_old.shape[1], dim=1)

            # updated probabilities
            p_new = torch.round(p_old * self.m_old / m_average, decimals=3)
            p_new = p_new / torch.repeat_interleave(p_new.sum(dim=1).unsqueeze(1), repeats=self.total_paths, dim=1)

            # updated payoff
            prod = self.n_users * p_new
            self.m_new = self.K - self.travel_time * (
                          1 + self.alpha * (torch.repeat_interleave(prod.sum(dim=0).unsqueeze(0), repeats=self.n_od,
                                                                    dim=0) / self.q) ** self.beta) - self.costs
            self.m_new[:, -1] = self.K - self.travel_time[:, -1] * (
                        1 + self.alpha * (prod[:, -1] / self.q[:, -1]) ** self.beta)

            iter += 1
        self.n_iter.append(iter)

        if self.reuse_p:
            self.p_old = p_new
        return p_old

    def compute_fitness(self, probs):
        fitness = (self.costs[:, :-1] * probs[:, :-1] * self.n_users[:, :-1]).sum()
        self.data_fit.append(float(torch.abs(fitness)))
        return fitness

    def eval(self, T):
        t = time.time()

        probs = self.compute_probs(T)
        fit = self.compute_fitness(probs)

        self.data_time.append(time.time() - t)
        self.total_time.append(time.time())
        return fit
