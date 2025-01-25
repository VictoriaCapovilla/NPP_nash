import time

import numpy as np
import torch

from Instance.instance import Instance


class LowerTorch:

    def __init__(self, instance: Instance, eps, mat_size, device, M, save):

        self.save = save
        self.device = device

        # network data
        self.n_od = instance.n_od
        self.total_paths = instance.n_paths + 1
        self.n_users = torch.tensor(np.array([instance.n_users for _ in range(self.total_paths)]), device=self.device).T

        self.mat_size = mat_size
        self.travel_time = torch.tensor(instance.travel_time).to(self.device)
        self.costs = torch.zeros((instance.n_od, self.total_paths)).to(self.device)

        # capacity constraints
        self.q = torch.zeros_like(self.travel_time).to(self.device)
        q_p = torch.repeat_interleave(torch.tensor(instance.q_p, device=self.device).unsqueeze(0),
                                      repeats=self.n_od, dim=0)
        self.q[:, :-1] = q_p
        self.q[:, -1] = torch.tensor(instance.q_od, device=self.device)

        # adding a third dimension to the matrixes
        self.travel_time = torch.repeat_interleave(self.travel_time.unsqueeze(0), repeats=mat_size, dim=0)
        self.q = torch.repeat_interleave(self.q.unsqueeze(0), repeats=mat_size, dim=0)

        self.n_users = torch.repeat_interleave(self.n_users.unsqueeze(0), repeats=mat_size, dim=0)

        self.costs = torch.repeat_interleave(self.costs.unsqueeze(0), repeats=mat_size, dim=0)

        # parameters
        self.alpha = instance.alpha
        self.beta = instance.beta
        self.eps = eps

        self.K = (self.travel_time[0, :, :] * (1 + self.alpha * (self.n_users[0, :, 0].sum() / self.q[0, :, :])
                                               ** self.beta)).max() + M

        # payoff matrix
        self.m_new = torch.zeros_like(self.q)
        self.m_old = torch.zeros_like(self.q)

        self.p_old = None

        if self.save:
            self.n_iter = []
            self.data_payoffs = []
            self.data_probs = []
            self.data_time = []
            self.total_time = []

    def compute_probs(self, T):
        T = torch.repeat_interleave(T.unsqueeze(1), repeats=self.n_od, dim=1)

        self.costs[:, :, : -1] = T

        # initial probabilities
        p_old = torch.ones((self.n_od, self.total_paths), device=self.device) / self.total_paths
        p_old = torch.repeat_interleave(p_old.unsqueeze(0), repeats=self.mat_size, dim=0)

        p_new = p_old

        # payoff we want to maximize
        prod = self.n_users * p_new
        self.m_old = self.K - self.travel_time * (
                      1 + self.alpha * (torch.repeat_interleave(prod.sum(dim=1).unsqueeze(1), repeats=self.n_od, dim=1)
                                        / self.q) ** self.beta) - self.costs
        self.m_old[:, :, -1] = self.K - self.travel_time[:, :, -1] * (
                                1 + self.alpha * (prod[:, :, -1] / self.q[:, :, -1]) ** self.beta)
        self.m_new = self.m_old

        # updating probabilities
        star = False
        iter = 0
        while (torch.abs(p_old - p_new) > self.eps).any() or not star:
            p_old = p_new
            self.m_old = self.m_new
            star = True

            # average payoff
            m_average = (self.m_old * p_old).sum(dim=2).unsqueeze(2)
            m_average = torch.repeat_interleave(m_average, repeats=p_old.shape[2], dim=2)

            # updated probabilities
            p_new = p_old * self.m_old / m_average

            # users flow on each path
            prod = self.n_users * p_new

            # updated payoff
            self.m_new = self.K - self.travel_time * (
                          1 + self.alpha * (torch.repeat_interleave(prod.sum(dim=1).unsqueeze(1), repeats=self.n_od,
                                                                    dim=1) / self.q) ** self.beta) - self.costs
            self.m_new[:, :, -1] = self.K - self.travel_time[:, :, -1] * (
                        1 + self.alpha * (prod[:, :, -1] / self.q[:, :, -1]) ** self.beta)

            iter += 1

        if self.save:
            self.n_iter.append(iter)
            self.data_payoffs.append(self.m_new[0].detach().cpu().numpy())

        return p_old

    def compute_fitness(self, probs):
        fitness = (self.costs[:, :, :-1] * probs[:, :, :-1] * self.n_users[:, :, :-1]).sum(dim=(1, 2))
        return fitness

    def eval(self, T):
        t = time.time()

        probs = self.compute_probs(T)
        fit = self.compute_fitness(probs)

        if self.save:
            self.data_probs.append(probs[0].detach().cpu().numpy())
            self.data_time.append(time.time() - t)
            self.total_time.append(time.time())
        return fit
