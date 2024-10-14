import numpy as np
import torch

from Instance.instance import Instance


class LowerTorch:

    def __init__(self, instance: Instance, eps, parameters, mat_size, device, gamma=None):

        # set require grad False
        self.device = device
        self.mat_size = mat_size
        self.tfp_costs = torch.tensor(instance.tfp_costs).to(self.device)
        self.n_od = instance.n_od
        self.total_paths = instance.n_paths + 1

        self.n_users = torch.tensor(np.array([instance.n_users for _ in range(self.total_paths)]), device=self.device).T

        self.K = (self.tfp_costs + self.n_users[:, 0]).max() + self.n_users[:, 0].sum()
        self.n_users = torch.repeat_interleave(self.n_users.unsqueeze(0), repeats=mat_size, dim=0)

        self.costs = torch.zeros((instance.n_od, self.total_paths)).to(self.device)
        self.costs[:, -1] = torch.tensor(instance.tfp_costs).to(self.device)
        self.costs = torch.repeat_interleave(self.costs.unsqueeze(0), repeats=mat_size, dim=0)

        self.eps = eps
        self.parameters = parameters

    def function(self, parameters, p):
        prod = self.n_users * p
        if self.parameters is None:
            m = self.K - self.costs - torch.repeat_interleave(prod.sum(dim=1).unsqueeze(1), repeats=self.n_od, dim=1)
            m[:, :, -1] = self.K - self.costs[:, :, -1] - prod[:, :, -1]
        else:
            alpha, beta = self.parameters

            delta = torch.ones((self.n_od, self.total_paths)).to(self.device)
            delta[:, -1] = 0
            delta = torch.repeat_interleave(delta.unsqueeze(0), repeats=self.mat_size, dim=0)

            # gamma

            # tr = travel time in unconstrained conditions

            q_a = torch.repeat_interleave(self.n_users.sum(dim=1).unsqueeze(1), repeats=self.n_od, dim=1)
            q_a[:, :, -1] = self.n_users[:, :, -1]

            m = self.K - alpha * (torch.repeat_interleave(prod.sum(dim=1).unsqueeze(1), repeats=self.n_od, dim=1) /
                                  q_a) ** beta - delta * self.costs
            m[:, :, -1] = (self.K - alpha * (prod[:, :, -1] / q_a[:, :, -1]) ** beta -
                           delta[:, :, -1] * self.costs[:, :, -1])
        return m

    def compute_probs(self, T):
        # T = torch.tensor(T).to(self.device).unsqueeze(1).to(self.device)
        T = torch.repeat_interleave(T.unsqueeze(1), repeats=self.n_od, dim=1)

        self.costs[:, :, : -1] = T

        # initial probabilities
        p_old = torch.ones((self.n_od, self.total_paths), device=self.device) / self.total_paths
        p_old = torch.repeat_interleave(p_old.unsqueeze(0), repeats=self.mat_size, dim=0)
        p_new = p_old

        # payoff we want to maximize
        m_old = self.function(self.parameters, p_old)
        m_new = m_old

        star = False
        iter = 0
        while (torch.abs(p_old - p_new) > self.eps).any() or not star:
            p_old = p_new
            m_old = m_new
            star = True

            # average payoff
            m_average = (m_old * p_old).sum(dim=2).unsqueeze(2)
            m_average = torch.repeat_interleave(m_average, repeats=p_old.shape[2], dim=2)

            # updated probabilities
            p_new = p_old * m_old / m_average

            # updated payoff
            m_new = self.function(self.parameters, p_new)
            iter += 1
        # print(iter)

        return p_old

    def compute_fitness(self, probs):
        fitness = (self.costs[:, :, :-1] * probs[:, :, :-1] * self.n_users[:, :, :-1]).sum(dim=(1, 2))
        return fitness

    def eval(self, T):
        probs = self.compute_probs(T)
        return self.compute_fitness(probs)
