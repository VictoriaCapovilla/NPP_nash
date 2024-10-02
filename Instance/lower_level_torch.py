import numpy as np
import torch

from Instance.instance import Instance


class LowerTorch:

    def __init__(self, instance: Instance, eps, mat_size):


        # set require grad False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mat_size = mat_size
        self.tfp_costs = torch.tensor(instance.tfp_costs).to(self.device)
        self.n_od = instance.n_od
        self.total_paths = instance.n_paths + 1

        self.n_users = torch.tensor(np.array([instance.n_users for _ in range(self.total_paths)]), device=self.device).T

        self.K = (self.tfp_costs + self.n_users[:, 0]).max() + self.n_users[:, 0].sum()
        self.n_users = torch.repeat_interleave(self.n_users.unsqueeze(0), repeats=mat_size, dim=0)

        self.costs = torch.zeros((instance.n_od, instance.n_paths + 1)).to(self.device)
        self.costs[:, -1] = torch.tensor(instance.tfp_costs).to(self.device)
        self.costs = torch.repeat_interleave(self.costs.unsqueeze(0), repeats=mat_size, dim=0)


        self.eps = eps

    def compute_probs(self, T):
        T = torch.tensor(T).to(self.device).unsqueeze(1).to(self.device)
        T = torch.repeat_interleave(T, repeats=self.n_od, dim=1)

        self.costs[:, :, : -1] = T

        # initial probabilities
        p_old = torch.ones((self.n_od, self.total_paths), device=self.device) / self.total_paths
        p_old = torch.repeat_interleave(p_old.unsqueeze(0), repeats=self.mat_size, dim=0)
        p_new = p_old

        # payoff we want to maximize
        prod = self.n_users * p_old
        # a = prod.sum(dim=1).unsqueeze(1)
        # a = torch.repeat_interleave(a, repeats=self.n_od, dim=1)
        # m_old = self.K - self.costs - a

        m_old = self.K - self.costs - torch.repeat_interleave(prod.sum(dim=1).unsqueeze(1), repeats=self.n_od, dim=1)
        m_old[:, :, -1] = self.K - self.costs[:, :, -1] - prod[:, :, -1]  # toll-free paths payoffs: they differ bcs the toll-free paths are different bwn ODs
        m_new = m_old

        star = False

        while (torch.abs(m_old - m_new) > self.eps).any() or not star:
            p_old = p_new
            m_old = m_new
            star = True

            # average payoff
            m_average = (m_old * p_old).sum(dim=2).unsqueeze(2)
            m_average = torch.repeat_interleave(m_average, repeats=p_old.shape[2], dim=2)
            # updated probabilities
            p_new = p_old * m_old / m_average
            p_old = p_new

            # updated payoff
            prod = self.n_users * p_old

            a = prod.sum(dim=1).unsqueeze(1)
            a = torch.repeat_interleave(a, repeats=self.n_od, dim=1)
            m_new = self.K - self.costs - a
            m_new[:, :, -1] = self.K - self.costs[:, :, -1] - prod[:, :,
                                                              -1]  # toll-free paths payoffs: they differ bcs the toll-free paths are different bwn ODs
            m_new = m_old

        return p_old

    def compute_fitness(self, probs):
        fitness = (self.costs[:, :, :-1] * probs[:, :, :-1] * self.n_users[:, :, :-1]).sum(dim=(1, 2))
        return fitness

    def eval(self, T):
        probs = self.compute_probs(T)
        return self.compute_fitness(probs)
