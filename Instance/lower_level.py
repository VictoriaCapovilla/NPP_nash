import numpy as np

from Instance.instance import Instance


class Lower:

    def __init__(self, instance: Instance, eps):
        self.tfp_costs = instance.tfp_costs
        self.n_od = instance.n_od
        self.total_paths = instance.n_paths + 1
        self.n_users = np.array([instance.n_users for _ in range(self.total_paths)]).T
        self.costs = np.zeros((instance.n_od, instance.n_paths + 1))
        self.costs[:, -1] = instance.tfp_costs
        self.K = (self.tfp_costs + self.n_users[:, 0]).max() + self.n_users[:, 0].sum()
        self.eps = eps

    def compute_probs(self, T):
        for i in range(self.n_od):
            self.costs[i, : -1] = T

        # initial probabilities
        p_old = np.ones((self.n_od, self.total_paths)) / self.total_paths
        p_new = p_old

        # payoff we want to maximize
        # note that toll-free paths payoffs differ bcs the initial costs are different bwn ODs:
        prod = self.n_users * p_old
        m_old = self.K - self.costs - prod.sum(axis=0)
        m_old[:, -1] = self.K - self.costs[:, -1] - prod[:, -1]
        m_new = m_old

        star = False

        while (np.abs(m_old - m_new) > self.eps).any() or not star:
            p_old = p_new
            m_old = m_new
            star = True

            # average payoff
            m_average = (m_old * p_old).sum(axis=1)

            # updated probabilities
            for k in range(self.n_od):
                p_new[k] = p_old[k]*m_old[k]/m_average[k]
            p_old = p_new

            # updated payoff
            prod = self.n_users * p_old
            m_new = self.K - self.costs - prod.sum(axis=0)
            m_new[:, -1] = self.K - self.costs[:, -1] - prod[:, -1]

        return p_old

    def compute_fitness(self, probs):
        fitness = (self.costs[:, :-1] * probs[:, :-1] * self.n_users[:, :-1]).sum()
        return fitness

    def eval(self, T):
        probs = self.compute_probs(T)
        return self.compute_fitness(probs), probs
