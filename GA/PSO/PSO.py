import numpy as np
import torch

from fstpso import FuzzyPSO

from GA.PSO.lower_level import LowerTorch


class GeneticAlgorithmTorch:

    def __init__(self, instance, lower_eps=10**(-12), device=None, reuse_p=False):

        self.device = device
        self.instance = instance

        self.n_paths = self.instance.n_paths

        self.M = (self.instance.travel_time[:, -1] * (
                1 + self.instance.alpha * (self.instance.n_users / self.instance.q_od) ** self.instance.beta)).max()

        self.lower = LowerTorch(self.instance, lower_eps, device=device, M=self.M, reuse_p=reuse_p)

        self.obj_val = 0

    def fitness_evaluation(self, population):
        population = torch.repeat_interleave(torch.from_numpy(np.array(population)).unsqueeze(0),
                                             repeats=self.instance.n_od, dim=0)
        return - self.lower.eval(torch.from_numpy(np.array(population)))

    def run_PSO(self):
        dims = self.n_paths
        FP = FuzzyPSO()
        FP.set_search_space([[0, self.M]] * dims)
        FP.set_fitness(self.fitness_evaluation)
        result = FP.solve_with_fstpso(max_iter=300)
        print("Best solution:", result[0])
        self.obj_val = torch.abs(result[1]).detach().cpu().numpy()
        print("Whose fitness is:", self.obj_val)
