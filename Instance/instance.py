import numpy as np


class Instance:

    def __init__(self, n_paths, n_od, n_users=None, tfp_costs=None, scale_factor=100, users_range=(1, 9)):
        self.n_paths = n_paths
        self.n_od = n_od
        self.scale_factor = scale_factor
        self.users_range = users_range
        self.n_users = n_users if n_users is not None \
            else np.random.randint(self.users_range[0], self.users_range[1], self.n_od) * self.scale_factor
        self.tfp_costs = tfp_costs if tfp_costs is not None else np.random.uniform(size=self.n_od)*self.scale_factor

    def print_instance(self):
        print(self.n_paths)

    def __repr__(self):
        return "paths: " + str(self.n_paths) + "   users: " + str(self.n_od)
