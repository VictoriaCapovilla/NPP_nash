import numpy as np


class Instance:

    def __init__(self, n_paths, n_od, users_range=(50, 200),
                 od_transfer_range=(20, 30),  p_transfer_range=(5, 10), alpha=0.4, beta=0.8):
        # network data
        self.n_paths = n_paths
        self.n_od = n_od
        self.users_range = users_range
        self.n_users = np.random.randint(self.users_range[0], self.users_range[1], self.n_od)

        # users travel time in unconstrained conditions
        self.travel_time = (
            np.random.uniform(p_transfer_range[0], p_transfer_range[1], size=(self.n_od, self.n_paths + 1)))
        self.travel_time[:, -1] = np.random.uniform(od_transfer_range[0], od_transfer_range[1], size=self.n_od)

        # capacity constraints on toll-free paths and toll paths
        self.q_od = np.random.randint(self.users_range[0], self.users_range[1], self.n_od)
        self.q_p = np.random.randint(self.users_range[0], self.users_range[1], self.n_paths)

        # parameters
        self.alpha = alpha
        self.beta = beta

    def __repr__(self):
        return "paths: " + str(self.n_paths) + "   users: " + str(self.n_od)
