import numpy as np
import random

np.random.seed(1231)
random.seed(1231)

__all__ = ['MOG_1D', 'MOG_2D']

class MOG_1D(object):
    def __init__(self, rng, std):
        n = 3

        centers = [-4.0, 0.0, 4.0]

        p = [1./n for _ in range(n)]

        self.p = p
        self.size = 1
        self.n = n
        self.std = std
        self.centers = np.reshape(np.array(centers), [-1, 1])
        self.rng = rng

        self.num_samples = 5000
        ith_center = rng.choice(n, self.num_samples, p=self.p)
        sample_centers = self.centers[ith_center]
        self.sample_points = rng.normal(loc=sample_centers, scale=std)

    # switch to random distribution (harder)
    def random_distribution(self, p=None):
        if p is None:
            p = [self.rng.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    # switch to uniform distribution
    def uniform_distribution(self):
        p = [1./self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N):
        index = self.rng.permutation(self.num_samples)
        return self.sample_points[index[:N]]


class MOG_2D(object):
    def __init__(self, data, rng, std):
        n = 9
        if data == 'grid':
            centers_x = [-3.0, 0, 3.0, -3.0, 0, 3.0, -3.0, 0, 3.0]
            centers_y = [-3.0, -3.0, -3.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0]
        elif data == 'circle':
            centers_x, centers_y = [0.0], [0.0]
            for i in range(3):
                centers_x.append(2.5 * np.cos(i*np.pi*2/3.0))
                centers_y.append(2.5 * np.sin(i*np.pi*2/3.0))
            for i in range(5):
                centers_x.append(5.0 * np.cos(i*np.pi*2/5.0))
                centers_y.append(5.0 * np.sin(i*np.pi*2/5.0))
        elif data == 'circle2':
            centers_x, centers_y = [], []
            for i in range(n):
                centers_x.append(5.0 * np.cos(i*np.pi*2.0/n))
                centers_y.append(5.0 * np.sin(i*np.pi*2.0/n))
        else:
            raise NotImplementedError

        centers_x = np.expand_dims(np.array(centers_x), 1)
        centers_y = np.expand_dims(np.array(centers_y), 1)

        p = [1./n for _ in range(n)]

        self.p = p
        self.size = 2
        self.n = n
        self.std = std
        self.rng = rng
        self.centers = np.concatenate([centers_x, centers_y], 1)

        self.num_samples = 5000
        ith_center = rng.choice(n, self.num_samples, p=self.p)
        sample_centers = self.centers[ith_center]
        self.sample_points = rng.normal(loc=sample_centers, scale=std)

    # switch to random distribution (harder)
    def random_distribution(self, p=None):
        if p is None:
            p = [self.rng.uniform() for _ in range(self.n)]
            p = p / np.sum(p)
        self.p = p

    # switch to uniform distribution
    def uniform_distribution(self):
        p = [1./self.n for _ in range(self.n)]
        self.p = p

    def sample(self, N):
        index = self.rng.permutation(self.num_samples)
        return self.sample_points[index[:N]]
