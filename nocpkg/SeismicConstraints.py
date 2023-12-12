import numpy as np

from nocpkg.Seismic import Seismic
from nocpkg.utils import NOCError


class SeismicConstraints:
    """
    Class gathering seismic constraints of the optimization
    """
    matching = 'frequency'
    n = np.empty(0, dtype=np.int32)
    l = np.empty(0, dtype=np.int32)
    nu = np.empty(0)
    sigma = np.empty(0)
    y = np.empty(0)
    yn = np.empty(0)
    covar = np.empty(0)
    coef = np.empty(0)
    number = 0

    def __init__(self, file='', types=None, matching='frequency', data=None, lfirst=False):
        if types is None:
            types = ['nu']
        self.file = file
        self.types = types
        if matching not in ['frequency', 'order', 'continuous_frequency', 'continuous_order']:
            raise NOCError(f"Error in noc.SeismicConstraints: unhandled option matching : {matching}")
        self.matching = matching
        self.data = data
        self.lfirst = lfirst

        if file != '':
            print("Reading table of frequencies: " + file)
            self.data = np.loadtxt(file, comments='#')

        if self.data is None:
            return

        self.seismic = Seismic(self.data, lfirst=self.lfirst)

        self.n = self.seismic.n
        self.l = self.seismic.l
        self.nu = self.seismic.nu
        self.sigma = self.seismic.sigma
        # print(np.shape(self.sigma), self.number)
        self.lval = self.seismic.lval

        self.print_freqs()

        self.seismic.get_constraints(self.types)
        self.lsep_target = self.seismic.lsep_target
        self.number = self.seismic.number
        self.y = self.seismic.y
        self.covar = self.seismic.covar
        self.yn = self.seismic.yn
        self.print_constraints()

    def print_freqs(self):
        print(f"Total number of frequencies: {len(self.nu):d}")
        print('n\tl\tnu\t\tsigma')
        for i in range(len(self.nu)):
            print(f"{self.n[i]:d}\t{self.l[i]:d}\t{self.nu[i]:f}\t{self.sigma[i]:f}")

    def print_constraints(self):
        print(f"Type(s) of seismic constraints:")
        for t in self.types:
            print(t)
        print(f"Total number of seismic constraints: {self.number}")
        print("Seismic constraints (# = value sigma): ")
        for i in range(self.number):
            print(f"{i:2d} = {self.y[i]:8g} {np.sqrt(self.covar[i, i]):8g}")
