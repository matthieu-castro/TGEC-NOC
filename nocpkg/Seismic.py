import numpy as np

from nocpkg.utils import NOCError


class Seismic:
    """
    Calculation of the different seismic constraints from the frequencies table.
    """
    n = np.empty(0, dtype=np.int)
    l = np.empty(0, dtype=np.int)
    lval = np.empty(0, dtype=np.int)
    nu = np.empty(0)
    sigma = np.empty(0)
    y = np.empty(0)
    yn = np.empty(0)
    covar = np.empty(0)
    coef = np.empty(0)
    number = 0

    def __init__(self, modes, lfirst=False):
        self.lfirst = lfirst
        self.modes = modes
        self.nb_f = self.modes[:, 0].size
        self.process_data()
        self.dict_funcs = {
            'nu': self.nuu, 'dnu': self.dnu, 'dnu0': self.dnu,
            'dnu1': self.dnu, 'dnu2': self.dnu, 'd01': self.d01,
            'd02': self.d02, 'rd02': self.rd02, 'sd': self.sd,
            'sd01': self.sd01, 'sd10': self.sd10, 'rsd10': self.rsd10,
            'rsd01': self.rsd01
        }
        self.lsep_target = None

        if self.nu.size < 1:
            raise NOCError("Error in NOC / SeismicConstraints: more than one frequency is required")

    def process_data(self):
        if self.lfirst:
            self.l = np.array(self.modes[:, 0], dtype=int)
            self.n = np.array(self.modes[:, 1], dtype=int)
        else:
            self.n = np.array(self.modes[:, 0], dtype=int)
            self.l = np.array(self.modes[:, 1], dtype=int)

        self.nu = self.modes[:, 2]
        self.sigma = self.modes[:, 3]
        self.lval = self.get_l_values(self.modes)

    def set_data(self, data):
        self.n = data[:, 0]
        self.l = data[:, 1]
        self.nu = data[:, 2]
        self.sigma = data[:, 3]
        self.nb_f = data[:, 0].size
        self.lval = self.get_l_values(data)

    def get_l_values(self, modes):
        """
        Get different l values in the frequencies data  
        :param modes: Mode characteristics (n, l, freq, dfreq)
        :return: l values 
        """
        nb_modes = modes.shape[0]
        if nb_modes > 0:
            lval = np.unique(modes[:, 1])
        else:
            lval = []

        return lval

    def get_constraints(self, types, source='file'):
        """
        Calculate the seismic constraints for observed frequencies (source='file')
        and theoretical frequencies (source='model')
        :param types: types of constraints to be calculated
        :param source: source of the frequencies (file or model)
        """
        self.y = np.array([])
        self.yn = np.array([])
        self.coef = np.empty((0, self.nu.size))
        for t in types:
            if t in ['nu', 'dnu', 'd01', 'd02', 'rd02', 'sd', 'sd01', 'sd10', 'rsd10', 'rsd01']:
                self.apply_func(self.dict_funcs[t])
            elif t in ['dnu0', 'dnu1', 'dnu2']:
                n = len(self.y)
                if source == 'file':
                    lval = self.lval[int(t[-1])]
                elif source == 'model':
                    lval = [int(t[-1])]
                self.apply_func(self.dict_funcs[t], param=lval)

                if t == 'dnu0':
                    self.lsep_target = np.mean(self.y[n:])
            else:
                raise NOCError(f"Error in noc.Seismic.get_constraints: seismic constraints of type {t} is not handled.")

        # Calculation of the covariant matrix
        self.number = self.coef.shape[0]
        self.covar = np.empty((self.number, self.number))
        for i in range(self.number):
            for j in range(self.number):
                self.covar[i, j] = (self.coef[i, :] * self.coef[j, :] * self.sigma ** 2).sum()

        # print("Theoretical constraints:")
        # print(self.y)
        # Eliminate zero elements
        non_zero = np.nonzero(self.y)
        self.y = self.y[non_zero]
        self.yn = self.yn[non_zero]

        # For covar, we keep only lines with at least one non-zero element
        self.covar = self.covar[np.any(self.covar != 0, axis=1)]
        self.covar = self.covar[:, np.any(self.covar != 0, axis=0)]

    def apply_func(self, func, param=None):
        result = func(lval=param)
        y, coef, yn = result[0:3]

        self.y = np.append(self.y, y)
        self.yn = np.append(self.yn, yn)
        self.coef = np.append(self.coef, coef, axis=0)

    def nuu(self, *args, **kwargs):
        """
        Return the frequencies as seismic constraints
        :return: y, coef, yn = nu, identity matrix, n
        """
        coef = np.identity(self.nu.size)

        return self.nu, coef, self.n

    def dnu(self, *args, **kwargs):
        """
        Calculate the large separation Delta_nu (n) = nu_{n,l} - nu_{n-1,l}
        :param args:
        :param kwargs: keyword arguments
        :return: y, covar, yn
        """
        if kwargs.get('modes', None) is not None:
            modes = kwargs.get('modes')
            n = modes[:, 0]
            l = modes[:, 1]
            nu = modes[:, 2]
            nb_f = np.size(n)
            lval = self.get_l_values(modes)
        else:
            n = self.n
            l = self.l
            nu = self.nu
            nb_f = self.nb_f
            if kwargs.get('lval', None) is not None:
                lval = np.array([kwargs.get('lval')], dtype=int)
            else:
                lval = self.lval

        # Number of modes for each l degree
        nb_l = []
        nb = 0
        for s in lval:
            j = np.where(l == s)
            m = len(j[0])
            nb_l.append(m)
            nb += m - 1

        coef = np.zeros((nb, nb_f))
        yn = np.empty(nb)
        p = 0
        # for each l value, if there is more than one frequency, we verify for each n if it exists a frequency
        # with (n-1) and build the coef matrix with a -1 for the frequency (n-1) and 1 with the frequency n.
        for k in range(len(lval)):
            if nb_l[k] > 1:
                i = np.where(l == lval[k])[0]
                for m in i[1:]:
                    for j in range(nb_f):
                        # print('n['+str(j)+']='+str(n[j])+' , n['+str(m)+']='+str(n[m]))
                        # print('l['+str(j)+']='+str(l[j])+' , lval['+str(k)+']='+str(lval[k]))
                        if n[j] == n[m] and l[j] == lval[k]:
                            coef[p, j] = 1.
                            yn[p] = n[j]
                        if n[j] == n[m] - 1 and l[j] == lval[k]:
                            coef[p, j] = -1.
                    p += 1

        for i in range(np.shape(coef)[0]):
            if np.count_nonzero(coef[i, :]) != 2:
                coef[i, :] = 0.0

        y = np.empty(nb)
        for i in range(nb):
            y[i] = (coef[i, :] * nu).sum()

        return y, coef, yn

    def d01(self):
        """
        Calculate the small separations $d01 = nu_{n,0} - \frac{nu_{n-1,1} + nu_{n,1}}{2}$
        :return:  y, coef, yn
            y : d01 [in muHz] as a function of frequency
        """
        # Index arrays where l=0 and l=1
        l0 = np.where(self.l == 0)[0]
        l1 = np.where(self.l == 1)[0]

        # Verify that l=0 stands before l=1 in data
        if self.l[l0[0]] != 0 and self.l[l1[0]] != 1:
            raise NOCError('Error in noc.Seismic.d01: '
                           'the l=1 modes must be placed after the l=0 modes')

        n0min = np.min(self.n[l0])  # Minimal n for l=0
        n0max = np.max(self.n[l0])  # Maximal n for l=0
        n1min = np.min(self.n[l1])  # Minimal n for l=1
        n1max = np.max(self.n[l1])  # Maximal n for l=1

        n0 = int(max([n0min, n1min + 1]))
        n1 = int(min([n0max, n1max]))

        l00 = l0[np.argmin(np.abs(self.n[l0] - n0))]
        l01 = l0[np.argmin(np.abs(self.n[l0] - n1))]

        N = n1 - n0 + 1
        yn = np.empty(N)
        coef = np.zeros((N, self.nb_f))
        for i in range(l00, l01 + 1):
            for j in range(self.nb_f):
                if self.n[j] == self.n[i] and self.l[j] == 0:
                    coef[i - l00, j] = 1.
                    yn[i - l00] = self.n[j]
                elif self.n[j] in [self.n[i] - 1, self.n[i]] and self.l[j] == 1:
                    coef[i - l00, j] = -0.5

        y = np.empty(N)
        for i in range(N):
            y[i] = (coef[i, :] * self.nu).sum()

        return y, coef, yn

    def d02(self, **kwargs):
        """
        Calculate the small separations $d02 = nu_{n,0} - nu_{n-1,2}$
        :return: y,coef,yn:
         y : d02 [in muHz] as a function of frequency
        """

        l0 = np.where(self.l == 0)[0]
        l2 = np.where(self.l == 2)[0]

        if self.l[l0[0]] != 0 and self.l[l2[0]] != 2:
            raise NOCError('Error in noc.Seismic.d02: \
                the l=2 modes must be placed after the l=0 modes')

        n0min = np.min(self.n[l0])
        n0max = np.max(self.n[l0])
        n2min = np.min(self.n[l2])
        n2max = np.max(self.n[l2])

        n0 = int(max([n0min, n2min + 1]))
        n2 = int(min([n0max, n2max + 1]))

        N = n2 - n0 + 1
        yn = np.empty(N)
        coef = np.zeros((N, self.nb_f))
        for i in range(N):
            for j in range(self.nb_f):
                if self.n[j] == i + n0 and self.l[j] == 0:
                    coef[i, j] = 1.0
                    yn[i] = self.n[j]
                elif self.n[j] == i + n0 - 1 and self.l[j] == 2:
                    coef[i, j] = -1.0
            if np.count_nonzero(coef[i, :]) != 2:
                coef[i, :] = 0.0

        y = np.empty(N)
        for i in range(N):
            y[i] = (coef[i, :] * self.nu).sum()

        return y, coef, yn

    def sd(self, lval=None):
        """
        Calculate the second difference as a function of frequency (Gough 1990, Houdek & Gough 2007)
        $sd = nu_{n-1,l} - 2 nu_{n,l} + nu_{n+1,l}$

        :return: y,coef,yn:
         y : second difference [in muHz] as a function of frequency
        """

        # number of modes for each l degree
        nb_l = []
        N = 0
        for s in self.lval:
            j = np.where(self.l == s)
            m = len(j[0])
            nb_l.append(m)
            N += m - 2

        coef = np.zeros((N, self.nb_f))
        yn = np.empty(N)
        p = 0
        for k in range(len(self.lval)):
            if nb_l[k] > 1:
                i = np.where(self.l == self.lval[k])[0]
                for m in i[1:-1]:
                    for j in range(self.nb_f):
                        if self.n[j] == self.n[m] and self.l[j] == self.lval[k]:
                            coef[p, j] = -2.0
                            yn[p] = self.n[j]
                        elif self.n[j] == self.n[m] - 1 and self.l[j] == self.lval[k]:
                            coef[p, j] = 1.0
                        elif self.n[j] == self.n[m] + 1 and self.l[j] == self.lval[k]:
                            coef[p, j] = 1.0
                    p += 1

        y = np.empty(N)
        for i in range(N):
            y[i] = (coef[i, :] * self.nu).sum()

        return y, coef, yn

    def sd01(self, **kwargs):
        """
        Calculate the second difference 01 as a function of frequency
        as defined in Eq. (4) of Roxburgh & Vorontsov (2003,A&A)
        $sd01 (n) = \frac{1}{8} ( nu_{n-1,0} - 4 nu_{n-1,1} + 6 nu_{n,0} - 4 nu_{n,1} + nu_{n+1,0})$

        :return: y,coef,yn:
         y : sd01 second difference [in muHz]
        """

        i0 = np.where(self.l == 0)[0]
        i1 = np.where(self.l == 1)[0]

        if self.l[i0[0]] != 0 and self.l[i1[0]] != 2:
            raise NOCError('Error in noc.Seismic.sd01: \
                the l=1 modes must be placed after the l=0 modes')

        n0min = np.min(self.n[i0])
        n0max = np.max(self.n[i0])
        n1min = np.min(self.n[i1])
        n1max = np.max(self.n[i1])

        n0 = int(max([n0min, n1min])) + 1
        n1 = int(min([n0max, n1max])) - 1

        N = n1 - n0 + 1
        if N < 1:
            raise NOCError('Error in noc.Seismic.sd01: \
                not enough l=0 and/or l=1 modes')

        yn = np.empty(N)
        coef = np.zeros((N, self.nb_f))
        for i in range(N):
            for j in range(self.nb_f):
                if self.n[j] == i + n0 - 1 and self.l[j] == 0:
                    coef[i, j] = 1.0 / 8
                elif self.n[j] == i + n0 and self.l[j] == 0:
                    coef[i, j] = 0.75
                    yn[i] = self.n[j]
                elif self.n[j] == i + n0 + 1 and self.l[j] == 0:
                    coef[i, j] = 1.0 / 8
                elif self.n[j] == i + n0 - 1 and self.l[j] == 1:
                    coef[i, j] = -0.5
                elif self.n[j] == i + n0 and self.l[j] == 1:
                    coef[i, j] = -0.5

        y = np.empty(N)
        for i in range(N):
            y[i] = (coef[i, :] * self.nu).sum()

        return y, coef, yn

    def sd10(self, **kwargs):
        """
        Calculate the second difference "10" as a function of frequency
        as defined in Eq. (5) of Roxburgh & Vorontsov (2003,A&A)
        $sd10 (n) = -\frac{1}{8} ( nu_{n-1,1} - 4 nu_{n,0} + 6 nu_{n,1} - 4 nu_{n+1,0} + nu_{n+1,1})$

        :return: y,coef,yn:
         y : sd10 second difference [in muHz]
        """

        i0 = np.where(self.l == 0)[0]
        i1 = np.where(self.l == 1)[0]

        if self.l[i0[0]] != 0 and self.l[i1[0]] != 2:
            raise NOCError('Error in noc.Seismic.sd10: \
                the l=1 modes must be placed after the l=0 modes')

        n0min = np.min(self.n[i0])
        n0max = np.max(self.n[i0])
        n1min = np.min(self.n[i1])
        n1max = np.max(self.n[i1])

        n0 = int(max([n0min, n1min])) + 1
        n1 = int(min([n0max, n1max])) - 1

        N = n1 - n0 + 1
        if (N < 1):
            raise NOCError('Error in noc.Seismic.sd10: \
                not enough l=0 and/or l=1 modes')

        yn = np.empty(N)
        coef = np.zeros((N, self.nb_f))
        for i in range(N):
            for j in range(self.nb_f):
                if self.n[j] == i + n0 - 1 and self.l[j] == 1:
                    coef[i, j] = -1.0 / 8
                elif self.n[j] == i + n0 and self.l[j] == 1:
                    coef[i, j] = -0.75
                    yn[i] = self.n[j]
                elif self.n[j] == i + n0 + 1 and self.l[j] == 1:
                    coef[i, j] = -1.0 / 8
                elif self.n[j] == i + n0 and self.l[j] == 0:
                    coef[i, j] = 0.5
                elif self.n[j] == i + n0 + 1 and self.l[j] == 0:
                    coef[i, j] = 0.5

        y = np.empty(N)
        for i in range(N):
            y[i] = (coef[i, :] * self.nu).sum()

        return y, coef, yn

    def rsd01(self, **kwargs):
        """
        Calculate the ratio $sd01(n) / dnu1(n)$
        as defined in Eq. 1 of Lebreton & Goupil (2012, A&A)
        with $dnu1 = nu_{n,1} - nu_{n-1,1}$

        :return y, coefy, yn, coef, coefd
        """
        i0 = np.where(self.l == 0)[0]
        i1 = np.where(self.l == 1)[0]

        if self.l[i0[0]] != 0 and self.l[i1[0]] != 2:
            raise NOCError('Error in noc.Seismic.rsd01: \
                the l=1 modes must be placed after the l=0 modes')

        n0min = np.min(self.n[i0])
        n0max = np.max(self.n[i0])
        n1min = np.min(self.n[i1])
        n1max = np.max(self.n[i1])

        n0 = int(max([n0min, n1min])) + 1
        n1 = int(min([n0max, n1max])) - 1

        N = n1 - n0 + 1
        if N < 1:
            raise NOCError('Error in noc.Seismic.rsd01: \
                not enough l=0 and/or l=1 modes')

        yn = np.empty(N)
        coef = np.zeros((N, self.nb_f))
        coefd = np.zeros((N, self.nb_f))
        for i in range(N):
            for j in range(self.nb_f):
                # sd01
                if self.n[j] == i + n0 - 1 and self.l[j] == 0:
                    coef[i, j] = 1.0 / 8
                elif self.n[j] == i + n0 and self.l[j] == 0:
                    coef[i, j] = 0.75
                    yn[i] = self.n[j]
                elif self.n[j] == i + n0 + 1 and self.l[j] == 0:
                    coef[i, j] = 1.0 / 8
                elif self.n[j] == i + n0 - 1 and self.l[j] == 1:
                    coef[i, j] = -0.5
                elif self.n[j] == i + n0 and self.l[j] == 1:
                    coef[i, j] = -0.5
                # dnu1
                if self.n[j] == i + n0 - 1 and self.l[j] == 1:
                    coefd[i, j] = -1.
                elif self.n[j] == i + n0 and self.l[j] == 1:
                    coefd[i, j] = 1.

        y = np.empty(N)
        sd01 = np.empty(N)
        dnu1 = np.empty(N)
        coefy = np.zeros((N, self.nb_f))
        for i in range(N):
            sd01[i] = (coef[i, :] * self.nu).sum()
            dnu1[i] = (coefd[i, :] * self.nu).sum()
            y[i] = sd01[i] / dnu1[i]
            coefy[i, :] = y[i] * (coef[i, :] / sd01[i] - coefd[i, :] / dnu1[i])

        return y, coefy, yn, coef, coefd

    def rsd10(self, **kwargs):
        """
        ratio $sd10 (n) / dnu0 (n)$
        as defined in Eq. 1 of Lebreton & Goupil (2012, A&A)
        with $dnu0 =  nu_{n+1,0} - nu_{n,0}$

        :return y, coefy, yn, coef, coefd
        """
        i0 = np.where(self.l == 0)[0]
        i1 = np.where(self.l == 1)[0]

        if self.l[i0[0]] != 0 and self.l[i1[0]] != 2:
            raise NOCError('Error in noc.Seismic.rsd10: \
                the l=1 modes must be placed after the l=0 modes')

        n0min = np.min(self.n[i0])
        n0max = np.max(self.n[i0])
        n1min = np.min(self.n[i1])
        n1max = np.max(self.n[i1])

        n0 = int(max([n0min, n1min])) + 1
        n1 = int(min([n0max, n1max])) - 1

        N = n1 - n0 + 1
        if N < 1:
            raise NOCError('Error in noc.Seismic.rsd10: \
                not enough l=0 and/or l=1 modes')

        yn = np.empty(N)
        coef = np.zeros((N, self.nb_f))
        coefd = np.zeros((N, self.nb_f))
        for i in range(N):
            for j in range(self.nb_f):
                # sd10
                if self.n[j] == i + n0 - 1 and self.l[j] == 1:
                    coef[i, j] = -1.0 / 8
                elif self.n[j] == i + n0 and self.l[j] == 1:
                    coef[i, j] = -0.75
                    yn[i] = self.n[j]
                elif self.n[j] == i + n0 + 1 and self.l[j] == 1:
                    coef[i, j] = -1.0 / 8
                elif self.n[j] == i + n0 and self.l[j] == 0:
                    coef[i, j] = 0.5
                elif self.n[j] == i + n0 + 1 and self.l[j] == 0:
                    coef[i, j] = 0.5
                # dnu0
                if self.n[j] == i + n0 and self.l[j] == 0:
                    coefd[i, j] = -1.0
                elif self.n[j] == i + n0 + 1 and self.l[j] == 0:
                    coefd[i, j] = 1.0

        y = np.empty(N)
        sd10 = np.empty(N)
        dnu0 = np.empty(N)
        coefy = np.zeros((N, self.nb_f))
        for i in range(N):
            sd10[i] = (coef[i, :] * self.nu).sum()
            dnu0[i] = (coefd[i, :] * self.nu).sum()
            y[i] = sd10[i] / dnu0[i]
            coefy[i, :] = y[i] * (coef[i, :] / sd10[i] - coefd[i, :] / dnu0[i])

        return y, coefy, yn, coef, coefd

    def rd02(self, **kwargs):
        """
        Calculate the ratio $d02 (n) / dnu1 (n)$
        with $dnu1 =  nu_{n,1} - nu_{n-1,1}$

        :return y, coefy, yn, coef, coefd
        """

        i0 = np.where(self.l == 0)[0]
        i1 = np.where(self.l == 1)[0]
        i2 = np.where(self.l == 2)[0]

        if self.l[i0[0]] != 0 and self.l[i1[0]] != 1 and self.l[i2[0]] != 2:
            raise NOCError('Error in noc.Seismic.rd02: \
                the l=0, l=1 and l=2 modes must placed in increasing l degree')

        n0min = np.min(self.n[i0])
        n0max = np.max(self.n[i0])
        n1min = np.min(self.n[i1])
        n1max = np.max(self.n[i1])
        n2min = np.min(self.n[i2])
        n2max = np.max(self.n[i2])

        n0 = int(max([n0min, n1min + 1, n2min + 1]))
        n1 = int(min([n0max, n1max + 1, n2max + 1]))

        N = n1 - n0 + 1
        if N < 1:
            raise NOCError('Error in noc.Seismic.rd02: \
                not enough l=0, l=1,  and l=2 modes')

        yn = np.empty(N)
        coef = np.zeros((N, self.nb_f))
        coefd = np.zeros((N, self.nb_f))
        for i in range(N):
            for j in range(self.nb_f):
                # d02
                if self.n[j] == i + n0 and self.l[j] == 0:
                    coef[i, j] = 1.0
                if self.n[j] == i + n0 - 1 and self.l[j] == 2:
                    coef[i, j] = -1.0
                # dnu1
                if self.n[j] == i + n0 - 1 and self.l[j] == 1:
                    coefd[i, j] = -1.0
                if self.n[j] == i + n0 and self.l[j] == 1:
                    coefd[i, j] = 1.0
                    yn[i] = self.n[j]

        y = np.empty(N)
        d02 = np.empty(N)
        dnu1 = np.empty(N)
        coefy = np.zeros((N, self.nb_f))
        for i in range(N):
            d02[i] = (coef[i, :] * self.nu).sum()
            dnu1[i] = (coefd[i, :] * self.nu).sum()
            y[i] = d02[i] / dnu1[i]
            coefy[i, :] = y[i] * (coef[i, :] / d02[i] - coefd[i, :] / dnu1[i])

        return y, coefy, yn, coef, coefd
