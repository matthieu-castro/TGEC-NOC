import os
import struct

import numpy as np
from tgec.constants import ggrav

from nocpkg.Seismic import Seismic
from nocpkg.utils import NOCError

# def fromstring(endian):
#     """
#     Handle the problems caused by endian types:
#     - Big endian: Sun, HP, etc
#     - Little endian: Intel machines
#     Giving the endian type of the file, the endian type of the machine
#     is automatically determined by the constant LittleEndian from numpy
#     The conventions are as those given in 'struct' module:
#     '@','=' : File is native (was produced on the same machine)
#     '<'     : File is little endian
#     '>','!' : File is big endian (network)
#     :param endian: endianness
#     :return: tuple
#     """
#     littleendian = False
#     try:
#         if np.little_endian:
#             littleendian = True
#     except:


class SeismicModel:

    def __init__(self, name, model, setup):
        self.name = name
        self.model = model
        self.age_model = model.age_model

        self.init_freq_files()

        self.setup = setup
        self.seismic = None
        self.modes = None
        self.y = np.array([])
        self.yn = np.array([])
        self.covar = np.array([])

    def init_freq_files(self):
        """
        Initializes parameters for frequency calculations
        """

        # self.seismic = SeismicModel(self.name, self)

        # adipls
        self.amdl = self.name + '.amdl'
        self.agsm = []
        self.ssm = []
        self.amde = self.name + '.amde'
        self.rkr = self.name + '.rkr'
        self.gkr = self.name + '.gkr'

        # pulse
        self.pulse_input = 'model.out_p'
        self.pulse_output = 'pulse.out'

    # def read_settings(self):
    #     pass

    def seismic_model(self, setup):
        self.setup = setup
        seismic_constraints = setup.seismic_constraints
        settings = setup.settings
        lsep_target = seismic_constraints.lsep_target

        modesset = settings['modes']
        oscprog = modesset['oscprog']

        if oscprog is None:
            raise NOCError("Unspecified pulsation program")
        if oscprog == 'pulse':
            path = f"{self.name}/freqs/{self.pulse_output}"
            self.read_pulse(path)
        elif oscprog == 'adipls':
            path = f"{self.name}/freqs/{self.name}.agsm"
            self.read_agsm(path)
            self.modes = self.modes[:, 0:4]
        else:
            print(f"{oscprog}: unknown oscillation program")
            return np.empty(0)

        # Adding surface effects
        if modesset['surface_effects'] is not None:
            lsep = self.largesep(lsep_target, settings, se=True)
            dnu_se = self.get_surface_effects(modesset['surface_effects'], lsep)
            self.modes[:, 2] = self.modes[:, 2] + dnu_se

        modes_n = np.empty((0, 4))

        # We check that the theoretical frequencies cover the observed ones
        lval0 = self.seismic.get_l_values(self.modes)
        # print('lval0=', end=' ')
        # print(lval0)
        # print('lval=', end=' ')
        # print(seismic_constraints.lval)
        for lc in seismic_constraints.lval:
            found = False
            for l in lval0:
                if lc == l:
                    found = True
                    j = (np.where(self.modes[:, 1] == lc))[0]
                    jc = (np.where(seismic_constraints.l == lc))[0]
                    LS = (seismic_constraints.nu[jc[1:]] - seismic_constraints.nu[jc[0:-1]]).mean()
                    # print(f"{0.5 * LS}")
                    count = 1
                    imin = 0
                    if seismic_constraints.matching == 'continuous_frequency':
                        if np.max(seismic_constraints.nu[jc]) - 0.5 * LS > np.min(self.modes[j, 2]) > \
                                np.min(seismic_constraints.nu[jc]) + 0.5 * LS:
                            print(f"Warning in SeismicModel/seismic_model: in {self.name}, l={lc} modes do not "
                                  f"cover the observational range")

                        imin = np.argmin(np.abs(self.modes[j, 2] - np.min(seismic_constraints.nu[jc])))
                    elif seismic_constraints.matching == 'continuous_order':
                        imin = np.argmin(np.abs(self.modes[j, 0] - seismic_constraints.nu[jc][0]))
                    elif seismic_constraints.matching == 'frequency':
                        count = 0
                        for k in j:
                            for i in jc:
                                if np.abs(seismic_constraints.nu[i] - self.modes[k, 2]) < 0.5 * LS:
                                    print(f"{seismic_constraints.nu[i]} - {self.modes[k, 2]}")
                                    modes_n = np.append(modes_n, [self.modes[k, :]], axis=0)
                                    count += 1
                    elif seismic_constraints.matching == 'order':
                        count = 0
                        for k in j:
                            for i in jc:
                                if not int(seismic_constraints.n[i]) - int(self.modes[k, 0]):
                                    modes_n = np.append(modes_n, [self.modes[k, :]], axis=0)
                                    count += 1
                    if count <= 0:
                        print(f"Warning in SeismicModel/seismic_model: in {self.name}, no matching between theoretical "
                              f"and observed l={lc} modes")

                    if seismic_constraints.matching in ['continuous_frequency', 'continuous_order']:
                        print(f"First l={lc} modes: n={self.modes[j[imin], 0]:0.0f}, nu={self.modes[j[imin], 2]:g}")
                        imax = imin + len(jc) - 1
                        if len(jc) > len(j) - imin:
                            print(f"Error in SeismicModel/seismic_model: in {self.name}, insufficient number of "
                                  f"l={lc}, {len(jc)} needed.")
                            return np.empty(0)
                        else:
                            print(f"Last l={lc} modes: n={self.modes[j[imax], 0]:0.0f}, nu={self.modes[j[imax], 2]:g}")
                            modes_n = np.append(modes_n, self.modes[j[imin]:j[imax] + 1, :], axis=0)

            if not found:
                print(f"Error in SeismicModel/seismic_model: in {self.name}, l={lc} modes are missing")
                return np.empty(0)

        print("Selected theoretical frequencies:")
        print("n\t l\t nu [muHz]")
        for i in range(modes_n.shape[0]):
            print(f"{modes_n[i, 0]:2.0f}\t{modes_n[i, 1]:2.0f}\t{modes_n[i, 2]:11.5f}")

        self.seismic.set_data(modes_n)
        self.seismic.get_constraints(seismic_constraints.types, source='model')
        self.y, self.covar, self.yn = self.seismic.y, self.seismic.covar, self.seismic.yn

        return self.y

    def largesep(self, lsep_target, settings, se=True):
        """
        Compute large separation of the theoretical frequencies
        :param lsep_target: value of the targeted large separations
        :param settings: settings of the optimization
        :param se: if True, large separations are calculated with surface effects corrections
        :return: calculated large separations
        """
        name = self.model.name
        modesset = settings['modes']
        oscprog = modesset['oscprog']
        if oscprog is None:
            raise NOCError("Unspecified pulsation program")
        if oscprog == 'adipls':
            path = f"{self.name}/freqs/{self.name}.agsm"
            self.read_agsm(path)
            self.modes = self.modes[:, 0:4]
        elif oscprog == 'pulse':
            path = f"{self.name}/freqs/{self.pulse_output}"
            self.read_pulse(path)
        else:
            print(f"{oscprog}: unknown pulsation program")
            return np.empty(0)

        if modesset is not None:
            return self.select_mode_set(modesset, lsep_target)

        # Large separations not corrected by surface_effects
        y, covar, yn = self.seismic.dnu(modes=self.modes)
        lsep = y.mean()

        # Adding surface effects
        if se and settings['modes']['surface_effects'] is not None:
            dnu_se = self.get_surface_effects(settings['modes']['surface_effects'], lsep)
            self.modes[:, 2] = self.modes[:, 2] + dnu_se
            y, covar, yn = self.seismic.dnu(modes=self.modes)
            return y.mean()
        else:
            return lsep

    def select_mode_set(self, modesset, lsep_target):
        """
        Select the oscillation modes following the modes settings of the JSON file and return the large separation
        corresponding to the target
        :param modesset: modes settings
        :param lsep_target: large separations target value

        :return: large separations of the models corresponding to the target
        """
        l = modesset['l']
        print("We select the degrees l: ", np.array_str(np.array(l)))
        dn = modesset['dn']
        nmin = modesset['nmin']
        nmax = modesset['nmax']
        print(f"We select the orders from nmin={nmin} to nmax={nmax} with a tolerance dn={dn}")
        nm = self.modes.shape[0]
        lsep = np.zeros(dn * 2 + 1)

        print("Selected modes:")
        for i in range(dn * 2 + 1):
            shift = i - dn
            print(f"shift = {shift}")
            selected = []
            for li in l:
                nl = 0
                for k in range(nm):
                    for n in range(nmin + shift, nmax + shift + 1):
                        if abs(self.modes[k, 0] - n) < 1e-5 and abs(self.modes[k, 1] - li) < 1e-5:
                            print(f"selected modes (l,n,nu,w2) = ({li},{n},{self.modes[k, 2]},{self.modes[k, 3]})")
                            selected.append(k)
                            nl += 1
                if nl == 0:
                    print(f"l={li} mode missing")
                else:
                    if nl < 2:
                        print(f"WARNING ! not enough l={li} modes")
            nms = len(selected)
            print(f"{nms} modes selected")
            if nms > 1:
                y, covar, yn = self.seismic.dnu(modes=self.modes[selected, :])
                lsep[i] = y.mean()
            else:
                if nms == 0:
                    print("WARNING ! no modes matching the requirements")
                if nms < 2:
                    print("WARNING ! not enough modes matching the requirements")
                lsep[i] = 0.0
            print(f"Large separation dnu = {lsep[i]}")

        i = np.argmin(abs(lsep_target - lsep))
        print(f"Closest value of large separation dnu = {lsep[i]} (shift = {i - dn})")
        return lsep[i]

    def get_surface_effects(self, surface_effects, lsep):
        """
        Provide correction of the surface effects
        :param surface_effects: surface effects settings
        :param lsep: large separation
        :return: corrections of the frequencies due to surface effects
        """
        nu = self.modes[:, 2]
        se_params = surface_effects['parameters'].copy()
        parameters = self.setup.parameters
        formula = surface_effects['formula']
        dict_se = {'_a': 0, '_b': 1, '_c': 2}
        dict_se_ = {0: '_a', 1: '_b', 2: '_c'}
        if surface_effects['prescription']:
            # surface effects parameters were set to 0.0, we use the Manchon et al. (2018) prescription
            for i, p in enumerate(se_params):
                se_params[i] = self.surface_effects_prescription(formula + dict_se_[i], lsep)
        # elif se_params is None:
            # surface effects parameters were set as tunable parameters
            se_params = []
        else:
            for p in parameters:
                if p.name in ['se_a', 'se_b', 'se_c']:
                    s = p.name[2:4]
                    i = dict_se[s]
                    se_params[i] = p.value

        # Otherwise, surface effects parameters were set in modes settings

        if surface_effects['numax'] is None:
            numax = self.model.numax
        else:
            numax = surface_effects['numax']

        if formula == 'lorentz3':
            dnu_se = numax * (se_params[2] + se_params[0] / (1. + (nu / numax) ** se_params[1]))
        elif formula in ['lorentz', 'lorentz2']:
            dnu_se = se_params[0] * numax * (1. / (1. + (nu / numax) ** se_params[1]) - 1.)
        elif formula == 'kb2008':
            dnu_se = se_params[0] * numax * (nu / numax) ** se_params[1]
        elif formula == 'bg1':
            dnu_se = se_params[0] * numax * (nu / numax) ** 3 / self.modes[:, 7]
        elif formula == 'bg2':
            dnu_se = (se_params[0] * numax / nu + se_params[1] * (nu / numax) ** 3) * numax / self.modes[:, 7]

        n = nu.size
        print("Surface effects:")
        print("\tParams: [" + " ".join('{:>10}'.format('{:>8g}'.format(p)) for p in se_params) + " ]")
        print("\tnu\tdnu_se")
        for i in range(n):
            print(f"\t{nu[i]}\t{dnu_se[i]}")

        return dnu_se

    def surface_effects_prescription(self, coeff, lsep):
        """
        Calculate the coefficient of the surface effects prescription
        :param coeff: coefficient to be calculated
        :param lsep: large separations

        :return: value of the coefficient
        """
        # Solar values from CO5BOLD models (Manchon et al. 2018)
        dnu_s, teff_s, g_s, kappa_s = 137, 5776, 27511, 0.415
        # Test if in the domain of validity
        teff = self.model.teff[-1]
        Dlogteff = self.model.logteff[-1] - np.log10(teff_s)
        logg = self.model.logg[-1]
        Dlogg = logg - np.log10(g_s)
        Dnu = lsep / dnu_s
        Dkappa = 1.0
        Z = self.model.z_s[-1]

        if not (4500 < teff < 6743 and 3.5 < logg < 4.5 and 0.000205 < Z < 0.0414):
            print("WARNING: Prescriptions for surface effects correction coefficients are given for a model out of "
                  "the range of validity. Results must be handled carefully.")

        if coeff == 'kb2008_a':
            return 10 ** (1.03 * np.log10(Dnu) + 3.26 * Dlogteff - 1.75 * Dlogg + 0.655 * Dkappa - 2.72)
        elif coeff == 'kb2008_b':
            return 10 ** (-0.185 * np.log10(Dnu) - 0.584 * Dlogteff + 0.313 * Dlogg - 0.117 * Dkappa + 0.289)
        elif coeff == 'lorentz2_a':
            return 10 ** (0.999 * np.log10(Dnu) + 3.15 * Dlogteff - 1.69 * Dlogg + 0.635 * Dkappa - 2.36)
        elif coeff == 'lorentz2_b':
            return 10 ** (-0.477 * np.log10(Dnu) - 1.51 * Dlogteff + 0.808 * Dlogg - 0.303 * Dkappa + 0.787)
        elif coeff == 'bg1_a':
            return 10 ** (1.93 * np.log10(Dnu) + 6.09 * Dlogteff - 3.26 * Dlogg + 1.22 * Dkappa - 11.9)
        elif coeff == 'bg2_a':
            return 10 ** (2.13 * np.log10(Dnu) + 6.72 * Dlogteff - 3.6 * Dlogg + 1.35 * Dkappa - 12)
        elif coeff == 'bg2_b':
            return 10 ** (1.8 * np.log10(Dnu) + 5.67 * Dlogteff - 3.04 * Dlogg + 1.14 * Dkappa - 12)
        else:
            raise NOCError("Prescription for required surface effects correction law is not supported.\n\
                You must provide your own value for the free parameters.")
        # TODO: verify the coefficient calculations

    def read_pulse(self, file):
        """
        Read the Pulse outputs
        :param file: file with calculated theoretical frequencies

        modes[:,0] : n order
        modes[:,1] : l degree
        modes[:,2] : nu frequency in muHz (eigenvalue)
        """

        data = np.genfromtxt(file, skip_header=5, filling_values=99)

        self.modes = np.zeros((data.shape[0], 4))

        self.modes[:, 0] = data[:, 1]  # n order
        self.modes[:, 1] = data[:, 0]  # l degree
        self.modes[:, 2] = data[:, 2] * 1000  # nu frequency in muHz
        # self.modes[:, 3] = abs(data[:, 2] - data[:, 5])*1000  # sigma
        self.modes[:, 3] = data[:, 2]*100  # sigma
        self.__reorder_modes()
        self.seismic = Seismic(self.modes)

    def read_agsm(self, file):
        """
        Read agsm output file from ADIPLS frequencies calculation
        :param file: agsm file
        OUTPUTS:
        modes[:,0] : n order
        modes[:,1] : l degree
        modes[:,2] : nu frequency in muHz (eigenvalue)
        modes[:,3] : square normalised frequency (w2)
        modes[:,4] : Richardson frequency in muHz
        modes[:,5] : variationnal frequency in muHz
        modes[:,6] : icase
        modes[:,7] : Inertia
        """
        f = FortranBinaryFile(file, mode='r')
        cs = []

        while True:
            try:
                csi = f.readRecordNative('d')
                cs.append(csi)
            except IndexError:
                break

        nmodes = len(cs)
        self.modes = np.zeros((nmodes, 8))
        mstar = cs[0][1]
        rstar = cs[0][2]
        for i in range(nmodes):
            self.modes[i, 0] = cs[i][18]  # n
            self.modes[i, 1] = cs[i][17]  # l
            sigma2 = cs[i][19]
            omega2 = sigma2*ggrav*mstar/(rstar**3)
            self.modes[i, 2] = np.sqrt(omega2)/(2*np.pi)*1e6  # nu in muHz
            self.modes[i, 3] = sigma2
            self.modes[i, 4] = cs[i][36]*1e3  # Richardson frequency in muHz
            self.modes[i, 5] = cs[i][26]*1e3  # Variational frequency in muHz
            self.modes[i, 7] = cs[i][23]
        # Reading the 'ics' integer variables
        f.close()

        f = FortranBinaryFile(file, mode='r')
        n = 38*2
        for i in range(nmodes):
            data = f.readRecordNative('i')
            ics = data[n: n+8]
            self.modes[i, 6] = ics[4]

        f.close()

        self.seismic = Seismic(self.modes, lfirst=False)

    def __reorder_modes(self):
        """
        Reorder modes to have ascendant n order for each l degree
        """
        nb_modes = self.modes.shape[0]
        # print('nb_modes=' + str(nb_modes))
        if nb_modes > 0:
            lval = np.unique(self.modes[:, 1])
        else:
            lval = []
        for k in range(len(lval)):
            i_list = np.where(self.modes[:, 1] == lval[k])[0]
            for i in [0, 2, 3]:
                inverted = np.flip(self.modes[i_list[0]:i_list[-1]+1, i])
                self.modes[:, i] = np.concatenate((self.modes[:i_list[0], i], inverted, self.modes[i_list[-1]+1:, i]))


class FortranBinaryFile:

    def __init__(self, filename, mode='r', endian='@'):
        """
        Fortran binary file support
        :param filename: name of the Fortran binary file
        :param mode: 'r' for read, 'w' for write. Default: 'r'
        :param endian: endian in binary file
        """
        self.filename = os.path.expanduser(filename)
        self.endian = endian
        # self.swap, self.fromstring = fromstring(endian)
        setattr(FortranBinaryFile, 'readRecord', FortranBinaryFile.__dict__['readRecordNative'])

        if mode == 'r':
            if not os.path.exists(filename):
                raise IOError(2, 'No such file or directory: ' + filename)
            if filename[-2:] == '.Z':
                self.file = os.popen("uncompress -c " + filename, 'rb')
            if filename[-3:] == '.gz':
                self.file = os.popen("gunzip -c " + filename, 'rb')
            else:
                try:
                    self.file = open(filename, 'rb')
                except:
                    raise IOError

            self.filesize = os.path.getsize(filename)

        else:
            raise IOError(0, 'Illegal mode: ' + repr(mode))

    def close(self):
        if not self.file.closed:
            self.file.close()

    def readRecordNative(self, dtype=None):
        a = self.file.read(4)  # record size in bytes
        recordsize = np.frombuffer(a, 'i')
        record = self.file.read(recordsize[0])
        self.file.read(4)  # record size in bytes

        if dtype in ('f', 'i', 'I', 'b', 'B', 'h', 'H', 'l', 'L', 'd'):
            return np.frombuffer(record, dtype)
        elif dtype in ('c', 'x'):
            return struct.unpack(self.endian+'1'+dtype, record)
        else:
            return None, record
