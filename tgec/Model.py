import glob
import os.path
import sys
import time
from os import path

import numpy as np

import tgec.constants
from tgec.Parameters import *
from tgec.RunModel import RunModel
from tgec.SeismicModel import SeismicModel

ggrav = tgec.constants.ggrav
pi = tgec.constants.pi
msun = tgec.constants.msun
rsun = tgec.constants.rsun


def extract_column(list, column):
    """
    The original file is an array of values in a file. After reading the file, we have a list of string, each string
    being a line of the array. The function extract a column of the original array form the list of string.
    :param list: list of string
    :param column: column to extract
    :return: extracted column in a ndarray
    """
    return np.array([line.split()[column] for line in list], dtype=float)


class Model:

    def __init__(self, name, setup, reinit=False, read=True, verbose=True):
        """
        Class representing the TGEC model

        :param name: name of the model
        :param setup: setup given by JSON file
        :param reinit: if true, delete all calculated files (default=false)
        :param read: if true, read files .g and .in (default=true)
        :param verbose: if true, print additional information
        """
        # Files used by the model
        self.name = name
        self.com_file = name + '.com'
        self.tgec_log = name + '_tgec.log'
        self.pulse_log = name + '_pulse.log'
        self.stop_file = 'stop'

        # Setup of the model optimization
        self.setup = setup
        self.targets = self.setup.targets
        self.parameters = self.setup.parameters
        self.settings = None
        self.nseismic = self.setup.nseismic
        # Initial age of the model
        for p in self.parameters:
            if p.name == 'age':
                self.age_model = p.value * 1e6  # Converted from Myr to yr

        self.verbose = verbose

        # Parameters of the model
        self.params = Parameters(self.name)
        self.run = RunModel(self.name)

        # Output directory of tgec models
        self.evol_path = self.name + '/evolution/'

        # Output variables in file .in
        self.age, self.lstar, self.lum, self.teff, self.mstar, self.mass, self.rad \
            = [np.array([], dtype=float) for _ in range(7)]
        # Output variables in file .g
        self.nmod = np.array([], dtype=int)
        self.logl, self.logteff, self.x_s, self.y_s = [np.array([], dtype=float) for _ in range(4)]
        # Output variables in file .ab
        self.li_s = np.array([], dtype=float)
        # Output variables in file .zc
        self.r_cz = np.array([], dtype=float)
        # Calculated fundamental outputs
        self.logg, self.rho, self.z_s, self.zox_s = [np.array([], dtype=float) for _ in range(4)]
        # Calculated seismic outputs
        self.large_sep = 0.0
        self.numax = np.array([], dtype=float)

        # Dictionary of the evolution model output variables. Filled in the function read_output
        self.dict_output = {}

        # If the directory 'evolution' has already been created, the evolution has finished
        self.finished = os.path.isdir(self.evol_path)

        # print(f"read = {read}, finished = {self.finished}")
        if reinit and self.finished:
            self.reinit()  # TODO: reinit function
        elif read and self.finished:
            print(f"Model already calculated. Reading outputs")
            self.read_output(verbose=verbose)

        # TODO: complete the seismic part
        self.seismic = SeismicModel(self.name, self, setup)

        # if os.path.exists(self.seismic.frun_file):
        #     self.seismic.read_settings()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"TGEC model \'{self.name}\' \n\t Parameters:\n" \
               f"\t\tMass: {self.params.mass} M\u2299\n" \
               f"\t\tX0: {self.params.x0}\n" \
               f"\t\tY0: {self.params.y0}\n" \
               f"\t\tZ0: {self.params.z0}\n"

    def __call__(self, update_com=True, debug=False, log=False, verbose=True):
        """
        Calculate model or frequencies of the model
        :param update_com: update the .com file. default=true
        :param debug: run in debug mode. default=false
        :param log: direct standard output to .log file. default=false
        :param verbose: print additional information on standard output
        """
        self.finished = False
        if update_com:
            self.params.update_com(self.age_model)

        if verbose:
            print("\n------------------------------------------")
            print(f"Computing evolution of model {self.name}... ")
        t1 = time.time()

        # Empty output buffer
        sys.stdout.flush()

        # Run TGEC
        self.run.load_tables()
        # output = self.run.run_tgec(debug=debug, log=log)
        self.run.run_tgec(debug=debug, log=log)
        self.finished = not os.path.isfile('stop')
        # self.__process_output(output, t1, verbose, debug, log)
        if self.finished:
            self.__process_output(t1, verbose)

    def __process_output(self, tstart, verbose):
        """
        Process the output of TGEC model calculation and read the output file if needed
        :param output: output messages printed to standard output
        :param tstart: time of calculation start
        :param verbose: if true, additional information is printed
        """
        t2 = time.time()
        hr, mn, sc = time.gmtime(t2 - tstart)[3:6]
        if verbose:
            print("DONE")
            print(f"Finished at {time.asctime()}")
            if hr == 0:
                if mn == 0:
                    print(f"\tTime: {sc:02.2f}")
                else:
                    print(f"\tTime: {mn:02d}m {sc:02.2f}s")
            else:
                print(f"\tTime: {hr:d}h {mn:02d}m {sc:02.2f}s")

        cmd = f"mkdir -p {self.name}/evolution && cp {self.name}.com {self.name} " \
              f"&& mv etoile_* {self.params.model_name}.* {self.name}"
        os.system(cmd)
        ext_file = ['in', 'g', 'l', 'J', 'zc', 'ab']
        for i in ext_file:
            cmd = f"cat {self.name}/{self.params.model_name}.{i}0* > {self.name}/evolution/{self.params.model_name}.{i}"
            os.system(cmd)

        # os.system("./clean")
        self.clean_links()

        self.read_output(verbose=verbose)

        # TODO: treat when there is an error

    def setup_com_file(self, setup=None, parameters=None, settings=None, verbose=True):
        """
        Update model parameters from the .com file with the parameters of the JSON file
        or following the update of the optimization process
        :param setup: optimization setup from the JSON file
        :param parameters: optimization parameters from the JSON file
        :param settings: optimization settings from the JSON file
        :param verbose: print details option (default=True)
        """
        if setup is None and (parameters is None or settings is None):
            raise NOCError("Model.setup_com_file must be called with setup or, parameters and settings provided")
        elif setup is not None:
            parameters = setup.parameters
            settings = setup.settings
        self.settings = settings

        # We check if the free parameters mentioned in the JSON file are parameters of the .com file
        # and update the value for the .com file
        for p in parameters:
            found = False
            for name_input in self.params.params_dict.keys():
                if name_input.lower() == p.name:
                    found = True
                    self.params.params_dict[name_input] = p.value
            # Parameters used for the surface effects
            if (p.name == 'se_a') | (p.name == 'se_b') | (p.name == 'se_c') | (p.name == 'log_f'):
                found = True
            # Age is a parameter but is not in the .com file
            if p.name == 'age':
                self.age_model = p.value * 1e6  # Converted from Myr to yr
                found = True
            if found and verbose:
                print(f"{p.name.upper()} = {p.value:8g}")
            if not found:
                raise NOCError(f"Error in NOC/setup_com_file: free parameter '{p.name}' not found in the .com file")

        # We update the start age for the .com file according to the model settings in the JSON file
        # or to the update in the optimization process
        start = self.params.start
        if self.settings['models'] is not None:
            start = self.settings['models']['start']
            if start == 'pms':
                self.params.params_dict['IZAMS'] = 0
            elif start == 'zams':
                self.params.params_dict['IZAMS'] = 2
        print(f"Model starts at {start.upper()}")

        if self.params.diffusion and verbose:
            print("Diffusion included")
        # Initial chemical abundances X0, Y0 and Z0 are parametrized following the prescription of the JSON file
        self.get_initial_chem(parameters, verbose=verbose)

        if verbose:
            print(f"X0 = {self.params.x0}, Y0 = {self.params.y0}, Z0 = {self.params.z0}, [Z/X]0 = {self.params.zox0}")

        return self

    def get_initial_chem(self, parameters, verbose=True):
        """
        Calculate the initial abundances X0, Y0 and Z0 following the prescription of the JSON file or the derivatives:
        - if both Y0 and [Z/X]0 are mentioned as free parameters (free=2) or not mentioned at all (free=0),
          we use the values of the .com file
        - if only one of them is mentioned as free parameter (free=1), the other one is calculated from the
          enrichment mentioned in the model settings
        :param parameters: free parameters mentioned in the JSON file
        :param verbose: print details option (default=True)
        """
        # TODO: include FESURHINI
        free = 0
        y0 = -1.0
        zox0 = -1.0
        for p in parameters:
            if p.name == 'y0':
                free += 1
                y0 = self.params.y0
            elif p.name == 'zox0':
                free += 1
                zox0 = self.params.zox0

        if free == 0:
            y0 = self.params.y0
            zox0 = self.params.zox0
        elif free == 1:
            y0, zox0 = self.get_enrichment(y0, zox0, verbose=verbose)

        # Parameters of the model are updated
        self.params.y0 = y0
        self.params.zox0 = zox0
        self.params.x0 = (1.0 - self.params.y0) / (1.0 + self.params.zox0)
        self.params.z0 = (1.0 - self.params.x0 - self.params.y0)

    def get_enrichment(self, y0, zox0, verbose=True):
        """
        Calculate Y0 or [Z/X]0 from the primitive abundance and the enrichment parametrized in the model settings
        in the JSON file
        :param y0: Initial helium abundance
        :param zsx0: Initial Z over X abundance
        :param verbose: print details option (default=True)

        :return: y0 and zox0 updated
        """
        diff = self.settings['models']['dy_dz']
        yp = self.settings['models']['yp']
        zp = self.settings['models']['zp']

        # If y0 is not a free parameter, it is calculated from the primitive abundances and the enrichment
        # TODO: verify these equations
        if y0 < 0.0:
            y0 = 1.0 - (1 + zox0) * (1.0 + diff * zp - yp) / (1.0 + (1.0 + diff) * zox0)
        if zox0 < 0.0:
            zox0 = (yp - y0 - diff * zp) / ((1.0 + diff) * y0 - (1.0 - diff) * zp)

        if verbose:
            print(f"yp = {yp}, zp = {zp}, dy_dz = {diff}")

        return y0, zox0

    def get_output(self, lsep_target, settings, name):
        """
        Get the model output corresponding to observable targets

        :param lsep_target: value of the large separation target
        :param settings: settings of the optimization
        :param name: name of the target to evaluate

        :return: value of the target at the age age_model
        """
        if name == 'largesep':
            self.large_sep = self.seismic.largesep(lsep_target.value, settings)
            return self.large_sep

        return self.dict_output[name][-1]

        # for i in range(self.age.size - 1):
        #     # While age_model doesn't lie between two successive values, we continue and increase i
        #     if self.age[i] < self.age_model and self.age[i + 1] < self.age_model:
        #         continue
        #     elif self.age[i] <= self.age_model <= self.age[i + 1]:
        #         # When age_param lies between two successive values, we check the closer one and return it
        #         if (self.age_model - self.age[i]) < (self.age[i + 1] - self.age_model):
        #             return self.dict_output[name][i]
        #         else:
        #             return self.dict_output[name][i + 1]
        #     else:
        #         # If age_param is larger than the largest age value, we return it
        #         return self.dict_output[name][-1]

    def read_output(self, verbose=True):
        """
        Read the output .in and .g evol file
        :param verbose: If True, print some details.
        """
        in_file = self.params.model_name + '.in'
        g_file = self.params.model_name + '.g'
        ab_file = self.params.model_name + '.ab'
        zc_file = self.params.model_name + '.zc'

        for file in [in_file, g_file, ab_file, zc_file]:
            if not os.path.exists(self.evol_path + file):
                print(f"Error: file {file} not found")
                return
            if not os.path.getsize(self.evol_path + file):
                print(f"Error: file {file} exists but is empty")
                return

        # Reading .in file
        if verbose:
            print(f"Reading {in_file}... ", end='')
        with open(self.evol_path + in_file, 'r') as f:
            content = f.readlines()
        size = len(content)

        i = 0
        while i < size:
            try:
                self.age = np.append(self.age, float(content[i][:14]))
                self.lstar = np.append(self.lstar, float(content[i][15:24]))  # L_\star (L_\odot)
                self.lum = np.append(self.lum, float(content[i][25:39]))  # L_\star (erg/s)
                self.teff = np.append(self.teff, float(content[i][40:48]))
                self.mstar = np.append(self.mstar, float(content[i][49:58]))  # M_\star (M_\odot)
                self.mass = np.append(self.mass, float(content[i][59:73]))  # M_\star (g)
                self.rad = np.append(self.rad, float(content[i][74:88]))  # R_\star (cm)
            except (IndexError, ValueError):
                print(f"Problem at line: {i}.")
                return

            i += 1

        self.logg = np.log10(ggrav * self.mass / self.rad ** 2)
        self.rho = (3 * self.mass) / (4 * pi * self.rad ** 3)
        self.numax = 3104.0 * ((self.mass / msun) * (rsun / self.rad) ** 2) * np.sqrt(5777.0 / self.teff)

        if verbose:
            print("Done.")

        # Reading .g file
        if verbose:
            print(f"Reading {g_file}... ", end='')
        with open(self.evol_path + g_file, 'r') as f:
            content = f.readlines()
        size = len(content)

        i = 0
        while i < size:
            try:
                self.nmod = np.append(self.nmod, int(content[i][:5]))
                self.logl = np.append(self.logl, float(content[i][30:37]))
                self.logteff = np.append(self.logteff, float(content[i][39:46]))
                self.x_s = np.append(self.x_s, float(content[i][47:57]))
                self.y_s = np.append(self.y_s, float(content[i][58:68]))
            except (IndexError, ValueError):
                print(f"Problem at line {i}.")
                return

            i += 1

        self.z_s = 1 - self.x_s - self.y_s
        self.zox_s = self.z_s / self.x_s

        if verbose:
            print("Done.")

        # Reading .ab file
        if verbose:
            print(f"Reading {ab_file}... ", end='')
        with open(self.evol_path + ab_file, 'r') as f:
            content = f.readlines()
        size = len(content)

        i = 0
        while i < size:
            try:
                self.li_s = np.append(self.li_s, float(content[i][32:45]))
            except IndexError:
                print(f"Index error at line {i}.")
                return
            except ValueError:
                print(f"Value error at line {i}.")
                return

            i += 1

        if verbose:
            print("Done.")

        # Reading .zc file
        if verbose:
            print(f"Reading {zc_file}... ", end='')
        with open(self.evol_path + zc_file, 'r') as f:
            content = f.readlines()
        size = len(content)

        i = 0
        while i < size:
            try:
                self.r_cz = np.append(self.r_cz, float(content[i][84:94]))
            except (IndexError, ValueError):
                print(f"Problem at line {i}.")
                return

            i += 1

        if verbose:
            print("Done.")

        self.dict_output = {
            "age": self.age,
            "logg": self.logg,
            "logteff": self.logteff,
            "teff": self.teff,
            "rad": self.rad,
            "lum": self.lum,
            "lstar": self.lstar,
            "logl": self.logl,
            "mean_density": self.rho,
            "y_s": self.y_s,
            "z_s": self.z_s,
            "zox_s": self.zox_s,
            "li_s": self.li_s,
            "r_cz": self.r_cz,
            "numax": self.numax,
            "largesep": self.large_sep
        }

    def tgec2pulse(self):
        """
        Convert TGEC output files to a Pulse input file
        """
        files_pattern = ["etoile_mo.0?????", "etoile_ft.0?????", "etoile_ab.0?????"]
        files = []
        # Initialization of static variables
        rmass, rdens, rray, rtemp, rhot, hp, n2mu, xpress, delrad, delad, gradthermreal, gamma1, abond4, vmu, deltamux \
            = [np.array([], float) for _ in range(15)]
        sigma = 5.669e-9
        boltz = 1.3806e-16
        avo = 6.0221e23
        clum = 2.99793e10
        constant = 4 * sigma / (3 * clum * avo * boltz)
        nlayers = 0

        for pattern in files_pattern:
            path = self.name + '/' + pattern
            files += glob.glob(path)
        # Filtering of files ending with '000000' we don't want to read
        files = [file for file in files if not file.endswith('000000')]
        for file in files:
            print(f"Reading {file}...")
            with open(file, "r") as f:
                content = f.readlines()

            nlayers = len(content)

            # Reading the etoile_mo file
            # if file.startswith('etoile_mo'):
            if 'etoile_mo' in file:
                rmass = extract_column(content, 2)
                rdens = extract_column(content, 3)
                rray = extract_column(content, 4)
                rtemp = extract_column(content, 8)
                rhot = extract_column(content, 11)
                hp = extract_column(content, 15)
                n2mu = extract_column(content, 25)

            # Reading the etoile_ft file
            # if file.startswith('etoile_ft'):
            if 'etoile_ft' in file:
                xpress = extract_column(content, 2)
                delrad = extract_column(content, 5)
                delad = extract_column(content, 6)
                gradthermreal = extract_column(content, 7)
                gamma1 = extract_column(content, 14)

            # Reading the etoile_ab file
            # if file.startswith('etoile_ab'):
            if 'etoile_ab' in file:
                abond4 = extract_column(content, 3)
                vmu = extract_column(content, 23)
                deltamux = extract_column(content, 24)

        # print(n2mu)
        itypezone = np.array([1 if n2mu[i] < 0 else 0 for i in range(nlayers)])
        rhot = np.array([-element for element in rhot])
        xkhirho = gamma1 / (1 + gamma1 * rhot * delad)
        xkhitemp = rhot * xkhirho

        xlogQ = np.array([np.log(1 - rmass[i] / rmass[0]) if rmass[i] != rmass[0] else 0. for i in range(nlayers)])
        mode = np.array([1 for _ in range(nlayers)])

        Bledoux = -(hp * deltamux / xkhitemp)/(1. + constant * rtemp**3. * vmu / rdens)

        # rnorm = rray/rray[0]

        # Calculation of real thermic gradient
        for i in range(nlayers):
            if itypezone[i] == 1:
                gradthermreal[i] = delad[i]
            else:
                if gradthermreal[i] > delrad[i] or gradthermreal[i] < 0:
                    # tmp = gradthermreal[i]
                    gradthermreal[i] = delrad[i]

        columns = np.column_stack((np.arange(1, nlayers+1, dtype=int), rray[::-1], rmass[::-1], rdens[::-1],
                                   xpress[::-1], rtemp[::-1], xkhirho[::-1], xkhitemp[::-1], gradthermreal[::-1],
                                   delad[::-1], abond4[::-1], Bledoux[::-1], xlogQ[::-1], mode[::-1], itypezone[::-1]))

        # Write variables in Pulse input file
        with open(self.seismic.pulse_input, 'w') as f:
            f.write("format: 1\n")
            f.write(f"{nlayers:>5d} layers\n")
            np.savetxt(f, columns, fmt='%5d   %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e %18.10e '
                                       '%18.10e %18.10e %18.10e %3d %3d')

    def clean_links(self):
        """
        Remove the symbolic links to the opacity and eos tables
        """
        if os.path.isfile('stop'):
            os.system("rm stop")

        links = ["EOS5_0?z?x", "IEOS0?z?x", "neos0*z?x", "peos0*z?x", "kappa92.dat", "kappa95.bin", "paq_*", "tabeos*",
                 "fort.9", "Asp09hz", "A09photo"]

        for link in links:
            os.system(f"rm -f {link}")

    def reinit(self):
        pass
