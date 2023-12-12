#!/usr/bin/python
# coding: utf-8

import os
import re
import sys
import time
import traceback

import numpy as np

from nocpkg.ComputeOptimal import ComputeOptimal
from tgec.Model import Model
from nocpkg.Setup import Setup
from nocpkg.utils import NOCError

__copyright__ = """
Natal Optimization Code (NOC)

Copyright (c) 2023 M. Castro (Universidade Federal do Rio Grande do Norte - Brazil)

This is a free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this code.  If not, see <http://www.gnu.org/licenses/>.
"""

__version__ = "1.1"


class NOCMain:

    def __init__(self, parser):
        self.parser = parser
        self.args = parser.parse_args()

        tstart = time.time()

        # Create file name
        self.name = re.sub('.json', '', os.path.basename(self.args.name))
        self.json_file = self.name + '.json'
        self.com_file = self.name + '.com'
        self.log_file = self.name + '.log'
        self.err_file = self.name + '.err'

        # Process arguments
        self.__process_args()
        print(__copyright__)
        print("This is NOC version: " + __version__)
        print("==================================================")
        cwd = os.getcwd()

        # Parse JSON setup file
        self.setup = Setup(self.json_file)
        self.setup.read_setup()
        self.parameters = self.setup.parameters
        self.targets = self.setup.targets
        self.settings = self.setup.settings
        self.nparams = self.setup.nparams

        # Initialize the model
        self.model = None
        self.model_name = self.name
        self.teff = None

        if not os.access(self.com_file, os.F_OK):
            raise NOCError("Error in NOC: missing file " + self.com_file)

        # Initialization of the model optimization
        try:
            self.noc_init()
        except NOCError:
            self.results = traceback.format_exc()
            print(self.results)
            raise

        # Model run
        try:
            self.results = self.noc_run()
        except NOCError:
            self.results = traceback.format_exc()
            print(self.results)
            raise

        tend = time.time()
        hr, mn, sc = time.gmtime(tend - tstart)[3:6]
        print(f"Finished at {time.asctime()}")
        if hr == 0:
            if mn == 0:
                print(f"\tTime: {sc:02.2f}")
            else:
                print(f"\tTime: {mn:02d}m {sc:02.2f}s")
        else:
            print(f"\tTime: {hr:d}h {mn:02d}m {sc:02.2f}s")



        os.chdir(cwd)
        sys.stdout.flush()
        sys.stderr.flush()

    def __process_args(self):
        if self.args.version:
            print(f"NOC {__version__}")
            sys.exit(1)

        if not self.args.name:
            self.parser.print_help()
            sys.exit(1)

        self.verbose = self.args.verbose

        # if self.args.guess:
        #     guess = Guess(self.args.name.strip())
        #     sys.exit(1)

    def noc_init(self, resume=False):

        print("\n\n---------- Initialization ----------\n\n")

        self.model = Model(self.model_name, self.setup, verbose=self.verbose)

        # Create the working directory and copy useful files in it
        self.create_wd()
        # print("Work directory created")

        # # If we resume from a previous optimization, we retrieve the previous parameters
        # if resume:
        #     self.parameters = self.get_param_resume(self.parameters)

        # Initiate model
        self.model_initialization()
        # print("Initial model created")
        # print(os.listdir("."))

        # check that teff is among targets
        self.teff = self.get_teff()
        # print("Teff is among targets")

    def noc_run(self):
        return ComputeOptimal(self.name, self.setup, verbose=self.verbose, debug=self.args.debug)

    def create_wd(self):
        """
        Create the working directory and copy useful files in it, then go in
        """
        work_dir = self.name

        if not os.access(work_dir, os.F_OK):
            os.mkdir(work_dir)

        used_files = ['circmerid.dat', 'circulation.dat', 'diffusion.dat', 'param.dat', 'structure', 'clean',
                      self.com_file]
        for file in used_files:
            if os.access(file, os.F_OK):
                os.system(f"cp -p {file} {work_dir}")
            else:
                raise NOCError(f"File {file} could not be found.")

        os.chdir(work_dir)

    # def get_param_resume(self, parameters):
    #     print("Initial parameters retrieved from previous calculation")
    #     param_prev = np.loadtxt('params.txt', usecols=(1,))  # TODO: verify creation of params.txt file
    #     for i, p in enumerate(parameters):
    #         p.value = param_prev[i]
    #
    #     return parameters

    def model_initialization(self):
        """
        Create an initial model taking into account the .com file and the settings in the JSON file,
        and update the .com file according to the JSON file
        """
        self.model.setup_model_params(setup=self.setup)

        self.model.finished = False

        self.model.params.update_params(self.model.age_model)

    def get_teff(self):
        """
        Check that Teff is among targets, and if so, return the value
        :return: teff
        """
        teff = -1.0
        for t in self.targets:
            if t.name == 'log_teff':
                teff = 10.0**(t.value)
            elif t.name == 'teff':
                teff = t.value
        if teff < 0:
            raise NOCError("Teff must be among the targets")

        return teff
