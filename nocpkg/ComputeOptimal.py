import os
import re
import sys

import numpy as np

from nocpkg.utils import NOCError, get_instance
from nocpkg.LevMarAlgo import LevMar
from tgec.Model import Model


class ComputeOptimal:

    def __init__(self, name, setup, verbose=False, debug=False):
        """
        Launch the computation of an optimal model matching best the set of constraints, using the 
        Levenberg-Marquardt method
        :param name: name of the model
        :param setup: setup of the optimization
        :param verbose: print details option (default=False)
        :param debug: run in debug mode (default=False)
        """
        self.name = name
        self.setup = setup
        self.parameters = setup.parameters
        self.targets = setup.targets
        self.settings = setup.settings
        self.seismic_constraints = setup.seismic_constraints

        self.nparams = setup.nparams
        self.ntargs = setup.ntargs
        self.nseismic = setup.nseismic
        self.ny = self.ntargs + self.nseismic  # Total of non-seismic and seismic targets

        self.verbose = verbose
        self.debug = debug
        self.cwd = os.getcwd()

        self.log_file = self.name + '.log'

        # Initialize data structures for the calculation
        self.model_refs = {}
        self.y = np.empty(self.ny)
        self.sigma = np.empty(self.ny)
        self.covar = np.zeros((self.ny, self.ny))
        self.y_name = []
        self.maxiter = 0
        self.ftol = 0
        self.chi2min = 0
        self.cov_cdtnb_thr = 1e13
        self.hess_cdtnb_thr = 1e13

        with open(self.log_file, 'w') as self.lgf:
            self.init_models()
            self.print_init_params()
            self.init_levmar()
            # Create the levmar instance through the __init__ method of the LevMar class
            self.levmar = LevMar(self.setup, self.name, self.y, self.sigma, self.covar, self.y_name, self.lgf)
            # Execute the levmar instance as a function through the __call__ method of the LevMar class
            msg = self.call_levmar()
            self.print_results(msg)
            self.clean()

    def init_models(self):
        """
        Create a dictionary to store the references to the central model with the current optimal parameters,
        and to the model with derivatives with respect to the tunable parameters.
        """
        # center_name = f"{self.name}-000"
        self.model_refs['center'] = Model(self.name, self.setup, verbose=self.verbose)
        # self.model_refs['center'] = Model(center_name, self.setup, verbose=self.verbose)
        self.model_refs['deriv'] = {}
        ii = 0
        for p in self.parameters:
            # If surface effects is a parameter -> seismic parameter
            if re.match('^se_', p.name):
                p.seismic = True
            if not (p.name in self.model_refs['deriv'].keys()):
                # if p.seismic:
                #     self.model_refs['deriv'][p.name] = self.model_refs['center']
                # else:
                #     # Create a new .com file and a new Model instance for each parameter
                #     deriv_name = f"{self.name}-{ii+1:03}"
                #     cmd = f"cp {self.name}.com {deriv_name}.com"
                #     os.system(cmd)
                #     self.model_refs['deriv'][p.name] = Model(deriv_name, self.setup, verbose=self.verbose)
                #     ii += 1
                deriv_name = f"{self.name}-{ii+1:03}"
                # if not p.seismic:
                    # Create a new .com file for each non-seismic parameter
                cmd = f"cp {self.name}.com {deriv_name}.com"
                os.system(cmd)
                # Create a new Model instance for each parameter
                self.model_refs['deriv'][p.name] = Model(deriv_name, self.setup, verbose=self.verbose)
                ii += 1


    def print_init_params(self):
        """
        Print the initial choice of parameters and the targets to be optimized
        """
        text = "\n\n---------- Start optimization process ----------\n\n"
        text += "Initial parameters:\n"
        text += "Name          Value     Step      min      max     rate\n"
        for p in self.parameters:
            text += f"{p.name:10} {p.value:8g} {p.step:8g} {p.bounds[0]:8g} {p.bounds[1]:8g} {p.rate:8g}\n"

        text += "\nTargets:\n"
        text += "Name       Value        sigma\n"
        for t in self.targets:
            text += str(t) + '\n'
        text += '\n'

        print(text)
        self.lgf.write(text)
        sys.stdout.flush()

    def init_levmar(self):
        """
        Create initial data structure (non-seismic and seismic targets y with error sigma, covariance matrix, etc.)
        used by the Levenberg-Marquardt algorithm
        """

        for i in range(self.ntargs):
            self.y[i] = self.targets[i].value
            self.sigma[i] = self.targets[i].sigma
            self.covar[i, i] = self.targets[i].sigma**2
            self.y_name.append(self.targets[i].name)

        # If there exists seismic constraints, we complete the data structures
        if self.nseismic > 0:
            i0 = self.ntargs
            i1 = self.ny
            self.y[i0:i1] = self.seismic_constraints.y
            self.sigma[i0:i1] = np.sqrt(np.diag(self.seismic_constraints.covar))
            self.covar[i0:i1, i0:i1] = self.seismic_constraints.covar

        self.maxiter = self.settings['method']['levmar']['maxiter']
        self.ftol = self.settings['method']['levmar']['ftol']
        self.chi2min = self.settings['method']['levmar']['chi2min']
        self.cov_cdtnb_thr = self.settings['method']['levmar']['cov_cdtnb_thr']
        self.hess_cdtnb_thr = self.settings['method']['levmar']['hess_cdtnb_thr']

        self.y_name = np.array(self.y_name)

    def call_levmar(self):
        """
        Execute the LevMar class __call__ method
        :return: text written in the log file
        """
        (self.chi2, self.parout, self.outputs, self.iter, error, msg) \
            = self.levmar(self.compute_model, (self.model_refs, self.targets), verbose=True, ftol=self.ftol, maxiter=self.maxiter,
                          chi2min=self.chi2min, cov_cdtnb_thr=self.cov_cdtnb_thr, hess_cdtnb_thr=self.hess_cdtnb_thr)

        if error:
            raise NOCError("Error in LevMar")

        return msg

    def compute_model(self, parameters, args, status):
        """
        Compute one model for the Levenberg-Marquardt algorithm. This is the function evaluated in the class LevMar
        Set the parameters of the model, its initial conditions, launch the calculation and return the log probability
        that this model is the optimal one.
        :param parameters: list of Parameters instances used for the model
        :param args: additional arguments passed to the function, tuple (model, model_center, targets)
        :param status: index corresponding to the parameter tested. status = -1 for the central model.
        Otherwise, the model is computed in order to evaluate the derivative with respect to parameter[status]

        :return: the value of target observables

        :raises: NOCError
        """
        model, model_center, targets = args
        index = status + 1
        name = model.name

        print(50*'*')
        print(f'Model {name}')
        model.setup_model_params(parameters=parameters, settings=self.settings)
        model.params.update_params(model.age_model)

        # Empty output buffer
        sys.stdout.flush()

        if status > -1 and parameters[status].seismic:
            # We just copy the central model to compute seismic quantities with different seismic parameters
            # model = Model(model_center.name, self.setup, verbose=self.verbose)
            os.system(f"cp -r {model_center.name} {name} && rm -r {name}/freqs")
            model = Model(name, self.setup, verbose=self.verbose)
        else:
            model(update_com=False, debug=self.debug, log=True, verbose=self.verbose)

            if not model.finished:
                with open('stop') as f:
                    stop_cause = f.readline()
                print("Evolution did not complete: " + stop_cause)
                model.clean_links()
                return 0.0, True

            print(f"-> Computation of model {index} successfully finished")

        if self.settings['modes'] is None:
            return self.__get_outputs(model, parameters)

        # else, we calculate the eigenfrequencies
        oscprog = self.settings['modes']['oscprog']

        if self.verbose:
            print(f"Computing model frequencies at age {model.age_model/1e6} Myr with {oscprog}... ")

        if oscprog == 'pulse':
            model.tgec2pulse()
            model.run.run_pulse(log=True)
        # TODO: add adipls commands
        elif oscprog == 'adipls':
            data, aa = model.tgec2amdl(bv=True)
            # data, aa = model.tgec2amdl()

            if not model.write_amdl(f"{name}.amdl", data, aa):
                raise NOCError("Unable to build the amdl file")

            model.run.run_adipls()

        return self.__get_outputs(model, parameters)

    def print_results(self, msg):
        """
        Print final results of the NOC run
        :param msg: message to be written
        """
        text = msg
        text += f"\nChi2 = {self.chi2:8g}"
        text += f"\nReduced Chi2 = {self.chi2/float(self.ny):8g}\n"

        if self.nseismic > 0:
            chi2s = self.chi2
            for i in range(self.ntargs):
                chi2s -= (self.outputs[i] - self.y[i])**2/self.covar[i, i]
            text += f"\nSeismic Chi2 = {chi2s:8g}"
            text += f"\nReduced seismic Chi2 = {chi2s/float(self.ny - self.ntargs):8g}\n"
            text += f"\nNon-seismic Chi2 = {self.chi2 - chi2s:8g}"
            text += f"\nReduced non-seismic chi2 = {(self.chi2 - chi2s)/float(self.ntargs):8g}\n"

        text += "\nFinal parameters:\n"
        for p in self.parout:
            text += f"{p.name:10s} = {p.value:8g} +/- {p.sigma:8g}\n"

        text += "\nDistances to global constraints (name = model data sigma model-data): \n"
        for i in range(self.ntargs):
            text += f"{self.targets[i].name:10s} = {self.outputs[i]:8g} {self.y[i]:8g} {self.targets[i].sigma:8g} " \
                    f"{self.outputs[i] - self.y[i]:8g}\n"
        if self.nseismic > 0:
            text += "\nDistances to seismic targets (# = model data sigma model-data): \n"
            for i in range(self.ntargs, self.ny):
                text += f"{i - self.ntargs:3d} = {self.outputs[i]:8g} {self.y[i]:8g} {np.sqrt(self.covar[i,i])} " \
                        f"{self.outputs[i] - self.y[i]:8g}\n"

        if self.verbose:
            print(text)

        self.lgf.write(text)
        self.lgf.flush()

    def clean(self):
        """
        Remove temporary models
        """
        for model in self.model_refs['deriv'].values():
            os.system(f"rm -rf {model.name}")  # TODO: check the right path

    def __get_outputs(self, model, new_parameters):
        """
        Fetch properties of the evolution model corresponding to the target observables
        :param model: model to be analyzed
        :param new_parameters: set of modified tunable parameters

        :return: list of parameters of the model corresponding to the targets observables
        """
        outputs = np.zeros(self.ny)

        error = False
        lsep_target = get_instance(self.targets, 'name', 'largesep')

        i = 0
        for t in self.targets:
            if not t.is_input:
                t_name = t.name.lower()
                try:
                    outputs[i] = model.get_output(lsep_target, self.settings, t_name)
                except KeyError:
                    raise NOCError(f"Error in NOC/ComputeOptimal.__get_outputs: target named {t_name} not found")
                except NOCError as e:
                    print(e)
                    error = True

                print(f"{t_name:10s} = {outputs[i]:g}")
                i += 1

        if self.nseismic > 0:
            outputs[self.ny-self.nseismic:] = self.__get_seismic_outputs(model, new_parameters)

        return outputs, error

    def __get_seismic_outputs(self, model, new_parameters):
        """
        Fetch seismic properties of the model, in the correct range of frequencies
        :param model: computed model
        :param new_parameters: set of modified parameters

        :return: list of observable corresponding to the seismic constraints
        """
        new_setup = self.setup.copy()
        new_setup.parameters = new_parameters
        seismic_model = model.seismic.seismic_model(new_setup)
        ns = self.nseismic
        # print(f"seismic model = {seismic_model}")
        if seismic_model.size == ns:
            return seismic_model

        print(f"Problem with model {self.name}: ")
        print("Theoretical frequencies do not cover the observational range.")
        print(f"{ns} seismic constraints were expected while {seismic_model.size} were computed.")
        print("We consider zero values.")
        return 0.0
