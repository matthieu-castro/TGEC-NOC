import json

import numpy as np

from nocpkg.Parameter import Parameter
from nocpkg.Target import Target
from nocpkg.SeismicConstraints import SeismicConstraints
from nocpkg.utils import NOCError


class Setup:
    dict_se_laws = {
        'bg1': 'Ball & Gizon (2014) inverse', 'bg2': 'Ball & Gizon (2014) inverse+cubic',
        'lorentz3': 'Lorentzian', 'lorentz2': 'Lorentzian with two parameters',
        'lorentz': 'Lorentzian with two parameters', 'kb2008': 'Kjeldsen et Bedding (2008)'}
    dict_se_n = {'bg1': 1, 'bg2': 2, 'lorentz3': 3, 'lorentz2': 2, 'lorentz': 2, 'kb2008': 2}

    def __init__(self, json_file=None):

        self.parameters = []
        self.targets = []
        self.seismic_constraints = SeismicConstraints('')
        self.settings = {}
        self.calc_freqs = False
        self.lsep_target = None
        if json_file is not None:
            with open(json_file) as f:
                try:
                    self.data = json.load(f)
                except FileNotFoundError:
                    raise NOCError(f"JSON file {json_file} not found")
                except json.JSONDecodeError:
                    raise NOCError("Some error occurred while reading the JSON file. Please check it.")

    def __get_nparams(self):
        """
        Count the number of parameters defined in the JSON file
        """
        self.nparams = len(self.parameters)

    def __get_ntargs(self):
        """
        Count the number of targets defined in the JSON file except for targets defined as inputs.
        """
        self.ntargs = sum(not t.is_input for t in self.targets)

    def __get_nseismic(self):
        """
        Count the number of seismic constraints defined in the JSON file
        """

        self.nseismic = self.seismic_constraints.number

    def __get_levmar_settings(self, jsonlevmar):
        """
        Extract information relative to the Levenberg-Marquardt method from the JSON file
        :param jsonlevmar: Instance of JSON element from the 'levmar' field, read in the input JSON file.
        :return: dictionary of options
        """
        settings_levmar = {
            'maxiter': jsonlevmar.get('maxiter'),
            'ftol': jsonlevmar.get('ftol'),
            'chi2min': jsonlevmar.get('chi2min'),
            'autostep': jsonlevmar.get('autostep', 0),
            'cov_cdtnb_thr': jsonlevmar.get('cov_cdtnb_thr', 1e13),
            'hess_cdtnb_thr': jsonlevmar.get('hess_cdtnb_thr', 1e13)
        }

        return {'levmar': settings_levmar}

    def __read_seismic_constraints(self, jsonseismic_constraints):
        """
        Read seismic constraints from JSON file passed in parameter
        :return: SeismicConstraints object
        """
        filename = jsonseismic_constraints.get('file').strip()
        types = jsonseismic_constraints.get('types')
        matching = jsonseismic_constraints.get('matching', 'frequency')
        print(f"matching strategy: {matching}")
        lfirst = jsonseismic_constraints.get('lfirst', False)
        if lfirst:
            print("l degree is assumed to be given in the first column")

        return SeismicConstraints(filename, types, matching=matching, lfirst=lfirst)

    def __get_modes_settings(self, jsonmodes):
        """
        Extract information relative to the modes computation of the models from the JSON file
        :param jsonmodes: Instance of JSON element from the 'modes' field, read in the input JSON file.
        :return: dictionary of options:
            l: l degrees of the computed modes. If not specified, we use the l list of the frequency file.
            nmin: minimum n order of the computed modes. If not specified, we use the min n of the frequency file.
            nmax: maximum n order of the computed modes. If not specified, we use the max n of the frequency file.
            dn: step in n. If not specified, we use 1
        """
        l = jsonmodes.get('l')
        if l is None:
            l = np.unique(self.seismic_constraints.l)
        else:
            l = np.fromstring(l, sep=',', dtype=int)

        nmin = jsonmodes.get('nmin')
        if nmin is None:
            nmin = int(min(self.seismic_constraints.n))
        nmax = jsonmodes.get('nmax')
        if nmax is None:
            nmax = int(max(self.seismic_constraints.n))
        dn = jsonmodes.get('dn', 1)

        oscprog = jsonmodes.get('oscprog')
        if oscprog is None or oscprog.strip() not in ['adipls', 'pulse']:
            raise NOCError("You must specify a value for 'oscprog': 'adipls' or 'pulse'.")

        jsonsurface_effects = jsonmodes.get('surface_effects')
        if jsonsurface_effects is not None:
            surface_effects_settings = self.__get_surface_effects_settings(jsonsurface_effects)
        else:
            surface_effects_settings = None

        settings_modes = {
            'l':  l,
            'nmin': nmin,
            'nmax': nmax,
            'dn': dn,
            'oscprog': oscprog.strip(),
            'surface_effects': surface_effects_settings
        }

        return settings_modes

    def __get_surface_effects_settings(self, jsonsurface_effects):
        """
        Extract information relative to the frequency correction of surface effects from the JSON file
        :param jsonsurface_effects: Instance of JSON element 'surface effects' from the 'modes' field,
                                    read in the input JSON file.
        :return: dictionary of options
            formula: surface effects correction function
            parameters: parameters of the function
            numax: numax value
            prescription: surface effects prescription
        """
        formula = jsonsurface_effects.get('formula')
        params = jsonsurface_effects.get('parameters')
        if params is not None:
            params = np.array(params, dtype=float)
        numax = jsonsurface_effects.get('numax')
        if numax is not None:
            numax = float(numax)

        if formula is None:
            raise NOCError("Unspecified formula for the surface effects.")
        else:
            formula = formula.lower()
            if formula not in self.dict_se_laws:
                raise NOCError(f"{formula}: unhandled formula for surface effects")

        surface_effects_settings = {
            'formula': formula,
            'parameters': params,
            'numax': numax,
            'prescription': False
        }

        # Check if values of surface effects parameters have been properly provided
        # If no parameters field have been provided, we consider the surface effects correction parameters as
        # tunable, and they must be provided in the configuration parameters
        if params is None:
            param_names = np.array([getattr(p, "name") for p in self.parameters])
            error = False
            if formula == 'bg1':
                if 'se_a' not in param_names:
                    error = True
            elif formula == 'bg2' or formula == 'lorentz2' or formula == 'lorentz' or formula == 'kb2008':
                if 'se_a' not in param_names or 'se_b' not in param_names:
                    error = True
            elif formula == 'lorentz3':
                if 'se_a' not in param_names or 'se_b' not in param_names or 'se_c' not in param_names:
                    error = True
            if error:
                raise NOCError("You did not specify the 'parameters' field inside 'surface_effects'.\n \
                               You must provide 'se_a' or 'se_b' or 'se_c', depending of the chosen formula,\n \
                               as a tunable parameter, or allow to derive a value from prescription by setting \n \
                               0 inside 'parameters'.")
        else:
            if params.size != self.dict_se_n[formula]:
                nb = int(self.dict_se_n[formula])
                raise NOCError(f"{nb} parameter{' is' if nb == 1 else 's are'} expected with the "
                               f"'{self.dict_se_laws[formula]}' function.")
            else:
                print(f"Modeling the surface effects with '{self.dict_se_laws[formula]}' formula.")
                surface_effects_settings['prescription'] = np.all(params == 0)
                if not surface_effects_settings['prescription'] and np.any(params == 0):
                    raise NOCError("Please set all surface effects parameters to 0 or to a value.")
                if surface_effects_settings['prescription']:
                    print("Surface effects coefficients computed with prescriptions taken from Manchon et al. (2018)")

        return surface_effects_settings

    def __get_model_settings(self, jsonmodels):
        """
        Extract information relative to the initialization of the TGEC model computation from the JSON file
        :param jsonmodels: Instance of JSON element 'models' from the 'settings' field,
                                    read in the input JSON file.
        :return: dictionary of options
        """
        start = jsonmodels.get('start', 'zams').strip().lower()  # Age of the model starting, zams or pms
        dy_dz = jsonmodels.get('dy_dz')  # Galactic variation of Y relative to Z
        yp = jsonmodels.get('yp')  # Primitive Y
        zp = jsonmodels.get('zp')  # Primitive Z
        retry = jsonmodels.get('retry', 5)

        if start not in ['zams', 'pms']:
            raise NOCError("Model start can only take the value 'zams' or 'pms'.")

        return {
            'dy_dz': float(dy_dz),
            'yp': float(yp),
            'zp': float(zp),
            'start': start,
            'retry': int(retry)
        }

    def copy(self):
        """
        Return a copy of Setup object
        :return: new instance of Setup object
        """
        clas = self.__class__
        new_setup = clas.__new__(clas)
        new_setup.__dict__.update(self.__dict__)
        return new_setup

    def read_setup(self):
        """
        Read input JSON file and extract setup parameters
        """
        jsonconfig = self.data.get('config')
        # process parameters #
        jsonparameters = jsonconfig.get('parameter')
        self.parameters = [Parameter(elem) for elem in jsonparameters]
        # If age is not in Parameters, we add it with the solar value
        if 'age' not in [p.name for p in self.parameters]:
            self.parameters.append(Parameter({'name': 'age', 'value': float(4.57e3), 'step': float(20.),
                                              'rate': float(5.), 'bounds': [float(20.), float(10000.)]}))
        self.__get_nparams()

        # process targets #
        jsontargets = jsonconfig.get('target')
        self.targets = [Target(elem) for elem in jsontargets]
        # If numax or largesep are in Targets, we must calculate the frequencies of the model
        # targets_name = [t.name for t in self.targets]
        # if 'largesep' in [t.name for t in self.targets]:
        #     self.calc_freqs = True
        for t in self.targets:
            if t.name == 'largesep':
                self.calc_freqs = True
                self.lsep_target = t.value
        self.__get_ntargs()  # Calculate number of targets

        # process settings #
        jsonsettings = jsonconfig.get('settings')
        # process Levenberg-Marquardt method settings
        jsonlevmar = jsonsettings.get('levmar')
        if jsonlevmar is not None:
            settings_method = self.__get_levmar_settings(jsonlevmar)
        else:
            raise NOCError("You must provide settings for the Levenberg-Marquardt algorithm")

        # process seismic constraints
        jsonseismic_constraints = jsonconfig.get('seismic_constraints')
        if jsonseismic_constraints is not None:
            self.calc_freqs = True
            self.seismic_constraints = self.__read_seismic_constraints(jsonseismic_constraints)
        else:
            self.seismic_constraints = SeismicConstraints('')

        self.__get_nseismic()  # Calculate number of seismic constraints

        # process mode computation options
        jsonmodes = jsonsettings.get('modes')
        if jsonmodes is not None:
            settings_modes = self.__get_modes_settings(jsonmodes)
        else:
            settings_modes = None
            if 'largesep' in np.array([getattr(t, 'name') for t in self.targets]):
                raise NOCError("You specified seismic target but did not provide mode settings.")

        if self.nseismic and settings_modes is None:
            raise NOCError("You have given seismic constraints, you must fill the 'modes' field.")

        # process model settings
        jsonmodels = jsonsettings.get('models')
        settings_models = self.__get_model_settings(jsonmodels) if jsonmodels is not None else None

        self.settings = {
            'method': settings_method,
            'modes': settings_modes,
            'models': settings_models
        }
