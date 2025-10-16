import multiprocessing
import os

import numpy as np

from nocpkg.utils import NOCError

multiproc = False


class LevMar:

    def __init__(self, setup, name, y, sigma, covar, y_name, log_file):
        """
        Class that implements the Levenberg-Marquardt algorithm
        :param setup: optimization setup read from the JSON file
        :param name: name of the model
        :param y: list of target values to be reached by optimization
        :param sigma: standard error on classical (non-seismic) target values
        :param covar: covariance matrix
        :param y_name: name of classical target values
        :param log_file: log file
        """
        self.setup = setup
        self.parameters = setup.parameters
        self.nparams = setup.nparams

        self.targets = setup.targets
        self.y = y
        self.y_name = y_name
        self.ny = y.size
        self.ns = setup.nseismic
        self.nt = setup.ntargs

        self.covar = covar
        self.sigma = sigma
        self.eps = 1e-20

        self.name = name
        self.lgf = log_file

        # Initialization of computation
        self.W = None
        self.new_param = None
        self.jacob = []
        self.jacobn = []
        self.init_optimal = []
        self.current_optimal = []
        self.next_optimal = []

        # Initialization of multiprocessing
        self.nproc = 0
        self.proc_index = {}
        self.processes, self.parents, self.childs = [[] for _ in range(3)]

    def __call__(self, func, levmar_args, verbose=True, ftol=1e-3, maxiter=30, chi2min=1e-4,
                 lamb0=1e-4, cov_cdtnb_thr=1e13, hess_cdtnb_thr=1e13):
        """
        Execute the Levenberg-Marquardt algorithm
        :param func: function that computes the model for which we are searching for the optimal parameters
        :param levmar_args: arguments of the function, tuple (model_refs, targets)
        :param verbose: print additional information (default=true)
        :param ftol: minimum variation of chi2 allowed, below which we consider that the optimization
                     has reached a local minimum (default=1e-3)
        :param maxiter: maximum number of iteration (default=30)
        :param chi2min: chi2 threshold below which the solution is accepted (default=1e-4)
        :param lamb0: initial damping parameter (default=1e-4)
        :param cov_cdtnb_thr: threshold for the condition number associated with the covariance matrix (default=1e13)
        :param hess_cdtnb_thr: threshold for the condition number associated with the Hessian matrix (default=1e13)

        :return: tuple composed of the best chi2, the best parameter, the best observables, the number of iterations,
                an error flag and the text to be written in the log file

        :raises: NOCError: cannot invert the Hessian matrix due to a null eigenvalue
        :raises: NOCError: unable to compute a model
        """
        self.func = func
        self.verbose = verbose
        self.maxiter = maxiter
        self.ftol = ftol
        self.chi2min = chi2min
        self.lamb = lamb0
        self.cov_cdtnb_thr = cov_cdtnb_thr
        self.hess_cdtnb_thr = hess_cdtnb_thr

        self.print_init()
        self.levmar_init(levmar_args)

        continued = 1
        it = 0
        msg = ''
        error = False

        chi2i = self.chi2(self.current_optimal, self.y, self.W)
        chi2n = chi2i

        self.__organize_files(levmar_args[0])

        text = self.print_result(chi2i)
        if self.verbose:
            print(text)
        self.lgf.write(text)
        self.lgf.flush()
        self.next_optimal = self.current_optimal

        while continued:
            text = '\n' + 50*'#' + '\n\n'
            text += f"Iteration #{it}\n"

            alpha, beta = self.levmar_coef(self.current_optimal, self.nparams, self.jacob, self.lamb)
            text += "Hessian matrix:\n"
            text += self.hessian2str(alpha, self.new_param)
            # Singular value decomposition
            U, s, V = np.linalg.svd(alpha)
            text += "\ns = [" + " ".join('{:>10}'.format('{:>8g}'.format(i)) for i in s) + "]\n"
            # text += f"epsilon = {self.eps}\n"

            if min(np.abs(s)) > self.eps:
                cdtn = max(np.abs(s))/min(np.abs(s))  # condition number
                text += f"\nHessian matrix conditioning number: {cdtn:8g}\n"
            else:
                text_err = "Error in NOC/levmar: there is a null eigenvalue in the Hessian matrix, " \
                           "it cannot be inverted"
                text += text_err
                if self.verbose:
                    print(text)
                self.lgf.write(text)
                raise NOCError(text_err)

            # if self.verbose:
            #     print(text)

            alpha = np.linalg.inv(np.matrix(alpha))
            dparam = np.array((alpha*np.matrix(beta.reshape((self.nparams, 1)))).flatten())[0]
            old_param = np.array([p.value for p in self.new_param])

            for i, p in enumerate(self.parameters):
                self.new_param[i].value = max(min(p.value + np.sign(dparam[i])*min(np.abs(dparam[i]),
                                                  np.abs(p.value*p.rate/100.0)), p.bounds[1]), p.bounds[0])

            text += "\nNew parameters:\n"
            text += "".join(str(p) + f" +/- {p.step}\n" for p in self.new_param)
            diff_param = np.array([p.value for p in self.new_param]) - old_param
            text += f"Change (new-old): {diff_param}\n"

            if self.verbose:
                print(text)
            self.lgf.write(text)
            self.lgf.flush()

            error, self.next_optimal, self.jacobn = self.levmar_step(self.next_optimal, self.new_param, levmar_args,
                                                                     self.jacobn)
            if error:
                text = f"Error in NOC/levmar: unable to compute the model at iteration #{it}. Exit."
                self.lgf.write(text)
                raise NOCError(text)

            chi2n = self.chi2(self.next_optimal, self.y, self.W)
            text = self.print_result(chi2i, new_chi2=chi2n)

            # Relative variation of chi2
            dchi2 = (chi2n - chi2i)/chi2i
            # We check if we continue the iterations or if the optimal model has been found
            continued = (it < self.maxiter) & (abs(dchi2) > self.ftol) & (chi2n > self.chi2min)
            if chi2n >= chi2i:
                self.lamb = self.lamb*10.0
                text += "Leaving the parameters unchanged:\n"
            else:
                self.lamb = 0.25*self.lamb
                for i in range(self.nparams):
                    self.parameters[i].value = self.new_param[i].value
                chi2i = chi2n
                self.current_optimal = self.next_optimal
                self.jacob = self.jacobn
                text += "Adopted parameters:\n"
                text += '\n'.join(str(p) for p in self.parameters) + '\n'

            self.__organize_files(levmar_args[0])

            text += f"chi2 = {chi2i:8g}; dchi2/chi2 = {dchi2:8g}; lambda = {self.lamb:8g}\n"
            if self.verbose:
                print(text)
            self.lgf.write(text)
            self.lgf.flush()
            it += 1

        text = '\n' + 50*'#' + '\n'

        text_end = "Calculation stopped because "
        if it > self.maxiter:
            text_end += f"the number of iterations exceeded {self.maxiter}.\n"
        if abs(dchi2) < self.ftol:
            text_end += f"the relative variation of the chi2 became lower than {self.ftol:g}.\n"
        if chi2n <= self.chi2min:
            text_end += f"chi2 became lower than {self.chi2min:g}.\n"
        msg += text_end
        text += text_end
        text += f"chi2 = {chi2i:g}\n"
        text += f"lambda = {self.lamb:g}\n"
        text += f"dchi2/chi2 = {dchi2:g}\n"
        text += f"number of iterations = {it}\n\n"

        self.lamb = 0.0
        alpha, beta = self.levmar_coef(self.current_optimal, self.nparams, self.jacob, self.lamb)
        text += "Hessian matrix:\n"
        text += self.hessian2str(alpha, self.parameters)

        # Singular value decomposition
        U, s, V = np.linalg.svd(alpha)
        text += "\ns = [" + " ".join('{:>10}'.format('{:>8g}'.format(i)) for i in s) + "]\n"

        print(text)
        text = ""

        if min(np.abs(s)) > self.eps:
            cdtn = max(np.abs(s))/min(np.abs(s)) # condition number
            text += f"Final Hessian matrix conditioning number = {cdtn:8g}\n"
        else:
            text_err = "Error in NOC/levmar: there is a null eigenvalue in the Hessian matrix, it cannot be inverted"
            text += text_err
            self.lgf.write(text)
            raise NOCError(text_err)
        # We may need to truncate the matrix
        limit = max(s)/self.hess_cdtnb_thr
        # Diagonal matrix that contain the singular values (eigenvalues)
        Si = np.zeros((self.nparams, self.nparams))
        truncated = [s[i] < limit for i in range(self.nparams)]
        # We keep only eigenvalues s > limit
        for i in range(self.nparams):
            Si[i, i] = 0.0 if truncated[i] else 1.0/s[i]

        if any(truncated):
            text_warn = "\nWARNING ! the final Hessian matrix was truncated using SVD\n\n"
            msg += text_warn
            text += text_warn

        # Inverse matrix A^-1 = V^t S^-1 U^t
        # alpha = np.dot(np.transpose(V), np.dot(np.linalg.inv(Si), np.transpose(U)))
        alpha = np.dot(np.transpose(V), np.dot(Si, np.transpose(U)))
        text += self.hessian2str(alpha, self.parameters)
        # Calculation of the errors on optimized parameters
        for i in range(self.nparams):
            self.parameters[i].sigma = np.sqrt(alpha[i, i])

        text += "\nFinal values:\n"
        text += '\n'.join(f"{p.name:10} = {p.value:8f} +/- {p.sigma:8f}" for p in self.parameters) + '\n'
        text += "\n---------- End optimization process ----------\n\n"

        if self.verbose:
            print(text)
        self.lgf.write(text)

        return chi2i, self.parameters, self.current_optimal, it, error, msg

    def print_init(self):
        """
        Print information about initial state of the calculation and settings
        """
        text = "\n---------- Levenberg-Marquardt method started ----------\n"
        text += "Initial values:\n"
        for p in self.parameters:
            text += str(p) + '\n'
        text += f"lambda     = {self.lamb:8g}\n"
        
        if self.verbose:
            print(text)
        self.lgf.write(text)

    def print_result(self, best_chi2, new_chi2=None):
        """
        Print a summary of the result of each iteration of the algorithm
        :param best_chi2: best chi2 value
        :param new_chi2: new chi2 value (default=None)
        :return: text to be printed out and written in log file
        """
        text = "\n" + 50*'#' + "\n"
        if new_chi2 is None:
            text += "\nInitial model:\n"
            text += f"chi2 = {best_chi2:8g}\n"
        else:
            text += f"New chi2: {new_chi2:8g}\n"
            text += f"Best chi2: {best_chi2:8g}\n"

        # Print new parameters values
        text += "\nParameters:\n"
        if new_chi2 is None:
            text += "\n".join(str(p) for p in self.parameters)
        else:
            text += "\n".join(str(p) for p in self.new_param)
        text += '\n'

        # Print target values and best optimal values
        t_names = [t.name for t in self.targets if not t.is_input]
        text += f"Constraints: {t_names}\n"
        if new_chi2 is None:
            ym = self.y - self.current_optimal
            text += f"Model value: {self.current_optimal}\n"
        else:
            ym = self.y - self.next_optimal
            text += f"Model value: {self.next_optimal}\n"
        text += f"Target values: {self.y}\n"
        text += f"Sigma: {self.sigma}\n"
        text += f"Distances (target - model): {ym}\n"

        return text

    def hessian2str(self, alpha, parameters):
        """
        Convert the Hessian matrix to printable string
        :param alpha: coefficients of the Hessian matrix
        :param parameters: list of parameters

        :return: string representing the Hessian matrix
        """
        param_names = np.array([p.name for p in parameters])
        text = "            "
        text += " ".join('{:>12}'.format('{:>12}'.format(name)) for name in param_names)
        text += "\n"
        for i in range(self.nparams):
            text += f"{parameters[i].name:>12}"
            text += " ".join('{:>12}'.format('{:>8g}'.format(alpha)) for alpha in alpha[i, :])
            text += "\n"
        return text

    def levmar_init(self, levmar_args):
        """
        Initialize some quantities from the given arguments and compute the Jacobian matrix for the 1st iteration
        :param levmar_args: tuple containing the list of targets
        :return: message written in the log file
        """
        text = ""
        # Singular value decomposition (SVD) A = U S V
        (U, s, V) = np.linalg.svd(self.covar)
        # Diagonal matrix containing the singular values (eigenvalues)
        Si = np.zeros((self.ny, self.ny))
        # Allow p_pertw to be optimized
        self.eps = 1e-80 if 'p_pertw' in [p.name for p in self.parameters] else 1e-20

        msg = ""

        if min(np.abs(s)) == 0:
            text_err = "Error in NOC/LevMar: there is a null eigenvalue in the covariant matrix." \
                       "It cannot be inverted."
            text += text_err
            self.lgf.write(text)
            self.lgf.close()
            raise NOCError(text_err)
        else:
            # Condition number
            cdt = max(np.abs(s))/min(np.abs(s))
            text += f"Covariance matrix conditioning number = {cdt:8g}\n"

        # We may need to truncate the matrix
        limit = max(s)/self.cov_cdtnb_thr
        truncated = [s[i] < limit for i in range(self.ny)]
        for i in range(self.ny):
            # We keep only the eigenvalues s > limit
            Si[i, i] = 0.0 if truncated[i] else 1.0/s[i]

        if any(truncated):
            text_warn = "\n\n WARNING: the covariance matrix is truncated using SVD !!\n\n"
            msg += text_warn
            text += text_warn

        # Inverse matrix: A^(-1) = V^T S^(-1) U^T
        self.W = np.dot(np.transpose(V), np.dot(Si, np.transpose(U)))

        if self.verbose:
            print(text)
        self.lgf.write(text)

        text = ""
        self.new_param = [p.copy() for p in self.parameters]

        # Initialization of Jacobian matrix
        self.jacob, self.jacobn = [np.zeros((self.nparams, self.ny)) for _ in range(2)]

        # Initialization of results structure
        self.init_optimal, self.current_optimal, self.next_optimal = [np.zeros(self.ny) for _ in range(3)]

        # Execute one step of the Levenberg-Marquardt algorithm
        error, self.init_optimal, self.jacob = self.levmar_step(self.init_optimal, self.parameters, levmar_args, self.jacob)
        self.current_optimal = self.init_optimal

        if error:
            text_err = "Error in NOC/LevMar: unable to compute the first model. Forced exit"
            text += text_err
            if self.verbose:
                print(text)
            self.lgf.write(text)
            raise NOCError(text_err)

        return msg

    def levmar_step(self, center_model, parameters, levmar_args, jacob):
        """
        Execute one step of the Levenberg-Marquardt algorithm and find the next optimal model
        :param center_model: Values of the central model (not shifted)
        :param parameters: list of Parameter instances
        :param levmar_args: arguments of the function that computes the model
        :param jacob: Jacobian matrix
            $\partial T_j/\partial P_i = (T_j(P_i+\delta p_i) - T_j(p_i)) / (\delta p_i).$

        :return: tuple containing an error flag, the central values and the jacobian
        """
        error = False

        # Calculation of the central model and the shifted models
        if multiproc:
            self.__add_process(0, self.func_shell, parameters, levmar_args)
            self.nproc += 1
            for i, p in enumerate(self.parameters):
                if not p.seismic:
                    self.__add_process(i + 1, self.func_deriv_shell, parameters, levmar_args)
                    self.nproc += 1

            shift_model = np.ones((self.ny, self.nproc))
            steps = np.zeros(self.nproc)
            res = []

            if any([proc.exitcode for proc in self.processes]):
                raise NOCError("LevMar.levmar_step failed")

            for i, parent in enumerate(self.parents):
                res.append(parent.recv())
                center_model[:] = res[0][0][:]
                if res[-1][2]:
                    return True, [], []

            i = 0
            for p in self.parameters:
                func_args_tmp = self.__get_func_args_tmp(levmar_args, p.name)
                shift, steps[i], error = self.func_deriv(self.func, parameters, func_args_tmp, i)
                if error:
                    return True, [], []
                self.__reorder_outputs(shift, shift_model[:, i])
                i += 1

        else:
            func_args_tmp = levmar_args[0]['center'], levmar_args[0]['center'], levmar_args[1]
            center_model[:], error = self.func(parameters, func_args_tmp, -1)
            if error:
                return True, [], []
            shift_model = np.ones((self.ny, self.nparams))
            steps = np.zeros(self.nparams)
            for i, p in enumerate(self.parameters):
                func_args_tmp = self.__get_func_args_tmp(levmar_args, p.name)
                shift_model[:, i], steps[i], error = self.func_deriv(self.func, parameters, func_args_tmp, i)
                if error:
                    return True, [], []

        # Calculation of the Jacobian matrix:
        j = 0
        for i, p in enumerate(self.parameters):
            mask_nan = np.argwhere(np.isfinite(shift_model[:, j]))
            jacob_tmp = (shift_model[:, j] - center_model[:])/steps[i]
            # # if p.name == 'age':
            # print(f"p = {p.name}: jacob_tmp = {jacob_tmp}, shift_model[:, {j}] = {shift_model[:, j]}, "
            #       f"center_model = {center_model[:]}, steps = {steps[i]}")
            self.__reorder_outputs(jacob_tmp[mask_nan].flatten(), jacob[i, :])
            # self.__reorder_outputs(np.compress(mask_nan, jacob_tmp), jacob[i, :])
            j += 1

        return error, center_model, jacob

    def levmar_coef(self, y_model, nparams, jacob, lamb):
        """
        Compute coefficients alpha and beta of the Levenberg-Marquardt algorithm
        :param y_model: value of the observable from the model
        :param nparams: number of tunable parameters, size of y_model
        :param jacob: Jacobian matrix of the system
        :param lamb: damping parameter lambda

        :return: alpha and beta
        """
        alpha = np.zeros((nparams, nparams))
        beta = np.zeros(nparams)

        for i in range(nparams):
            tmp = 0.0
            for l in range(self.ny):
                for m in range(self.ny):
                    tmp += 0.5*self.W[l, m] * ((self.y[l] - y_model[l])*jacob[i, l] +
                                               (self.y[m] - y_model[m])*jacob[i, m])
            beta[i] = tmp
            for j in range(i, nparams):
                tmp = 0.0
                for l in range(self.ny):
                    for m in range(self.ny):
                        tmp += 0.5*self.W[l, m] * (jacob[i, l]*jacob[j, m] + jacob[i, m]*jacob[j, l])
                        # if tmp == 0 and l == m:
                        #      print(f"W[{l}, {m}] = {self.W[l, m]}, jacob[{i}, {l}] = {jacob[i, l]}"
                        #            f", jacob[{j}, {m}] = {jacob[j, m]}, jacob[{i}, {m}] = {jacob[i, m]}"
                        #            f", jacob[{j}, {l}] = {jacob[j, l]}")
                alpha[i, j] = tmp * (1.0 + lamb*(i == j))

        for i in range(1, nparams):
            for j in range(i):
                alpha[i, j] = alpha[j, i]

        return alpha, beta

    def func_deriv(self, func, parameters, args, index):
        """
        Computes a model with a shifted parameter
        :param func: function computing a TGEC model and returning the value of targets
        :param parameters: list of Parameter instances
        :param args: additional arguments passed to the function, tuple (model_refs, targets)
        :param index: index corresponding to the shifted parameter

        :return: tuple (shifted model, value of the step, error)
        :raises: NOCError if model computation does not finish
        """
        error = True
        param_copy = [p.copy() for p in parameters]
        iter = 0

        setting_models = self.setup.settings['models']
        max_iter = 3 if setting_models is None else setting_models['retry']

        while error and iter < max_iter:
            for i in range(len(param_copy)):
                param_copy[i].value = parameters[i].value
            step = parameters[index].step/(1.0 + iter)
            param_copy[index].value = parameters[index].value + step
            y_model, error = func(param_copy, args, index)
            iter += 1
            if error and iter < max_iter:
                print(f"Error in NOC/LevMar: unable to compute the derivative "
                      f"(model #{index+1} with respect to the parameter {param_copy[index].name}, "
                      f"we reduce the step by a factor {iter+1}")
        if error:
            raise NOCError(f"Error in NOC/LevMar: unable to compute the derivative "
                           f"(model #{index+1} with respect to the parameter {param_copy[index].name} "
                           f"after {iter} iterations")

        return y_model, step, error

    def func_shell(self, func, parameters, args, index, pipe):
        """
        Wrapper used to run the function self.func in parallel
        :param func: function to be evaluated
        :param parameters: list of Parameter instances
        :param args: additional arguments passed to the function, tuple (model_refs, targets)
        :param index: index corresponding to the shifted parameter (index=-1 for central model)
        :param pipe: pipe that connect two connection objects
        """
        (model, error) = func(parameters, args, index)
        pipe.send((model, 0.0, error))
        pipe.close()

    def func_deriv_shell(self, func, parameters, args, index, pipe):
        """
        Wrapper used to run the function self.func in parallel, to compute derivative with respect
        to a given parameter
        :param func: function to be evaluated
        :param parameters: list of Parameter instances
        :param args: additional arguments passed to the function, tuple (model_refs, targets)
        :param index: index corresponding to the shifted parameter (index=-1 for central model)
        :param pipe: pipe that connect two connection objects
        """
        (model, step, error) = self.func_deriv(func, parameters, args, index)
        pipe.send((model, step, error))
        pipe.close()

    def chi2(self, y_model, y, W):
        """
        Compute the chi2 value
        :param y_model: values of targets from the model
        :param y: target values
        :param W: inverse of the covariance matrix

        :return: chi2 value
        """
        return (y - y_model).dot(W.dot((y - y_model)))

    def __add_process(self, iparam, target_func, parameters, levmar_args):
        """
        For each tunable parameter, add the necessary processes, the computation of a TGEC model given the input
        parameters, to the queue
        :param iparam: index of the parameter. If iparam=0, it corresponds to the central model
        :param target_func: function called to compute the model
        :param parameters: list of tunable parameters defined in the JSON file
        :param levmar_args: arguments passed to the __call__ method
        """
        if iparam == 0:
            model = levmar_args[0]['center']
            pname = 'center'
        else:
            pname = parameters[iparam-1].name
            model = levmar_args[0]['deriv'][pname]

        model_ref = levmar_args[0]['center']
        func_args_tmp = (model, model_ref, levmar_args[1])

        # We create a pipe of processes to launch a parallel computation of models
        pipe = multiprocessing.Pipe()
        self.proc_index[f"{parameters[iparam-1].name}"] = self.nproc
        self.parents.append(pipe[0])
        self.childs.append(pipe[1])
        self.processes.append(multiprocessing.Process(target=target_func, args=(self.func, parameters, func_args_tmp,
                                                                                iparam-1, self.childs[-1])))
        self.processes[-1].start()

    def __get_func_args_tmp(self, levmar_args, pname):
        """
        Build the function arguments for func_deriv
        :param levmar_args: tuple containing the arguments for the LevMar method
        :param pname: name of the parameter

        :return: tuple (shifted model, central model, targets)
        """
        model = levmar_args[0]['deriv'][pname]
        model_ref = levmar_args[0]['center']

        return model, model_ref, levmar_args[1]

    def __reorder_outputs(self, array_in, array_out):
        """
        Reorder output values stored in an array, so that each element corresponds to the right target
        :param array_in: input data
        :param array_out: output data
        """

        j = 0
        for i, t in enumerate(self.targets):
            if not t.is_input:
                array_out[i] = array_in[j]
                j += 1

        if self.ns > 0:
            array_out[self.nt:] = array_in[j:]

    def __organize_files(self, model_refs):
        """
        Rename the file of the optimal model and remove the derivative models
        """
        # pass
        os.system(f"rm -rf {model_refs['center'].name}_opt")
        os.system(f"mv {model_refs['center'].name} {model_refs['center'].name}_opt")
        for model in model_refs['deriv'].values():
            os.system(f"rm -rf {model.name}")
        # TODO: see how to write this function
